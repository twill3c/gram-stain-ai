"""Grad-CAM に対する検査(TEST_SPEC T-210・T-256〜T-259)。

期待値の出所:
  - 健全性の閾値(順位相関の中央値 < 0.5): Adebayo et al., "Sanity Checks for Saliency Maps",
    NeurIPS 2018 の model randomization test。重みを乱数化しても地図が変わらない手法は、
    モデルではなく入力の輪郭を写している。閾値 0.5 は測る前に置いた(SPEC 5.0 の G-10)
  - 本体の判定: SPEC G-10(loop_008 で**測る前に**引き直した)

**実装より先に書く**(HC-253)。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def _require(path: Path, how: str) -> Path:
    if path.exists():
        return path
    if not list((ROOT / "dataset" / "raw").glob("*.zip")):
        pytest.skip("dataset/raw に素材が無い環境")
    pytest.fail(f"{path.relative_to(ROOT)} が無い。素材はあるので生成できるはず: {how}")


@pytest.fixture(scope="module")
def cam() -> dict:
    return json.loads(_require(REPORTS / "gradcam.json", "python -m ml.gradcam").read_text(encoding="utf-8"))


@pytest.mark.unit
def test_t256_gradcam_matches_a_hand_computed_map() -> None:
    """T-256 — Grad-CAM の実装が、定義から手で計算した値と一致する。

    小さな決定論的ネット(畳み込み 1 層 + 全域平均 + 線形)では、
    クラス c の Grad-CAM は解析的に書ける:
      dy_c / dA_k(i,j) = W[c,k] / (H*W)   ← 全域平均の勾配は空間で一様
      alpha_k = W[c,k] / (H*W)
      CAM = ReLU( sum_k alpha_k * A_k )
    hook で取った勾配から作った地図が、これと一致しなければ実装が誤っている。
    """
    import torch
    from torch import nn

    from ml.gradcam import grad_cam

    torch.manual_seed(0)

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, padding=1)
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            a = self.conv(x)
            return self.fc(a.mean(dim=(2, 3)))

    net = Tiny().eval()
    x = torch.randn(1, 3, 6, 6)
    got = grad_cam(net, net.conv, x, target_class=1, out_size=None)

    with torch.no_grad():
        a = net.conv(x)[0]                      # (4, 6, 6)
    h, w = a.shape[1:]
    alpha = net.fc.weight[1].detach() / (h * w)  # (4,)
    want = torch.relu((alpha[:, None, None] * a).sum(0)).numpy()

    assert got.shape == want.shape
    assert np.abs(got - want).max() < 1e-6, f"最大差 {np.abs(got - want).max():.3e}"


@pytest.mark.integration
def test_t257_randomization_sanity_holds(cam: dict) -> None:
    """T-257 / G-10(a) — 重みを乱数化すると地図が変わる。

    変わらないなら、その手法はモデルではなく入力の輪郭を写しているだけで、
    本体の判定(T-258)は意味を持たない。
    """
    s = cam["sanity"]
    assert s["n_images"] > 0
    assert s["median_spearman_trained_vs_random"] < 0.5, (
        f"学習済みと乱数化の CAM の順位相関の中央値が {s['median_spearman_trained_vs_random']:.3f}。"
        "手法がモデルを見ていない疑いがある"
    )


@pytest.mark.integration
def test_t258_verdict_follows_the_preregistered_rule(cam: dict) -> None:
    """T-258 / G-10(b) — 判定が、測る前に書いた規則から機械で出ている。

    規則: 学習済みモデルの集中度比の 95% 区間の下限が、乱数化モデルの区間の上限を上回る。
    判定をこちらの言葉で書かない(T-235 と同じ趣旨)。
    """
    t = cam["trained"]
    r = cam["randomized"]
    expected = t["ci_low"] > r["ci_high"]
    assert cam["verdict"]["looks_at_stain_beyond_architecture"] == expected
    assert cam["verdict"]["sanity_passed"] == (cam["sanity"]["median_spearman_trained_vs_random"] < 0.5)
    for d in (t, r):
        assert d["ci_low"] <= d["mean_ratio"] <= d["ci_high"]


@pytest.mark.integration
def test_t259_degenerate_maps_are_counted_not_hidden(cam: dict) -> None:
    """T-259 — 全面 0 の地図(比が定義できない画像)を黙って捨てていない。

    乱数化モデルは ReLU が死んで地図が全面 0 になりうる。捨てた枚数を記録していなければ、
    残った画像だけの平均が母集団の平均に見えてしまう。
    """
    for key in ("trained", "randomized"):
        d = cam[key]
        assert "n_degenerate" in d, f"{key} に全面 0 の枚数が記録されていない"
        assert d["n_used"] + d["n_degenerate"] == cam["n_images"], (
            f"{key}: 使った {d['n_used']} + 捨てた {d['n_degenerate']} ≠ 全 {cam['n_images']}"
        )


@pytest.mark.integration
def test_t210_stain_definition_matches_the_controls(cam: dict) -> None:
    """T-210 / G-10 — 「染まった画素」の定義が色相ベースラインと同じ。

    CAM と対照が別の『菌体』を指していると、両者を並べて語れない。
    """
    controls = json.loads((REPORTS / "controls.json").read_text(encoding="utf-8"))
    assert cam["stain_quantile"] == controls["stain_quantile"]
