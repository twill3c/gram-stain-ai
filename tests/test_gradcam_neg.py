"""見ていない陰性球菌でモデルが見ている場所を測る道具に対する検査(TEST_SPEC T-301〜T-304)。

期待値の出所:
  - 集中度比の定義(SPEC §3.20): (マスクに乗った地図の割合) / (マスクの面積の割合)
  - Grad-CAM は対象層の解像度の地図を返す(Selvaraju et al. 2017)
  - 判定規則: SPEC §3.20(測る前に書いた)

**実装より先に書く。**
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


@pytest.mark.unit
def test_t301_concentration_ratio_matches_hand_calculation() -> None:
    """T-301 / §3.20 — 一様な地図で 1、マスクだけに乗った地図で 1/面積割合、全面 0 は None。"""
    from ml.gradcam import concentration

    mask = np.zeros((28, 28), dtype=bool)
    mask[4:11, 5:19] = True                       # 面積割合 = 7*14/784 = 0.125
    frac = mask.mean()
    assert abs(frac - 0.125) < 1e-12
    uniform = np.ones((28, 28))
    only = mask.astype(float)
    assert abs(concentration(uniform, mask) - 1.0) < 1e-9
    assert abs(concentration(only, mask) - 1.0 / frac) / (1.0 / frac) < 1e-9
    assert concentration(np.zeros((28, 28)), mask) is None


@pytest.mark.unit
def test_t302_grad_cam_returns_layer_resolution() -> None:
    """T-302 / §3.20 — 任意の層で、その層の解像度の地図を返す。out_size を渡すと 224。"""
    import torch
    from torch import nn

    from ml.gradcam import grad_cam

    torch.manual_seed(0)
    # 224 → 28 → 14 → 7 と解像度が落ちる小さなネット
    feats = nn.Sequential(
        nn.Conv2d(3, 4, 3, stride=8, padding=1), nn.ReLU(),   # 28
        nn.Conv2d(4, 4, 3, stride=2, padding=1), nn.ReLU(),   # 14
        nn.Conv2d(4, 4, 3, stride=2, padding=1), nn.ReLU(),   # 7
    )
    net = nn.Sequential(feats, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2)).eval()
    x = torch.randn(1, 3, 224, 224)
    for layer, side in ((feats[1], 28), (feats[3], 14), (feats[5], 7)):
        cam = grad_cam(net, layer, x, 0, out_size=None)
        assert cam.shape == (side, side), f"層の地図が {cam.shape}(期待 {side}x{side})"
        up = grad_cam(net, layer, x, 0, out_size=(224, 224))
        assert up.shape == (224, 224)


@pytest.mark.unit
def test_t303_randomized_copy_differs_from_trained() -> None:
    """T-303 / §3.20 — 乱数化モデルは重みも出力も学習済みと違う。"""
    import torch

    from ml.gradcam import randomized_copy
    from ml.train import build_model

    torch.manual_seed(1)
    trained = build_model().eval()
    random_model = randomized_copy(trained, 20260917)
    diff = sum(float((a - b).abs().sum()) for a, b in zip(trained.state_dict().values(),
                                                           random_model.state_dict().values())
               if a.dtype.is_floating_point)
    assert diff > 0, "乱数化しても重みが変わっていない"
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert not torch.allclose(trained(x), random_model(x)), "乱数化しても出力が同じ"


# ---------------------------------------------------------------- 報告


@pytest.mark.integration
def test_t304_gradcam_neg_verdict_is_mechanical() -> None:
    """T-304 / §3.20 — 判定を、報告の数と測る前に書いた規則から計算し直して一致を見る。"""
    report = json.loads(_require(REPORTS / "gradcam_negative_cocci.json",
                                 "python -m ml.gradcam_neg measure").read_text(encoding="utf-8"))
    for layer, lr in report["layers"].items():
        sane = lr["sanity"]["median_spearman"] < 0.5
        for ruler, r in lr["rulers"].items():
            nc, rnd, pc = r["neg_cocci_trained"], r["neg_cocci_randomized"], r["pos_cocci_trained"]
            if not sane:
                want = {"location": "判定しない", "vs_correct": "判定しない"}
            else:
                if nc["ci_low"] > rnd["ci_high"]:
                    loc = "誤答の根拠は菌体の上に寄っている"
                elif nc["ci_high"] < rnd["ci_low"]:
                    loc = "誤答の根拠は菌体の外(背景)に寄っている"
                else:
                    loc = "区別できない"
                if pc["n_used"] == 0:
                    # 物差し C の 2 モデルは試験側に陽性球菌が無い(SPEC §3.20)
                    vs = "比べる正答が無い"
                else:
                    overlap = not (nc["ci_high"] < pc["ci_low"] or pc["ci_high"] < nc["ci_low"])
                    vs = "違いは見えない" if overlap else "見る場所が違う"
                want = {"location": loc, "vs_correct": vs}
            assert r["verdict"] == want, f"{layer} / 物差し {ruler}: 記録 {r['verdict']} 対 再計算 {want}"
            for s in (nc, rnd, pc):
                if s["n_used"]:
                    assert s["ci_low"] <= s["mean_ratio"] <= s["ci_high"], (layer, ruler)
