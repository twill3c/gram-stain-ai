"""背景の効果を確かめる道具に対する検査(TEST_SPEC T-296〜T-300)。

期待値の出所:
  - 三つの操作の定義(SPEC §3.18): 染まった画素は一画素も変えない
  - 縁: チェビシェフ距離 3 以内(期待値は scipy の距離変換で、実装とは別の道筋で求める)
  - 並べ替え: 染まっていない画素の値の多重集合を保つ

**実装より先に書く。** 合成の塊は縁から離し、背景には必ずむらを入れる(平らな背景では
「平ら」も「並べ替え」も恒等になって、検査が何も確かめなくなる)。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

PINK = (220, 110, 160)


def _require(path: Path, how: str) -> Path:
    if path.exists():
        return path
    if not list((ROOT / "dataset" / "raw").glob("*.zip")):
        pytest.skip("dataset/raw に素材が無い環境")
    pytest.fail(f"{path.relative_to(ROOT)} が無い。素材はあるので生成できるはず: {how}")


def _textured(seed: int = 0) -> np.ndarray:
    """むらのある明るい背景(横方向の勾配 + 小さな雑音)に、縁から離した桃色の塊を置く。"""
    # 勾配の幅は 40 階調。15 階調だと、並べ替えで隣どうしの差が増える比が約 2.5 倍しかなく、
    # T-298 の「2 倍より大きい」が余裕なく揺れる(走らせる前の点検で見積もって広げた)。
    # 背景の彩度は約 0.08〜0.10 のままで、桃色の塊(0.5)とは大津の閾値で分かれる
    rng = np.random.default_rng(seed)
    h = w = 256
    xx = np.linspace(0, 1, w)[None, :].repeat(h, 0)
    base = np.stack([200 + 40 * xx, 195 + 40 * xx, 180 + 40 * xx], -1)
    img = np.clip(base + rng.normal(0, 2.0, (h, w, 3)), 0, 255).astype(np.uint8)
    yy, xg = np.mgrid[0:h, 0:w]
    for cy, cx in ((60, 60), (60, 190), (190, 60), (190, 190), (128, 128)):
        img[((yy - cy) / 9) ** 2 + ((xg - cx) / 9) ** 2 <= 1] = PINK
    return img


def _mask(img: np.ndarray) -> np.ndarray:
    from ml.separate import stained_mask

    return stained_mask(img)


@pytest.mark.unit
def test_t296_flat_keeps_stained_and_flattens_the_rest() -> None:
    """T-296 / §3.18 — 「平ら」は染まった画素を変えず、残りをすべて背景色の中央値にする。"""
    from ml.background import flat

    img = _textured()
    m = _mask(img)
    assert m.sum() > 0 and (~m).sum() > 0
    assert len(np.unique(img[~m].reshape(-1, 3), axis=0)) > 10, "合成の背景にむらが無い(検査が何も確かめなくなる)"
    out = flat(img)
    assert np.array_equal(out[m], img[m]), "染まった画素が変わった"
    bg = out[~m].reshape(-1, 3)
    assert len(np.unique(bg, axis=0)) == 1, "染まっていない画素が一色になっていない"
    want = np.median(img[~m].reshape(-1, 3), axis=0).round().astype(np.uint8)
    assert np.array_equal(bg[0], want), "塗った色が染まっていない画素の中央値でない"


@pytest.mark.unit
def test_t297_ring_keeps_chebyshev_3_around_stained() -> None:
    """T-297 / §3.18 — 「縁を残して平ら」は、染まった画素からチェビシェフ距離 3 以内をそのまま残す。"""
    from ml.background import flat_keep_ring

    img = _textured(1)
    m = _mask(img)
    # 期待値は距離変換で求める(実装が膨張で書いても、別の道筋で照合する)
    dist = ndimage.distance_transform_cdt(~m, metric="chessboard")
    keep = dist <= 3
    out = flat_keep_ring(img)
    assert np.array_equal(out[keep], img[keep]), "縁 3 画素以内が保たれていない"
    rest = out[~keep].reshape(-1, 3)
    assert len(rest) > 0 and len(np.unique(rest, axis=0)) == 1, "縁より外が一色になっていない"


@pytest.mark.unit
def test_t298_shuffle_keeps_values_and_breaks_arrangement() -> None:
    """T-298 / §3.18 — 「並べ替え」は値の多重集合を保ち、並びを壊す。"""
    from ml.background import shuffle_background

    img = _textured(2)
    m = _mask(img)
    out = shuffle_background(img, "img-a")
    assert np.array_equal(out[m], img[m]), "染まった画素が変わった"

    def as_sorted(a: np.ndarray) -> np.ndarray:
        v = a.reshape(-1, 3)
        return v[np.lexsort(v.T[::-1])]

    assert np.array_equal(as_sorted(out[~m]), as_sorted(img[~m])), "背景の値の多重集合が変わった"

    def neighbour_diff(a: np.ndarray) -> float:
        # 横に隣り合う画素のうち、両方とも染まっていないものの差の平均
        both = (~m[:, :-1]) & (~m[:, 1:])
        d = np.abs(a[:, 1:].astype(int) - a[:, :-1].astype(int)).sum(-1)
        return float(d[both].mean())

    assert neighbour_diff(out) > 2.0 * neighbour_diff(img), "並べ替えても隣どうしの差が増えていない(並びが壊れていない)"


@pytest.mark.unit
def test_t299_shuffle_is_deterministic_per_image_id() -> None:
    """T-299 / §3.18 — 同じ画像 ID なら同じ結果、違う ID なら違う結果。"""
    from ml.background import shuffle_background

    img = _textured(3)
    a1 = shuffle_background(img, "img-a")
    a2 = shuffle_background(img, "img-a")
    b = shuffle_background(img, "img-b")
    assert np.array_equal(a1, a2)
    assert not np.array_equal(a1, b)


# ---------------------------------------------------------------- 報告


@pytest.mark.integration
def test_t300_background_verdict_is_mechanical() -> None:
    """T-300 / §3.18 — 判定を、報告の数と測る前に書いた規則から計算し直して一致を見る。"""
    report = json.loads(_require(REPORTS / "background_negative_cocci.json",
                                 "python -m ml.background measure").read_text(encoding="utf-8"))
    side_ok = all(v < 0.20 for v in report["checks"]["side_effect_increase"].values())
    for ruler, r in report["primary"].items():
        d_flat = r["flat"]["drop"]
        replicated = d_flat >= 0.10 and r["flat"]["ci_low"] > 0
        if not side_ok:
            want = {"replication": "判定しない", "location": None, "nature": None}
        elif not replicated:
            want = {"replication": "再現しない", "location": None, "nature": None}
        else:
            want = {
                "replication": "再現した",
                "location": "菌体のすぐまわり(3 画素の縁)が効いている" if r["ring"]["drop"] <= d_flat / 2
                else "縁より外の背景が効いている",
                "nature": "背景の並び(模様)が効いている" if r["shuffle"]["drop"] >= d_flat / 2
                else "並びではなく、ばらつきが消えたこと(平らさ)が効いている",
            }
        assert r["verdict"] == want, f"物差し {ruler}: 記録 {r['verdict']} 対 再計算 {want}"
        for k in ("flat", "ring", "shuffle"):
            assert r[k]["ci_low"] <= r[k]["drop"] <= r[k]["ci_high"], (ruler, k)
