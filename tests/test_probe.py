"""色相以外の手がかりを絞る道具に対する検査(TEST_SPEC T-288〜T-291)。

期待値の出所:
  - 拡大の定義(面積は s² 倍・細長さは不変)
  - 彩度だけを動かす操作の定義(色相は不変)
  - 判定規則: SPEC §3.14(測る前に書いた)

**実装より先に書く。**
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

BACKGROUND = (235, 230, 215)
PINK = (220, 110, 160)


def _require(path: Path, how: str) -> Path:
    if path.exists():
        return path
    if not list((ROOT / "dataset" / "raw").glob("*.zip")):
        pytest.skip("dataset/raw に素材が無い環境")
    pytest.fail(f"{path.relative_to(ROOT)} が無い。素材はあるので生成できるはず: {how}")


def _field(n_side: int = 4, spacing: int = 110, a: int = 8, b: int = 8) -> np.ndarray:
    """512px の視野に、同じ形の塊を格子状に置く。"""
    img = np.empty((512, 512, 3), dtype=np.uint8)
    img[:] = BACKGROUND
    yy, xx = np.mgrid[0:512, 0:512]
    for i in range(n_side):
        for j in range(n_side):
            cy, cx = 90 + i * spacing, 90 + j * spacing
            img[((yy - cy) / a) ** 2 + ((xx - cx) / b) ** 2 <= 1] = PINK
    return img


@pytest.mark.unit
def test_t288_zoom_keeps_elongation_and_scales_area() -> None:
    """T-288 / §3.14 — 拡大は細長さを変えず、成分の面積を s² 倍にする。"""
    from ml.probe import zoom
    from ml.stage_b import otsu_threshold

    def area_and_elongation(img: np.ndarray) -> tuple[float, float]:
        from scipy import ndimage

        from ml.stage_b import EIGHT, LAMBDA_FLOOR

        a = img.astype(np.float32) / 255.0
        mx, mn = a.max(2), a.min(2)
        sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
        labels, n = ndimage.label(sat > otsu_threshold(sat), structure=EIGHT)
        flat = labels.ravel()
        area = np.bincount(flat)[1:]
        yy, xx = np.indices(labels.shape)
        y, x = yy.ravel().astype(float), xx.ravel().astype(float)
        size = n + 1
        sy, sx = np.bincount(flat, y, size)[1:], np.bincount(flat, x, size)[1:]
        syy, sxx = np.bincount(flat, y * y, size)[1:], np.bincount(flat, x * x, size)[1:]
        sxy = np.bincount(flat, x * y, size)[1:]
        keep = area > 20
        ar = area[keep]
        my, mxx = sy[keep] / ar, sx[keep] / ar
        vyy, vxx = syy[keep] / ar - my * my, sxx[keep] / ar - mxx * mxx
        vxy = sxy[keep] / ar - my * mxx
        half = (vyy + vxx) / 2
        disc = np.sqrt(np.maximum(half ** 2 - (vyy * vxx - vxy ** 2), 0.0))
        elong = np.sqrt((half + disc) / np.maximum(half - disc, LAMBDA_FLOOR))
        return float(np.median(ar)), float(np.median(elong))

    for shape_kw in ({"a": 8, "b": 8}, {"a": 18, "b": 6}):
        base = _field(**shape_kw)
        a0, e0 = area_and_elongation(base)
        for s in (1.5, 2.0):
            a1, e1 = area_and_elongation(zoom(base, s))
            assert abs(e1 - e0) < 0.05, f"s={s} で細長さが変わった({e0:.3f} → {e1:.3f})"
            assert 0.85 * s ** 2 <= a1 / a0 <= 1.15 * s ** 2, f"s={s} で面積比が {a1/a0:.2f}(期待 {s**2:.2f})"


@pytest.mark.unit
def test_t289_sham_zoom_is_near_identity() -> None:
    """T-289 / §3.14 — s = 1.0 の拡大はほぼ恒等。"""
    from ml.probe import zoom

    img = _field()
    out = zoom(img, 1.0)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert int(np.abs(out.astype(int) - img.astype(int)).max()) <= 1


@pytest.mark.unit
def test_t290_saturation_scaling_keeps_hue() -> None:
    """T-290 / §3.14 — 彩度だけを動かし、色相は変えない。"""
    from ml.mechanism import rgb_to_hsv
    from ml.probe import scale_saturation

    img = _field()
    before = rgb_to_hsv(img.astype(np.float64) / 255.0)
    stained = before[..., 1] > 0.3
    target = 0.30
    out = scale_saturation(img, target)
    after = rgb_to_hsv(out.astype(np.float64) / 255.0)
    d = np.abs(after[..., 0][stained] - before[..., 0][stained]) % 360.0
    assert np.minimum(d, 360.0 - d).max() < 0.5, "色相が動いた"
    assert abs(float(np.median(after[..., 1][stained])) - target) <= 0.02, "彩度の中央値が目標に入らない"


# ---------------------------------------------------------------- 報告


@pytest.mark.integration
def test_t291_probe_verdict_is_mechanical() -> None:
    """T-291 / §3.14 — 判定を、報告の数と測る前に書いた規則から計算し直して一致を見る。"""
    report = json.loads(_require(REPORTS / "probe_negative_cocci.json",
                                 "python -m ml.probe measure").read_text(encoding="utf-8"))
    sham_ok = all(abs(v) < 0.05 for v in report["checks"]["sham_zoom_error_change"].values())
    for ruler, r in report["primary"].items():
        best_zoom = min(r["zoom"].values(), key=lambda d: d["error"])
        sat = r["saturation"]
        base = r["error_original"]

        def qualifies(cand: dict) -> bool:
            side = cand["side_effect"]
            return (base - cand["error"] >= 0.50 and cand["error"] <= 0.40
                    and side["positive_cocci_error_increase"] < 0.20
                    and side["negative_bacilli_error_increase"] < 0.20)

        if not sham_ok:
            want = "判定しない"
        elif qualifies(best_zoom):
            want = "大きさが主因"
        elif qualifies(sat):
            want = "濃さが主因"
        elif (base - best_zoom["error"]) < 0.20 and (base - sat["error"]) < 0.20:
            want = "どちらでもない"
        else:
            want = "混合"
        assert r["verdict"] == want, f"物差し {ruler}: 記録 {r['verdict']} 対 再計算 {want}"
        assert r["drop"]["ci_low"] <= r["drop"]["point"] <= r["drop"]["ci_high"]
