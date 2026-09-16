"""拡大の二つの効果を分ける道具に対する検査(TEST_SPEC T-292〜T-295)。

期待値の出所:
  - 数だけの操作: 中心を等倍で残すので、残った塊の面積と細長さは変わらない。数は約 1/s²
  - 大きさだけの操作: 塊ごとに重心で s 倍にするので、数は同じで面積は s² 倍
  - 判定規則: SPEC §3.16(測る前に書いた)

**実装より先に書く。** loop_013/014 で検査側の前提を 4 度誤った(相関の無い群・縁で切れる塊・
離散化の偏りを絶対差で縛る・一回の抽選)。合成の塊は縁から離し、許容差は相対で置く。
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


def _field(centres, a: int, b: int) -> np.ndarray:
    img = np.empty((512, 512, 3), dtype=np.uint8)
    img[:] = BACKGROUND
    yy, xx = np.mgrid[0:512, 0:512]
    for cy, cx in centres:
        img[((yy - cy) / a) ** 2 + ((xx - cx) / b) ** 2 <= 1] = PINK
    return img


def _blobs(img: np.ndarray) -> list[tuple[float, float]]:
    """縁に接しない塊の (面積, 細長さ)。"""
    from scipy import ndimage

    from ml.stage_b import EIGHT, LAMBDA_FLOOR, otsu_threshold

    a = img.astype(np.float32) / 255.0
    mx, mn = a.max(2), a.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    labels, n = ndimage.label(sat > otsu_threshold(sat), structure=EIGHT)
    edge = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    out = []
    for lab in range(1, n + 1):
        if lab in edge:
            continue
        ys, xs = np.nonzero(labels == lab)
        if len(ys) < 10:
            continue
        cov = np.cov(np.stack([ys, xs]).astype(float))
        ev = np.sort(np.linalg.eigvalsh(cov))
        out.append((float(len(ys)), float(np.sqrt(ev[1] / max(ev[0], LAMBDA_FLOOR)))))
    return out


@pytest.mark.unit
def test_t292_count_only_keeps_inner_blobs_and_reduces_count() -> None:
    """T-292 / §3.16 — 数だけの操作は、窓の中を等倍のまま残し、外を背景色で埋める。

    期待値は**実装と独立に**ここで組み立てる: 元画像の窓の中だけを残し、外を背景色で塗った画像。
    塊の数で「約 1/s²」を確かめる書き方は、窓の境で切れた塊(縁に接しないので除外されない)を
    数えて外れる —— 走らせる前の点検で見つけて、この形にした。
    """
    from ml.separate import count_only

    centres = [(40 + i * 48, 40 + j * 48) for i in range(10) for j in range(10)]
    img = _field(centres, 14, 5)
    before = _blobs(img)
    for s in (2.0, 3.0):
        side = int(round(512 / s))
        top = (512 - side) // 2
        want = np.empty_like(img)
        want[:] = BACKGROUND
        want[top : top + side, top : top + side] = img[top : top + side, top : top + side]
        got = count_only(img, s)
        assert got.shape == img.shape and got.dtype == np.uint8
        assert np.array_equal(got, want), f"s={s}: 窓の中を等倍で残し外を背景色で埋めた画像と一致しない"
        # 窓の中に丸ごと残った塊は、面積も細長さも変わらない(引き伸ばしていない)
        # 楕円は中心の上下 14・左右 5 画素まで塗られる。丸ごと窓に入る条件は
        # cy-14 >= top かつ cy+14 <= top+side-1(横も同じ)。**端にぴったり接する塊も入る**
        inner = [(cy, cx) for cy, cx in centres
                 if cy - 14 >= top and cy + 14 <= top + side - 1
                 and cx - 5 >= top and cx + 5 <= top + side - 1]
        full = [x for x in _blobs(got) if abs(x[0] - np.median([b[0] for b in before])) < 1.0]
        assert len(full) == len(inner), f"s={s}: 丸ごと残った塊 {len(full)} 個 ≠ 幾何から数えた {len(inner)} 個"


@pytest.mark.unit
def test_t293_size_only_keeps_count_and_scales_area() -> None:
    """T-293 / §3.16 — 大きさだけの操作は、塊の数を保ち、面積を s² 倍にする。"""
    from ml.separate import size_only

    centres = [(96 + i * 160, 96 + j * 160) for i in range(3) for j in range(3)]  # 9 個・3 倍でも重ならない間隔
    for a, b in ((8, 8), (15, 5)):
        img = _field(centres, a, b)
        before = _blobs(img)
        assert len(before) == 9
        for s in (2.0, 3.0):
            out, _overlap = size_only(img, s)
            after = _blobs(out)
            assert len(after) == len(before), f"s={s}: 塊の数が {len(before)} → {len(after)}"
            ratio = np.median([x[0] for x in after]) / np.median([x[0] for x in before])
            assert 0.85 * s ** 2 <= ratio <= 1.15 * s ** 2, f"s={s}: 面積比 {ratio:.2f}(期待 {s ** 2:.2f})"
            e0, e1 = np.median([x[1] for x in before]), np.median([x[1] for x in after])
            assert abs(e1 - e0) / e0 < 0.05, f"s={s}: 細長さ {e0:.3f} → {e1:.3f}"


@pytest.mark.unit
def test_t294_size_only_sham_is_near_identity_on_flat_background() -> None:
    """T-294 / §3.16 P1 — 大きさだけの操作を s = 1.0 で通すと、平らな背景の合成画像はほぼ変わらない。"""
    from ml.separate import size_only

    centres = [(96 + i * 160, 96 + j * 160) for i in range(3) for j in range(3)]
    img = _field(centres, 12, 6)
    out, overlap = size_only(img, 1.0)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert int(np.abs(out.astype(int) - img.astype(int)).max()) <= 1
    assert overlap == 0.0


# ---------------------------------------------------------------- 報告


@pytest.mark.integration
def test_t295_separate_verdict_is_mechanical() -> None:
    """T-295 / §3.16 — 判定を、報告の数と測る前に書いた規則から計算し直して一致を見る。"""
    report = json.loads(_require(REPORTS / "separate_negative_cocci.json",
                                 "python -m ml.separate measure").read_text(encoding="utf-8"))
    checks = report["checks"]
    ok = (all(abs(v) < 0.05 for v in checks["p1_sham_size_error_change"].values())
          and all(v < 0.20 for v in checks["p2_side_effect_increase"].values()))
    for ruler, r in report["primary"].items():
        size_ok = r["size_only"]["drop"] >= 0.15 and r["size_only"]["ci_low"] > 0
        count_ok = r["count_only"]["drop"] >= 0.15 and r["count_only"]["ci_low"] > 0
        if not ok:
            want = "判定しない"
        elif size_ok and count_ok:
            want = "両方"
        elif size_ok:
            want = "大きさが効いている"
        elif count_ok:
            want = "数が効いている"
        else:
            want = "どちらでもない"
        assert r["verdict"] == want, f"物差し {ruler}: 記録 {r['verdict']} 対 再計算 {want}"
        for k in ("size_only", "count_only"):
            assert r[k]["ci_low"] <= r[k]["drop"] <= r[k]["ci_high"], (ruler, k)
