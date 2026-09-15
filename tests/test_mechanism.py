"""陰性球菌で崩れる理由を測る道具に対する検査(TEST_SPEC T-281〜T-287)。

期待値の出所:
  - HSV の色相を回しても彩度と明度は変わらない(色空間の定義)
  - 円周平均の定義(350 度と 10 度の平均は 0 度)
  - 判定規則: SPEC §3.12(測る前に書いた)

**実装より先に書く。**
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "dataset" / "metadata"
REPORTS = ROOT / "reports"

BACKGROUND = (235, 230, 215)
PINK = (220, 110, 160)
VIOLET_HUE = 282.9


def _require(path: Path, how: str) -> Path:
    if path.exists():
        return path
    if not list((ROOT / "dataset" / "raw").glob("*.zip")):
        pytest.skip("dataset/raw に素材が無い環境")
    pytest.fail(f"{path.relative_to(ROOT)} が無い。素材はあるので生成できるはず: {how}")


def _pink_blobs() -> np.ndarray:
    img = np.empty((256, 256, 3), dtype=np.uint8)
    img[:] = BACKGROUND
    yy, xx = np.mgrid[0:256, 0:256]
    for cy, cx, a, b in ((60, 60, 6, 6), (60, 160, 15, 4), (180, 90, 6, 6), (190, 190, 12, 5)):
        img[((yy - cy) / a) ** 2 + ((xx - cx) / b) ** 2 <= 1] = PINK
    return img


def _hsv(img: np.ndarray) -> np.ndarray:
    from ml.mechanism import rgb_to_hsv

    return rgb_to_hsv(img.astype(np.float64) / 255.0)


def _circ_diff(a: np.ndarray, b: float) -> np.ndarray:
    d = (a - b) % 360.0
    return np.minimum(d, 360.0 - d)


@pytest.mark.unit
def test_t281_sham_recolour_is_near_identity() -> None:
    """T-281 / §3.12 M3 — Δ = 0 の変換は画像をほぼ変えない。"""
    from ml.mechanism import rotate_hue

    rng = np.random.default_rng(1)
    noise = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    for img in (_pink_blobs(), noise):
        out = rotate_hue(img, 0.0)
        assert out.dtype == np.uint8 and out.shape == img.shape
        assert int(np.abs(out.astype(int) - img.astype(int)).max()) <= 1


@pytest.mark.unit
def test_t282_recolour_moves_only_hue() -> None:
    """T-282 / §3.12 — 色相だけを動かし、彩度と明度はそのまま。"""
    from ml.mechanism import rotate_hue

    img = _pink_blobs()
    src = _hsv(img)
    stained = src[..., 1] > 0.3
    source_hue = float(np.median(src[..., 0][stained]))
    out = rotate_hue(img, VIOLET_HUE - source_hue)
    dst = _hsv(out)
    assert np.abs(dst[..., 1] - src[..., 1]).max() <= 2 / 255 + 1e-9, "彩度が変わった"
    assert np.abs(dst[..., 2] - src[..., 2]).max() <= 2 / 255 + 1e-9, "明度が変わった"
    assert _circ_diff(dst[..., 0][stained], VIOLET_HUE).max() <= 3.0, "染まった画素の色相が目標に入らない"


@pytest.mark.unit
def test_t283_recolour_keeps_shape_rule_features_synthetic() -> None:
    """T-283 / §3.12 M2 — 塗り替えは形の規則の特徴量を変えない(合成)。"""
    from ml.mechanism import rotate_hue
    from ml.stage_b import shape_feature

    img = _pink_blobs()
    before = shape_feature(img)
    after = shape_feature(rotate_hue(img, 120.0))
    assert before["n_components"] == after["n_components"]
    assert abs(before["elongation"] - after["elongation"]) < 1e-9


@pytest.mark.unit
def test_t284_circular_mean_wraps_around_zero() -> None:
    """T-284 — 円周平均が 0 度をまたいでも正しい。算術平均だと 180 度になる。"""
    from ml.mechanism import circular_mean_deg

    assert _circ_diff(np.array([circular_mean_deg([350.0, 10.0])]), 0.0)[0] < 1e-9
    assert abs(circular_mean_deg([280.0, 286.0]) - 283.0) < 1e-9


@pytest.mark.unit
def test_t287_drop_interval_is_over_groups() -> None:
    """T-287 / §3.12 — 下がり幅の区間は group 単位で、点推定を含む。"""
    from ml.mechanism import bootstrap_drop

    # **group の中で正誤が揃っている**合成データにする(同じスライドは似た間違え方をする)。
    # 画像ごとに独立な正誤を束ねても区間は広がらない —— loop_013 でその前提の無いデータで書いて落とした
    n, size = 120, 20
    wrong_before = np.ones(n, dtype=int)
    coarse = np.arange(n) // size                       # 独立単位 6
    wrong_after = (coarse % 2 == 0).astype(int)         # group ごとに全部正解か全部誤り(平均 0.5)
    fine = np.arange(n)                                 # 画像ごとに別 group と扱う(独立単位 120)
    p, lo, hi = bootstrap_drop(wrong_before, wrong_after, fine, n_boot=500, seed=1)
    assert lo <= p <= hi
    _, lo2, hi2 = bootstrap_drop(wrong_before, wrong_after, coarse, n_boot=500, seed=1)
    assert (hi2 - lo2) > (hi - lo), (
        f"group の中で揃った誤りなのに、group 単位の区間 {hi2 - lo2:.3f} が画像単位 {hi - lo:.3f} より広くない"
    )


# ---------------------------------------------------------------- 素材がある環境


@pytest.mark.integration
def test_t283_recolour_keeps_shape_rule_features_on_negative_cocci() -> None:
    """T-283 / §3.12 M2 — 陰性球菌 126 枚すべてで、塗り替え前後の形の規則の特徴量が一致する。"""
    from PIL import Image

    from ml.mechanism import load_hue, target_hues, rotate_hue
    from ml.stage_b import shape_feature

    prepared = _require(META / "prepared.jsonl", "python -m ml.prepare_dataset")
    rows = [json.loads(line) for line in prepared.read_text(encoding="utf-8").splitlines()]
    nc = [r for r in rows if r["stage_b"] and r["gram"] == "negative" and r["shape"] == "coccus"]
    assert len(nc) == 126
    hue = load_hue()
    violet, _ = target_hues(rows, hue)
    for r in nc:
        with Image.open(ROOT / "dataset" / "processed" / r["path"]) as im:
            a = np.asarray(im.convert("RGB"))
        b = shape_feature(a)
        c = shape_feature(rotate_hue(a, violet - hue[r["image_id"]]))
        assert b["n_components"] == c["n_components"], r["image_id"]
        assert (b["elongation"] is None) == (c["elongation"] is None), r["image_id"]
        if b["elongation"] is not None:
            assert abs(b["elongation"] - c["elongation"]) < 1e-9, r["image_id"]


@pytest.fixture(scope="module")
def report() -> dict:
    path = _require(REPORTS / "mechanism_negative_cocci.json", "python -m ml.mechanism measure")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.integration
def test_t285_retrained_models_never_saw_the_held_out_taxon(report: dict) -> None:
    """T-285 / §3.12 / G-08 — 学習し直したモデルは、抜いた陰性球菌の分類群を一枚も見ていない。"""
    from ml.stage_b import restrict_folds

    rows = {json.loads(line)["image_id"]: json.loads(line)
            for line in (META / "prepared.jsonl").read_text(encoding="utf-8").splitlines()}
    splits = json.loads((META / "splits.json").read_text(encoding="utf-8"))
    keep = {i for i, r in rows.items() if r["stage_b"]}
    for m in report["models"]:
        ids = json.loads((ROOT / m["ids_file"]).read_text(encoding="utf-8"))
        fold = restrict_folds(splits[m["splits_key"]]["folds"], keep)[m["fold"]]
        assert sorted(ids["test"]) == sorted(fold["test"]), m["name"]
        assert sorted(ids["train"]) == sorted(fold["train"]), m["name"]
        for taxon in m["held_out_negative_cocci"]:
            assert not any(rows[i]["folder"] == taxon for i in ids["train"]), f"{m['name']} が {taxon} を学習で見ている"


@pytest.mark.integration
def test_t286_verdict_is_mechanical(report: dict) -> None:
    """T-286 / §3.12 — 判定を、報告の数と測る前に書いた規則から計算し直して一致を見る。"""
    checks = report["checks"]
    ok = (checks["m1_hue_rule_positive_rate"] >= 0.90
          and checks["m2_shape_features_identical"] is True
          and all(abs(v) < 0.05 for v in checks["m3_sham_error_change"].values()))
    for ruler, r in report["primary"].items():
        drop = r["error_original"] - r["error_violet"]
        assert abs(drop - r["drop"]["point"]) < 1e-9, ruler
        if not ok:
            want = "判定しない"
        elif drop >= 0.50 and r["error_violet"] <= 0.40:
            want = "色が主因"
        elif drop < 0.20:
            want = "見た目が主因"
        else:
            want = "混合"
        assert r["verdict"] == want, f"物差し {ruler}: 記録 {r['verdict']} 対 再計算 {want}"
        assert r["drop"]["ci_low"] <= r["drop"]["point"] <= r["drop"]["ci_high"]
