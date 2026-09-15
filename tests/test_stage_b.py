"""Stage B(形: 球菌 / 桿菌)に対する検査(TEST_SPEC T-262〜T-271)。

期待値の出所:
  - 形の規則の合成画像: 描いた図形の幾何から決まる。円の軸比は 1、
    半軸 4 と 15 の楕円の軸比は 3.75(離散化で少し動くので 1.3 / 2.5 で切る)
  - 対照と CNN の報告: SPEC §3.7 / §3.8 に**測る前に**書いた定義と条件

**実装より先に書く。** 走らせて赤であることを確かめてから ml/stage_b.py を書く。
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
VIOLET = (110, 60, 150)   # crystal violet 様
PINK = (220, 110, 160)    # safranin 様


def _require(path: Path, how: str) -> Path:
    if path.exists():
        return path
    if not list((ROOT / "dataset" / "raw").glob("*.zip")):
        pytest.skip("dataset/raw に素材が無い環境")
    pytest.fail(f"{path.relative_to(ROOT)} が無い。素材はあるので生成できるはず: {how}")


def _canvas(size: int = 512) -> np.ndarray:
    img = np.empty((size, size, 3), dtype=np.uint8)
    img[:] = BACKGROUND
    return img


def _ellipse(img: np.ndarray, cy: float, cx: float, a: float, b: float, angle: float, colour) -> None:
    """半軸 a(長)・b(短)の楕円を、角度 angle(ラジアン)で塗る。"""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    dy, dx = yy - cy, xx - cx
    u = dx * np.cos(angle) + dy * np.sin(angle)
    v = -dx * np.sin(angle) + dy * np.cos(angle)
    img[(u / a) ** 2 + (v / b) ** 2 <= 1.0] = colour


def _cocci(colour, n_side: int = 5, spacing: int = 90, offset: int = 40) -> np.ndarray:
    img = _canvas()
    assert offset + (n_side - 1) * spacing + 6 < img.shape[0], "図形がキャンバスからはみ出す"
    for i in range(n_side):
        for j in range(n_side):
            _ellipse(img, offset + i * spacing, offset + j * spacing, 6, 6, 0.0, colour)
    return img


def _rods(colour, n_side: int = 5, spacing: int = 90) -> np.ndarray:
    img = _canvas()
    rng = np.random.default_rng(3)
    for i in range(n_side):
        for j in range(n_side):
            _ellipse(img, 45 + i * spacing, 45 + j * spacing, 15, 4, float(rng.uniform(0, np.pi)), colour)
    return img


# ---------------------------------------------------------------- 形の規則(合成画像)


@pytest.mark.unit
def test_t262_shape_rule_measures_shape() -> None:
    """T-262 / G-12 — 円だけの視野は丸く、楕円だけの視野は細長く測れる。"""
    from ml.stage_b import shape_feature

    round_ = shape_feature(_cocci(VIOLET))
    long_ = shape_feature(_rods(VIOLET))
    assert round_["n_components"] == 25 and long_["n_components"] == 25, (round_, long_)
    assert round_["elongation"] < 1.3, f"円の細長さが {round_['elongation']:.3f}"
    assert long_["elongation"] > 2.5, f"8x30 の楕円の細長さが {long_['elongation']:.3f}"


@pytest.mark.unit
def test_t263_shape_rule_is_colour_blind() -> None:
    """T-263 / G-12 — 同じ図形を紫と桃で描くと、特徴量が一致する。

    形の規則が色に反応すると、Gram と形の偏りを「形を見た」と取り違える。
    そのための対照は別に置いてある(色の規則を形に当てたもの)。
    """
    from ml.stage_b import shape_feature

    for draw in (_cocci, _rods):
        v = shape_feature(draw(VIOLET))
        p = shape_feature(draw(PINK))
        assert v["n_components"] == p["n_components"], (v, p)
        assert abs(v["elongation"] - p["elongation"]) < 1e-9, (v, p)


@pytest.mark.unit
def test_t264_area_outliers_are_dropped() -> None:
    """T-264 / G-12 — 面積 30 未満の点と 2,000 超の塊は数えない。"""
    from ml.stage_b import shape_feature

    img = _canvas()
    for k in range(10):
        _ellipse(img, 30 + k * 45, 30, 2, 2, 0.0, VIOLET)       # 面積 約 13
    _ellipse(img, 300, 300, 60, 40, 0.3, VIOLET)                 # 面積 約 7,500
    got = shape_feature(img)
    assert got["n_components"] == 0, got
    assert got["elongation"] is None, got


@pytest.mark.unit
def test_t265_stain_threshold_follows_density() -> None:
    """T-265 / SPEC §3.7 — 疎な視野でも密な視野でも、描いた数だけ成分を数える。

    Stage A の「彩度の上位 10%」は、疎な視野では背景を拾って成分がつながり、
    密な視野では菌体を取りこぼす。形はそこで壊れる。
    """
    from ml.stage_b import shape_feature

    sparse = _cocci(VIOLET, n_side=5, spacing=90)                # 25 個・約 1%
    dense = _cocci(VIOLET, n_side=25, spacing=20, offset=12)     # 625 個・約 27%
    for name, img, n in (("疎", sparse, 25), ("密", dense, 625)):
        stained = float((img != np.array(BACKGROUND, dtype=np.uint8)).any(2).mean())
        got = shape_feature(img)
        assert got["n_components"] == n, f"{name}な視野(染色 {stained:.1%}): {got['n_components']} 個 ≠ {n}"


# ---------------------------------------------------------------- fold


@pytest.mark.unit
def test_t266_stage_b_folds_only_remove_images() -> None:
    """T-266 / SPEC §3.7 — Stage B の fold は Stage A の fold から対象外を抜くだけ。"""
    from ml.stage_b import restrict_folds

    folds = [{"train": ["a", "b", "c", "d"], "test": ["e", "f"]},
             {"train": ["c", "d", "e", "f"], "test": ["a", "b"]}]
    keep = {"a", "c", "e", "f"}
    got = restrict_folds(folds, keep)
    assert len(got) == len(folds)
    for old, new in zip(folds, got):
        for part in ("train", "test"):
            assert set(new[part]) <= set(old[part]), f"{part} に元に無い画像が足された"
            assert set(new[part]) == set(old[part]) & keep, f"{part} の抜き方が違う"


@pytest.fixture(scope="module")
def prepared() -> list[dict]:
    path = _require(META / "prepared.jsonl", "python -m ml.prepare_dataset")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.integration
def test_t267_every_stage_b_fold_trains_on_both_classes(prepared: list[dict]) -> None:
    """T-267 / F-09 — Stage B の全 fold の train に球菌と桿菌の両方がいる。"""
    from ml.stage_b import restrict_folds

    splits = json.loads(_require(META / "splits.json", "python -m ml.prepare_dataset").read_text(encoding="utf-8"))
    shape = {r["image_id"]: r["shape"] for r in prepared if r["stage_b"]}
    for key in ("ruler_a_cv", "ruler_b", "ruler_c"):
        for i, fold in enumerate(restrict_folds(splits[key]["folds"], set(shape))):
            classes = {shape[x] for x in fold["train"]}
            assert classes == {"coccus", "bacillus"}, f"{key} fold {i} の train が {classes}"


@pytest.mark.unit
def test_t272_gram_by_shape_cells_are_tallied_from_predictions() -> None:
    """T-272 — Gram × 形の 4 セルの誤りを、fold の予測から数える(事後の分析・合否ではない)。

    期待値は手で数えられる小さな合成データから取る。
    """
    from ml.stage_b import tally_cells

    rows = {
        "p1": {"gram": "positive", "shape": "coccus", "folder": "S.aureus"},
        "p2": {"gram": "positive", "shape": "coccus", "folder": "S.aureus"},
        "p3": {"gram": "positive", "shape": "bacillus", "folder": "L.casei"},
        "n1": {"gram": "negative", "shape": "coccus", "folder": "Neisseria"},
        "n2": {"gram": "negative", "shape": "coccus", "folder": "Veillonella"},
        "n3": {"gram": "negative", "shape": "bacillus", "folder": "E.coli"},
    }
    # 球菌 = 0 / 桿菌 = 1。陰性球菌を 2 枚とも桿菌と答え、陽性球菌を 1 枚だけ外す
    preds = {"p1": 0, "p2": 1, "p3": 1, "n1": 1, "n2": 1, "n3": 1}
    got = tally_cells(preds, rows)
    assert got["negative x coccus"] == {"n": 2, "wrong": 2, "error_rate": 1.0, "n_taxa": 2}
    assert got["positive x coccus"] == {"n": 2, "wrong": 1, "error_rate": 0.5, "n_taxa": 1}
    assert got["positive x bacillus"] == {"n": 1, "wrong": 0, "error_rate": 0.0, "n_taxa": 1}
    assert got["negative x bacillus"] == {"n": 1, "wrong": 0, "error_rate": 0.0, "n_taxa": 1}
    assert sum(v["n"] for v in got.values()) == len(preds)


# ---------------------------------------------------------------- 対照の報告


@pytest.fixture(scope="module")
def controls_shape() -> dict:
    path = _require(REPORTS / "controls_shape.json", "python -m ml.stage_b controls")
    return json.loads(path.read_text(encoding="utf-8"))


CONTROLS = ("shape", "hue_on_shape", "canvas_size", "majority")


@pytest.mark.integration
def test_t268_controls_cover_four_controls_three_rulers(controls_shape: dict) -> None:
    """T-268 / G-12 — 対照が四種 × 三物差しで記録されている。"""
    for name in CONTROLS:
        assert name in controls_shape["baselines"], f"対照 {name} が無い"
        for ruler in ("a", "b", "c"):
            r = controls_shape["baselines"][name].get(ruler)
            assert r is not None, f"対照 {name} に物差し {ruler} が無い"
            assert r["ci_low"] <= r["macro_f1"] <= r["ci_high"], (name, ruler, r)


@pytest.mark.integration
def test_t269_each_stage_b_image_once_with_train_thresholds(controls_shape: dict, prepared: list[dict]) -> None:
    """T-269 / F-09 / G-08 — Stage B 対象の全画像を一度ずつ、train で決めた閾値で予測している。"""
    n_stage_b = sum(1 for r in prepared if r["stage_b"])
    assert controls_shape["n_images_total"] == n_stage_b
    for name in CONTROLS:
        for ruler in ("a", "b", "c"):
            got = controls_shape["baselines"][name][ruler]["n_images"]
            assert got == n_stage_b, f"{name} / {ruler}: {got} 枚 ≠ {n_stage_b}"
    for name in ("shape", "hue_on_shape"):
        b = controls_shape["baselines"][name]
        assert b["threshold_fitted_on"] == "train"
        for ruler in ("a", "b", "c"):
            assert b[ruler]["macro_f1"] <= b["oracle_upper_bound_macro_f1"] + 1e-9, (name, ruler)
    assert "n_missing_feature" in controls_shape["baselines"]["shape"], "特徴量なしの枚数が報告されていない"


# ---------------------------------------------------------------- CNN


@pytest.fixture(scope="module")
def cnn_shape() -> dict:
    path = _require(REPORTS / "cnn_shape.json", "python -m ml.train --target shape")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.integration
def test_t270_stage_b_verdict_is_mechanical(cnn_shape: dict, controls_shape: dict) -> None:
    """T-270 / SPEC §3.8 — 判定を、区間と分類群ごとの誤りから計算し直して一致を見る。"""
    base = controls_shape["baselines"]
    strong = {}
    for ruler in ("a", "b", "c"):
        s, h = base["shape"][ruler], base["hue_on_shape"][ruler]
        strong[ruler] = s if s["macro_f1"] >= h["macro_f1"] else h
    cond1 = all(cnn_shape["rulers"][r]["ci_low"] > strong[r]["ci_high"] for r in ("a", "b", "c"))

    hard_err = base["shape"]["b"]["per_taxon_errors"]
    hard = sorted(k for k, v in hard_err.items() if v["error_rate"] >= 0.10)
    cnn_b = cnn_shape["per_taxon"]["b"]
    wins = sum(1 for t in hard if cnn_b.get(t, {}).get("error_rate", 1.0) < hard_err[t]["error_rate"])
    total_rule = sum(hard_err[t]["wrong"] for t in hard)
    total_cnn = sum(cnn_b.get(t, {}).get("wrong", 0) for t in hard)
    cond2 = bool(hard) and wins > len(hard) / 2 and total_cnn < total_rule

    v = cnn_shape["verdict"]
    assert v["condition_1_beats_strong_control_on_all_rulers"] == cond1
    assert v["condition_2_beats_on_hard_taxa"] == cond2
    assert v["hard_taxa"] == hard
    assert v["adds_nothing"] == ((not cond1) and (not cond2))


@pytest.mark.unit
def test_t273_shuffle_null_separates_leak_from_chance() -> None:
    """T-273 / G-09 — 同じ予測を fold 内で入れ替えた分布が、漏れを見分ける。

    多数派クラス(全部を同じ答え)の macro F1 は、ばらけて答える予測の偶然水準ではない。
    入れ替えは予測の割合を保ったまま画像との対応だけを壊すので、前提を置かずに偶然水準が作れる。
    """
    from ml.stage_b import shuffle_null

    # **一回の抽選で断定しない**(loop_011 で踏んだ)。ラベルと無関係な予測でも、
    # 上側 97.5% を超える事象は 2.5% の確率で起きる。確かめるべきは一回の合否ではなく、
    # 超える割合が名目の水準に収まること(較正)と、漏れを毎回見分けること(検出力)である
    rng = np.random.default_rng(20260915)
    trials, false_alarms, missed_leaks = 200, 0, 0
    for t in range(trials):
        y = rng.integers(0, 2, 400)
        blind = rng.integers(0, 2, 400)                                              # ラベルと無関係
        leak = np.where(rng.random(400) < 0.8, y, 1 - y)                             # 8 割がラベルを写す漏れ
        folds_blind = [(y[i:i + 100], blind[i:i + 100]) for i in range(0, 400, 100)]
        folds_leak = [(y[i:i + 100], leak[i:i + 100]) for i in range(0, 400, 100)]
        got_blind = shuffle_null(folds_blind, n=200, seed=t)
        got_leak = shuffle_null(folds_leak, n=200, seed=t)
        false_alarms += int(got_blind["observed"] > got_blind["q975"])
        missed_leaks += int(got_leak["observed"] <= got_leak["q975"])
        assert got_blind["q025"] <= got_blind["median"] <= got_blind["q975"]
    rate = false_alarms / trials
    assert rate <= 0.06, f"ラベルと無関係な予測を漏れと判定した割合が {rate:.1%}(名目 2.5%)"
    assert missed_leaks == 0, f"漏れのある予測を {missed_leaks}/{trials} 回見逃した"


@pytest.fixture(scope="module")
def permutation_null() -> dict:
    path = _require(REPORTS / "permutation_null.json", "python -m ml.stage_b perm-null")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.integration
def test_t271_stage_b_cnn_learning_controls(cnn_shape: dict, permutation_null: dict) -> None:
    """T-271 / G-09 / G-08 — ラベル置換が偶然水準を上に外れず、予算は val で決めている。

    **loop_011 で引き直した。** 旧基準「多数派クラスの区間と重なる」は、多数派が
    偶然水準ではないので、漏れの無い Stage B を落とした(0.4963 対 0.4119)。
    偶然水準は同じ予測の fold 内入れ替えで作る。漏れは上に外れる形で出るので片側で見る。
    """
    for stage in ("shape", "gram"):
        s = permutation_null[stage]
        assert s["observed"] <= s["q975"], (
            f"{stage}: ラベル置換 {s['observed']:.4f} が入れ替え分布の上側 97.5% {s['q975']:.4f} を超える —— 学習系に漏れがある"
        )
    assert abs(permutation_null["shape"]["observed"] - cnn_shape["permutation_control"]["a"]["macro_f1"]) < 1e-3
    assert cnn_shape["epoch_budget"]["chosen_on"] == "ruler_a_val"
    assert cnn_shape["epoch_budget"]["curve"], "予算を選んだ曲線が残っていない"
