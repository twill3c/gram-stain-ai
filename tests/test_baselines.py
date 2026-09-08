"""陽性対照(非学習ベースライン)と評価器に対する検査(TEST_SPEC T-207・T-211・T-227〜T-234)。

期待値の出所:
  - macro F1 の定義: scikit-learn の f1_score(average="macro") との一致で検算する
  - 多数派クラスの macro F1: 解析的に計算できる。陽性 1,407 / 陰性 559 のとき
    全部を陽性と答えると、陽性の F1 = 2*1407/(1407+1966)、陰性の F1 = 0
  - ブートストラップ: 区間が点推定を含むこと、group 単位であること
  - 色相ベースラインの分離度: 2026-09-08 実測(R/B 比の単一閾値で macro F1 の上限 0.9745)

**実装より先に書く。** 走らせて赤であることを確かめてから評価器を書く。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "dataset" / "metadata"
REPORTS = ROOT / "reports"


def _require(path: Path, how: str) -> Path:
    if path.exists():
        return path
    if not list((ROOT / "dataset" / "raw").glob("*.zip")):
        pytest.skip("dataset/raw に素材が無い環境")
    pytest.fail(f"{path.relative_to(ROOT)} が無い。素材はあるので生成できるはず: {how}")


@pytest.fixture(scope="module")
def controls() -> dict:
    path = _require(REPORTS / "controls.json", "python ml/run_controls.py")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 指標そのもの


@pytest.mark.unit
def test_t227_macro_f1_matches_sklearn() -> None:
    """T-227 — 自前の macro F1 が scikit-learn と一致する。

    指標を自分で書くなら、権威ある実装と突き合わせる。
    指標が違っていると、そのあとの比較がすべて意味を失う。
    """
    from sklearn.metrics import f1_score

    from ml.metrics import macro_f1

    rng = np.random.default_rng(20260908)
    for _ in range(50):
        n = int(rng.integers(10, 200))
        y = rng.integers(0, 2, n)
        p = rng.integers(0, 2, n)
        assert abs(macro_f1(y, p) - f1_score(y, p, average="macro", zero_division=0)) < 1e-12


@pytest.mark.unit
def test_t228_majority_class_macro_f1_is_analytic() -> None:
    """T-228 — 多数派クラスだけを答えたときの macro F1 が解析解と一致する。

    陽性 n_pos / 陰性 n_neg で全部を陽性(=0)と答えると:
      陽性の precision = n_pos/n, recall = 1 → F1 = 2*n_pos/(n_pos+n)
      陰性の F1 = 0
      macro F1 = 平均
    """
    from ml.metrics import macro_f1

    for n_pos, n_neg in ((1407, 559), (10, 90), (50, 50)):
        y = np.array([0] * n_pos + [1] * n_neg)
        p = np.zeros_like(y)
        n = n_pos + n_neg
        expected = (2 * n_pos / (n_pos + n) + 0.0) / 2
        assert abs(macro_f1(y, p) - expected) < 1e-12


@pytest.mark.unit
def test_t229_bootstrap_is_over_groups_not_images() -> None:
    """T-229 / G-06 — 信頼区間は group 単位で取る。

    画像単位で取ると、同じスライドの別視野が独立標本として数えられ、
    **区間が実際より狭くなる**。狭い区間は「差がある」と言いやすくする方向に効く。

    検算: 同じ 200 枚を 10 個の group に束ねると、独立な単位が 200 から 10 へ減るので、
    区間は画像ごとに別 group の場合より広くなる。

    **極端な場合(全部を 1 group)で確かめてはならない。** group が 1 つだと復元抽出は
    毎回その同じ group を選び、再標本が原標本と完全に一致するので区間幅は 0 になる。
    広がるのではなく消える。この誤った期待値は loop_003 で正しい実装を一度落とした
    (ループログの VERIF-FALSE)。
    """
    from ml.metrics import bootstrap_macro_f1

    rng = np.random.default_rng(7)
    n, group_size = 200, 20
    y = rng.integers(0, 2, n)
    p = y.copy()
    # 誤りは group の中にまとまって出る(同じスライドは似た間違え方をする)
    p[:group_size] = 1 - p[:group_size]

    per_image = np.arange(n)                        # 画像ごとに別 group(独立単位 200)
    clustered = np.arange(n) // group_size          # 10 個の group(独立単位 10)

    lo_i, hi_i = bootstrap_macro_f1(y, p, per_image, n_boot=600, seed=1)[1:]
    lo_g, hi_g = bootstrap_macro_f1(y, p, clustered, n_boot=600, seed=1)[1:]
    assert (hi_g - lo_g) > (hi_i - lo_i), (
        f"独立単位を 200 から 10 へ減らしたのに区間が広がっていない"
        f"(画像単位 {hi_i-lo_i:.4f} / 10 group {hi_g-lo_g:.4f})"
    )


@pytest.mark.unit
def test_t230_bootstrap_interval_contains_point_estimate() -> None:
    """T-230 / G-06 — 95% 区間が点推定を含む。"""
    from ml.metrics import bootstrap_macro_f1

    rng = np.random.default_rng(11)
    n = 300
    y = rng.integers(0, 2, n)
    p = np.where(rng.random(n) < 0.8, y, 1 - y)
    groups = rng.integers(0, 60, n)
    point, lo, hi = bootstrap_macro_f1(y, p, groups, n_boot=500, seed=3)
    assert lo <= point <= hi, f"点推定 {point} が区間 [{lo}, {hi}] の外にある"
    assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------------------- 対照の中身


@pytest.mark.integration
def test_t207_controls_cover_all_three_rulers(controls: dict) -> None:
    """T-207 / G-05 — 対照の成績が三つの物差しすべてで記録されている。

    一つの物差しでしか測っていない対照は、物差しの違いを語る材料にならない。
    """
    for name in ("hue", "canvas_size", "majority"):
        assert name in controls["baselines"], f"対照 {name} が無い"
        for ruler in ("a", "b", "c"):
            r = controls["baselines"][name].get(ruler)
            assert r is not None, f"対照 {name} に物差し {ruler} の結果が無い"
            for key in ("macro_f1", "ci_low", "ci_high", "n_images", "n_groups"):
                assert key in r, f"対照 {name} / 物差し {ruler} に {key} が無い"
            assert r["ci_low"] <= r["macro_f1"] <= r["ci_high"]


@pytest.mark.integration
def test_t211_canvas_size_control_is_pixel_blind(controls: dict) -> None:
    """T-211 / G-11 — キャンバス寸法の対照が画素を一切見ていない。

    この対照の値打ちは「画素を見ずにどこまで当たるか」にある。
    画素を見てしまえば、ただの弱い分類器になって対照にならない。
    """
    feature = controls["baselines"]["canvas_size"]["feature"]
    assert feature == "source_width", f"想定外の特徴量: {feature}"
    src = (ROOT / "ml" / "run_controls.py").read_text(encoding="utf-8")
    assert "def canvas_size_baseline" in src
    body = src.split("def canvas_size_baseline", 1)[1].split("\ndef ", 1)[0]
    for forbidden in ("Image.open", "np.asarray(im", "path"):
        assert forbidden not in body, (
            f"キャンバス寸法の対照が画素に触れている疑い: {forbidden!r} が本体にある"
        )


@pytest.mark.integration
def test_t231_hue_threshold_is_fitted_on_train_only(controls: dict) -> None:
    """T-231 / G-08 — 色相ベースラインの閾値が train だけで決められている。

    test を見て閾値を決めると、それは対照ではなく test に当てた上限になる。
    実測(2026-09-08)の上限 0.9745 は同じデータで閾値を選んだ値であり、
    **train で決めた閾値の成績はそれ以下でなければならない**。
    """
    fitted = controls["baselines"]["hue"]
    assert fitted["threshold_fitted_on"] == "train", "閾値が train 以外で決められている"
    oracle = fitted["oracle_upper_bound_macro_f1"]
    for ruler in ("a", "b", "c"):
        got = fitted[ruler]["macro_f1"]
        assert got <= oracle + 1e-9, (
            f"物差し {ruler} の成績 {got} が、同じデータで閾値を選んだ上限 {oracle} を超えている。"
            "閾値の決め方に test が漏れている疑いがある"
        )


@pytest.mark.integration
def test_t232_controls_are_evaluated_on_the_same_population(controls: dict) -> None:
    """T-232 / F-09 — 三つの物差しが同じ母集団を評価している。

    評価する枚数が違うと、成績の差が「物差しの違い」なのか
    「測った対象の違い」なのか分けられなくなる。
    """
    for name, b in controls["baselines"].items():
        ns = {ruler: b[ruler]["n_images"] for ruler in ("a", "b", "c")}
        assert len(set(ns.values())) == 1, f"対照 {name} の評価枚数が物差しごとに違う: {ns}"


@pytest.mark.integration
def test_t233_hue_control_beats_majority_clearly(controls: dict) -> None:
    """T-233 / G-05 — 色相ベースラインが多数派クラスを明確に上回る。

    上回らないなら、この対照は「色は情報である」ことを示せておらず、
    対照としての役目を果たさない(そのときは対照の作り方を疑う)。
    """
    for ruler in ("a", "b", "c"):
        hue = controls["baselines"]["hue"][ruler]
        maj = controls["baselines"]["majority"][ruler]
        assert hue["ci_low"] > maj["ci_high"], (
            f"物差し {ruler}: 色相 {hue['macro_f1']:.4f} [{hue['ci_low']:.4f}, {hue['ci_high']:.4f}] が "
            f"多数派 {maj['macro_f1']:.4f} [{maj['ci_low']:.4f}, {maj['ci_high']:.4f}] を"
            "区間の重なりなく上回っていない"
        )


@pytest.mark.integration
def test_t234_every_image_is_predicted_exactly_once_per_ruler(controls: dict) -> None:
    """T-234 / F-09 — 各物差しで、全画像がちょうど一度ずつ予測されている。

    fold をまたいで同じ画像を二度数えると、成績も区間も歪む。
    """
    n_total = controls["n_images_total"]
    for name, b in controls["baselines"].items():
        for ruler in ("a", "b", "c"):
            assert b[ruler]["n_images"] == n_total, (
                f"対照 {name} / 物差し {ruler}: 予測 {b[ruler]['n_images']} 枚 ≠ 全 {n_total} 枚"
            )
