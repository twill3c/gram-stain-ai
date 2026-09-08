"""CNN の評価に対する検査(TEST_SPEC T-206・T-208・T-235〜T-239)。

期待値の出所:
  - 合否条件: SPEC §3.4(**測る前に**書いた二条件)
  - 対照の値: reports/controls.json(loop_003 実測。CNN より先に測った)
  - 配信サイズ: SPEC N-01(ONNX 30MB 以下)
  - チャンス水準: ラベルを乱数に置換したときの macro F1。解析的な期待値ではなく、
    実際に置換して測った値を使う(HC-163: 損失の比では対照にならない)

**実装より先に書く。**
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RULERS = ("a", "b", "c")


def _require(path: Path, how: str) -> Path:
    if path.exists():
        return path
    if not list((ROOT / "dataset" / "raw").glob("*.zip")):
        pytest.skip("dataset/raw に素材が無い環境")
    pytest.fail(f"{path.relative_to(ROOT)} が無い。素材はあるので生成できるはず: {how}")


@pytest.fixture(scope="module")
def cnn() -> dict:
    return json.loads(_require(REPORTS / "cnn.json", "python -m ml.train").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def controls() -> dict:
    return json.loads(_require(REPORTS / "controls.json", "python -m ml.run_controls").read_text(encoding="utf-8"))


@pytest.mark.integration
def test_t208_cnn_scored_on_all_three_rulers_with_ci(cnn: dict) -> None:
    """T-208 / G-06 — CNN の macro F1 が三つの物差しすべてで 95% 区間つきで記録されている。"""
    for ruler in RULERS:
        r = cnn["rulers"].get(ruler)
        assert r is not None, f"物差し {ruler} の結果が無い"
        for key in ("macro_f1", "ci_low", "ci_high", "n_images", "n_groups"):
            assert key in r, f"物差し {ruler} に {key} が無い"
        assert r["ci_low"] <= r["macro_f1"] <= r["ci_high"]


@pytest.mark.integration
def test_t236_cnn_and_controls_share_the_population(cnn: dict, controls: dict) -> None:
    """T-236 / F-09 — CNN と対照が同じ母集団で測られている。

    測った枚数が違えば、差の出所が「物差し」なのか「対象」なのか分けられない。
    """
    n = controls["n_images_total"]
    for ruler in RULERS:
        assert cnn["rulers"][ruler]["n_images"] == n, (
            f"物差し {ruler}: CNN {cnn['rulers'][ruler]['n_images']} 枚 ≠ 対照 {n} 枚"
        )


@pytest.mark.integration
def test_t206_label_permutation_falls_to_chance(cnn: dict) -> None:
    """T-206 / G-09 — ラベルを乱数に置換すると成績がチャンス水準へ落ちる。

    学習系そのものの陽性対照である。置換しても成績が出るなら、
    成績の出所は入力の信号ではなく、手続きのどこかにある漏れである。

    判定は**多数派クラスの水準を信頼区間の重なりなく上回らないこと**とする。
    ラベルが無作為なら、最善の戦略は多数派クラスを答えることだからである。
    """
    perm = cnn.get("permutation_control")
    assert perm is not None, "ラベル置換の対照が無い"
    for ruler, r in perm.items():
        maj = cnn["majority_reference"][ruler]
        assert r["ci_low"] <= maj["ci_high"], (
            f"物差し {ruler}: ラベルを乱数に置換したのに macro F1 "
            f"{r['macro_f1']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}] が "
            f"多数派 {maj['macro_f1']:.4f} [{maj['ci_low']:.4f}, {maj['ci_high']:.4f}] を"
            "区間の重なりなく上回った。学習系に漏れがある"
        )


@pytest.mark.integration
def test_t237_epoch_budget_was_chosen_without_test(cnn: dict) -> None:
    """T-237 / G-08 — エポック予算が test を見ずに決められている。

    予算を test で決めれば、それは test に当てた上限であって成績ではない。
    """
    e = cnn["epoch_budget"]
    assert e["chosen_on"] == "ruler_a_val", f"予算の決め方が想定と違う: {e['chosen_on']}"
    assert e["chosen"] >= 1
    assert e["curve"], "選定に使った曲線が残っていない(あとから検算できない)"


@pytest.mark.integration
def test_t238_model_fits_the_delivery_budget(cnn: dict) -> None:
    """T-238 / N-01 — 骨格の fp32 サイズが配信目標 30MB 以下。

    ここで落ちるなら、量子化するか骨格を変えるかであって、目標を動かすことではない。
    """
    mb = cnn["model"]["fp32_megabytes"]
    assert mb <= 30.0, f"骨格 {cnn['model']['architecture']} は fp32 {mb:.1f}MB で 30MB を超える"


@pytest.mark.integration
def test_t235_verdict_follows_the_preregistered_conditions(cnn: dict, controls: dict) -> None:
    """T-235 / SPEC §3.4 — 合否が、測る前に決めた二条件から機械的に出ている。

    **判定をこちらの言葉で書かない。** 書いてよいのは条件だけで、
    条件に当てはめる仕事は機械にさせる。そうしないと、数を見てから
    「効いた」と言える理由を探すことになる。

    条件 1: 三つの物差しすべてで、色の規則を信頼区間の重なりなく上回る
    条件 2: 色の規則が落ちる分類群で明確に上回る
    """
    v = cnn["verdict"]
    hue = controls["baselines"]["hue"]

    expected_1 = all(
        cnn["rulers"][r]["ci_low"] > hue[r]["ci_high"] for r in RULERS
    )
    assert v["condition_1_beats_hue_on_all_rulers"] == expected_1, (
        f"条件 1 の判定が実際の区間と合わない(記録 {v['condition_1_beats_hue_on_all_rulers']} / "
        f"実際 {expected_1})"
    )

    # 条件 2 は「色の規則が落ちる分類群」でのみ比べる。対象の選び方も記録されていること
    assert "hard_taxa" in v, "条件 2 の対象分類群が記録されていない"
    assert v["hard_taxa"], "対象分類群が空。色の規則が落ちる分類群は実測で存在する"
    assert "condition_2_beats_on_hard_taxa" in v

    # 二条件のどちらも成立しないなら、結論は「足していない」でなければならない
    if not v["condition_1_beats_hue_on_all_rulers"] and not v["condition_2_beats_on_hard_taxa"]:
        assert v["adds_nothing"] is True, (
            "二条件のどちらも成立していないのに、結論が『何も足していない』になっていない"
        )


@pytest.mark.integration
def test_t239_shipped_model_never_saw_its_test_set(cnn: dict) -> None:
    """T-239 / G-08 — 出荷するモデルの学習に test が入っていない。

    出荷モデルは物差し A の 70/15/15 で学習する。
    train と val で作り、test は最後に一度だけ当てる。
    """
    s = cnn["shipped_model"]
    assert s["trained_on"] == ["train"], f"学習に使った分割が想定と違う: {s['trained_on']}"
    assert s["selected_on"] == "val"
    assert s["test_evaluations"] == 1, (
        f"test に {s['test_evaluations']} 回当てている。1 回でなければ、"
        "その成績はモデル選択の産物を含む"
    )
