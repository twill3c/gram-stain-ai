"""ONNX 変換と二実装照合の検査(TEST_SPEC T-204・T-240〜T-244)。

期待値の出所:
  - 照合の許容差: SPEC 較正ゲート G-01(max abs 差 < 1e-4・argmax 一致率 100%)
  - 配信サイズ: SPEC N-01(ONNX 30MB 以下)
  - 出荷モデルの予測: reports/cnn.json(loop_004 で測った成績の出どころ)
  - 帰属表示: DATA_LICENSE.md(CC BY 4.0 は帰属を要求する)

**二実装照合は非循環オラクルである。** PyTorch と ONNX Runtime は同じ重みを別実装で走らせる。
片方が壊れれば一致しない。こちらが期待値を書いているのではない。

**実装より先に書く。**
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PUBLIC = ROOT / "public" / "models"


def _require(path: Path, how: str) -> Path:
    if path.exists():
        return path
    if not list((ROOT / "dataset" / "raw").glob("*.zip")):
        pytest.skip("dataset/raw に素材が無い環境")
    pytest.fail(f"{path.relative_to(ROOT)} が無い。素材はあるので生成できるはず: {how}")


@pytest.fixture(scope="module")
def parity() -> dict:
    return json.loads(_require(REPORTS / "onnx_parity.json",
                               "python -m ml.export_onnx").read_text(encoding="utf-8"))


@pytest.mark.integration
def test_t204_pytorch_and_onnxruntime_agree(parity: dict) -> None:
    """T-204 / G-01 — PyTorch と ONNX Runtime の出力が一致する。

    測る量は **argmax 一致率**と **softmax 確率の差**である(SPEC §5.1 で引き直した)。
    生 logits の絶対差では測らない —— logits は有界でないので閾値がモデルの確信度に比例し、
    しかも製品が表示するのは確率であって logits ではない。
    """
    assert parity["argmax_agreement"] == 1.0, (
        f"argmax 一致率 {parity['argmax_agreement']:.6f}。"
        f"食い違い {parity['argmax_mismatches']} 件"
    )
    assert parity["max_prob_diff"] < 1e-3, (
        f"softmax 確率の最大差 {parity['max_prob_diff']:.3e} が許容差 1e-3 以上。"
        "同じ重みを別実装で走らせて一致しないなら、どちらかが壊れている"
    )
    assert parity["n_images"] > 0, "照合に使った画像が 0 枚"


@pytest.mark.integration
def test_t245_superseded_gate_is_recorded_not_hidden(parity: dict) -> None:
    """T-245 / SPEC §5.1 — 引き直す前のゲートの結果が出荷物に残っている。

    **ゲートを引き直したこと自体を隠さない。** 旧い書き方(生 logits の絶対差 < 1e-4)は
    実測 1.0133e-4 で通らなかった。その事実と、引き直した理由が記録されていなければ、
    「最初から通っていた」ように読めてしまう。
    """
    sup = parity["gate"]["superseded"]
    assert "rule" in sup and "measured" in sup and "why_replaced" in sup
    assert sup["passed"] == (sup["measured"] < 1e-4), (
        "旧ゲートの合否が、記録された実測値と整合していない"
    )
    assert parity["max_abs_diff_logits"] == sup["measured"], (
        "生 logits の絶対差が二か所で食い違っている"
    )


@pytest.mark.integration
def test_t240_onnx_fits_the_delivery_budget(parity: dict) -> None:
    """T-240 / N-01 — 配信する ONNX が 30MB 以下。"""
    mb = parity["onnx_megabytes"]
    assert mb <= 30.0, f"ONNX が {mb:.2f}MB で配信目標 30MB を超える"


@pytest.mark.integration
def test_t241_onnx_signature_matches_what_the_web_layer_expects(parity: dict) -> None:
    """T-241 / F-01 — 入出力の名前と形が、Web 層が前提とする形になっている。

    **境界をまたぐ契約は、テストできる場所に置く**(HC-190)。
    名前や形が変わったとき、Web 側で動かして初めて気づくのでは遅い。
    """
    sig = parity["signature"]
    assert sig["input_names"] == ["input"], f"入力名が想定と違う: {sig['input_names']}"
    assert sig["output_names"] == ["logits"], f"出力名が想定と違う: {sig['output_names']}"
    assert sig["input_shape"][1:] == [3, 224, 224], f"入力形が想定と違う: {sig['input_shape']}"
    assert sig["output_shape"][1:] == [2], f"出力形が想定と違う: {sig['output_shape']}"
    assert sig["dynamic_batch"] is True, "バッチ次元が動的でない"


@pytest.mark.integration
def test_t242_shipped_artifacts_exist_with_attribution() -> None:
    """T-242 / N-04 — 配信物が揃い、CC BY 4.0 の帰属が同梱されている。

    モデルは CC BY 4.0 のデータの派生物である。帰属を配信物から切り離さない。
    """
    for name in ("model.onnx", "labels.json", "model_metadata.json"):
        _require(PUBLIC / name, "python -m ml.export_onnx")

    meta = json.loads((PUBLIC / "model_metadata.json").read_text(encoding="utf-8"))
    for key in ("model_version", "architecture", "input_size", "task",
                "dataset", "dataset_doi", "dataset_license", "attribution",
                "trained_at", "git_commit", "metrics"):
        assert key in meta, f"model_metadata.json に {key} が無い"
    assert "10.17632/cvkgfzp7ck.1" in meta["dataset_doi"]
    assert meta["dataset_license"] == "CC BY 4.0"
    assert "Jamshidi" in meta["attribution"], "配布元の帰属表示が入っていない"

    labels = json.loads((PUBLIC / "labels.json").read_text(encoding="utf-8"))
    assert labels["classes"] == ["gram_positive", "gram_negative"], (
        f"クラスの並びが想定と違う: {labels['classes']}"
    )


@pytest.mark.integration
def test_t243_metadata_carries_all_three_rulers(parity: dict) -> None:
    """T-243 / F-09 — 配信するメタデータが三つの物差しをすべて載せている。

    出荷モデルの test macro F1(1.0)だけを載せると、
    **Web 画面がその数だけを示すことになる**。本プロジェクトが警告している当のものである。
    数を配るなら、三つの物差しも一緒に配る。
    """
    meta = json.loads((PUBLIC / "model_metadata.json").read_text(encoding="utf-8"))
    m = meta["metrics"]
    for ruler in ("a", "b", "c"):
        assert ruler in m["rulers"], f"物差し {ruler} が配信メタデータに無い"
        assert "macro_f1" in m["rulers"][ruler]
    assert "hue_baseline" in m, "非学習の対照が配信メタデータに無い"
    for ruler in ("a", "b", "c"):
        assert ruler in m["hue_baseline"], f"対照の物差し {ruler} が無い"


@pytest.mark.integration
def test_t244_exported_model_reproduces_the_measured_predictions(parity: dict) -> None:
    """T-244 / G-08 — 出荷する ONNX が、成績を測ったときと同じ予測を返す。

    **配るものと測ったものが同じでなければ、測った数は配るものの成績ではない。**
    出荷モデルの test 予測を PyTorch 側で取り直し、ONNX と argmax まで一致させる。
    """
    t = parity["shipped_test"]
    assert t["n_images"] > 0
    assert t["argmax_agreement"] == 1.0, (
        f"出荷モデルの test 予測が ONNX と一致しない(食い違い {t['mismatches']} 件)"
    )
    cnn = json.loads((REPORTS / "cnn.json").read_text(encoding="utf-8"))
    assert abs(t["onnx_macro_f1"] - cnn["shipped_model"]["test"]["macro_f1"]) < 1e-9, (
        f"ONNX で測り直した test macro F1 {t['onnx_macro_f1']} が "
        f"記録 {cnn['shipped_model']['test']['macro_f1']} と違う"
    )
