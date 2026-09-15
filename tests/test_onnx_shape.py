"""Stage B(形)の出荷物に対する検査(TEST_SPEC T-274〜T-276)。

期待値の出所:
  - 照合の許容差: SPEC 較正ゲート G-01(argmax 一致率 100%・確率の最大差 < 1e-3)
  - 配信サイズ: SPEC N-01(30MB 以下)
  - 陰性球菌の誤り率: reports/cnn_shape_cells.json(loop_011 で fold の予測から数えた)
  - 画面の約束: SPEC §3.11

**実装より先に書く。**
"""

from __future__ import annotations

import json
from pathlib import Path

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
    return json.loads(_require(REPORTS / "onnx_parity_shape.json",
                               "python -m ml.export_onnx --target shape").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def meta() -> dict:
    return json.loads(_require(PUBLIC / "model_shape_metadata.json",
                               "python -m ml.export_onnx --target shape").read_text(encoding="utf-8"))


@pytest.mark.integration
def test_t274_shape_model_parity_and_signature(parity: dict) -> None:
    """T-274 / G-01 / N-01 — 形のモデルも PyTorch と ONNX Runtime で一致し、Web 層の前提の形をしている。"""
    assert parity["n_images"] == 1842, f"照合した枚数 {parity['n_images']} が Stage B の 1,842 枚と違う"
    assert parity["argmax_agreement"] == 1.0, f"argmax の食い違い {parity['argmax_mismatches']} 件"
    assert parity["max_prob_diff"] < 1e-3, f"確率の最大差 {parity['max_prob_diff']:.3e}"
    sig = parity["signature"]
    assert sig["input_names"] == ["input"] and sig["output_names"] == ["logits"]
    assert sig["input_shape"][1:] == [3, 224, 224] and sig["output_shape"][1:] == [2]
    assert sig["dynamic_batch"] is True
    assert parity["onnx_megabytes"] <= 30.0


@pytest.mark.integration
def test_t275_shape_shipped_model_touches_test_once(parity: dict) -> None:
    """T-275 / G-08 — 形の出荷モデルは train だけで学習し、test には一度だけ当てた。"""
    cnn = json.loads((REPORTS / "cnn_shape.json").read_text(encoding="utf-8"))
    s = cnn["shipped_model"]
    assert s["trained_on"] == ["train"]
    assert s["test_evaluations"] == 1
    assert s["checkpoint"].endswith("best_model_shape.pth")
    t = parity["shipped_test"]
    assert t["argmax_agreement"] == 1.0, f"出荷モデルの test 予測が ONNX と食い違う({t['mismatches']} 件)"
    assert abs(t["onnx_macro_f1"] - s["test"]["macro_f1"]) < 1e-9


@pytest.mark.integration
def test_t276_shape_metadata_does_not_hide_the_negative_cocci_collapse(meta: dict) -> None:
    """T-276 / SPEC §3.11 — 配信メタデータが三物差し・対照・陰性球菌の崩れを載せている。

    出荷モデルの test の数と三物差しだけを配ると、画面は「科を抜いても 0.83」とだけ言える。
    §3.10 の崩れは、配るものに付けて配る。
    """
    labels = json.loads((PUBLIC / "labels_shape.json").read_text(encoding="utf-8"))
    assert labels["classes"] == ["coccus", "bacillus"], f"並びが違う: {labels['classes']}"
    assert meta["classes"] == labels["classes"]

    m = meta["metrics"]
    for r in ("a", "b", "c"):
        assert {"macro_f1", "ci_low", "ci_high"} <= set(m["rulers"][r])
        assert {"macro_f1", "ci_low", "ci_high"} <= set(m["shape_rule_baseline"][r])
    assert m["verdict"]["condition_1"] is True and m["verdict"]["condition_2"] is True
    perm = m["label_permutation_control"]
    assert perm["observed"] <= perm["null_q975"]

    cells = json.loads((REPORTS / "cnn_shape_cells.json").read_text(encoding="utf-8"))["rulers"]
    nc = meta["negative_cocci"]
    for r in ("a", "b", "c"):
        assert nc["error_rate"][r] == cells[r]["cnn"]["negative x coccus"]["error_rate"], r
    assert nc["n_images"] == cells["a"]["cnn"]["negative x coccus"]["n"]
    assert sorted(nc["taxa"]) == ["Neisseria gonorrhoeae", "Veillonella"]
    pct = f"{round(nc['error_rate']['c'] * 100)}%"
    assert pct in nc["caveat"], f"注意書きに C の誤り率 {pct} が入っていない: {nc['caveat']}"
    assert "診断" not in nc["caveat"].replace("診断確率ではない", "")
    assert "not_a_medical_device" in meta
