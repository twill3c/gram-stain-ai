"""ラベル典拠と取得物のメタデータに対する検査(TEST_SPEC T-101〜T-113)。

期待値の出所:
  - Gram 反応 / 形態: BacDive v2 API(Leibniz Institute DSMZ・CC BY 4.0)。2026-09-08 取得
  - Candida albicans の除外: NCBI Taxonomy taxid 5476 の Lineage。2026-09-08 取得
  - zip のサイズ・SHA-256: 配布元の実測(2026-09-08)

件数は定数で書かない。データが動くたびに壊れる数ではなく、
「欠けが無い」「食い違いが無い」という不変量で書く。
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "dataset" / "metadata"


@pytest.fixture(scope="module")
def labels() -> dict:
    path = META / "labels.json"
    if not path.exists():
        pytest.skip("labels.json が無い。先に python ml/build_labels.py を実行する")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def included(labels: dict) -> list[dict]:
    return [t for t in labels["taxa"] if t["included"]]


@pytest.mark.unit
def test_t101_every_included_taxon_has_gram_authority(included: list[dict]) -> None:
    """T-101 / G-07 — 採用した分類群はすべて、Gram の出所を名指しできる。"""
    missing = [
        t["folder"]
        for t in included
        if not (t.get("gram_source") or {}).get("authority")
        or not (t.get("gram_source") or {}).get("authority_level")
    ]
    assert missing == [], f"Gram の典拠が欠けている分類群: {missing}"


@pytest.mark.unit
def test_t102_exclusions_cite_an_external_authority(labels: dict) -> None:
    """T-102 / G-07 — 除外は理由だけでなく、外部権威の URL を伴う。

    除外はこちらの判断に見えやすい。判断の根拠が外にあることを、URL の実在で示す。
    """
    excluded = [t for t in labels["taxa"] if not t["included"]]
    assert excluded, "除外された分類群が 1 件も無い。Candida albicans は細菌でないため除外されるはず"
    for t in excluded:
        assert t.get("reason"), f"{t['folder']} に除外理由が無い"
        url = t.get("authority_url", "")
        assert url.startswith("http"), f"{t['folder']} に外部権威の URL が無い"

    candida = next((t for t in excluded if t["folder"] == "Candida albicans"), None)
    assert candida is not None, "Candida albicans が除外されていない"
    assert "5476" in candida["authority_url"], "NCBI Taxonomy taxid 5476 を指していない"
    # 権威が返した Lineage をそのまま持っていること(要約ではなく本文)
    assert "Fungi" in candida["evidence"], "権威の Lineage に Fungi が含まれていない"


@pytest.mark.unit
def test_t103_gram_values_are_binary(included: list[dict]) -> None:
    """T-103 / G-07 — Gram の値は positive / negative のみ。"""
    bad = {t["folder"]: t["gram"] for t in included if t["gram"] not in ("positive", "negative")}
    assert bad == {}, f"想定外の Gram 値: {bad}"


@pytest.mark.unit
def test_t104_all_shape_terms_are_normalised(labels: dict) -> None:
    """T-104 / SPEC §2.1 — 権威が使う形態の語がすべて正規化表で解決されている。

    未知の語を黙って捨てると、捨てた分だけ一致率が上がって見える。
    """
    unmapped = labels["summary"]["unmapped_shape_terms"]
    assert unmapped == [], f"正規化表に無い形態の語がある: {unmapped}"


@pytest.mark.unit
def test_t105_stage_b_exclusions_are_explained(included: list[dict]) -> None:
    """T-105 / SPEC §2.1 — Stage B の対象外には理由が書かれている。"""
    silent = [t["folder"] for t in included if not t["stage_b"] and not t["stage_b_excluded_reason"]]
    assert silent == [], f"Stage B 対象外の理由が書かれていない分類群: {silent}"


@pytest.mark.unit
def test_t106_grouping_keys_exist_for_every_taxon(included: list[dict]) -> None:
    """T-106 / F-09 — 三つの物差しが切る単位(種・属・科)が全分類群で埋まっている。

    ここが欠けると、物差し B・C は「切ったつもりで切れていない」状態になる。
    """
    for key in ("group_species", "group_genus", "group_family"):
        missing = [t["folder"] for t in included if not t.get(key)]
        assert missing == [], f"{key} が欠けている分類群: {missing}"


@pytest.mark.unit
def test_t107_folder_names_include_outdated_genus(included: list[dict]) -> None:
    """T-107 / F-09 — フォルダ名の属と現行の属が食い違う分類群が存在する。

    **食い違うのが正常である。** データセットのフォルダ名は 2020 年の
    Lactobacillus 属分割(Zheng et al. 2020)より前の名前を含む。
    フォルダ名をそのまま属とみなす実装は、物差し C を誤って切る。
    この検査が落ちるとしたら、権威の解決が効いていないということである。
    """
    renamed = [t for t in included if t["name_is_current"]]
    assert renamed, "旧名のフォルダが 1 件も検出されていない。属の解決が効いていない疑いがある"

    # フォルダ名の属だけで切ると、現行の属より粗い単位になることを示す
    folder_genera = {t["folder"].split()[0] for t in included}
    current_genera = {t["group_genus"] for t in included}
    assert len(current_genera) > len(folder_genera), (
        f"フォルダ名の属 {len(folder_genera)} 種類に対し現行の属 {len(current_genera)} 種類。"
        "分割後の属が反映されていない"
    )


@pytest.mark.unit
def test_t108_phylum_alone_cannot_decide_gram(included: list[dict]) -> None:
    """T-108 / SPEC §2.1 — 門だけから Gram を決められない反例が実在する。

    これは SPEC が「種名→Gram は外部権威に取る」と定める理由そのものである。
    近道(門で決め打つ)が成立しないことを、データに対して示しておく。
    """
    by_phylum: dict[str, set[str]] = {}
    for t in included:
        ph = (t.get("lineage") or {}).get("phylum")
        if ph:
            by_phylum.setdefault(ph, set()).add(t["gram"])
    mixed = {ph: sorted(g) for ph, g in by_phylum.items() if len(g) > 1}
    assert mixed, (
        "門と Gram が一対一に対応してしまっている。近道が成立するように見えるが、"
        "Veillonella(Bacillota 門・Gram 陰性)が権威で確認できているはずである"
    )


@pytest.mark.integration
def test_t109_declared_and_actual_sizes_agree() -> None:
    """T-109 / N-03 — 配布元の公表サイズと実測サイズが一致する。"""
    path = META / "raw_manifest.json"
    if not path.exists():
        pytest.skip("raw_manifest.json が無い。先に python ml/download_dataset.py を実行する")
    m = json.loads(path.read_text(encoding="utf-8"))
    bad = [
        (f["filename"], f["declared_size"], f["actual_size"])
        for f in m["files"]
        if f["declared_size"] != f["actual_size"]
    ]
    assert bad == [], f"サイズ不一致: {bad}"
    for f in m["files"]:
        assert re.fullmatch(r"[0-9a-f]{64}", f["sha256"]), f"SHA-256 の形が不正: {f['filename']}"


@pytest.mark.integration
def test_t110_duplicate_zip_is_detected() -> None:
    """T-110 / N-03 — 配布 zip の重複が検出されている。

    配布元には同一内容の zip が 2 本ある(2026-09-08 実測)。
    展開して両方を学習に入れると、同じ画像が train と test の両方に入りうる。
    **重複を「見つけていること」を検査対象にする** —— 見つけていなければ、
    後段の leakage 検査は重複を正常と見なして通ってしまう。
    """
    path = META / "raw_manifest.json"
    if not path.exists():
        pytest.skip("raw_manifest.json が無い。先に python ml/download_dataset.py を実行する")
    m = json.loads(path.read_text(encoding="utf-8"))
    if len(m["files"]) < 34:
        pytest.skip(f"取得が未完了({len(m['files'])}/34 本)")
    dups = m["duplicate_sha256_groups"]
    assert dups, "同一 SHA-256 の zip が検出されていない。重複検出が働いていない疑いがある"


@pytest.mark.unit
def test_t111_mutation_removing_normalisation_breaks_stage_b(labels: dict) -> None:
    """T-111 — 変異検査: 形態の正規化を外すと Stage B から Gram 陰性球菌が消える。

    この層は赤から始まっていない(loop_001 の PROC-SKIP)。
    代わりに「規則を壊すと結果が変わる」ことを示し、検査が規則を実際に見ていることを確かめる。

    正規化なし(生の語のまま多数決)にすると:
      Veillonella      球 8 / 卵 5 / 卵円 1 / 桿 1 → 一致率 53.3% で閾値 90% 未満
      Neisseria gonorrhoeae  球 14 / 卵 2        → 一致率 87.5% で閾値 90% 未満
    この 2 件は採用分類群で唯一の Gram 陰性球菌なので、Stage B の 4 クラスが 3 クラスになる。
    """
    threshold = labels["rules"]["shape_min_agreement"]
    included = [t for t in labels["taxa"] if t["included"]]

    def stage_b_pairs(use_normalised: bool) -> set[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        for t in included:
            counts = t["shape_counts_normalised"] if use_normalised else t["shape_counts_raw"]
            total = sum(counts.values())
            if not total:
                continue
            top = max(counts, key=lambda k: counts[k])
            if counts[top] / total < threshold:
                continue
            if t.get("stage_b_excluded_reason") and not use_normalised:
                pass
            shape = {"coccus-shaped": "coccus", "oval-shaped": "coccus", "ovoid-shaped": "coccus",
                     "rod-shaped": "bacillus", "curved-shaped": "bacillus"}.get(top, top)
            out.add((t["gram"], shape))
        return out

    with_norm = stage_b_pairs(True)
    without_norm = stage_b_pairs(False)

    assert ("negative", "coccus") in with_norm, "正規化ありでも Gram 陰性球菌が居ない"
    assert ("negative", "coccus") not in without_norm, (
        "正規化を外しても Gram 陰性球菌が残った。正規化が結果を変えていないなら、"
        "SPEC §2.1 の説明(正規化しないと 4 クラスが 3 クラスになる)が実態と合っていない"
    )
    assert len(without_norm) < len(with_norm), (
        f"正規化なし {sorted(without_norm)} が 正規化あり {sorted(with_norm)} を下回っていない"
    )


@pytest.fixture(scope="module")
def measured() -> dict:
    path = META / "dataset_measured.json"
    if not path.exists():
        pytest.skip("dataset_measured.json が無い。先に python ml/verify_dataset.py を実行する")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.integration
def test_t114_declared_and_measured_counts_are_both_recorded(measured: dict) -> None:
    """T-114 / SPEC §2.0 — 記載枚数と実測枚数の両方が記録され、食い違いが隠れていない。

    数そのものは定数で書かない(データが動けば変わる)。
    検査するのは「両方が記録されていること」と「食い違いを実測が上書きしていること」である。
    """
    assert measured["declared_images"] > 0, "配布元の記載枚数が記録されていない"
    assert measured["measured_images"] > 0, "実測枚数が記録されていない"
    # 食い違いがあるなら、それが見える形で残っていること
    if measured["declared_images"] != measured["measured_images"]:
        spec = (ROOT / "SPEC.md").read_text(encoding="utf-8")
        assert str(measured["measured_images"]) in spec.replace(",", ""), (
            "記載と実測が食い違うのに、実測値が SPEC に書かれていない"
        )


@pytest.mark.integration
def test_t115_byte_identical_images_are_detected(measured: dict) -> None:
    """T-115 / G-03 — バイト完全一致の画像重複が検出されている。

    2026-09-08 実測: 66 組。うち 60 組は重複 zip 由来、**6 組は同一 zip の中**にある。
    後者があるため、重複 zip を取り除くだけでは重複は消えない。
    重複を見つけていない状態で分割すると、同じ画像が train と test の両方に入る。
    """
    assert measured["duplicate_groups"] > 0, (
        "バイト一致の重複が 1 件も検出されていない。重複検出が働いていない疑いがある"
    )
    assert measured["unique_images_by_sha256"] < measured["measured_images"], (
        "一意枚数が実測枚数と同じ。重複が数えられていない"
    )
    assert measured["duplicate_extra_copies"] == (
        measured["measured_images"] - measured["unique_images_by_sha256"]
    ), "余分な複製の数と、実測枚数マイナス一意枚数が合わない"


@pytest.mark.integration
def test_t116_filenames_are_not_safe_as_group_ids(measured: dict) -> None:
    """T-116 / G-04 — ファイル名が group ID として使えないことを、実データで示す。

    連番の隣どうしがバイト一致する組が実在する(`Enterococcus faecium/24.png` = `25.png` ほか)。
    **番号が違うことは中身が違うことを意味しない。**
    この検査が落ちるなら、ファイル名を group ID にする設計を SPEC から外してよいことになる。
    """
    path = META / "images.jsonl"
    if not path.exists():
        pytest.skip("images.jsonl が無い")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    by_digest: dict[str, list[dict]] = {}
    for r in rows:
        by_digest.setdefault(r["sha256"], []).append(r)

    within_zip_dups = [
        [x["member"] for x in v]
        for v in by_digest.values()
        if len(v) > 1 and len({x["zip"] for x in v}) == 1
    ]
    assert within_zip_dups, (
        "同一 zip 内のバイト一致が 1 件も無い。ファイル名を group ID にしてよいことになるので、"
        "SPEC §6.1 の根拠を測り直すこと"
    )


@pytest.mark.validation
def test_t112_text_hygiene_is_clean() -> None:
    """T-112 / N-07 — 日本語本文に字種違反が無い。"""
    targets = [
        str(ROOT / "SPEC.md"),
        str(ROOT / "TEST_SPEC.md"),
        str(ROOT / "README.md"),
        str(ROOT / "docs"),
    ]
    targets = [t for t in targets if Path(t).exists()]
    res = subprocess.run(
        [sys.executable, str(ROOT / "harness" / "text_hygiene.py"), *targets],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert res.returncode == 0, f"字種検査が落ちた:\n{res.stdout}\n{res.stderr}"


@pytest.mark.validation
def test_t113_every_spec_gate_is_referenced_by_a_case() -> None:
    """T-113 — SPEC のゲート表の各 G-xx が TEST_SPEC のケース一覧から参照されている。

    宣言しただけのゲートは、誰も守っていないのに守られて見える(scaffold 規約)。
    """
    spec = (ROOT / "SPEC.md").read_text(encoding="utf-8")
    test_spec = (ROOT / "TEST_SPEC.md").read_text(encoding="utf-8")
    gates = set(re.findall(r"^\| (G-\d+) \|", spec, flags=re.MULTILINE))
    assert gates, "SPEC にゲート表が見つからない"
    referenced = set(re.findall(r"G-\d+", test_spec))
    orphans = sorted(gates - referenced)
    assert orphans == [], f"どのケースからも参照されていないゲート: {orphans}"
