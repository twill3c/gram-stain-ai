"""labels.json / raw_manifest.json から docs/DATASET.md を生成する。

手で書くと、ラベル表を作り直したときに文書だけ古いまま残る(SPEC-DRIFT)。
数の入る文書は生成物にして、出所を一つにしておく。
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "dataset" / "metadata"
DOCS = ROOT / "docs"
JST = timezone(timedelta(hours=9))

GRAM_JA = {"positive": "陽性", "negative": "陰性"}
SHAPE_JA = {"coccus": "球菌", "bacillus": "桿菌", None: "—"}


def main() -> int:
    labels = json.loads((META / "labels.json").read_text(encoding="utf-8"))
    taxa = labels["taxa"]
    included = [t for t in taxa if t["included"]]
    s = labels["summary"]

    manifest_path = META / "raw_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    measured_path = META / "dataset_measured.json"
    measured = json.loads(measured_path.read_text(encoding="utf-8")) if measured_path.exists() else None

    L: list[str] = []
    add = L.append

    add("# DATASET.md — 採録した分類群とラベルの典拠")
    add("")
    add(f"<!-- ml/build_docs.py が生成。手で編集しない。生成日時 {datetime.now(JST).isoformat(timespec='seconds')} -->")
    add("")
    add("ラベル(Gram 反応・形態)は画像からではなく種名から決まる。")
    add("その対応表は外部権威に取っている(SPEC §2.1・較正ゲート G-07)。")
    add("")
    add(f"- 権威: **{labels['authority']['primary']}**({labels['authority']['primary_license']})")
    add(f"- 取得日時: {labels['authority']['accessed_at']}")
    add(f"- 引用: {labels['authority']['primary_citation']}")
    add("")

    add("## 採用規則")
    add("")
    add(f"- Gram: {labels['rules']['gram_rule']}")
    add(f"- 形態: 正規化後の一致率 {labels['rules']['shape_min_agreement']:.0%} 以上のときだけ採用")
    add("- 形態の語の正規化:")
    add("")
    add("| 権威が使う語 | 正規化後 |")
    add("|---|---|")
    for raw, norm in labels["rules"]["shape_normalisation"].items():
        add(f"| `{raw}` | {SHAPE_JA.get(norm, norm)} |")
    add("")
    add("**この正規化は結果を変える。** 正規化しないと `Veillonella`(53.3%)と")
    add("`Neisseria gonorrhoeae`(87.5%)が一致率不足で落ち、Stage B から Gram 陰性球菌が")
    add("全部消えて 4 クラスが 3 クラスになる(検査 T-111)。")
    add("")

    add("## 採録の要約")
    add("")
    add("| 項目 | 実測 |")
    add("|---|---|")
    add(f"| 配布元の分類群 | {s['taxa_total']} |")
    add(f"| 採用 | {s['taxa_included']} |")
    add(f"| Gram 陽性 / 陰性 | {s['gram_positive']} / {s['gram_negative']} |")
    add(f"| 一致率 100% でない Gram ラベル | {', '.join('`'+x+'`' for x in s['gram_tier_b']) or 'なし'} |")
    add(f"| Stage B 対象 | {s['stage_b_taxa']} |")
    add(f"| Stage B 対象外 | {', '.join('`'+x+'`' for x in s['stage_b_excluded']) or 'なし'} |")
    add(f"| 科の数 | {len(s['families'])} |")
    if manifest:
        files = manifest["files"]
        total = sum(f["actual_size"] for f in files)
        dups = manifest["duplicate_sha256_groups"]
        add(f"| 配布 zip | {len(files)} 本 / {total:,} バイト |")
        add(f"| 同一内容の重複 zip | {len(dups)} 組 |")
    add("")

    if measured:
        add("## 配布物の実測(展開せず zip の中身を数えた)")
        add("")
        add("| 項目 | 実測 |")
        add("|---|---|")
        add(f"| 走査した zip | {measured['zips_scanned']} 本 |")
        add(f"| フォルダ(分類群) | {measured['folders']} |")
        add(f"| 配布元の記載枚数 | {measured['declared_images']:,} |")
        add(f"| **実測枚数** | **{measured['measured_images']:,}** |")
        add(f"| 一意な画像(SHA-256) | {measured['unique_images_by_sha256']:,} |")
        add(f"| バイト完全一致の重複 | {measured['duplicate_groups']} 組 / 余分な複製 {measured['duplicate_extra_copies']} 枚 |")
        add(f"| 画像形式 | 全て正方: {measured['all_square']} / モード: {', '.join(measured['modes'])} |")
        add(f"| キャンバス幅 | {measured['width_min']}〜{measured['width_max']} px({measured['distinct_widths']} 通り) |")
        add("")
        gap = measured["declared_images"] - measured["measured_images"]
        if gap:
            add(f"**記載 {measured['declared_images']:,} 枚に対し実測 {measured['measured_images']:,} 枚"
                f"(差 {gap:,} 枚)。** 記載が結合前後のどちらを指すのか判別できないため、")
            add("本プロジェクトは実測値のみを使う。")
            add("")
        add("### キャンバス寸法は分類群の指紋になるか(G-11 の材料)")
        add("")
        add(f"- 寸法が 1 通りしかないフォルダ: {len(measured['folders_with_single_canvas_size'])} 個")
        add(f"- 単一のフォルダにしか現れない寸法: {measured['canvas_sizes_unique_to_one_folder']} 通り")
        add(f"- その寸法を持つ画像: {measured['images_with_fingerprint_size']} 枚"
            f"({measured['images_with_fingerprint_size']/measured['measured_images']:.1%})")
        add("")
        add("寸法だけで分類群が特定できる画像は少数だが、**少数でもゼロではない**。")
        add("画素を一切見ない分類器の成績を G-11 で実測してから、CNN の成績を語る。")
        add("")

    fam = Counter(t["group_family"] for t in included)
    add("## 科ごとの分類群数(物差し C が切る単位)")
    add("")
    add("| 科 | 分類群数 |")
    add("|---|---|")
    for f, c in fam.most_common():
        add(f"| {f} | {c} |")
    add("")
    add(f"最大の科は **{fam.most_common(1)[0][0]}** の {fam.most_common(1)[0][1]} 分類群。")
    add("種を 1 つ抜いても近縁が学習側に残るため、物差し B は「未知の種」ではなく")
    add("近縁の見覚えを測りうる。物差し C を科の単位に置いたのはこのためである。")
    add("")

    add("## 分類群ごとの採録内容")
    add("")
    add("| フォルダ名 | 現行の学名 | Gram | 一致率 | 形態 | 一致率 | 科 | Stage B |")
    add("|---|---|---|---|---|---|---|---|")
    for t in sorted(included, key=lambda x: x["folder"]):
        renamed = "**" + t["resolved_name"] + "**" if t["name_is_current"] else t["resolved_name"]
        shape = SHAPE_JA.get(t["shape"], t["shape"] or "—")
        sb = "○" if t["stage_b"] else "×"
        add(
            f"| {t['folder']} | {renamed} | {GRAM_JA[t['gram']]} | {t['gram_agreement']:.1%} | "
            f"{shape} | {t['shape_agreement']:.1%} | {t['group_family']} | {sb} |"
        )
    add("")
    add("太字は、フォルダ名が旧名で現行の学名と違うもの。")
    add("`Lactobacillus` 属は 2020 年に分割され、11 フォルダは現在 5 属にまたがる。")
    add("**フォルダ名をそのまま属として切ると、物差し C が意図した単位で切れない**(検査 T-107)。")
    add("")

    excluded = [t for t in taxa if not t["included"]]
    if excluded:
        add("## 除外した分類群")
        add("")
        for t in excluded:
            add(f"### {t['folder']}")
            add("")
            add(f"- 理由: {t['reason']}")
            add(f"- 典拠: {t['authority']} — {t['authority_url']}(取得日 {t.get('accessed', '—')})")
            add(f"- 権威の回答: {t['evidence']}")
            if t.get("note"):
                add(f"- 備考: {t['note']}")
            add("")

    add("## 種水準の典拠が無かったもの")
    add("")
    genus_level = [t for t in included if (t.get("gram_source") or {}).get("authority_level") == "genus"]
    if not genus_level:
        add("なし。")
    for t in genus_level:
        src = t["gram_source"]
        add(f"### {t['folder']}")
        add("")
        add(f"- 補いの水準: **{src['authority_level']}**({src['authority']})")
        add(f"- 権威の回答: {src['evidence']}")
        add(f"- 注意: {src['caveat']}")
        add("")

    add("## 門と Gram の対応(近道が成立しないことの実測)")
    add("")
    add("| 門 | Gram | 分類群数 |")
    add("|---|---|---|")
    by_phylum: dict[tuple[str, str], int] = Counter(
        ((t.get("lineage") or {}).get("phylum"), t["gram"]) for t in included
    )
    for (ph, g), c in sorted(by_phylum.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        add(f"| {ph} | {GRAM_JA[g]} | {c} |")
    add("")
    mixed = {ph for ph, _ in by_phylum} & {
        ph for ph in {p for p, _ in by_phylum}
        if len({g for (p2, g) in by_phylum if p2 == ph}) > 1
    }
    add(f"**門から Gram は決められない。** 陽性と陰性が同居する門: {', '.join(sorted(mixed)) or 'なし'}。")
    add("採用した他の Bacillota はすべて陽性なので、門で決め打つ実装はここだけ静かに間違える(検査 T-108)。")
    add("")

    DOCS.mkdir(exist_ok=True)
    (DOCS / "DATASET.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"docs/DATASET.md を生成({len(L)} 行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
