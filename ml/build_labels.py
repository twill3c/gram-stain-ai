"""権威(BacDive ほか)の生の回答から、学習に使うラベル表 labels.json を作る。

ここでやっているのは「決めること」であって「調べること」ではない。調べた結果は
authority_bacdive.json にあり、この層は**採用規則を明示して当てるだけ**にしてある。
規則を規則として書き出しておかないと、あとから「なぜこの種を外したのか」が
思い出せなくなり、除外がこちらの都合に見えてしまう。

採用規則(SPEC §2.1 / G-07):

  1. 語の正規化。権威は形態に 5 通りの語を使う。球形・卵形・卵円形はまとめて球菌、
     桿形・湾曲形はまとめて桿菌とする。**この正規化は結果を変える** —— 正規化しないと
     Veillonella(球 8 / 卵 5 / 卵円 1 / 桿 1)と Neisseria gonorrhoeae(球 14 / 卵 2)が
     一致率不足で落ち、Stage B から Gram 陰性球菌が全部消えて 4 クラスが 3 クラスになる。
  2. Gram ラベルは、権威記録が 1 件以上あり多数派が過半数のとき採用する。
     一致率 100% を tier A、100% 未満を tier B として印を残す(除外はしない)。
  3. 形態ラベルは Gram より厳しく、正規化後の一致率 90% 以上のときだけ採用する。
     満たさない分類群は Stage B の対象外とし、理由を残す。
  4. 権威記録が 0 件の分類群は不採用。ただし「無い」で終わらせず、
     別の権威(属水準の合議・NCBI Taxonomy)に当たった結果を必ず書く。

使い方:
    python ml/build_labels.py
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "dataset" / "metadata"
JST = timezone(timedelta(hours=9))

SHAPE_NORMALISATION = {
    "coccus-shaped": "coccus",
    "oval-shaped": "coccus",
    "ovoid-shaped": "coccus",
    "rod-shaped": "bacillus",
    "curved-shaped": "bacillus",
}
SHAPE_MIN_AGREEMENT = 0.90

# 権威に species 水準の記録が無かった分類群の、明示的な補い。
# ここに書くのは「こちらの知識」ではなく「別の権威に当たった結果」である。
FALLBACKS: dict[str, dict] = {
    "Porphyromonas gingivalis": {
        "gram": "negative",
        "shape": "bacillus",
        "authority_level": "genus",
        "authority": "BacDive — 属 Porphyromonas の株記録",
        "authority_url": "https://api.bacdive.dsmz.de/v2/taxon/Porphyromonas",
        "evidence": "属 43 株のうち細胞形態の記録がある 3 株はすべて gram=negative / rod-shaped "
                    "(BacDive 133139 P. loveana / 12513 P. bennonis / 12505 P. asaccharolytica)",
        "accessed": "2026-09-08",
        "caveat": "P. gingivalis そのものには BacDive に Gram の記録が無い。属水準の合議で補っており、"
                  "種水準の典拠ではない。形態は権威の記述が short rod / coccobacilli と揺れるため "
                  "Stage B の対象外とする",
        "stage_b": False,
    },
}

# 学習・評価から外す分類群。外す理由は必ず権威で述べる。
EXCLUSIONS: dict[str, dict] = {
    "Candida albicans": {
        "reason": "細菌ではない(酵母・真菌)。Gram 陽性菌として扱うと、"
                  "Gram 陽性クラスに真菌の見た目が混ざる",
        "authority": "NCBI Taxonomy taxid 5476",
        "authority_url": "https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=5476",
        "evidence": "Lineage: cellular organisms; Eukaryota; Opisthokonta; Fungi; Dikarya; "
                    "Ascomycota; saccharomyceta; Saccharomycotina; Pichiomycetes; Serinales; "
                    "Debaryomycetaceae; Candida",
        "accessed": "2026-09-08",
        "note": "細菌の権威である BacDive に該当分類群が無かったことも、独立の傍証になっている",
    },
}


def normalise_shape(counts: dict[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for raw, n in counts.items():
        key = SHAPE_NORMALISATION.get(raw)
        if key is None:
            out.setdefault("__unmapped__:" + raw, 0)
            out["__unmapped__:" + raw] += n
        else:
            out[key] = out.get(key, 0) + n
    return out


def main() -> int:
    src = json.loads((META / "authority_bacdive.json").read_text(encoding="utf-8"))
    taxa = {t["folder"]: t for t in src["taxa"]}

    labels: list[dict] = []
    unmapped: set[str] = set()

    for folder, t in taxa.items():
        if folder in EXCLUSIONS:
            labels.append({"folder": folder, "included": False, **EXCLUSIONS[folder]})
            continue

        gram_counts = t.get("gram_counts") or {}
        gram_total = sum(gram_counts.values())
        raw_shape = t.get("shape_counts") or {}
        shape_counts = normalise_shape(raw_shape)
        unmapped |= {k.split(":", 1)[1] for k in shape_counts if k.startswith("__unmapped__")}
        shape_counts = {k: v for k, v in shape_counts.items() if not k.startswith("__unmapped__")}
        shape_total = sum(shape_counts.values())

        fb = FALLBACKS.get(folder)
        if gram_total == 0 and fb is None:
            labels.append(
                {"folder": folder, "included": False, "reason": "権威に Gram の記録が無く、補いも用意していない"}
            )
            continue

        if gram_total:
            gram = max(gram_counts, key=lambda k: gram_counts[k])
            gram_agreement = gram_counts[gram] / gram_total
            gram_src = {"authority_level": "species", "authority": "BacDive",
                        "strains_with_gram": gram_total,
                        "gram_counts": gram_counts,
                        "evidence": t.get("evidence", [])[:3]}
        else:
            gram = fb["gram"]
            gram_agreement = 1.0
            gram_src = {k: fb[k] for k in
                        ("authority_level", "authority", "authority_url", "evidence", "accessed", "caveat")}

        if shape_total:
            shape = max(shape_counts, key=lambda k: shape_counts[k])
            shape_agreement = shape_counts[shape] / shape_total
        else:
            shape, shape_agreement = (fb or {}).get("shape"), 0.0

        stage_b = bool(shape) and shape_agreement >= SHAPE_MIN_AGREEMENT
        if fb is not None and fb.get("stage_b") is False:
            stage_b = False
            stage_b_reason = fb["caveat"]
        elif stage_b:
            stage_b_reason = None
        else:
            stage_b_reason = (
                f"正規化後の形態一致率 {shape_agreement:.1%} が閾値 {SHAPE_MIN_AGREEMENT:.0%} 未満"
                if shape else "権威に形態の記録が無い"
            )

        lineage = t.get("lineage") or {}
        labels.append(
            {
                "folder": folder,
                "included": True,
                "resolved_name": t.get("resolved_name"),
                "name_is_current": t.get("resolved_name") != folder,
                "gram": gram,
                "gram_agreement": round(gram_agreement, 4),
                "gram_tier": "A" if gram_agreement == 1.0 else "B",
                "gram_source": gram_src,
                "shape": shape if stage_b else None,
                "shape_agreement": round(shape_agreement, 4),
                "shape_counts_raw": raw_shape,
                "shape_counts_normalised": shape_counts,
                "stage_b": stage_b,
                "stage_b_excluded_reason": stage_b_reason,
                "lineage": lineage,
                # 目玉の物差し B / C が切る単位。folder は旧名なので、切る単位に使わない
                "group_species": t.get("resolved_name"),
                "group_genus": lineage.get("genus"),
                "group_family": lineage.get("family"),
            }
        )

    included = [x for x in labels if x["included"]]
    doc = {
        "generated_at": datetime.now(JST).isoformat(timespec="seconds"),
        "rules": {
            "shape_normalisation": SHAPE_NORMALISATION,
            "shape_min_agreement": SHAPE_MIN_AGREEMENT,
            "gram_rule": "権威記録 1 件以上・多数派が過半数で採用。一致率 100% を tier A、未満を tier B",
        },
        "authority": {
            "primary": src["source"],
            "primary_license": src["source_license"],
            "primary_citation": src["source_citation"],
            "accessed_at": src["accessed_at"],
        },
        "summary": {
            "taxa_total": len(labels),
            "taxa_included": len(included),
            "gram_positive": sum(1 for x in included if x["gram"] == "positive"),
            "gram_negative": sum(1 for x in included if x["gram"] == "negative"),
            "gram_tier_b": [x["folder"] for x in included if x.get("gram_tier") == "B"],
            "stage_b_taxa": sum(1 for x in included if x["stage_b"]),
            "stage_b_excluded": [x["folder"] for x in included if not x["stage_b"]],
            "families": sorted({x["group_family"] for x in included if x["group_family"]}),
            "unmapped_shape_terms": sorted(unmapped),
        },
        "taxa": labels,
    }
    (META / "labels.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    s = doc["summary"]
    print(f"採用 {s['taxa_included']}/{s['taxa_total']} 分類群")
    print(f"  Gram 陽性 {s['gram_positive']} / 陰性 {s['gram_negative']}")
    print(f"  tier B(一致率 100% 未満): {s['gram_tier_b']}")
    print(f"  Stage B 対象 {s['stage_b_taxa']} / 対象外 {s['stage_b_excluded']}")
    print(f"  科の数 {len(s['families'])}")
    if s["unmapped_shape_terms"]:
        print(f"  !! 正規化表に無い形態の語: {s['unmapped_shape_terms']}")
    for x in labels:
        if not x["included"]:
            print(f"  除外 {x['folder']}: {x['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
