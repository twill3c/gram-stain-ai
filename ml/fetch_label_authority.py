"""種名 → Gram 反応 / 形態 の対応表を、外部権威(BacDive)から取る。

なぜ外部権威が要るのか(SPEC §2.1・G-07):
  ラベルは画像からではなく種名から決まる。その対応をこちらの知識で書いてしまうと、
  「モデルが当てた」ことの検算がこちらの思い込みの検算になる。循環を断つには、
  対応表の出所がプロジェクトの外になければならない。

なぜ系統分類まで取るのか(SPEC §3.1):
  目玉の「属ホールドアウト」は、どの階層で切るかで難易度が変わる。
  とくに Lactobacillus は 2020 年に 25 属へ分割された(Zheng et al. 2020)。
  データセットのフォルダ名は旧名のままなので、**フォルダ名の属は現行の属ではない**。
  切る階層を自分で決めず、権威が返す現行の分類(LPSN)を持ち帰ってから決める。

出所:
  BacDive (Leibniz Institute DSMZ) https://api.bacdive.dsmz.de/
  2026-02 以降は登録不要。データは CC BY 4.0。レコードごとに DOI が付く。

使い方:
    python ml/fetch_label_authority.py
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

API = "https://api.bacdive.dsmz.de/v2"
UA = "gram-stain-ai/1.0 (research) python-urllib"
ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "dataset" / "metadata"
JST = timezone(timedelta(hours=9))

# 1 分類群あたりに問い合わせる株数の上限。多い種(S. aureus は 506 株)を全部引くと
# API に無用な負荷をかける。合議が割れているかどうかは 100 株で十分に見える。
MAX_STRAINS = 100

# データセットのフォルダ名(= 検索の出発点)と、BacDive へ渡す学名の候補。
# 候補を複数持つのは、フォルダ名が旧名のことがあるため(HC-012: 名前は候補生成にのみ使う)。
#   Propionibacterium acnes → Cutibacterium acnes (Scholz & Kilian 2016)
#   Lactobacillus の一部    → Lacticaseibacillus / Lactiplantibacillus /
#                             Limosilactobacillus / Ligilactobacillus (Zheng et al. 2020)
TAXA: list[dict] = [
    {"folder": "Acinetobacter baumannii", "queries": [("Acinetobacter", "baumannii")]},
    {"folder": "Actinomyces israelii", "queries": [("Actinomyces", "israelii")]},
    {"folder": "Bacteroides fragilis", "queries": [("Bacteroides", "fragilis")]},
    {"folder": "Bifidobacterium spp", "queries": [("Bifidobacterium", None)], "rank": "genus"},
    {"folder": "Candida albicans", "queries": [("Candida", "albicans")], "note": "酵母(真菌)"},
    {"folder": "Clostridium perfringens", "queries": [("Clostridium", "perfringens")]},
    {"folder": "Enterococcus faecalis", "queries": [("Enterococcus", "faecalis")]},
    {"folder": "Enterococcus faecium", "queries": [("Enterococcus", "faecium")]},
    {"folder": "Escherichia coli", "queries": [("Escherichia", "coli")]},
    {"folder": "Fusobacterium", "queries": [("Fusobacterium", None)], "rank": "genus"},
    {"folder": "Lactobacillus casei", "queries": [("Lacticaseibacillus", "casei"), ("Lactobacillus", "casei")]},
    {"folder": "Lactobacillus crispatus", "queries": [("Lactobacillus", "crispatus")]},
    {"folder": "Lactobacillus delbrueckii", "queries": [("Lactobacillus", "delbrueckii")]},
    {"folder": "Lactobacillus gasseri", "queries": [("Lactobacillus", "gasseri")]},
    {"folder": "Lactobacillus jensenii", "queries": [("Lactobacillus", "jensenii")]},
    {"folder": "Lactobacillus johnsonii", "queries": [("Lactobacillus", "johnsonii")]},
    {"folder": "Lactobacillus paracasei", "queries": [("Lacticaseibacillus", "paracasei"), ("Lactobacillus", "paracasei")]},
    {"folder": "Lactobacillus plantarum", "queries": [("Lactiplantibacillus", "plantarum"), ("Lactobacillus", "plantarum")]},
    {"folder": "Lactobacillus reuteri", "queries": [("Limosilactobacillus", "reuteri"), ("Lactobacillus", "reuteri")]},
    {"folder": "Lactobacillus rhamnosus", "queries": [("Lacticaseibacillus", "rhamnosus"), ("Lactobacillus", "rhamnosus")]},
    {"folder": "Lactobacillus salivarius", "queries": [("Ligilactobacillus", "salivarius"), ("Lactobacillus", "salivarius")]},
    {"folder": "Listeria monocytogenes", "queries": [("Listeria", "monocytogenes")]},
    {"folder": "Micrococcus spp", "queries": [("Micrococcus", None)], "rank": "genus"},
    {"folder": "Neisseria gonorrhoeae", "queries": [("Neisseria", "gonorrhoeae")]},
    {"folder": "Porphyromonas gingivalis", "queries": [("Porphyromonas", "gingivalis")]},
    {"folder": "Propionibacterium acnes", "queries": [("Cutibacterium", "acnes"), ("Propionibacterium", "acnes")]},
    {"folder": "Proteus", "queries": [("Proteus", None)], "rank": "genus"},
    {"folder": "Pseudomonas aeruginosa", "queries": [("Pseudomonas", "aeruginosa")]},
    {"folder": "Staphylococcus aureus", "queries": [("Staphylococcus", "aureus")]},
    {"folder": "Staphylococcus epidermidis", "queries": [("Staphylococcus", "epidermidis")]},
    {"folder": "Staphylococcus saprophyticus", "queries": [("Staphylococcus", "saprophyticus")]},
    {"folder": "Streptococcus agalactiae", "queries": [("Streptococcus", "agalactiae")]},
    {"folder": "Veillonella", "queries": [("Veillonella", None)], "rank": "genus"},
]


def get(url: str, tries: int = 5) -> dict:
    """1 件取る。

    HC-219: 再試行の except を例外クラスの列挙で書かない。列挙は「自分が思いついた
    壊れ方」しか受けず、実際に来た http.client.IncompleteRead は URLError の系統では
    ないので素通りした。ネットワーク境界では網を絞らない。
    """
    last: Exception | None = None
    for i in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=90) as res:
                return json.load(res)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return {}
            last = exc
            time.sleep(2 * (i + 1))
        except Exception as exc:  # noqa: BLE001 — 網を絞らないことが趣旨(HC-219)
            last = exc
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"取得できない: {url} ({last})")


def taxon_ids(genus: str, species: str | None) -> list[int]:
    path = f"{API}/taxon/{urllib.parse.quote(genus)}"
    if species:
        path += f"/{urllib.parse.quote(species)}"
    ids: list[int] = []
    url: str | None = path
    while url and len(ids) < MAX_STRAINS:
        d = get(url)
        if not d:
            break
        ids.extend(d.get("results") or [])
        url = d.get("next")
    return ids[:MAX_STRAINS]


def fetch_records(ids: list[int]) -> dict:
    out: dict = {}
    for i in range(0, len(ids), 100):
        chunk = ids[i : i + 100]
        d = get(f"{API}/fetch/" + ";".join(str(x) for x in chunk))
        out.update(d.get("results") or {})
        time.sleep(0.3)
    return out


def summarise(records: dict) -> dict:
    gram: Counter = Counter()
    shape: Counter = Counter()
    lineage: Counter = Counter()
    evidence: list[dict] = []
    for bid, rec in records.items():
        cm = (rec.get("Morphology") or {}).get("cell morphology")
        entries = cm if isinstance(cm, list) else ([cm] if cm else [])
        g = s = None
        for e in entries:
            g = e.get("gram stain") or g
            s = e.get("cell shape") or s
        if g:
            gram[g] += 1
        if s:
            shape[s] += 1

        tax = (rec.get("Name and taxonomic classification") or {})
        lpsn = tax.get("LPSN") or {}
        key = (
            lpsn.get("phylum") or tax.get("phylum"),
            lpsn.get("class") or tax.get("class"),
            lpsn.get("order") or tax.get("order"),
            lpsn.get("family") or tax.get("family"),
            lpsn.get("genus") or tax.get("genus"),
        )
        if any(key):
            lineage[key] += 1

        if (g or s) and len(evidence) < 5:
            evidence.append(
                {
                    "bacdive_id": int(bid),
                    "species": tax.get("species"),
                    "gram_stain": g,
                    "cell_shape": s,
                    "record_doi": (rec.get("General") or {}).get("doi"),
                    "url": f"https://bacdive.dsmz.de/strain/{bid}",
                }
            )

    top_lineage = lineage.most_common(1)[0][0] if lineage else (None,) * 5
    return {
        "gram_counts": dict(gram),
        "shape_counts": dict(shape),
        "gram_majority": gram.most_common(1)[0][0] if gram else None,
        "shape_majority": shape.most_common(1)[0][0] if shape else None,
        "gram_unanimous": len(gram) == 1,
        "shape_unanimous": len(shape) == 1,
        "lineage": dict(zip(("phylum", "class", "order", "family", "genus"), top_lineage)),
        "lineage_variants": len(lineage),
        "evidence": evidence,
    }


def main() -> int:
    META.mkdir(parents=True, exist_ok=True)
    # HC-219: 多数件を順に取る処理は件ごとに保存して再開可能にする。
    # 外部取得は必ず途中で落ちる。落ちること自体ではなく、落ちたときに成果が残らないことが失敗である。
    cache_dir = META / ".authority_cache"
    cache_dir.mkdir(exist_ok=True)

    out: list[dict] = []
    for t in TAXA:
        folder = t["folder"]
        cache = cache_dir / (folder.replace(" ", "_") + ".json")
        if cache.exists():
            out.append(json.loads(cache.read_text(encoding="utf-8")))
            print(f"  {folder:32s} → キャッシュ")
            continue
        used = None
        ids: list[int] = []
        for genus, species in t["queries"]:
            ids = taxon_ids(genus, species)
            if ids:
                used = f"{genus} {species}" if species else genus
                break
        if not ids:
            print(f"  !! 権威が見つからない: {folder}")
            rec = {"folder": folder, "resolved_name": None, "status": "needs_review",
                   "reason": "BacDive に該当する分類群が無い"}
            cache.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            out.append(rec)
            continue

        recs = fetch_records(ids)
        s = summarise(recs)
        s.update(
            {
                "folder": folder,
                "resolved_name": used,
                "queried_name_is_current": used == " ".join(x for x in t["queries"][0] if x),
                "rank": t.get("rank", "species"),
                "strains_queried": len(ids),
                "strains_with_gram": sum(s["gram_counts"].values()),
                "strains_with_shape": sum(s["shape_counts"].values()),
                "note": t.get("note"),
            }
        )
        cache.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
        out.append(s)
        print(
            f"  {folder:32s} → {used:34s} gram={s['gram_majority']} "
            f"({sum(s['gram_counts'].values())}株) shape={s['shape_majority']} "
            f"family={s['lineage'].get('family')}"
        )

    doc = {
        "source": "BacDive (Leibniz Institute DSMZ)",
        "source_api": API,
        "source_license": "CC BY 4.0",
        "source_citation": (
            "Schober et al., BacDive in 2025: the core database for prokaryotic strain data, "
            "Nucleic Acids Research 53(D1):D748, https://doi.org/10.1093/nar/gkae959"
        ),
        "accessed_at": datetime.now(JST).isoformat(timespec="seconds"),
        "max_strains_per_taxon": MAX_STRAINS,
        "taxa": out,
    }
    (META / "authority_bacdive.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n{len(out)} 分類群 → dataset/metadata/authority_bacdive.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
