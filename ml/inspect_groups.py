"""知覚ハッシュの閾値を目で確かめるための貼り合わせを作る。

SPEC §6.1 は「近傍の閾値は、**まとめすぎ / まとめ足りない**の両方向を目視で確かめてから決める」
と定める。片方向だけ見ると、閾値は必ず自分の都合のいい方へ寄る。

出すもの:
  1. まとめた対のうち、距離が**大きい**ほうから N 組(まとめすぎの候補)
  2. まとめなかった対のうち、距離が**小さい**ほうから N 組(まとめ足りない候補)

どちらも同じ分類群の中の対だけを見る(別分類群はそもそもつながない)。

使い方:
    python ml/inspect_groups.py --n 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "dataset" / "metadata"
PROC = ROOT / "dataset" / "processed"
OUT = ROOT / "reports" / "group_calibration"

THUMB = 220


def load() -> tuple[list[dict], int]:
    rows = [json.loads(line) for line in (META / "prepared.jsonl").read_text(encoding="utf-8").splitlines()]
    for r in rows:
        r["dhash_int"] = int(r["dhash"], 16)
    summary = json.loads((META / "prepare_summary.json").read_text(encoding="utf-8"))
    return rows, summary["hamming_threshold"]


def pairs_by_distance(rows: list[dict], threshold: int) -> tuple[list, list]:
    """同一分類群の全対を距離つきで集め、閾値の内と外に分ける。"""
    merged: list[tuple[int, dict, dict]] = []
    apart: list[tuple[int, dict, dict]] = []
    by_folder: dict[str, list[dict]] = {}
    for r in rows:
        by_folder.setdefault(r["folder"], []).append(r)
    for items in by_folder.values():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                d = bin(items[i]["dhash_int"] ^ items[j]["dhash_int"]).count("1")
                (merged if d <= threshold else apart).append((d, items[i], items[j]))
    merged.sort(key=lambda t: -t[0])   # まとめた中で最も遠い = まとめすぎの候補
    apart.sort(key=lambda t: t[0])     # 離した中で最も近い = まとめ足りない候補
    return merged, apart


def sheet(pairs: list, title: str, path: Path) -> None:
    if not pairs:
        return
    cols = 2
    rows_n = len(pairs)
    header = 26
    img = Image.new("RGB", (THUMB * cols + 30, (THUMB + header) * rows_n + 30), "white")
    draw = ImageDraw.Draw(img)
    for k, (d, a, b) in enumerate(pairs):
        y = 15 + k * (THUMB + header)
        draw.text((15, y), f"距離 {d}  {a['folder']}  {Path(a['source_member']).name} / "
                           f"{Path(b['source_member']).name}", fill="black")
        for c, rec in enumerate((a, b)):
            p = PROC / rec["path"]
            if p.exists():
                with Image.open(p) as im:
                    img.paste(im.convert("RGB").resize((THUMB, THUMB)), (15 + c * THUMB, y + header))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=88)
    print(f"  {title} → {path.relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="各方向に出す対の数")
    ap.add_argument("--threshold", type=int, default=None,
                    help="閾値を上書きして試す(前処理をやり直さずに較正するため)")
    args = ap.parse_args()

    rows, threshold = load()
    if args.threshold is not None:
        threshold = args.threshold
    merged, apart = pairs_by_distance(rows, threshold)
    print(f"閾値 {threshold} / 同一分類群の対 {len(merged) + len(apart):,} 組")
    print(f"  まとめた {len(merged):,} 組(距離 {merged[-1][0] if merged else '—'}〜{merged[0][0] if merged else '—'})")
    print(f"  離した   {len(apart):,} 組(距離 {apart[0][0] if apart else '—'}〜{apart[-1][0] if apart else '—'})")

    sheet(merged[: args.n], "まとめすぎの候補(まとめた中で最も遠い)", OUT / "merged_farthest.jpg")
    sheet(apart[: args.n], "まとめ足りない候補(離した中で最も近い)", OUT / "apart_nearest.jpg")

    # 距離の分布。閾値の位置が谷にあるのか、山の途中を切っているのかを見る
    hist: dict[int, int] = {}
    for d, _, _ in merged + apart:
        hist[d] = hist.get(d, 0) + 1
    dist = {
        "hamming_threshold": threshold,
        "pairs_total": len(merged) + len(apart),
        "pairs_merged": len(merged),
        "histogram": {str(k): hist[k] for k in sorted(hist)},
    }
    (OUT / "distance_histogram.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "distance_histogram.json").write_text(
        json.dumps(dist, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    lo = sorted(hist)[:40]
    print("\n距離の分布(0〜39):")
    for d in lo:
        mark = "  ←閾値" if d == threshold else ""
        print(f"  {d:3d}: {hist[d]:6d}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
