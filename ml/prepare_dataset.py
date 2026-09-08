"""配布 zip から学習用の画像を作り、group をまとめ、三つの物差しの分割を生成する。

## 前処理

配布画像は正方キャンバスに視野円が内接し、四隅が黒い。実測(loop_002・102 枚の標本)では
直径がキャンバス幅と完全に一致し、中心のずれは 0.00 px、面積は理論円の 0.9999〜1.0001 倍。
つまり円の検出は要らず、内接正方形は中心の一辺 W/√2 で決まる。**この前提は全画像で assert する。**

切り出しは **中心の 512x512 をリサイズなし**で取る。

  - なぜ内接正方形をそのまま使わないか: 一辺は 522〜1083 px と画像ごとに違う。
    一律に縮めると縮小率が画像ごとに変わり、**菌の見かけの大きさが変わる**。
    球菌と桿菌を見分ける課題では、見かけの大きさは情報である
  - なぜ 512 か: キャンバス幅の最小は 739 px で、その内接正方形は 522 px。
    512 なら全画像に入る(余裕 10 px)。これ以上大きくすると最小の画像で円をはみ出す
  - 限界: 対物倍率は全画像 100 倍だが、配布元の説明は複数機関・複数機材の結合だという。
    画素あたりの実寸が機材ごとに違えば、画素で揃えても実寸では揃わない。
    **配布物にスケールバーが無いので確かめようがない。** 確かめていないことを記録しておく

## group

同じ視野、あるいは同じスライドを写した画像が train と test に分かれると、
test は「見たことのある標本」を当てることになる。

loop_002 で全 59,539 対を実測して分かったこと:

  - **同一視野と呼べる対は 3 組しかない**(dhash 距離 19 / 34 / 35。次に近いのが 55)。
    loop_001 に「近接ショットが多数ある」と書いたのは、12 枚の貼り合わせを見た印象からの
    一般化であり、実測は支持しなかった。分布には 36〜54 の谷があるので閾値はそこに置く
  - 代わりに別の構造が見つかった。**同一キャンバス寸法の 3 枚組が 264 個ある**
    (1225px を除く 421 群のうち 264 群がちょうど 3 枚、76% が 3 の倍数)。
    ただしこれは同一視野ではない —— 群内の dhash 距離は中央値 126 で乱数対と変わらず、
    目視でも明らかに別の視野である。染色の濃さ・背景・ピントは揃っている。
    **同じスライドの別視野**と読むのが自然で、これは撮影セッション ID の代わりになる
  - 例外は 1225px で、22 分類群にまたがって 539 枚(27.4%)ある。
    一枚のスライドから 55 視野という規模ではないので、これは別の書き出し経路と見られる。
    **1225px の大きな群にはスライドの手がかりが無い**

したがって group は次の連結成分とする。

  1. SHA-256 の完全一致は 1 枚に畳む(重複そのものを消す)
  2. 同一 (分類群, キャンバス寸法) のセルは、**枚数が SLIDE_CELL_MAX 以下ならまとめる**
     (スライドの代理)。それを超えるセルは一枚のスライドとは考えにくいのでまとめない
  3. dhash が近い対はどこであってもつなぐ(同一視野)

**限界: 撮影セッション ID は配布物に無い。** 上の 2 は代理であって本物ではなく、
大きなセル(実測で全体の約 4 分の 1)は group を作れないまま残る。
その分だけ物差し A は甘い。**物差し B と C を置く理由の一つがこれである。**

使い方:
    python ml/prepare_dataset.py --hamming 45 --no-images
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dataset" / "raw"
META = ROOT / "dataset" / "metadata"
PROC = ROOT / "dataset" / "processed"
JST = timezone(timedelta(hours=9))

CROP = 512
SEED = 20260908
SQRT2 = math.sqrt(2.0)


# ------------------------------------------------------------------ 前処理


def centre_crop(img: Image.Image) -> Image.Image:
    """中心 CROP x CROP を切り出す。リサイズしない。"""
    w, h = img.size
    left = (w - CROP) // 2
    top = (h - CROP) // 2
    return img.crop((left, top, left + CROP, top + CROP))


def check_circle(alpha: np.ndarray) -> dict:
    """視野円がキャンバスに内接し中心にあることを確かめ、幾何を返す。"""
    mask = alpha > 127
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    return {
        "diameter_x": int(mask.sum(0).max()),
        "diameter_y": int(mask.sum(1).max()),
        "centre_x": float(xs.mean()),
        "centre_y": float(ys.mean()),
        "fill_ratio": float(mask.sum() / (math.pi * (mask.sum(0).max() / 2) ** 2)),
        "width": w,
        "height": h,
    }


def dhash(img: Image.Image, size: int = 16) -> int:
    """差分ハッシュ。隣接画素の大小関係だけを見るので、明るさの違いに鈍い。

    Gram 染色は色が情報なので、**色を落とさない**よう輝度ではなく
    3 チャンネルの平均ではなく緑チャンネルを使う……のではなく、
    ここで欲しいのは「同じ視野か」であって「同じ色か」ではない。
    構図の一致を見たいので輝度で取る。色の違いは後段の分類器の仕事である。
    """
    small = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(small, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def popcount(x: int) -> int:
    return bin(x).count("1")


# ------------------------------------------------------------------ group


def build_groups(items: list[dict], hamming: int, slide_cell_max: int) -> dict:
    """スライドの代理と同一視野の両方でまとめ、items へ group_id を書き込む。

    総当たりは n^2 だが n は 2,000 程度なので 200 万回で済む。
    バケット法で速くすることもできるが、**取りこぼしの無さを優先**する。
    近傍を取りこぼすと leakage が残り、それは成績を良く見せる方向に効く。
    """
    n = len(items)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # (1) スライドの代理 — 同一 (分類群, キャンバス寸法) の小さいセル
    cells: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        cells[(it["folder"], it["source_width"])].append(i)
    slide_cells = 0
    ungrouped_large = 0
    for members in cells.values():
        if len(members) <= slide_cell_max:
            if len(members) > 1:
                slide_cells += 1
            for x in members[1:]:
                union(members[0], x)
        else:
            ungrouped_large += len(members)

    # (2) 同一視野 — dhash の近傍。分類群をまたぐ結合はしない(T-221)
    hashes = [it["dhash"] for it in items]
    folders = [it["folder"] for it in items]
    hash_pairs = 0
    for i in range(n):
        hi, fi = hashes[i], folders[i]
        for j in range(i + 1, n):
            if folders[j] != fi:
                continue
            if popcount(hi ^ hashes[j]) <= hamming:
                union(i, j)
                hash_pairs += 1

    for i, it in enumerate(items):
        it["group_id"] = f"g{find(i):05d}"

    return {
        "slide_cells_merged": slide_cells,
        "hash_pairs_merged": hash_pairs,
        "images_in_ungroupable_large_cells": ungrouped_large,
        "slide_cell_max": slide_cell_max,
    }


# ------------------------------------------------------------------ 分割


def split_ruler_a(items: list[dict], rng: random.Random) -> dict:
    """物差し A — 分類群の中で group 単位に 70/15/15。

    分類群ごとに割るので、どの分割にも全分類群が現れる(T-222)。
    割る単位は画像ではなく group(T-202)。
    """
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for folder in sorted({it["folder"] for it in items}):
        groups = sorted({it["group_id"] for it in items if it["folder"] == folder})
        rng.shuffle(groups)
        n = len(groups)
        n_val = max(1, round(n * 0.15))
        n_test = max(1, round(n * 0.15))
        assign = {}
        for g in groups[:n_test]:
            assign[g] = "test"
        for g in groups[n_test : n_test + n_val]:
            assign[g] = "val"
        for g in groups[n_test + n_val :]:
            assign[g] = "train"
        for it in items:
            if it["folder"] == folder:
                out[assign[it["group_id"]]].append(it["image_id"])
    return out


def split_ruler_a_cv(items: list[dict], k: int, rng: random.Random) -> dict:
    """物差し A の交差検証版 — 分類群の中で group を k 個へ配る。

    なぜ 70/15/15 と別に要るのか(loop_003):
      三つの物差しを比べるとき、**評価する母集団が違うと差の出所が分からなくなる**。
      70/15/15 の A は 286 枚しか評価しないが、B と C は fold を通せば全画像を評価する。
      286 枚と 1,966 枚の成績を並べると、差が「物差しの違い」なのか「測った枚数の違い」なのか
      分けられない。A も k 分割にして、三つとも全画像を一度ずつ評価する形に揃える。

    70/15/15 のほうは残す。出荷するモデルは val でモデル選択をする必要があり、
    そのためには train / val / test の三分割が要るからである(G-08)。
    """
    folds: list[list[str]] = [[] for _ in range(k)]
    for folder in sorted({it["folder"] for it in items}):
        groups = sorted({it["group_id"] for it in items if it["folder"] == folder})
        rng.shuffle(groups)
        assign = {g: i % k for i, g in enumerate(groups)}
        for it in items:
            if it["folder"] == folder:
                folds[assign[it["group_id"]]].append(it["image_id"])

    all_ids = [it["image_id"] for it in items]
    out = []
    for f in folds:
        held = set(f)
        out.append({"held_out": [], "train": [i for i in all_ids if i not in held], "test": sorted(held)})
    return {"key": "group_within_taxon", "folds": out}


def _balanced_folds(units: list[str], unit_gram: dict[str, str], unit_size: dict[str, int],
                    k: int, rng: random.Random) -> list[list[str]]:
    """単位(分類群 or 科)を k 個の fold へ分ける。

    守る条件は二つ。

    1. **各 fold に Gram 陽性と陰性の両方が入る**(T-225)。片方しか無い fold で macro F1 を
       取ると、存在しないクラスの F1 が 0 として混ざり、成績が fold の作り方の産物になる。
       指標を壊さないのは fold を作る側の仕事である
    2. **fold ごとの枚数をできるだけ均す。** 単位の数で均等に配ると枚数が偏る ——
       科で切ると Lactobacillaceae だけで 912 枚(全体の 46%)あり、素朴に配ると
       その fold だけが巨大になる。巨大な fold の成績は平均を支配し、
       小さな fold の成績は信頼区間が広くなりすぎる

    手順: 大きい陰性を 1 つずつ各 fold へ、次に大きい陽性を 1 つずつ各 fold へ配ってから、
    残りを大きい順に「そのとき最も枚数の少ない fold」へ入れる。
    """
    pos = sorted((u for u in units if unit_gram[u] == "positive"),
                 key=lambda u: (-unit_size[u], u))
    neg = sorted((u for u in units if unit_gram[u] == "negative"),
                 key=lambda u: (-unit_size[u], u))
    if k > min(len(pos), len(neg)):
        # どちらかの単位数が fold 数を下回ると、どう配っても片方ゼロの fold が出る
        k = min(len(pos), len(neg))

    folds: list[list[str]] = [[] for _ in range(k)]
    totals = [0] * k
    for i in range(k):  # 各 fold に陰性を 1 つ
        folds[i].append(neg[i])
        totals[i] += unit_size[neg[i]]
    for i in range(k):  # 各 fold に陽性を 1 つ
        folds[i].append(pos[i])
        totals[i] += unit_size[pos[i]]

    rest = sorted(neg[k:] + pos[k:], key=lambda u: (-unit_size[u], u))
    for u in rest:
        j = min(range(k), key=lambda i: (totals[i], i))
        folds[j].append(u)
        totals[j] += unit_size[u]
    for f in folds:
        rng.shuffle(f)
    return folds


def split_holdout(items: list[dict], key: str, k: int, rng: random.Random) -> dict:
    """物差し B / C — key(folder or family)の単位でまるごと抜く。"""
    unit_gram: dict[str, str] = {}
    unit_size: Counter = Counter()
    for it in items:
        unit_gram.setdefault(it[key], it["gram"])
        unit_size[it[key]] += 1
    units = sorted(unit_gram)
    folds = _balanced_folds(units, unit_gram, unit_size, k, rng)

    out = []
    for held in folds:
        held_set = set(held)
        test = [it["image_id"] for it in items if it[key] in held_set]
        train = [it["image_id"] for it in items if it[key] not in held_set]
        out.append({"held_out": sorted(held), "train": train, "test": test})
    return {"key": key, "folds": out}


# ------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser()
    # 既定 45 は実測で決めた。距離 19/34/35 に 3 対があり、次が 55。36〜54 が谷なのでその中央。
    # 谷に置いているので、この値を数ずらしても結果は変わらない(loop_002 で確認)
    ap.add_argument("--hamming", type=int, default=45,
                    help="知覚ハッシュ(256bit)の近傍とみなすハミング距離の上限")
    # 既定 12 は実測で決めた。1225px を除くセルの最大が 12 枚で、そこまでは一枚のスライドから
    # 撮った視野の数として無理がない。1225px の大きなセル(21〜55 枚)は別の書き出し経路と見られ、
    # 一枚のスライドとは考えにくいのでまとめない
    ap.add_argument("--slide-cell-max", type=int, default=12,
                    help="同一 (分類群, キャンバス寸法) をスライドの代理としてまとめる上限枚数")
    ap.add_argument("--folds-b", type=int, default=8, help="物差し B の fold 数")
    ap.add_argument("--folds-c", type=int, default=6, help="物差し C の fold 数")
    ap.add_argument("--no-images", action="store_true", help="画像を書き出さず台帳だけ作り直す")
    ap.add_argument("--splits-only", action="store_true",
                    help="prepared.jsonl を読み直し、group と分割だけ作り直す(zip を開かない)")
    args = ap.parse_args()

    if args.splits_only:
        return rebuild_splits(args)

    labels_doc = json.loads((META / "labels.json").read_text(encoding="utf-8"))
    label_by_folder = {t["folder"]: t for t in labels_doc["taxa"] if t["included"]}

    PROC.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    seen: dict[str, str] = {}
    geometry_violations: list[str] = []
    skipped_excluded = 0

    for zp in sorted(RAW.glob("*.zip")):
        with zipfile.ZipFile(zp) as z:
            for info in sorted(z.infolist(), key=lambda i: i.filename):
                if info.is_dir() or not info.filename.lower().endswith(".png"):
                    continue
                folder = info.filename.split("/")[0]
                label = label_by_folder.get(folder)
                if label is None:  # 除外した分類群(Candida albicans)
                    skipped_excluded += 1
                    continue

                data = z.read(info)
                digest = hashlib.sha256(data).hexdigest()
                if digest in seen:  # バイト完全一致は 1 枚に畳む(T-219)
                    continue
                seen[digest] = info.filename

                with Image.open(io.BytesIO(data)) as im:
                    arr = np.asarray(im)
                    geo = check_circle(arr[..., 3])
                    w = geo["width"]
                    # 実測した前提を全画像で確かめる。破れたら黙って進まない
                    ok = (
                        geo["width"] == geo["height"]
                        and geo["diameter_x"] == w
                        and geo["diameter_y"] == w
                        and abs(geo["centre_x"] - (w - 1) / 2) < 1.0
                        and abs(geo["centre_y"] - (w - 1) / 2) < 1.0
                        and 0.99 < geo["fill_ratio"] < 1.01
                    )
                    if not ok:
                        geometry_violations.append(f"{zp.name}::{info.filename} {geo}")
                    if w / SQRT2 < CROP:
                        raise RuntimeError(
                            f"{info.filename}: 内接正方形 {w/SQRT2:.0f}px が切り出し {CROP}px より小さい"
                        )
                    rgb = im.convert("RGB")
                    crop = centre_crop(rgb)

                image_id = digest[:16]
                if not args.no_images:
                    (PROC / folder).mkdir(parents=True, exist_ok=True)
                    crop.save(PROC / folder / f"{image_id}.png", optimize=True)

                items.append(
                    {
                        "image_id": image_id,
                        "source_zip": zp.name,
                        "source_member": info.filename,
                        "source_sha256": digest,
                        "source_width": w,
                        "crop_size": CROP,
                        "path": f"{folder}/{image_id}.png",
                        "folder": folder,
                        "species": label["group_species"],
                        "genus": label["group_genus"],
                        "family": label["group_family"],
                        "gram": label["gram"],
                        "shape": label["shape"],
                        "stage_b": label["stage_b"],
                        "dhash": dhash(crop),
                    }
                )
        print(f"  {zp.name:52s} 累計 {len(items):5d} 枚", flush=True)

    if geometry_violations:
        print(f"\n!! 視野円の前提を満たさない画像 {len(geometry_violations)} 枚")
        for v in geometry_violations[:5]:
            print("   ", v)

    print(f"\n除外分類群でスキップ: {skipped_excluded} 枚")
    print(f"一意な画像: {len(items)} 枚")

    return finish(items, args, skipped_excluded, len(geometry_violations))


def rebuild_splits(args) -> int:
    """zip を開かずに prepared.jsonl から group と分割だけ作り直す。

    分割の作り方を変えるたびに 3.4 GB を読み直すのは無駄であり、
    **読み直しが高くつくと、作り方を試す回数が減る**。試す回数が減れば、
    閾値や fold の切り方は「最初に思いついた形」のまま出荷されることになる。
    """
    path = META / "prepared.jsonl"
    if not path.exists():
        print("prepared.jsonl が無い。--splits-only を使う前に一度そのまま実行する")
        return 1
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for it in items:
        it["dhash"] = int(it["dhash"], 16)
    prev = json.loads((META / "prepare_summary.json").read_text(encoding="utf-8"))
    print(f"prepared.jsonl から {len(items)} 枚を読み直した(zip は開かない)")
    return finish(items, args, prev.get("images_skipped_excluded_taxa", 0),
                  prev.get("geometry_violations", 0))


def finish(items: list[dict], args, skipped_excluded: int, geometry_violations: int) -> int:
    gstat = build_groups(items, args.hamming, args.slide_cell_max)
    groups = Counter(it["group_id"] for it in items)
    multi = {g: n for g, n in groups.items() if n > 1}
    print(f"スライドの代理(同一 分類群+寸法・{args.slide_cell_max} 枚以下)で "
          f"{gstat['slide_cells_merged']} セルを結合")
    print(f"知覚ハッシュ(ハミング ≤ {args.hamming})で {gstat['hash_pairs_merged']} 対を結合")
    print(f"手がかりの無い大きなセルに残った画像: {gstat['images_in_ungroupable_large_cells']} 枚 "
          f"({gstat['images_in_ungroupable_large_cells']/len(items):.1%})")
    print(f"group {len(groups)} 個 / うち 2 枚以上を含む group {len(multi)} 個 "
          f"(最大 {max(groups.values())} 枚)")

    rng = random.Random(SEED)
    splits = {
        "seed": SEED,
        "hamming_threshold": args.hamming,
        "ruler_a": split_ruler_a(items, rng),
        "ruler_a_cv": split_ruler_a_cv(items, args.folds_b, rng),
        "ruler_b": split_holdout(items, "folder", args.folds_b, rng),
        "ruler_c": split_holdout(items, "family", args.folds_c, rng),
    }

    with (META / "prepared.jsonl").open("w", encoding="utf-8") as fh:
        for it in items:
            row = dict(it)
            row["dhash"] = f"{it['dhash']:064x}"
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (META / "splits.json").write_text(
        json.dumps(splits, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    summary = {
        "generated_at": datetime.now(JST).isoformat(timespec="seconds"),
        "crop_size": CROP,
        "seed": SEED,
        "hamming_threshold": args.hamming,
        "images_unique": len(items),
        "images_skipped_excluded_taxa": skipped_excluded,
        "geometry_violations": geometry_violations,
        "groups": len(groups),
        "groups_with_multiple_images": len(multi),
        "largest_group": max(groups.values()),
        "grouping": gstat,
        "ruler_a_counts": {k: len(v) for k, v in splits["ruler_a"].items()},
        "ruler_b_folds": len(splits["ruler_b"]["folds"]),
        "ruler_c_folds": len(splits["ruler_c"]["folds"]),
        "ruler_b_held_out": [f["held_out"] for f in splits["ruler_b"]["folds"]],
        "ruler_c_held_out": [f["held_out"] for f in splits["ruler_c"]["folds"]],
        "limitation": "撮影セッション ID は配布物に無い。同一 (分類群, キャンバス寸法) を"
                      "スライドの代理として使っているが、これは代理であって本物ではない。"
                      "1225px のような大きなセルは group を作れないまま残り、その分だけ物差し A は甘い",
    }
    (META / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print()
    print("物差し A:", summary["ruler_a_counts"])
    print(f"物差し B: {summary['ruler_b_folds']} fold(分類群を抜く)")
    print(f"物差し C: {summary['ruler_c_folds']} fold(科を抜く)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
