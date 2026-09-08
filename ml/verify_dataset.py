"""配布 zip の中身を、展開せずに実測する。

SPEC §2 は「記載枚数と実測枚数は別物として扱う」と定める。
配布元の説明は 2,722 枚だが、その数がそのまま学習に使える枚数とは限らない。
重複・空フォルダ・想定外の拡張子が混じりうるので、数える前に数えられる形にする。

ここで測るもの:
  - 分類群ごとの画像枚数
  - 画像バイト列の SHA-256(**zip をまたぐ重複**を見つける。配布元には同一内容の zip が 2 本ある)
  - キャンバス寸法(G-11 の非画像ベースラインの材料。寸法が分類群と相関するなら、
    寸法を揃える処理そのものが近道を作る)
  - アルファチャンネルの有無(視野円マスクとして使えるか)

使い方:
    python ml/verify_dataset.py
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dataset" / "raw"
META = ROOT / "dataset" / "metadata"
JST = timezone(timedelta(hours=9))


def main() -> int:
    zips = sorted(RAW.glob("*.zip"))
    if not zips:
        print("dataset/raw に zip が無い。先に python ml/download_dataset.py を実行する")
        return 1

    rows: list[dict] = []
    by_digest: dict[str, list[str]] = defaultdict(list)
    non_png: Counter = Counter()

    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                ext = info.filename.rsplit(".", 1)[-1].lower()
                if ext != "png":
                    non_png[ext] += 1
                    continue
                data = z.read(info)
                digest = hashlib.sha256(data).hexdigest()
                with Image.open(io.BytesIO(data)) as im:
                    w, h = im.size
                    mode = im.mode
                # 展開後の相対パス(zip 内のフォルダ名)を分類群名として使う。
                # ただしこれは候補生成であって、ラベルの根拠ではない(SPEC §2.1)
                folder = info.filename.split("/")[0]
                rows.append(
                    {
                        "zip": zp.name,
                        "member": info.filename,
                        "folder": folder,
                        "sha256": digest,
                        "bytes": info.file_size,
                        "width": w,
                        "height": h,
                        "mode": mode,
                    }
                )
                by_digest[digest].append(f"{zp.name}::{info.filename}")
        print(f"  {zp.name:52s} 累計 {len(rows):5d} 枚", flush=True)

    dup_groups = {d: names for d, names in by_digest.items() if len(names) > 1}
    dup_extra = sum(len(v) - 1 for v in dup_groups.values())
    unique = len(by_digest)

    per_folder = Counter(r["folder"] for r in rows)
    square = sum(1 for r in rows if r["width"] == r["height"])
    modes = Counter(r["mode"] for r in rows)
    sizes = sorted({r["width"] for r in rows})

    # 分類群ごとのキャンバス寸法の異なり数。1 なら、その分類群の画像は寸法で識別できてしまう
    size_sets = {f: {r["width"] for r in rows if r["folder"] == f} for f in per_folder}
    size_unique_folders = [f for f, s in size_sets.items() if len(s) == 1]
    # ある寸法が単一の分類群にしか現れないなら、その寸法は分類群の指紋になる
    size_to_folders: dict[int, set[str]] = defaultdict(set)
    for r in rows:
        size_to_folders[r["width"]].add(r["folder"])
    fingerprint_sizes = {w: sorted(fs)[0] for w, fs in size_to_folders.items() if len(fs) == 1}
    fingerprinted_images = sum(1 for r in rows if r["width"] in fingerprint_sizes)

    doc = {
        "generated_at": datetime.now(JST).isoformat(timespec="seconds"),
        "zips_scanned": len(zips),
        "declared_images": 2722,
        "measured_images": len(rows),
        "unique_images_by_sha256": unique,
        "duplicate_extra_copies": dup_extra,
        "duplicate_groups": len(dup_groups),
        "folders": len(per_folder),
        "per_folder_counts": dict(sorted(per_folder.items())),
        "all_square": square == len(rows),
        "modes": dict(modes),
        "distinct_widths": len(sizes),
        "width_min": sizes[0],
        "width_max": sizes[-1],
        "folders_with_single_canvas_size": sorted(size_unique_folders),
        "canvas_sizes_unique_to_one_folder": len(fingerprint_sizes),
        "images_with_fingerprint_size": fingerprinted_images,
        "non_png_entries": dict(non_png),
        "duplicate_examples": {d: v for d, v in list(dup_groups.items())[:5]},
    }
    META.mkdir(parents=True, exist_ok=True)
    (META / "dataset_measured.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 画像 1 枚 1 行の台帳。分割・leakage 検査はこれを起点にする
    with (META / "images.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print()
    print(f"zip {len(zips)} 本 / フォルダ {len(per_folder)} 個")
    print(f"記載 {doc['declared_images']} 枚 / 実測 {len(rows)} 枚 / 一意 {unique} 枚")
    print(f"  重複グループ {len(dup_groups)} 件・余分な複製 {dup_extra} 枚")
    print(f"全て正方: {doc['all_square']} / モード: {dict(modes)}")
    print(f"キャンバス幅 {sizes[0]}〜{sizes[-1]} px({len(sizes)} 通り)")
    print(f"寸法が 1 通りしかないフォルダ: {len(size_unique_folders)} 個")
    print(f"単一フォルダにしか現れない寸法: {len(fingerprint_sizes)} 通り "
          f"→ その寸法を持つ画像 {fingerprinted_images} 枚({fingerprinted_images/len(rows):.1%})")
    if non_png:
        print(f"PNG 以外のエントリ: {dict(non_png)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
