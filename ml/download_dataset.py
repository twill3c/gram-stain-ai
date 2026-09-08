"""Mendeley Data の公式 API から zip を取得し、SHA-256 と取得日を metadata へ記録する。

規律(SPEC §2 / N-03):
  - 取得先は公式 API のファイル一覧が返す download_url のみ。HTML を回避するスクレイピングはしない
  - raw/ は .gitignore の下にある。リポジトリへ入るのは metadata だけ
  - 公表サイズと実測サイズが食い違えば、実測を採る(そして食い違いを記録する)

使い方:
    python ml/download_dataset.py            # 未取得のものだけ取る
    python ml/download_dataset.py --recheck  # 取得済みの SHA-256 を検算し直す
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

DATASET_ID = "cvkgfzp7ck"
VERSION = 1
DOI = "10.17632/cvkgfzp7ck.1"
FILES_API = (
    f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}"
    f"/files?folder_id=root&version={VERSION}"
)
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dataset" / "raw"
META = ROOT / "dataset" / "metadata"
JST = timezone(timedelta(hours=9))


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


# Mendeley は既定の Python-urllib User-Agent を 403 で拒否する(loop_001 stage 2 で実測)。
# curl では 200 が返っていたため、実測した経路と実装した経路の違いが失敗として現れた。
# 以後、この取得系は必ずこの opener を通す。
UA = "gram-stain-ai/1.0 (research; +https://github.com/) python-urllib"
_opener = urllib.request.build_opener()
_opener.addheaders = [("User-Agent", UA), ("Accept", "application/json")]
urllib.request.install_opener(_opener)


def fetch_listing() -> list[dict]:
    req = urllib.request.Request(FILES_API, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as res:
        return json.load(res)


def download(url: str, dest: Path, expected: int, tries: int = 6) -> None:
    """1 本落とす。途中まで取れていれば Range で続きから取る。

    urlretrieve を使わない理由(loop_001 stage 4 で実測):
      あの関数はタイムアウトを受け取らない。接続が黙って止まると**例外を上げずに永久に待つ**。
      実際 33.9MB/92MB で止まり、CPU ほぼ 0 のまま約 25 分待ち続けた。
      例外が上がらない止まり方は、再試行の網をどれだけ広げても捕まらない。
      止まらないようにするには、読みの一つ一つにタイムアウトを掛けるしかない。
    """
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, tries + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > expected:  # 取りすぎ = 前回の残骸。作り直す
            part.unlink()
            have = 0
        if have == expected:
            break

        headers = {"User-Agent": UA}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as res:
                # Range を無視して 200 で全部返すサーバもある。その場合は最初から書き直す
                mode = "ab" if (have and res.status == 206) else "wb"
                if mode == "wb":
                    have = 0
                with part.open(mode) as fh:
                    while chunk := res.read(1 << 20):  # 読みごとに timeout が効く
                        fh.write(chunk)
                        have += len(chunk)
        except Exception as exc:  # noqa: BLE001 — 網を絞らない(HC-219)
            got = part.stat().st_size if part.exists() else 0
            print(f"          中断 {got/1e6:.0f}/{expected/1e6:.0f} MB ({type(exc).__name__}) "
                  f"— 再試行 {attempt}/{tries}", flush=True)
            if attempt == tries:
                raise
            time.sleep(min(30, 3 * attempt))
            continue

        if part.stat().st_size == expected:
            break
        print(f"          短い ({part.stat().st_size}/{expected}) — 続きを取る", flush=True)

    got = part.stat().st_size
    if got != expected:
        raise RuntimeError(f"{dest.name}: {got} バイトしか取れない(公表 {expected})")
    part.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recheck", action="store_true", help="取得済みの SHA-256 を検算し直す")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)

    listing = fetch_listing()
    (META / "mendeley_files.json").write_text(
        json.dumps(listing, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    records: list[dict] = []
    for i, f in enumerate(sorted(listing, key=lambda x: x["filename"]), 1):
        name = f["filename"]
        declared = f["size"]
        url = f["content_details"]["download_url"]
        dest = RAW / name

        if dest.exists() and dest.stat().st_size == declared and not args.recheck:
            print(f"[{i:2}/{len(listing)}] skip  {name}", flush=True)
        else:
            if not (dest.exists() and dest.stat().st_size == declared):
                print(f"[{i:2}/{len(listing)}] get   {name} ({declared/1e6:.0f} MB)", flush=True)
                t0 = time.time()
                download(url, dest, declared)
                print(f"          done in {time.time()-t0:.0f}s", flush=True)

        actual = dest.stat().st_size
        digest = sha256(dest)
        if actual != declared:
            print(f"  !! サイズ不一致 {name}: 公表 {declared} / 実測 {actual}", file=sys.stderr)
        records.append(
            {
                "filename": name,
                "declared_size": declared,
                "actual_size": actual,
                "sha256": digest,
                "source_url": url,
                "dataset_doi": DOI,
                "download_date": datetime.now(JST).isoformat(timespec="seconds"),
            }
        )

    # 重複の実測。SPEC §2 が「zip 34 本のうち 1 本は重複」と述べる根拠をここで作る
    by_digest: dict[str, list[str]] = {}
    for r in records:
        by_digest.setdefault(r["sha256"], []).append(r["filename"])
    dups = {d: names for d, names in by_digest.items() if len(names) > 1}

    out = {
        "dataset": "Bacteria Data for Machine Vision and Digital Biology",
        "doi": DOI,
        "version": VERSION,
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "files": records,
        "duplicate_sha256_groups": dups,
        "generated_at": datetime.now(JST).isoformat(timespec="seconds"),
    }
    (META / "raw_manifest.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n{len(records)} ファイル / 重複グループ {len(dups)} 件 → dataset/metadata/raw_manifest.json")
    for d, names in dups.items():
        print(f"  重複: {names} (sha256 {d[:16]}…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
