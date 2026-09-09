"""画面に載せるサンプル画像と、ブラウザ側を照合するための期待値を作る。

## サンプルの選び方

**選び方を先に書く。** あとから「よく当たる画像」を選ぶと、画面が実力より良く見える。

  - Gram 陽性 / 陰性の両方
  - 球菌 / 桿菌の両方
  - **色の規則が破れる分類群を必ず入れる**(`Listeria monocytogenes`)。
    うまくいく例だけを並べない
  - 同じ group から二枚選ばない(同じスライドの別視野を別の例として見せない)
  - 各分類群の中では image_id の辞書順で先頭。**予測の当たり外れで選ばない**

## 期待値

ブラウザ側の前処理と推論が Python 側と一致することを確かめる材料(G-02)。

  - `expect/<id>.input.f32` — 224x224x3 の正規化済みテンソル(CHW・float32 little endian)
  - `expect/<id>.logits.f32` — Python の ONNX Runtime が返した logits(2 個)

配布物に入れるのは**一番小さい検体だけ**にはしない —— サンプルは画面が使うので全部要る。
ただし期待値は検査でしか使わないので `tests/fixtures/` に置き、配布物には入れない。

使い方:
    python -m ml.build_samples
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "dataset" / "metadata"
PROC = ROOT / "dataset" / "processed"
PUBLIC_SAMPLES = ROOT / "public" / "samples"
FIXTURES = ROOT / "tests" / "fixtures"
JST = timezone(timedelta(hours=9))

INPUT = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 選ぶ分類群と、その理由。**理由を書けないものは選ばない**
WANTED: list[tuple[str, str]] = [
    ("Staphylococcus aureus", "Gram 陽性・球菌。教科書どおりの紫のぶどう状"),
    ("Streptococcus agalactiae", "Gram 陽性・球菌。連鎖状に並ぶ"),
    ("Lactobacillus delbrueckii", "Gram 陽性・桿菌。細長い"),
    ("Clostridium perfringens", "Gram 陽性・桿菌。太い"),
    ("Escherichia coli", "Gram 陰性・桿菌。桃色に染まる"),
    ("Pseudomonas aeruginosa", "Gram 陰性・桿菌"),
    ("Neisseria gonorrhoeae", "Gram 陰性・球菌。双球菌"),
    ("Veillonella", "Gram 陰性・球菌。Bacillota 門なのに Gram 陰性という反例"),
    ("Listeria monocytogenes", "Gram 陽性だが脱色されやすい。**色の規則が最もよく破れる**分類群"),
    ("Porphyromonas gingivalis", "Gram 陰性だが染まり方が揺れる。種水準の典拠が無く属で補っている"),
]


def normalise(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("RGB").resize((INPUT, INPUT), Image.Resampling.BILINEAR),
                   dtype=np.float32) / 255.0
    a = (a - MEAN) / STD
    return np.ascontiguousarray(a.transpose(2, 0, 1))  # CHW


def main() -> int:
    rows = [json.loads(line) for line in (META / "prepared.jsonl").read_text(encoding="utf-8").splitlines()]
    labels = {t["folder"]: t for t in
              json.loads((META / "labels.json").read_text(encoding="utf-8"))["taxa"] if t["included"]}
    onnx = ROOT / "public" / "models" / "model.onnx"
    if not onnx.exists():
        print("public/models/model.onnx が無い。先に python -m ml.export_onnx を実行する")
        return 1
    sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    PUBLIC_SAMPLES.mkdir(parents=True, exist_ok=True)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "expect").mkdir(exist_ok=True)

    by_folder: dict[str, list[dict]] = {}
    for r in rows:
        by_folder.setdefault(r["folder"], []).append(r)

    used_groups: set[str] = set()
    samples = []
    for folder, why in WANTED:
        cands = sorted(by_folder.get(folder, []), key=lambda r: r["image_id"])
        pick = next((r for r in cands if r["group_id"] not in used_groups), None)
        if pick is None:
            print(f"  !! {folder}: 使える画像が無い")
            continue
        used_groups.add(pick["group_id"])

        src = PROC / pick["path"]
        dst = PUBLIC_SAMPLES / f"{pick['image_id']}.png"
        shutil.copyfile(src, dst)

        with Image.open(src) as im:
            x = normalise(im)
        logits = sess.run(None, {in_name: x[None].astype(np.float32)})[0][0]

        (FIXTURES / "expect" / f"{pick['image_id']}.input.f32").write_bytes(x.astype("<f4").tobytes())
        (FIXTURES / "expect" / f"{pick['image_id']}.logits.f32").write_bytes(
            logits.astype("<f4").tobytes())
        # 生の RGBA は書かない。同じ中身の PNG を配布物として既に追跡しており、
        # 期待値にも置くと 10.5 MB が二重にリポジトリへ残る。
        # TS 側は tests/helpers/png.ts で復号する(その復号器の正しさは、
        # 復号して作ったテンソルがここの期待値と一致することで確かめられる)

        lab = labels[folder]
        samples.append({
            "id": pick["image_id"],
            "file": f"{pick['image_id']}.png",
            "folder": folder,
            "scientific_name": lab["resolved_name"],
            "gram": lab["gram"],
            "shape": lab["shape"],
            "family": lab["group_family"],
            "why": why,
            "source_member": pick["source_member"],
        })
        print(f"  {folder:32s} {pick['image_id']}  logits {logits.round(3).tolist()}")

    doc = {
        "generated_at": datetime.now(JST).isoformat(timespec="seconds"),
        "selection_rule": "Gram 陽性/陰性・球菌/桿菌の両方を含め、色の規則が破れる分類群を必ず入れる。"
                          "同じ group から二枚選ばない。各分類群では image_id の辞書順で先頭。"
                          "**予測の当たり外れで選ばない**",
        "crop": "配布元の画像から視野円の内接正方形の中心 512x512 を切り出したもの(リサイズなし)",
        "license": "CC BY 4.0",
        "attribution": "Jamshidi, Mohammad (Behdad); Sargolzaee, Saleh; Foorginezhad, Salimeh; "
                       "Moztarzadeh, Omid (2023), Bacteria Data for Machine Vision and Digital "
                       "Biology, Mendeley Data, V1, doi: 10.17632/cvkgfzp7ck.1. "
                       "Licensed under CC BY 4.0. 視野円の内接正方形を切り出す加工を行っている。",
        "samples": samples,
    }
    (PUBLIC_SAMPLES / "index.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    (FIXTURES / "samples.json").write_text(
        json.dumps({"samples": [{"id": s["id"], "gram": s["gram"]} for s in samples]},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    total = sum((PUBLIC_SAMPLES / s["file"]).stat().st_size for s in samples)
    print(f"\nサンプル {len(samples)} 枚 / {total/1e6:.2f} MB → public/samples/")
    print(f"期待値 → tests/fixtures/expect/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
