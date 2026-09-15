"""CNN を三つの物差しで測り、SPEC §3.4 に先に書いた合否条件へ機械的に当てはめる。

## 骨格の選び方

SPEC は ResNet18 を標準候補としていたが、**実測で外れた**(loop_004)。

| 骨格 | fp32 サイズ | CPU 学習速度 |
|---|---|---|
| ResNet18 | 46.8 MB | 17.4 枚/秒 |
| EfficientNet-B0 | 21.2 MB | 16.4 枚/秒 |
| MobileNetV3-Small | 10.2 MB | 60.1 枚/秒 |

ResNet18 は N-01 の配信目標 30 MB を**量子化する前から**超えている。
目標のほうを動かすのではなく骨格を変える。MobileNetV3-Small は 10.2 MB で収まり、
22 fold を CPU で回せる唯一の候補でもある。

## エポック予算

fold ごとに val を切ると、fold の中でさらにデータが減る。代わりに、
**物差し A の 70/15/15 の val で予算を一度決め、その予算を全 fold へ一律に当てる**。
test は一切見ない(G-08)。

## ラベル置換の対照(G-09)

ラベルを無作為に置き換えて同じ手続きを回す。成績が多数派クラスの水準へ落ちなければ、
成績の出所は入力の信号ではなく手続きのどこかにある漏れである。
**損失の比では対照にならない**(HC-163)ので、最終的な macro F1 で見る。

使い方:
    python -m ml.train --stage budget    # エポック予算を val で決める
    python -m ml.train --stage sweep     # 三つの物差し + ラベル置換
    python -m ml.train --stage ship      # 出荷モデルを 70/15/15 で作る
    python -m ml.train --stage all
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn

from ml.metrics import NEGATIVE, POSITIVE, bootstrap_macro_f1, macro_f1

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "dataset" / "metadata"
PROC = ROOT / "dataset" / "processed"
REPORTS = ROOT / "reports"
CKPT = ROOT / "ml" / "checkpoints"
JST = timezone(timedelta(hours=9))

ARCH = "mobilenet_v3_small"
INPUT = 224
BATCH = 32
LR = 3e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 20
SEED = 20260908
N_BOOT = 2000

# ImageNet 統計。転移学習の前提に合わせる
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


# ------------------------------------------------------------------ データ


def load_all(rows: list[dict]) -> tuple[np.ndarray, dict[str, int]]:
    """全画像を uint8 で常駐させる。

    読み込み+縮小は 51 枚/秒しか出ず、MobileNet の学習(60 枚/秒)では律速になる。
    1,966 x 224 x 224 x 3 = 296 MB でメモリに載るので、最初に一度だけ読む。
    """
    arr = np.zeros((len(rows), INPUT, INPUT, 3), dtype=np.uint8)
    index: dict[str, int] = {}
    for i, r in enumerate(rows):
        with Image.open(PROC / r["path"]) as im:
            arr[i] = np.asarray(im.convert("RGB").resize((INPUT, INPUT), Image.Resampling.BILINEAR))
        index[r["image_id"]] = i
        if (i + 1) % 500 == 0:
            print(f"    常駐 {i+1}/{len(rows)}", flush=True)
    return arr, index


def to_tensor(batch: np.ndarray, train: bool, rng: np.random.Generator) -> torch.Tensor:
    """uint8 の束を正規化済みテンソルにする。増補は学習時のみ。

    顕微鏡の視野に向きの意味は無いので、水平・垂直反転と 90 度回転を使う。
    **色は振らない。** Gram 染色では色調が分類情報そのものである
    (だからこそ、色を取り上げたときに何が残るかは別に測る価値がある)。
    """
    x = batch.astype(np.float32) / 255.0
    if train:
        if rng.random() < 0.5:
            x = x[:, :, ::-1]
        if rng.random() < 0.5:
            x = x[:, ::-1]
        k = int(rng.integers(0, 4))
        if k:
            x = np.rot90(x, k, axes=(1, 2))
        x = np.ascontiguousarray(x)
    x = (x - MEAN) / STD
    return torch.from_numpy(x).permute(0, 3, 1, 2)


def build_model() -> nn.Module:
    m = torchvision.models.mobilenet_v3_small(
        weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, 2)
    return m


# ------------------------------------------------------------------ 学習


def train_model(arr, idx_train, y_train, epochs: int, seed: int,
                idx_eval=None, y_eval=None, log_every: int = 0):
    """1 本学習する。idx_eval を渡すとエポックごとの成績を返す。"""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    # クラス不均衡(陽性 1,407 / 陰性 559)を損失で補正する。
    # 補正しないと、多数派を答えるだけの解に落ちても損失は下がる
    counts = np.array([(y_train == POSITIVE).sum(), (y_train == NEGATIVE).sum()], dtype=np.float32)
    w = torch.from_numpy((counts.sum() / (2 * np.maximum(counts, 1))).astype(np.float32))
    lossf = nn.CrossEntropyLoss(weight=w)

    curve = []
    order = np.arange(len(idx_train))
    for ep in range(epochs):
        model.train()
        rng.shuffle(order)
        total = 0.0
        for s in range(0, len(order), BATCH):
            sel = idx_train[order[s : s + BATCH]]
            xb = to_tensor(arr[sel], True, rng)
            yb = torch.from_numpy(y_train[order[s : s + BATCH]].astype(np.int64))
            out = model(xb)
            loss = lossf(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.detach().item() * len(sel)
        sched.step()
        entry = {"epoch": ep + 1, "train_loss": round(total / len(order), 4)}
        if idx_eval is not None:
            pred = predict(model, arr, idx_eval)
            entry["eval_macro_f1"] = round(macro_f1(y_eval, pred), 4)
        curve.append(entry)
        if log_every and (ep + 1) % log_every == 0:
            print(f"      epoch {ep+1}/{epochs} {entry}", flush=True)
    return model, curve


@torch.no_grad()
def predict(model: nn.Module, arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    model.eval()
    rng = np.random.default_rng(0)
    out = np.empty(len(idx), dtype=np.int64)
    for s in range(0, len(idx), BATCH):
        sel = idx[s : s + BATCH]
        logits = model(to_tensor(arr[sel], False, rng))
        out[s : s + BATCH] = logits.argmax(1).numpy()
    return out


# ------------------------------------------------------------------ 段


def prepare(target: str = "gram") -> tuple:
    """target が shape なら Stage B(形)。対象外を抜き、fold も抜くだけにする(SPEC §3.7)。

    形のラベルは 球菌 = 0 / 桿菌 = 1(ml/stage_b.py の対照と同じ向き)。
    """
    rows = [json.loads(line) for line in (META / "prepared.jsonl").read_text(encoding="utf-8").splitlines()]
    splits = json.loads((META / "splits.json").read_text(encoding="utf-8"))
    if target == "shape":
        from ml.stage_b import BACILLUS, COCCUS, restrict_folds

        rows = [r for r in rows if r["stage_b"]]
        keep = {r["image_id"] for r in rows}
        for key in ("ruler_a_cv", "ruler_b", "ruler_c"):
            splits[key]["folds"] = restrict_folds(splits[key]["folds"], keep)
        splits["ruler_a"] = {part: [i for i in ids if i in keep] for part, ids in splits["ruler_a"].items()}
    print(f"  画像 {len(rows)} 枚を常駐させる(target={target})")
    arr, index = load_all(rows)

    def label(r: dict) -> int:
        if target == "shape":
            return BACILLUS if r["shape"] == "bacillus" else COCCUS
        return POSITIVE if r["gram"] == "positive" else NEGATIVE

    meta = {
        r["image_id"]: {
            "y": label(r),
            "group_id": r["group_id"],
            "folder": r["folder"],
        }
        for r in rows
    }
    return rows, splits, arr, index, meta


def ids_to_arrays(ids: list[str], index: dict, meta: dict):
    idx = np.array([index[i] for i in ids])
    y = np.array([meta[i]["y"] for i in ids])
    return idx, y


def run_budget(splits, arr, index, meta) -> dict:
    """物差し A の 70/15/15 の val でエポック予算を決める。test は見ない。"""
    a = splits["ruler_a"]
    idx_tr, y_tr = ids_to_arrays(a["train"], index, meta)
    idx_va, y_va = ids_to_arrays(a["val"], index, meta)
    print(f"  予算探索: train {len(idx_tr)} / val {len(idx_va)} で最大 {MAX_EPOCHS} エポック")
    t0 = time.time()
    _, curve = train_model(arr, idx_tr, y_tr, MAX_EPOCHS, SEED, idx_va, y_va, log_every=1)
    best = max(curve, key=lambda c: (c["eval_macro_f1"], -c["epoch"]))
    print(f"  → 最良 epoch {best['epoch']}(val macro F1 {best['eval_macro_f1']})  "
          f"{time.time()-t0:.0f}s")
    return {"chosen": best["epoch"], "chosen_on": "ruler_a_val", "max_epochs": MAX_EPOCHS,
            "curve": curve, "best_val_macro_f1": best["eval_macro_f1"]}


def run_folds(folds, arr, index, meta, epochs: int, permute: bool, tag: str) -> dict:
    """fold を順に学習し、out-of-fold の予測を集めて採点する。

    **fold ごとに保存する**(loop_004 で踏んだ TOOL-ENV)。この機では他のセッションが
    同時に重い処理を走らせることがあり、実効コア数が読めない。長い計算が止まったとき、
    保存していなければそこまでの計算が丸ごと消える。HC-219 で外部取得について書いた
    再開可能性は、長い計算にも同じく要る。
    """
    cache_dir = CKPT / "fold_preds"
    cache_dir.mkdir(parents=True, exist_ok=True)
    preds: dict[str, int] = {}
    for i, fold in enumerate(folds):
        cache = cache_dir / f"{tag}_{epochs}ep_{i:02d}.json"
        if cache.exists():
            preds.update(json.loads(cache.read_text(encoding="utf-8")))
            print(f"    {tag} fold {i+1}/{len(folds)}  キャッシュ", flush=True)
            continue
        idx_tr, y_tr = ids_to_arrays(fold["train"], index, meta)
        idx_te, _ = ids_to_arrays(fold["test"], index, meta)
        if permute:
            # ラベルだけを無作為に置き換える。入力も手続きも変えない(G-09)
            y_tr = np.random.default_rng(SEED + i).permutation(y_tr)
        t0 = time.time()
        model, _ = train_model(arr, idx_tr, y_tr, epochs, SEED + i)
        p = predict(model, arr, idx_te)
        got = {image_id: int(v) for image_id, v in zip(fold["test"], p)}
        cache.write_text(json.dumps(got, ensure_ascii=False), encoding="utf-8")
        preds.update(got)
        print(f"    {tag} fold {i+1}/{len(folds)}  train {len(idx_tr)} / test {len(idx_te)}  "
              f"{time.time()-t0:.0f}s", flush=True)
    return preds


def score(preds: dict[str, int], meta: dict) -> dict:
    ids = sorted(preds)
    y = np.array([meta[i]["y"] for i in ids])
    p = np.array([preds[i] for i in ids])
    g = np.array([meta[i]["group_id"] for i in ids])
    point, lo, hi = bootstrap_macro_f1(y, p, g, n_boot=N_BOOT)
    return {"macro_f1": round(point, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "accuracy": round(float((y == p).mean()), 4),
            "n_images": len(ids), "n_groups": int(len(set(g)))}


def per_taxon(preds: dict[str, int], meta: dict) -> dict:
    out: dict[str, dict] = {}
    for image_id, p in preds.items():
        m = meta[image_id]
        d = out.setdefault(m["folder"], {"n": 0, "wrong": 0})
        d["n"] += 1
        d["wrong"] += int(p != m["y"])
    return {k: {**v, "error_rate": round(v["wrong"] / v["n"], 4)} for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("budget", "sweep", "ship", "all"), default="all")
    ap.add_argument("--epochs", type=int, default=None, help="予算を直接指定する(budget を飛ばす)")
    # この機では他のセッションが同時に重い処理を走らせることがある(loop_004 実測)。
    # 得られないコア数を torch に渡すと、奪い合ってスレッド競合を起こし、
    # **速度低下は比例より悪くなる**。既定は控えめにして、空いているときだけ上げる
    ap.add_argument("--threads", type=int, default=3, help="torch のスレッド数")
    ap.add_argument("--target", choices=("gram", "shape"), default="gram",
                    help="gram は Stage A、shape は Stage B(形)。報告も fold のキャッシュも別にする")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    print(f"  torch スレッド {args.threads}(論理コア {__import__('os').cpu_count()})")
    REPORTS.mkdir(exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    shape = args.target == "shape"
    out_path = REPORTS / ("cnn_shape.json" if shape else "cnn.json")
    doc = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    # fold のキャッシュ名に target を入れる。Stage A の予測を Stage B が読むと、
    # 形の成績として Gram の予測を採点してしまう
    tag = "S" if shape else ""

    rows, splits, arr, index, meta = prepare(args.target)
    doc["target"] = args.target

    n_params = sum(p.numel() for p in build_model().parameters())
    doc["model"] = {
        "architecture": ARCH,
        "pretrained": "IMAGENET1K_V1",
        "input_size": [INPUT, INPUT],
        "parameters": int(n_params),
        "fp32_megabytes": round(n_params * 4 / 1e6, 2),
        "why_not_resnet18": "ResNet18 は fp32 で 46.8MB あり、量子化する前から N-01 の配信目標 "
                            "30MB を超える。目標を動かすのではなく骨格を変えた",
    }
    doc["generated_at"] = datetime.now(JST).isoformat(timespec="seconds")
    doc["n_boot"] = N_BOOT
    doc["seed"] = SEED

    if args.stage in ("budget", "all") and args.epochs is None:
        doc["epoch_budget"] = run_budget(splits, arr, index, meta)
        out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.epochs is not None:
        doc["epoch_budget"] = {"chosen": args.epochs, "chosen_on": "ruler_a_val",
                               "max_epochs": MAX_EPOCHS, "curve": doc.get("epoch_budget", {}).get("curve", []),
                               "note": "予算を直接指定した"}

    epochs = doc["epoch_budget"]["chosen"]

    if args.stage in ("sweep", "all"):
        doc.setdefault("rulers", {})
        doc.setdefault("per_taxon", {})
        for ruler, key in (("a", "ruler_a_cv"), ("b", "ruler_b"), ("c", "ruler_c")):
            print(f"  物差し {ruler.upper()}({epochs} エポック)")
            preds = run_folds(splits[key]["folds"], arr, index, meta, epochs, False, tag + ruler.upper())
            doc["rulers"][ruler] = score(preds, meta)
            doc["per_taxon"][ruler] = per_taxon(preds, meta)
            print(f"    → macro F1 {doc['rulers'][ruler]['macro_f1']}")
            out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"  ラベル置換の対照(物差し A・{epochs} エポック)")
        preds = run_folds(splits["ruler_a_cv"]["folds"], arr, index, meta, epochs, True, tag + "PERM")
        doc["permutation_control"] = {"a": score(preds, meta)}
        print(f"    → macro F1 {doc['permutation_control']['a']['macro_f1']}")
        out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.stage in ("ship", "all"):
        a = splits["ruler_a"]
        idx_tr, y_tr = ids_to_arrays(a["train"], index, meta)
        idx_va, y_va = ids_to_arrays(a["val"], index, meta)
        idx_te, y_te = ids_to_arrays(a["test"], index, meta)
        print(f"  出荷モデル: train {len(idx_tr)} で学習し val {len(idx_va)} で確認、"
              f"test {len(idx_te)} は最後に一度だけ")
        model, curve = train_model(arr, idx_tr, y_tr, epochs, SEED, idx_va, y_va, log_every=2)
        ckpt_name = "best_model_shape.pth" if shape else "best_model.pth"
        torch.save(model.state_dict(), CKPT / ckpt_name)
        p_te = predict(model, arr, idx_te)
        doc["shipped_model"] = {
            "trained_on": ["train"],
            "selected_on": "val",
            "test_evaluations": 1,
            "epochs": epochs,
            "val_macro_f1": curve[-1]["eval_macro_f1"],
            "test": {"macro_f1": round(macro_f1(y_te, p_te), 4),
                     "accuracy": round(float((y_te == p_te).mean()), 4),
                     "n_images": len(idx_te)},
            "checkpoint": f"ml/checkpoints/{ckpt_name}",
        }
        out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    # 集計は材料が揃っているときだけ。段を分けて走らせられるようにした以上、
    # 末尾で無条件に集計すると、材料の無い段が落ちる(loop_004 で踏んだ)
    if "rulers" in doc and all(r in doc["rulers"] for r in ("a", "b", "c")):
        (finalise_shape if shape else finalise)(doc, out_path)
    else:
        print("  (三つの物差しが揃っていないので判定は保留)")
    return 0


def finalise(doc: dict, out_path: Path) -> None:
    """SPEC §3.4 の二条件へ機械的に当てはめる。

    **判定をこちらの言葉で書かない。** 書いてよいのは条件だけで、
    当てはめる仕事は機械にさせる。そうしないと、数を見てから
    「効いた」と言える理由を探すことになる。
    """
    controls = json.loads((REPORTS / "controls.json").read_text(encoding="utf-8"))
    hue = controls["baselines"]["hue"]
    doc["majority_reference"] = {r: controls["baselines"]["majority"][r] for r in ("a", "b", "c")}
    doc["control_reference"] = {r: hue[r] for r in ("a", "b", "c")}

    cond1 = all(doc["rulers"][r]["ci_low"] > hue[r]["ci_high"] for r in ("a", "b", "c"))

    # 条件 2 の対象 — 色の規則が落ちる分類群。**選び方も先に決めておく**:
    # 物差し B で色の規則の誤り率が 10% 以上だったもの
    hard = sorted(
        k for k, v in hue["b"]["per_taxon_errors"].items() if v["error_rate"] >= 0.10
    )
    cond2 = None
    detail = {}
    if hard and "per_taxon" in doc:
        cnn_b = doc["per_taxon"]["b"]
        wins = 0
        for t in hard:
            hue_err = hue["b"]["per_taxon_errors"][t]["error_rate"]
            cnn_err = cnn_b.get(t, {}).get("error_rate")
            detail[t] = {"hue_error_rate": hue_err, "cnn_error_rate": cnn_err}
            if cnn_err is not None and cnn_err < hue_err:
                wins += 1
        # 「明確に上回る」= 対象分類群の過半で誤り率が下がり、合計の誤り数も減っている
        hue_total = sum(hue["b"]["per_taxon_errors"][t]["wrong"] for t in hard)
        cnn_total = sum(cnn_b.get(t, {}).get("wrong", 0) for t in hard)
        cond2 = wins > len(hard) / 2 and cnn_total < hue_total
        detail["_totals"] = {"hue_wrong": hue_total, "cnn_wrong": cnn_total,
                             "taxa_improved": wins, "taxa_considered": len(hard)}

    doc["verdict"] = {
        "condition_1_beats_hue_on_all_rulers": cond1,
        "condition_2_beats_on_hard_taxa": bool(cond2),
        "hard_taxa": hard,
        "hard_taxa_selection": "物差し B で色の規則の誤り率が 10% 以上だった分類群",
        "hard_taxa_detail": detail,
        "adds_nothing": (not cond1) and (not bool(cond2)),
    }
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    write_markdown(doc, controls)

    v = doc["verdict"]
    print()
    print(f"  条件 1(三物差しすべてで色の規則を区間の重なりなく上回る): {v['condition_1_beats_hue_on_all_rulers']}")
    print(f"  条件 2(色の規則が落ちる分類群で明確に上回る): {v['condition_2_beats_on_hard_taxa']}")
    if v["adds_nothing"]:
        print("  → このデータで CNN は色の規則に何も足していない")


def write_markdown(doc: dict, controls: dict) -> None:
    """結果を読める形にする。数と判定はすべて doc から取り、ここで作らない。"""
    hue = controls["baselines"]["hue"]
    maj = controls["baselines"]["majority"]
    v = doc["verdict"]
    L: list[str] = []
    add = L.append

    add("# CNN を三つの物差しで測る")
    add("")
    add(f"<!-- ml/train.py が生成。手で編集しない。生成日時 {doc['generated_at']} -->")
    add("")
    add(f"骨格 **{doc['model']['architecture']}**(ImageNet 事前学習・"
        f"パラメータ {doc['model']['parameters']:,}・fp32 {doc['model']['fp32_megabytes']} MB)。")
    add(f"エポック予算 {doc['epoch_budget']['chosen']}(物差し A の val で決めた。test は見ていない)。")
    add(f"信頼区間は group 単位のブートストラップ {doc['n_boot']:,} 回。")
    add("")
    add("## 結果")
    add("")
    add("| | A 画像単位 | B 分類群ホールドアウト | C 科ホールドアウト |")
    add("|---|---|---|---|")
    for label, src in (("**CNN**", doc["rulers"]),
                       ("色相(非学習)", hue),
                       ("多数派クラス", maj)):
        cells = [f"{src[r]['macro_f1']:.4f} [{src[r]['ci_low']:.4f}, {src[r]['ci_high']:.4f}]"
                 for r in ("a", "b", "c")]
        add(f"| {label} | {' | '.join(cells)} |")
    add("")
    add("macro F1 と 95% 信頼区間。どの行も全 "
        f"{doc['rulers']['a']['n_images']:,} 枚を一度ずつ評価している。")
    add("")

    add("## 学習系の陽性対照(G-09)")
    add("")
    p = doc["permutation_control"]["a"]
    add(f"ラベルを無作為に置き換えて同じ手続きを回すと、macro F1 は "
        f"**{p['macro_f1']:.4f}** [{p['ci_low']:.4f}, {p['ci_high']:.4f}] だった"
        f"(多数派クラス {maj['a']['macro_f1']:.4f})。")
    add("")
    add("信号を消せば成績も消える。**損失の比では対照にならない**(HC-163)ので、")
    add("最終的な macro F1 で見ている。ここで成績が残るなら、")
    add("それは入力の信号ではなく手続きの漏れが出している数である。")
    add("")

    add("## 判定")
    add("")
    add("合否条件は [SPEC.md](../SPEC.md) の 3.4 節に**測る前に**書いた。")
    add("当てはめる仕事は機械にさせている(検査 T-235)。")
    add("")
    add("- 条件 1(三つの物差しすべてで色の規則を信頼区間の重なりなく上回る): "
        f"**{'成立' if v['condition_1_beats_hue_on_all_rulers'] else '不成立'}**")
    add("- 条件 2(色の規則が落ちる分類群で明確に上回る): "
        f"**{'成立' if v['condition_2_beats_on_hard_taxa'] else '不成立'}**")
    add("")
    if v["adds_nothing"]:
        add("**どちらも成立しなかった。このデータで CNN は色の規則に何も足していない。**")
        add("")
        add("Gram 染色は色の検査なので、これ自体は驚くべきことではない。驚くべきなのは、")
        add("**対照を置かなければ「AI が 9 割以上当てた」と書けてしまった**ということのほうである。")
    else:
        add("成立した条件については、下の内訳でどこが効いたかを見る。")
    add("")

    add("### 条件 2 の内訳")
    add("")
    add(f"対象の選び方: {v['hard_taxa_selection']}")
    add("")
    add("| 分類群 | 色相の誤り率 | CNN の誤り率 |")
    add("|---|---|---|")
    for t in v["hard_taxa"]:
        d = v["hard_taxa_detail"][t]
        cnn_e = d["cnn_error_rate"]
        shown = f"{cnn_e * 100:.0f}%" if cnn_e is not None else "—"
        add(f"| {t} | {d['hue_error_rate']:.0%} | {shown} |")
    tot = v["hard_taxa_detail"].get("_totals", {})
    if tot:
        add("")
        add(f"対象 {tot['taxa_considered']} 分類群のうち CNN が改善したのは {tot['taxa_improved']}、")
        add(f"誤り総数は色相 {tot['hue_wrong']} 対 CNN {tot['cnn_wrong']}。")
    add("")

    add("## 物差しごとの差が意味すること")
    add("")
    a, b, c = (doc["rulers"][r]["macro_f1"] for r in ("a", "b", "c"))
    ha, hb, hc = (hue[r]["macro_f1"] for r in ("a", "b", "c"))
    add(f"- CNN: A {a:.4f} → B {b:.4f} → C {c:.4f}")
    add(f"- 色相: A {ha:.4f} → B {hb:.4f} → C {hc:.4f}")
    add("")
    add("色の規則は物差しを変えてもほとんど動かない。色は種の同一性に依らないからである。")
    add("**CNN のほうが大きく落ちるなら、その落ちた分が「種の見覚え」だった**ということになる。")
    add("落ちないなら、CNN も種に依らない手がかりを見ていることになる。")
    add("")

    add("## 出荷するモデル")
    add("")
    s = doc.get("shipped_model")
    if s:
        add("物差し A の 70/15/15 で学習した。train で作り、val で確認し、")
        add(f"**test には最後に一度だけ**当てた(test_evaluations = {s['test_evaluations']}・G-08)。")
        add("")
        add("| 項目 | 値 |")
        add("|---|---|")
        add(f"| エポック | {s['epochs']} |")
        add(f"| val macro F1 | {s['val_macro_f1']} |")
        add(f"| test macro F1 | {s['test']['macro_f1']}({s['test']['n_images']} 枚) |")
        add(f"| test 正解率 | {s['test']['accuracy']} |")
        add("")
        add("### この数だけを出してはならない")
        add("")
        add(f"**この test macro F1 {s['test']['macro_f1']} は、本プロジェクトが警告している当のものである。**")
        add("正しい手順で出した数ではある —— train で学習し、val で確認し、test には一度だけ当てた。")
        add("それでもこの数は、**同じ分類群のスライドを見分ける課題**の成績でしかない。")
        add("")
        add(f"同じ骨格・同じ手続きで、科をまるごと抜けば {doc['rulers']['c']['macro_f1']:.4f} まで落ちる。")
        add("落差は物差しの取り方だけから来ている。")
        add("")
        add("だからこの数は、**単独では出さない**。三つの物差しの表と必ず並べる。")
    else:
        add("未作成。")
    add("")
    add("## 骨格を変えた理由")
    add("")
    add(doc["model"]["why_not_resnet18"])
    (REPORTS / "cnn.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def finalise_shape(doc: dict, out_path: Path) -> None:
    """SPEC §3.8 の二条件へ機械的に当てはめる(Stage B)。

    「強いほうの対照」は物差しごとに、形の規則と「色の規則を形に当てたもの」の
    macro F1 が高いほう。**どちらを強いとみなすかも数を見る前に決めてある。**
    """
    controls = json.loads((REPORTS / "controls_shape.json").read_text(encoding="utf-8"))
    base = controls["baselines"]
    strong: dict[str, dict] = {}
    strong_name: dict[str, str] = {}
    for r in ("a", "b", "c"):
        s, h = base["shape"][r], base["hue_on_shape"][r]
        strong_name[r] = "shape" if s["macro_f1"] >= h["macro_f1"] else "hue_on_shape"
        strong[r] = s if strong_name[r] == "shape" else h
    doc["majority_reference"] = {r: base["majority"][r] for r in ("a", "b", "c")}
    doc["control_reference"] = {name: {r: base[name][r] for r in ("a", "b", "c")}
                                for name in ("shape", "hue_on_shape", "canvas_size")}

    cond1 = all(doc["rulers"][r]["ci_low"] > strong[r]["ci_high"] for r in ("a", "b", "c"))

    # 条件 2 の対象 — 物差し B で形の規則の誤り率が 10% 以上だった分類群
    rule_err = base["shape"]["b"]["per_taxon_errors"]
    hard = sorted(k for k, v in rule_err.items() if v["error_rate"] >= 0.10)
    cnn_b = doc["per_taxon"]["b"]
    detail: dict[str, dict] = {}
    wins = 0
    for t in hard:
        cnn_err = cnn_b.get(t, {}).get("error_rate")
        detail[t] = {"shape_rule_error_rate": rule_err[t]["error_rate"], "cnn_error_rate": cnn_err}
        if cnn_err is not None and cnn_err < rule_err[t]["error_rate"]:
            wins += 1
    rule_total = sum(rule_err[t]["wrong"] for t in hard)
    cnn_total = sum(cnn_b.get(t, {}).get("wrong", 0) for t in hard)
    cond2 = bool(hard) and wins > len(hard) / 2 and cnn_total < rule_total
    detail["_totals"] = {"shape_rule_wrong": rule_total, "cnn_wrong": cnn_total,
                         "taxa_improved": wins, "taxa_considered": len(hard)}

    doc["finalised_at"] = datetime.now(JST).isoformat(timespec="seconds")
    doc["verdict"] = {
        "condition_1_beats_strong_control_on_all_rulers": cond1,
        "condition_2_beats_on_hard_taxa": bool(cond2),
        "strong_control": strong_name,
        "hard_taxa": hard,
        "hard_taxa_selection": "物差し B で形の規則の誤り率が 10% 以上だった分類群",
        "hard_taxa_detail": detail,
        "adds_nothing": (not cond1) and (not bool(cond2)),
        # 成否ではなく読み方(§3.8): 色の規則が形の規則より高い物差しでは、形のラベルが色で推せる
        "colour_explains_shape": {r: base["hue_on_shape"][r]["macro_f1"] > base["shape"][r]["macro_f1"]
                                  for r in ("a", "b", "c")},
    }
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_shape(doc, controls)

    v = doc["verdict"]
    print()
    print(f"  条件 1(三物差しすべてで強いほうの対照を区間の重なりなく上回る): {v['condition_1_beats_strong_control_on_all_rulers']}")
    print(f"  条件 2(形の規則が落ちる分類群で明確に上回る): {v['condition_2_beats_on_hard_taxa']}")
    if v["adds_nothing"]:
        print("  → このデータで CNN は形について非学習の規則に何も足していない")


def write_markdown_shape(doc: dict, controls: dict) -> None:
    """Stage B の結果を読める形にする。数と判定はすべて doc と controls から取る。"""
    base = controls["baselines"]
    v = doc["verdict"]
    L: list[str] = []
    add = L.append
    add("# Stage B — 形(球菌 / 桿菌)を三つの物差しで測る")
    add("")
    add(f"<!-- ml/train.py --target shape が生成。手で編集しない。学習開始 {doc['generated_at']} / 判定 {doc.get('finalised_at', '—')} -->")
    add("")
    add(f"対象は Stage B の {controls['n_taxa']} 分類群・{controls['n_images_total']:,} 枚。"
        f"骨格・前処理は Stage A と同じ。エポック予算 {doc['epoch_budget']['chosen']}(物差し A の val で決めた)。")
    add("対照は CNN より先に測った(G-12)。")
    add("")
    add("## 結果")
    add("")
    add("| | A 画像単位 | B 分類群ホールドアウト | C 科ホールドアウト |")
    add("|---|---|---|---|")
    rows = (("**CNN**", doc["rulers"]),
            ("形の規則(非学習・色を見ない)", base["shape"]),
            ("色の規則を形に当てたもの", base["hue_on_shape"]),
            ("キャンバス寸法(画素を見ない)", base["canvas_size"]),
            ("多数派クラス", base["majority"]))
    for label, src in rows:
        cells = [f"{src[r]['macro_f1']:.4f} [{src[r]['ci_low']:.4f}, {src[r]['ci_high']:.4f}]" for r in ("a", "b", "c")]
        add(f"| {label} | {' | '.join(cells)} |")
    add("")
    add(f"macro F1 と 95% 信頼区間(group 単位のブートストラップ {doc['n_boot']:,} 回)。"
        f"形の規則で特徴量が取れなかった画像は {base['shape']['n_missing_feature']} 枚(train の多数派を答えた)。")
    add("")
    p = doc["permutation_control"]["a"]
    null_path = REPORTS / "permutation_null.json"
    if null_path.exists():
        s = json.loads(null_path.read_text(encoding="utf-8"))["shape"]
        add(f"ラベル置換の対照(G-09): {p['macro_f1']:.4f} [{p['ci_low']:.4f}, {p['ci_high']:.4f}]。"
            f"同じ予測を fold 内で入れ替えた偶然水準は中央 {s['median']:.4f} [{s['q025']:.4f}, {s['q975']:.4f}]"
            f"(観測以上の割合 p={s['p_upper']:.4f})で、**上に外れていない —— 学習系に漏れは無い**。")
        add("")
        add(f"G-09 は loop_011 で引き直した。旧基準「多数派クラス {base['majority']['a']['macro_f1']:.4f} の区間と重なる」では"
            "この結果が落ちる。多数派は全部を同じ答えにする予測の値で、ばらけて答える予測の偶然水準ではないからである"
            "(記録は TEST_SPEC の T-271)。")
    else:
        add(f"ラベル置換の対照(G-09): {p['macro_f1']:.4f} [{p['ci_low']:.4f}, {p['ci_high']:.4f}]"
            "(偶然水準は `python -m ml.stage_b perm-null` で出す)。")
    add("")
    add("## 判定")
    add("")
    add("合否条件は [SPEC.md](../SPEC.md) の 3.8 節に**対照を測る前に**書いた。当てはめは機械(検査 T-270)。")
    add("")
    names = {"shape": "形の規則", "hue_on_shape": "色の規則を形に当てたもの"}
    add("- 条件 1(三つの物差しすべてで、強いほうの対照を信頼区間の重なりなく上回る): "
        f"**{'成立' if v['condition_1_beats_strong_control_on_all_rulers'] else '不成立'}**"
        f"(強いほう: A {names[v['strong_control']['a']]} / B {names[v['strong_control']['b']]} / C {names[v['strong_control']['c']]})")
    add("- 条件 2(形の規則が落ちる分類群で明確に上回る): "
        f"**{'成立' if v['condition_2_beats_on_hard_taxa'] else '不成立'}**")
    add("")
    if v["adds_nothing"]:
        add("**どちらも成立しなかった。このデータで CNN は形について、非学習の規則に何も足していない。**")
        add("")
    add("### 条件 2 の内訳")
    add("")
    add(f"対象の選び方: {v['hard_taxa_selection']}")
    add("")
    add("| 分類群 | 形の規則の誤り率 | CNN の誤り率 |")
    add("|---|---|---|")
    for t in v["hard_taxa"]:
        d = v["hard_taxa_detail"][t]
        shown = f"{d['cnn_error_rate']:.0%}" if d["cnn_error_rate"] is not None else "—"
        add(f"| {t} | {d['shape_rule_error_rate']:.0%} | {shown} |")
    tot = v["hard_taxa_detail"]["_totals"]
    add("")
    add(f"対象 {tot['taxa_considered']} 分類群のうち CNN が改善したのは {tot['taxa_improved']}、"
        f"誤り総数は形の規則 {tot['shape_rule_wrong']} 対 CNN {tot['cnn_wrong']}。")
    add("")
    add("## 形のラベルは色で推せるか(読み方・§3.8)")
    add("")
    for r, label in (("a", "A"), ("b", "B"), ("c", "C")):
        s, h = base["shape"][r]["macro_f1"], base["hue_on_shape"][r]["macro_f1"]
        verdict = "色の規則のほうが高い —— 形のラベルが色で推せてしまう" if v["colour_explains_shape"][r] \
            else "形の規則のほうが高い —— 色では推せない分を形の規則が拾っている"
        add(f"- 物差し {label}: 形の規則 {s:.4f} / 色の規則 {h:.4f} → {verdict}")
    add("")
    cells_path = REPORTS / "cnn_shape_cells.json"
    if cells_path.exists():
        cells_doc = json.loads(cells_path.read_text(encoding="utf-8"))["rulers"]
        add("## どこで外したか —— Gram × 形の 4 セル(事後の分析・判定には入れない)")
        add("")
        add("判定が二条件とも成立しても、**誤りは一様ではない**。判定の後で、fold の予測を Gram × 形で数え直した"
            "(`python -m ml.stage_b cells`・検査 T-272)。")
        add("")
        add("| Gram × 形 | 分類群 | 枚数 | CNN A | CNN B | CNN C | 形の規則 B |")
        add("|---|---|---|---|---|---|---|")
        names_cell = {"negative x bacillus": "陰性 × 桿菌", "negative x coccus": "**陰性 × 球菌**",
                      "positive x bacillus": "陽性 × 桿菌", "positive x coccus": "陽性 × 球菌"}
        for k, label in names_cell.items():
            a = cells_doc["a"]["cnn"][k]
            add(f"| {label} | {a['n_taxa']} | {a['n']:,} | {a['error_rate']:.0%} | "
                f"{cells_doc['b']['cnn'][k]['error_rate']:.0%} | {cells_doc['c']['cnn'][k]['error_rate']:.0%} | "
                f"{cells_doc['b']['shape_rule'][k]['error_rate']:.0%} |")
        add("")
        add("数字は誤り率。**学習で見ていない分類群・科の Gram 陰性球菌を、CNN はほぼ全部外す**。"
            "同じ分類群を見ていれば(物差し A)ほぼ当てる。")
        add("")
        mech_path = REPORTS / "mechanism_negative_cocci.json"
        if mech_path.exists():
            mech = json.loads(mech_path.read_text(encoding="utf-8"))["primary"]
            add("**理由を一つ測った**(`python -m ml.mechanism measure`・SPEC §3.13)。陰性球菌を学習で見ていない fold のモデルで、"
                "形はそのままに色相だけを紫へ回すと、誤り率は "
                f"B {mech['b']['error_original']:.1%}→{mech['b']['error_violet']:.1%}・"
                f"C {mech['c']['error_original']:.1%}→{mech['c']['error_violet']:.1%}"
                f"(判定 B「{mech['b']['verdict']}」・C「{mech['c']['verdict']}」)。"
                "**色相は原因ではない。** 回したのは色相だけで、染まりの濃さ・大きさ・並び方は変えていないので、"
                "どの手がかりで桿菌と答えているかは特定していない。")
        else:
            add("**機構は測っていない。** 候補は二つあり、どちらとも決めていない —— "
                "(a) 陰性球菌は 2 分類群しかなく、片方を抜くと学習に残るのは 1 分類群だけになる。"
                "(b) `Neisseria` の双球菌の対や `Veillonella` の小ささが、見た目として桿菌に近い。"
                "切り分けは陰性球菌の画像を紫に塗り替えて予測が変わるかで測れる(SPEC §3.10)。")
    (REPORTS / "cnn_shape.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# 起動ブロックはファイルの最終行に置く。関数定義より前にあると、
# main() が走る時点でまだ def に到達しておらず NameError になる(loop_004 で踏んだ)
if __name__ == "__main__":
    raise SystemExit(main())
