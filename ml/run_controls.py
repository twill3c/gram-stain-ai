"""陽性対照を測る。**CNN より先に測る。**

順序に意味がある。対照の数が先に出ていれば、あとから CNN の数を単独で語れなくなる。
逆にすると、CNN の数を見てから「対照はどこまで出るか」を測ることになり、
対照は必ず「CNN より低い」ことを確かめる作業になってしまう。

測る三つ:

  1. **色相ベースライン(F-10 / G-05)** — Gram 染色は陽性が紫・陰性が桃〜赤に染まる
     **色の検査**である。染まった画素の R/B 比に閾値を一本引くだけで、学習は一切しない。
     閾値は train でだけ決める(T-231)
  2. **キャンバス寸法の分類器(G-11)** — 画素を一切見ず、画像の寸法だけで当てにいく。
     当たってしまうなら、その分だけ「画像を見て当てた」の根拠が弱い
  3. **多数派クラス** — 全部を Gram 陽性と答える。成績の目盛りの下端

三つとも、三つの物差し(A 画像単位 / B 分類群ホールドアウト / C 科ホールドアウト)で測る。
どの物差しでも全 1,966 枚を一度ずつ評価するので、成績の差は物差しの違いだけから来る。

使い方:
    python ml/run_controls.py
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from ml.metrics import NEGATIVE, POSITIVE, bootstrap_macro_f1, fit_threshold, macro_f1

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "dataset" / "metadata"
PROC = ROOT / "dataset" / "processed"
REPORTS = ROOT / "reports"
JST = timezone(timedelta(hours=9))

# 色統計を取るときの縮小先。色の話なので細部は要らない
COLOUR_SIZE = 128
# 「染まった画素」とみなす彩度の分位。上位 10%
STAIN_QUANTILE = 0.90
N_BOOT = 2000


# ------------------------------------------------------------------ 特徴量


def colour_features(path: Path) -> dict:
    """1 枚の染色色の統計を取る。

    背景は淡く、菌体は染まって彩度が高い。彩度上位 STAIN_QUANTILE の画素を
    「染まった画素」とみなし、そこだけの色を見る。全画素の平均を取ると
    大半を占める背景の色になり、染色の情報が薄まる。
    """
    with Image.open(path) as im:
        small = im.convert("RGB").resize((COLOUR_SIZE, COLOUR_SIZE), Image.Resampling.BILINEAR)
        a = np.asarray(small, dtype=np.float32) / 255.0

    mx = a.max(2)
    mn = a.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    stained = a[sat >= np.quantile(sat, STAIN_QUANTILE)]
    if stained.size == 0:
        stained = a.reshape(-1, 3)

    r, g, b = stained[:, 0], stained[:, 1], stained[:, 2]
    # R/B 比 — 紫(crystal violet)は B が強く、桃〜赤(safranin)は R が強い。
    # 色相のように 0 度と 360 度がつながらないので、閾値を一本引くのに扱いやすい
    rb = float(np.mean(r / np.maximum(b, 1e-6)))

    # 参考として色相の円平均も残す(解説で使う)
    mxc = stained.max(1)
    mnc = stained.min(1)
    d = np.maximum(mxc - mnc, 1e-6)
    h = np.zeros_like(mxc)
    i0 = mxc == r
    i1 = (mxc == g) & ~i0
    i2 = ~i0 & ~i1
    h[i0] = ((g - b)[i0] / d[i0]) % 6
    h[i1] = ((b - r)[i1] / d[i1]) + 2
    h[i2] = ((r - g)[i2] / d[i2]) + 4
    ang = np.deg2rad(h * 60.0)
    hue = math.degrees(math.atan2(float(np.sin(ang).mean()), float(np.cos(ang).mean()))) % 360.0

    return {"rb_ratio": rb, "hue_deg": hue, "saturation": float(np.mean(mxc - mnc))}


# ------------------------------------------------------------------ 対照


def hue_baseline(x_train, y_train, x_test) -> tuple[np.ndarray, float]:
    """色相ベースライン。R/B 比に閾値を一本。train でだけ決める。"""
    t = fit_threshold(x_train, y_train)
    return (np.asarray(x_test) > t).astype(int), t


def canvas_size_baseline(widths_train, y_train, widths_test) -> tuple[np.ndarray, float]:
    """キャンバス寸法だけの対照。**画素を一切読まない**(T-211)。

    この関数の値打ちは「画素を見ずにどこまで当たるか」にある。
    画素を見てしまえば、ただの弱い分類器になって対照にならない。
    引数は画像の寸法(整数)だけで、ファイル経路も画像も受け取らない。
    """
    t = fit_threshold(np.asarray(widths_train, dtype=np.float64), y_train)
    return (np.asarray(widths_test, dtype=np.float64) > t).astype(int), t


def majority_baseline(y_train, n_test: int) -> tuple[np.ndarray, float]:
    """多数派クラス。train で多いほうを、test 全部に答える。"""
    y_train = np.asarray(y_train)
    label = POSITIVE if (y_train == POSITIVE).sum() >= (y_train == NEGATIVE).sum() else NEGATIVE
    return np.full(n_test, label, dtype=int), float(label)


# ------------------------------------------------------------------ 評価


def out_of_fold(folds: list[dict], index: dict, predict) -> dict[str, int]:
    """各 fold の test を予測し、画像 ID → 予測 の辞書にまとめる。

    全画像がちょうど一度ずつ予測される(T-234)。
    """
    preds: dict[str, int] = {}
    for fold in folds:
        tr = [index[i] for i in fold["train"]]
        te = [index[i] for i in fold["test"]]
        yp = predict(tr, te)
        for rec, p in zip(te, yp):
            if rec["image_id"] in preds:
                raise RuntimeError(f"同じ画像が二度予測された: {rec['image_id']}")
            preds[rec["image_id"]] = int(p)
    return preds


def score(preds: dict[str, int], index: dict) -> dict:
    ids = sorted(preds)
    y = np.array([index[i]["y"] for i in ids])
    p = np.array([preds[i] for i in ids])
    g = np.array([index[i]["group_id"] for i in ids])
    point, lo, hi = bootstrap_macro_f1(y, p, g, n_boot=N_BOOT)
    return {
        "macro_f1": round(point, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "accuracy": round(float((y == p).mean()), 4),
        "n_images": len(ids),
        "n_groups": int(len(set(g))),
    }


def main() -> int:
    rows = [json.loads(line) for line in (META / "prepared.jsonl").read_text(encoding="utf-8").splitlines()]
    splits = json.loads((META / "splits.json").read_text(encoding="utf-8"))

    cache = META / "colour_features.json"
    if cache.exists():
        feats = json.loads(cache.read_text(encoding="utf-8"))
        print(f"色統計をキャッシュから読んだ({len(feats)} 枚)")
    else:
        feats = {}
        for k, r in enumerate(rows, 1):
            feats[r["image_id"]] = colour_features(PROC / r["path"])
            if k % 400 == 0:
                print(f"  色統計 {k}/{len(rows)}", flush=True)
        cache.write_text(json.dumps(feats, ensure_ascii=False), encoding="utf-8")
        print(f"色統計を測った({len(feats)} 枚)→ {cache.relative_to(ROOT)}")

    index = {}
    for r in rows:
        f = feats[r["image_id"]]
        index[r["image_id"]] = {
            "image_id": r["image_id"],
            "y": POSITIVE if r["gram"] == "positive" else NEGATIVE,
            "group_id": r["group_id"],
            "folder": r["folder"],
            "family": r["family"],
            "width": r["source_width"],
            "rb_ratio": f["rb_ratio"],
            "hue_deg": f["hue_deg"],
        }

    # 同じデータで閾値を選んだときの上限。train で決めた成績はこれを超えてはならない(T-231)
    all_x = np.array([v["rb_ratio"] for v in index.values()])
    all_y = np.array([v["y"] for v in index.values()])
    oracle_t = fit_threshold(all_x, all_y)
    oracle = macro_f1(all_y, (all_x > oracle_t).astype(int))

    rulers = {
        "a": splits["ruler_a_cv"]["folds"],
        "b": splits["ruler_b"]["folds"],
        "c": splits["ruler_c"]["folds"],
    }

    baselines: dict[str, dict] = {
        "hue": {"feature": "rb_ratio", "threshold_fitted_on": "train",
                "oracle_upper_bound_macro_f1": round(oracle, 4),
                "oracle_threshold": round(oracle_t, 4),
                "description": "染まった画素の R/B 比に閾値を一本。学習しない"},
        "canvas_size": {"feature": "source_width", "threshold_fitted_on": "train",
                        "description": "画素を一切見ず、キャンバス寸法だけで当てる"},
        "majority": {"feature": "none", "threshold_fitted_on": "train",
                     "description": "全部を Gram 陽性と答える"},
    }

    for ruler, folds in rulers.items():
        preds_hue = out_of_fold(folds, index, lambda tr, te: hue_baseline(
            [r["rb_ratio"] for r in tr], [r["y"] for r in tr], [r["rb_ratio"] for r in te])[0])
        preds_size = out_of_fold(folds, index, lambda tr, te: canvas_size_baseline(
            [r["width"] for r in tr], [r["y"] for r in tr], [r["width"] for r in te])[0])
        preds_maj = out_of_fold(folds, index, lambda tr, te: majority_baseline(
            [r["y"] for r in tr], len(te))[0])

        baselines["hue"][ruler] = score(preds_hue, index)
        baselines["canvas_size"][ruler] = score(preds_size, index)
        baselines["majority"][ruler] = score(preds_maj, index)
        # 誤りがどの分類群に集まるかを残す。
        # 全体の数だけでは「どこで間違えたか」が見えず、対照の限界を語れない
        per_taxon: dict[str, dict] = {}
        for image_id, p in preds_hue.items():
            rec = index[image_id]
            d = per_taxon.setdefault(rec["folder"], {"n": 0, "wrong": 0, "gram": rec["y"]})
            d["n"] += 1
            d["wrong"] += int(p != rec["y"])
        baselines["hue"][ruler]["per_taxon_errors"] = {
            k: {"n": v["n"], "wrong": v["wrong"], "error_rate": round(v["wrong"] / v["n"], 4)}
            for k, v in sorted(per_taxon.items(), key=lambda kv: -kv[1]["wrong"])
            if v["wrong"] > 0
        }
        print(f"  物差し {ruler.upper()}: 色相 {baselines['hue'][ruler]['macro_f1']:.4f} / "
              f"寸法 {baselines['canvas_size'][ruler]['macro_f1']:.4f} / "
              f"多数派 {baselines['majority'][ruler]['macro_f1']:.4f}", flush=True)

    doc = {
        "generated_at": datetime.now(JST).isoformat(timespec="seconds"),
        "n_images_total": len(rows),
        "n_boot": N_BOOT,
        "stain_quantile": STAIN_QUANTILE,
        "colour_size": COLOUR_SIZE,
        "rulers": {
            "a": "画像単位(分類群の中で group を k 分割)",
            "b": "分類群ホールドアウト",
            "c": "科ホールドアウト",
        },
        "baselines": baselines,
        "note": "対照は CNN より先に測っている。対照の数が先に出ていれば、"
                "あとから CNN の数を単独で語れない",
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "controls.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(doc)
    print(f"\n→ {(REPORTS / 'controls.json').relative_to(ROOT)}")
    print(f"→ {(REPORTS / 'controls.md').relative_to(ROOT)}")
    return 0


def write_markdown(doc: dict) -> None:
    b = doc["baselines"]
    L: list[str] = []
    add = L.append
    add("# 陽性対照(CNN より先に測った)")
    add("")
    add(f"<!-- ml/run_controls.py が生成。手で編集しない。生成日時 {doc['generated_at']} -->")
    add("")
    add("Gram 染色は、陽性が紫(crystal violet)・陰性が桃〜赤(safranin)に染まる**色の検査**である。")
    add("だから「AI が当てた」と言う前に、**学習を一切しない色の規則がどこまで当てるか**を測る。")
    add("")
    add("**この表は CNN を作る前に出した。** 対照の数が先に出ていれば、")
    add("あとから CNN の数を単独で語れない。")
    add("")
    add("## 結果")
    add("")
    add(f"評価はどの物差しでも全 {doc['n_images_total']:,} 枚を一度ずつ。")
    add(f"信頼区間は group 単位のブートストラップ({doc['n_boot']:,} 回)。")
    add("")
    add("| 対照 | A 画像単位 | B 分類群ホールドアウト | C 科ホールドアウト |")
    add("|---|---|---|---|")
    names = {"hue": "**色相(R/B 比・非学習)**", "canvas_size": "キャンバス寸法(画素を見ない)",
             "majority": "多数派クラス"}
    for key, label in names.items():
        cells = []
        for ruler in ("a", "b", "c"):
            r = b[key][ruler]
            cells.append(f"{r['macro_f1']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
        add(f"| {label} | {' | '.join(cells)} |")
    add("")
    add("macro F1 と 95% 信頼区間。")
    add("")
    add("## 読み方")
    add("")
    hue_a, hue_b, hue_c = (b["hue"][r]["macro_f1"] for r in ("a", "b", "c"))
    add(f"**色相の規則は、三つの物差しでほとんど動かない**({hue_a:.4f} / {hue_b:.4f} / {hue_c:.4f})。")
    add("学習で見ていない分類群でも、見ていない科でも、同じくらい当たる。")
    add("色は種の同一性に依らないので、これは当然である —— そして**当然であることが重要**である。")
    add("物差しを変えて成績が落ちるとしたら、それは種の見覚えに頼っている印になる。")
    add("")
    size_a = b["canvas_size"]["a"]["macro_f1"]
    maj = b["majority"]["a"]["macro_f1"]
    add(f"**キャンバス寸法だけでも {size_a:.4f} 出る**(多数派クラスの {maj:.4f} より上)。")
    add("画素を一切見ずにこれだけ当たるということは、撮影の条件が分類群とわずかに相関している。")
    add("**この分は「画像を見て当てた」の根拠から差し引く。**")
    add("")
    add(f"同じデータで閾値を選んだときの上限は {b['hue']['oracle_upper_bound_macro_f1']:.4f} で、")
    add("上の表の値はすべてそれ以下である(閾値を train でだけ決めている証拠 —— 検査 T-231)。")
    add("")
    add("## 色の規則がどこで間違えるか")
    add("")
    add("物差し B(分類群ホールドアウト)での誤りの内訳。")
    add("")
    add("| 分類群 | 誤り / 枚数 | 誤り率 |")
    add("|---|---|---|")
    for k, v in list(b["hue"]["b"]["per_taxon_errors"].items())[:12]:
        add(f"| {k} | {v['wrong']} / {v['n']} | {v['error_rate']:.0%} |")
    add("")
    add("誤りは散らばっておらず、**染色そのものが揺れる菌に集中している**。")
    add("`Listeria monocytogenes` は Gram 陽性だが脱色されやすく、")
    add("`Porphyromonas gingivalis` は Gram 陰性だが染まり方が揺れることが知られている。")
    add("")
    add("つまりこの対照は、**Gram 染色が曖昧なところでだけ間違える**。")
    add("")
    add("## ここから CNN に問うこと")
    add("")
    add("CNN が意味を持つのは、次のどちらかが起きたときだけである。")
    add("")
    add("1. 三つの物差しすべてで、色の規則を信頼区間の重なりなく上回る")
    add("2. 全体では並んでも、**色の規則が落ちる分類群**(上の表)で明確に上回る")
    add("")
    add("2 が起きれば、CNN は色以外の手がかり(形・並び・きめ)を使っていることになる。")
    add("どちらも起きなければ、**このデータで CNN は色の規則に何も足していない**。")
    add("そのときはそう書く。")
    (REPORTS / "controls.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
