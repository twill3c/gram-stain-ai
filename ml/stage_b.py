"""Stage B(形: 球菌 / 桿菌)の対照を測る。**CNN より先に測る**(G-12)。

SPEC §3.7 に、画像を分類する前に固定した定義をそのまま実装する。

形の規則(非学習):
  1. 512x512 の切り出しをそのまま使う(128px に縮めると球菌が 3px になり形が消える)
  2. 彩度 (max-min)/max に**画像ごとの大津の閾値**を引く。Stage A の「上位 10%」は
     疎な視野で背景を拾い、密な視野で菌体を取りこぼす(T-265)
  3. 8 近傍の連結成分のうち面積 30〜2,000px だけを残す(ごみと、融けた塊を落とす)
  4. 成分ごとに画素座標の共分散の固有値比 √(λ大/λ小) を細長さとする
  5. 画像の特徴量は細長さの中央値。成分が残らない画像は特徴量なしとして数える
  6. 閾値は train でだけ決め、細長さ > 閾値 を桿菌と予測する

**この規則は色を見ない**(T-263)。彩度は染まっているかどうかにだけ使い、
何色に染まっているかは使わない。Gram と形の偏りで当たる分は、
別の対照「色の規則を形に当てたもの」が測る。

使い方:
    python -m ml.stage_b controls
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from ml.metrics import fit_threshold, macro_f1
from ml.run_controls import (
    JST,
    META,
    N_BOOT,
    PROC,
    REPORTS,
    ROOT,
    canvas_size_baseline,
    colour_features,
    majority_baseline,
    out_of_fold,
    score,
)

COCCUS = 0
BACILLUS = 1
AREA_MIN = 30
AREA_MAX = 2000
LAMBDA_FLOOR = 0.25
EIGHT = np.ones((3, 3), dtype=int)


# ------------------------------------------------------------------ 形の特徴量


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """大津の閾値。クラス間分散が最大になる境目を返す(値 > 閾値 が前景)。"""
    hist, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    p = hist.astype(np.float64)
    total = p.sum()
    if total == 0:
        return 1.0
    p /= total
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    m0 = np.cumsum(p * centers)
    mt = m0[-1]
    w1 = 1.0 - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mt * w0 - m0) ** 2 / (w0 * w1)
    between[~np.isfinite(between)] = -1.0
    k = int(np.argmax(between))
    return float(edges[k + 1])


def shape_feature(rgb: np.ndarray) -> dict:
    """1 枚の視野から、染まった塊の細長さの中央値を取る。

    返り値: {"elongation": float | None, "n_components": int, "stain_threshold": float}
    """
    a = rgb.astype(np.float32) / 255.0
    mx = a.max(2)
    mn = a.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    t = otsu_threshold(sat)
    mask = sat > t

    labels, n = ndimage.label(mask, structure=EIGHT)
    if n == 0:
        return {"elongation": None, "n_components": 0, "stain_threshold": t}

    flat = labels.ravel()
    yy, xx = np.indices(labels.shape)
    y = yy.ravel().astype(np.float64)
    x = xx.ravel().astype(np.float64)
    size = n + 1
    area = np.bincount(flat, minlength=size).astype(np.float64)
    sy = np.bincount(flat, y, size)
    sx = np.bincount(flat, x, size)
    syy = np.bincount(flat, y * y, size)
    sxx = np.bincount(flat, x * x, size)
    sxy = np.bincount(flat, x * y, size)

    keep = (area >= AREA_MIN) & (area <= AREA_MAX)
    keep[0] = False  # 背景
    if not keep.any():
        return {"elongation": None, "n_components": 0, "stain_threshold": t}

    ar = area[keep]
    my, mxm = sy[keep] / ar, sx[keep] / ar
    vyy = syy[keep] / ar - my * my
    vxx = sxx[keep] / ar - mxm * mxm
    vxy = sxy[keep] / ar - my * mxm
    half_tr = (vyy + vxx) / 2
    disc = np.sqrt(np.maximum(half_tr ** 2 - (vyy * vxx - vxy ** 2), 0.0))
    lam_big = half_tr + disc
    lam_small = np.maximum(half_tr - disc, LAMBDA_FLOOR)
    elong = np.sqrt(lam_big / lam_small)
    return {"elongation": float(np.median(elong)), "n_components": int(keep.sum()), "stain_threshold": t}


# ------------------------------------------------------------------ fold


def restrict_folds(folds: list[dict], keep: set[str]) -> list[dict]:
    """Stage A の fold から、Stage B 対象外の画像を抜くだけ。分け直さない(SPEC §3.7)。"""
    out = []
    for f in folds:
        g = dict(f)
        g["train"] = [i for i in f["train"] if i in keep]
        g["test"] = [i for i in f["test"] if i in keep]
        out.append(g)
    return out


# ------------------------------------------------------------------ 対照


def fit_threshold_either(x, y) -> tuple[float, int]:
    """閾値一本を、向きごと train で決める。向き +1 は x > t を 1、-1 は x < t を 1。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    t_up = fit_threshold(x, y)
    t_dn = -fit_threshold(-x, y)
    s_up = macro_f1(y, (x > t_up).astype(int))
    s_dn = macro_f1(y, (x < t_dn).astype(int))
    return (t_up, 1) if s_up >= s_dn else (t_dn, -1)


def _apply(x, t: float, direction: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return ((x > t) if direction == 1 else (x < t)).astype(int)


def shape_rule(train: list[dict], test: list[dict]) -> np.ndarray:
    """形の規則。細長さ > 閾値 を桿菌。特徴量なしは train の多数派を答える。"""
    tr = [r for r in train if r["elongation"] is not None]
    t = fit_threshold(np.array([r["elongation"] for r in tr]), np.array([r["y"] for r in tr]))
    fallback = majority_baseline([r["y"] for r in train], 1)[0][0]
    return np.array([
        int(r["elongation"] > t) if r["elongation"] is not None else int(fallback) for r in test
    ])


def hue_on_shape(train: list[dict], test: list[dict]) -> np.ndarray:
    """Stage A の R/B 比に、形のラベルで閾値一本(向きも train で決める)。"""
    t, d = fit_threshold_either([r["rb_ratio"] for r in train], [r["y"] for r in train])
    return _apply([r["rb_ratio"] for r in test], t, d)


def _per_taxon_errors(preds: dict[str, int], index: dict) -> dict:
    acc: dict[str, dict] = {}
    for image_id, p in preds.items():
        rec = index[image_id]
        d = acc.setdefault(rec["folder"], {"n": 0, "wrong": 0})
        d["n"] += 1
        d["wrong"] += int(p != rec["y"])
    return {
        k: {"n": v["n"], "wrong": v["wrong"], "error_rate": round(v["wrong"] / v["n"], 4)}
        for k, v in sorted(acc.items(), key=lambda kv: -kv[1]["wrong"])
        if v["wrong"] > 0
    }


def load_features(rows: list[dict]) -> dict[str, dict]:
    """形の特徴量を測る。重いので画像ごとにキャッシュする。"""
    cache = META / "shape_features.json"
    feats: dict[str, dict] = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    todo = [r for r in rows if r["image_id"] not in feats]
    for k, r in enumerate(todo, 1):
        with Image.open(PROC / r["path"]) as im:
            feats[r["image_id"]] = shape_feature(np.asarray(im.convert("RGB")))
        if k % 200 == 0 or k == len(todo):
            cache.write_text(json.dumps(feats, ensure_ascii=False), encoding="utf-8")
            print(f"  形の特徴量 {k}/{len(todo)}", flush=True)
    return feats


def controls() -> int:
    rows_all = [json.loads(line) for line in (META / "prepared.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows_all if r["stage_b"]]
    splits = json.loads((META / "splits.json").read_text(encoding="utf-8"))
    keep = {r["image_id"] for r in rows}

    shape_feats = load_features(rows)
    colour_cache = META / "colour_features.json"
    colour = json.loads(colour_cache.read_text(encoding="utf-8")) if colour_cache.exists() else {}
    for r in rows:
        if r["image_id"] not in colour:
            colour[r["image_id"]] = colour_features(PROC / r["path"])

    index = {}
    for r in rows:
        index[r["image_id"]] = {
            "image_id": r["image_id"],
            "y": BACILLUS if r["shape"] == "bacillus" else COCCUS,
            "group_id": r["group_id"],
            "folder": r["folder"],
            "family": r["family"],
            "width": r["source_width"],
            "rb_ratio": colour[r["image_id"]]["rb_ratio"],
            "elongation": shape_feats[r["image_id"]]["elongation"],
            "n_components": shape_feats[r["image_id"]]["n_components"],
        }
    recs = list(index.values())
    n_missing = sum(1 for v in recs if v["elongation"] is None)

    # 同じデータで閾値を選んだときの上限。train で決めた成績はこれを超えてはならない(T-269)
    y_all = np.array([v["y"] for v in recs])
    has = [v for v in recs if v["elongation"] is not None]
    t_all = fit_threshold(np.array([v["elongation"] for v in has]), np.array([v["y"] for v in has]))
    maj_all = majority_baseline(y_all, 1)[0][0]
    oracle_shape = macro_f1(y_all, np.array([
        int(v["elongation"] > t_all) if v["elongation"] is not None else int(maj_all) for v in recs]))
    t_h, d_h = fit_threshold_either([v["rb_ratio"] for v in recs], y_all)
    oracle_hue = macro_f1(y_all, _apply([v["rb_ratio"] for v in recs], t_h, d_h))

    baselines: dict[str, dict] = {
        "shape": {"feature": "median_elongation", "threshold_fitted_on": "train",
                  "oracle_upper_bound_macro_f1": round(oracle_shape, 4),
                  "oracle_threshold": round(t_all, 4),
                  "n_missing_feature": n_missing,
                  "description": "染まった塊の細長さの中央値に閾値一本。色を見ない。学習しない"},
        "hue_on_shape": {"feature": "rb_ratio", "threshold_fitted_on": "train",
                         "oracle_upper_bound_macro_f1": round(oracle_hue, 4),
                         "oracle_direction": d_h,
                         "description": "Stage A の R/B 比に、形のラベルで閾値一本(向きも train で決める)"},
        "canvas_size": {"feature": "source_width", "threshold_fitted_on": "train",
                        "description": "画素を一切見ず、キャンバス寸法だけで当てる"},
        "majority": {"feature": "none", "threshold_fitted_on": "train",
                     "description": "train で多いほう(桿菌)を全部に答える"},
    }

    rulers = {"a": "ruler_a_cv", "b": "ruler_b", "c": "ruler_c"}
    for ruler, key in rulers.items():
        folds = restrict_folds(splits[key]["folds"], keep)
        preds = {
            "shape": out_of_fold(folds, index, shape_rule),
            "hue_on_shape": out_of_fold(folds, index, hue_on_shape),
            "canvas_size": out_of_fold(folds, index, lambda tr, te: canvas_size_baseline(
                [r["width"] for r in tr], [r["y"] for r in tr], [r["width"] for r in te])[0]),
            "majority": out_of_fold(folds, index, lambda tr, te: majority_baseline(
                [r["y"] for r in tr], len(te))[0]),
        }
        for name, p in preds.items():
            baselines[name][ruler] = score(p, index)
            if name in ("shape", "hue_on_shape"):
                baselines[name][ruler]["per_taxon_errors"] = _per_taxon_errors(p, index)
        print(f"  物差し {ruler.upper()}: " + " / ".join(
            f"{name} {baselines[name][ruler]['macro_f1']:.4f}" for name in preds), flush=True)

    doc = {
        "generated_at": datetime.now(JST).isoformat(timespec="seconds"),
        "target": "shape",
        "classes": {"0": "coccus", "1": "bacillus"},
        "n_images_total": len(rows),
        "n_taxa": len({r["folder"] for r in rows}),
        "n_boot": N_BOOT,
        "shape_rule": {"area_min": AREA_MIN, "area_max": AREA_MAX, "lambda_floor": LAMBDA_FLOOR,
                       "stain_mask": "彩度に画像ごとの大津の閾値", "connectivity": 8,
                       "input": "512x512 の切り出し(縮小しない)"},
        "baselines": baselines,
        "note": "対照は Stage B の CNN より先に測っている(G-12)。定義は SPEC §3.7 に測る前に固定した",
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "controls_shape.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  特徴量なし: {n_missing} / {len(rows)} 枚")
    print(f"→ {(REPORTS / 'controls_shape.json').relative_to(ROOT)}")
    return 0


# ------------------------------------------------------------------ Gram × 形の集計(事後の分析)


def tally_cells(preds: dict[str, int], rows: dict[str, dict]) -> dict[str, dict]:
    """予測を Gram × 形の 4 セルに分けて誤りを数える(T-272)。

    **合否ではない。** SPEC §3.8 の判定は測る前に決めた二条件だけで出す。
    これは判定の後で「どこで外したか」を読むための集計である。
    """
    acc: dict[str, dict] = {}
    for image_id, p in preds.items():
        r = rows[image_id]
        y = BACILLUS if r["shape"] == "bacillus" else COCCUS
        d = acc.setdefault(f"{r['gram']} x {r['shape']}", {"n": 0, "wrong": 0, "taxa": set()})
        d["n"] += 1
        d["wrong"] += int(p != y)
        d["taxa"].add(r["folder"])
    return {
        k: {"n": v["n"], "wrong": v["wrong"], "error_rate": round(v["wrong"] / v["n"], 4), "n_taxa": len(v["taxa"])}
        for k, v in sorted(acc.items())
    }


def cells() -> int:
    """三物差しの CNN と形の規則を、Gram × 形の 4 セルで並べる。"""
    rows = {r["image_id"]: r for r in (json.loads(line) for line in
            (META / "prepared.jsonl").read_text(encoding="utf-8").splitlines()) if r["stage_b"]}
    cnn = json.loads((REPORTS / "cnn_shape.json").read_text(encoding="utf-8"))
    ctrl = json.loads((REPORTS / "controls_shape.json").read_text(encoding="utf-8"))["baselines"]["shape"]
    epochs = cnn["epoch_budget"]["chosen"]
    fold_dir = ROOT / "ml" / "checkpoints" / "fold_preds"
    by_taxon = {r["folder"]: f"{r['gram']} x {r['shape']}" for r in rows.values()}

    out: dict[str, dict] = {}
    for ruler in ("a", "b", "c"):
        preds: dict[str, int] = {}
        for f in sorted(fold_dir.glob(f"S{ruler.upper()}_{epochs}ep_*.json")):
            preds.update(json.loads(f.read_text(encoding="utf-8")))
        if len(preds) != len(rows):
            raise SystemExit(f"物差し {ruler.upper()} の予測が {len(preds)} 枚で、Stage B の {len(rows)} 枚に足りない")
        cnn_cells = tally_cells(preds, rows)
        rule_cells = {k: {"wrong": 0} for k in cnn_cells}
        for taxon, v in ctrl[ruler]["per_taxon_errors"].items():
            rule_cells[by_taxon[taxon]]["wrong"] += v["wrong"]
        for k in rule_cells:
            rule_cells[k]["error_rate"] = round(rule_cells[k]["wrong"] / cnn_cells[k]["n"], 4)
        out[ruler] = {"cnn": cnn_cells, "shape_rule": rule_cells}

    doc = {
        "generated_at": datetime.now(JST).isoformat(timespec="seconds"),
        "note": "事後の分析。SPEC §3.8 の判定には入れない。fold の予測(ml/checkpoints/fold_preds)から数えた",
        "epochs": epochs,
        "rulers": out,
    }
    (REPORTS / "cnn_shape_cells.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    for ruler in ("a", "b", "c"):
        print(f"  物差し {ruler.upper()}: " + " / ".join(
            f"{k} CNN {v['error_rate']:.0%}・規則 {out[ruler]['shape_rule'][k]['error_rate']:.0%}"
            for k, v in out[ruler]["cnn"].items()))
    print(f"→ {(REPORTS / 'cnn_shape_cells.json').relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("controls", "cells"))
    args = ap.parse_args()
    if args.command == "controls":
        return controls()
    if args.command == "cells":
        return cells()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
