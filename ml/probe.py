"""色相以外の手がかりを絞る(SPEC §3.14・loop_014)。

§3.13 で色相は否定された。残る候補を**一つずつ変えて**測る。
学習し直しは要らない —— §3.12 で保存した 4 つのモデル(`ml/checkpoints/mechanism/*.pth`)に推論させるだけ。

  - 拡大: 中心 512/s を切り出して 512 へ。菌体の見かけの大きさだけが変わる(細長さは不変・面積は s² 倍)
  - 彩度: 染まった画素の彩度の中央値を、Stage B の Gram 陽性の中央値へ合わせる(**色相と明度は変えない**)

拡大は視野を狭めるので、**同じ操作を正しく答えられている群にも当てて**副作用を測る(SPEC §3.14)。

使い方:
    python -m ml.probe measure
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from ml.mechanism import (
    CKPT_DIR,
    IDS_DIR,
    MODELS,
    bootstrap_drop,
    rgb_to_hsv,
    to_input,
)
from ml.stage_b import BACILLUS, COCCUS, otsu_threshold

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "dataset" / "metadata"
PROC = ROOT / "dataset" / "processed"
REPORTS = ROOT / "reports"
ZOOMS = (1.0, 1.25, 1.5, 2.0, 3.0)


# ------------------------------------------------------------------ 操作


def zoom(img: np.ndarray, s: float) -> np.ndarray:
    """中心 (辺/s) を切り出して元の辺へ拡大する。細長さは変わらず、面積が s² 倍になる。"""
    h, w = img.shape[:2]
    side = int(round(min(h, w) / s))
    top, left = (h - side) // 2, (w - side) // 2
    crop = img[top : top + side, left : left + side]
    if crop.shape[0] == h and crop.shape[1] == w:
        return crop.copy()
    return np.asarray(Image.fromarray(crop).resize((w, h), Image.Resampling.BILINEAR))


def stained_mask_saturation(hsv: np.ndarray) -> np.ndarray:
    """染まった画素(彩度に画像ごとの大津の閾値)。形の規則と同じ決め方。"""
    return hsv[..., 1] > otsu_threshold(hsv[..., 1])


def median_stained_saturation(img: np.ndarray) -> float:
    hsv = rgb_to_hsv(img.astype(np.float64) / 255.0)
    m = stained_mask_saturation(hsv)
    return float(np.median(hsv[..., 1][m])) if m.any() else 0.0


def scale_saturation(img: np.ndarray, target_median: float) -> np.ndarray:
    """染まった画素の彩度の中央値が target_median になるよう、彩度を一様に掛ける。

    **色相と明度は変えない。** RGB の上で `v - (v - x) * f` と書けば、
    最大値(明度)と、最大最小の間の比(色相)はそのまま残る。
    """
    a = img.astype(np.float64) / 255.0
    hsv = rgb_to_hsv(a)
    m = stained_mask_saturation(hsv)
    med = float(np.median(hsv[..., 1][m])) if m.any() else 0.0
    if med <= 1e-9:
        return img.copy()
    f = target_median / med
    v = a.max(-1, keepdims=True)
    out = v - (v - a) * f
    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ 測る


def measure() -> int:
    import torch

    from ml.train import build_model, predict

    rows = [json.loads(line) for line in (META / "prepared.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {r["image_id"]: r for r in rows}
    cache: dict[str, np.ndarray] = {}

    def image(i: str) -> np.ndarray:
        if i not in cache:
            with Image.open(PROC / by_id[i]["path"]) as im:
                cache[i] = np.asarray(im.convert("RGB"))
        return cache[i]

    # 彩度の目標: Stage B の Gram 陽性画像の「染まった画素の彩度の中央値」の中央値
    pos = [r["image_id"] for r in rows if r["stage_b"] and r["gram"] == "positive"]
    target_sat = float(np.median([median_stained_saturation(image(i)) for i in pos]))
    print(f"  彩度の目標(陽性の中央値): {target_sat:.4f}", flush=True)

    def y_of(ids):
        return np.array([BACILLUS if by_id[i]["shape"] == "bacillus" else COCCUS for i in ids])

    def err(model, ids, transform) -> tuple[float, np.ndarray]:
        if not ids:
            return float("nan"), np.zeros(0, dtype=int)
        arr = np.stack([to_input(transform(image(i))) for i in ids])
        w = (predict(model, arr, np.arange(len(ids))) != y_of(ids)).astype(int)
        return float(w.mean()), w

    variants = {f"zoom_{s}": (lambda a, s=s: zoom(a, s)) for s in ZOOMS}
    variants["saturation"] = lambda a: scale_saturation(a, target_sat)

    per_model = {}
    for m in MODELS:
        model = build_model()
        model.load_state_dict(torch.load(CKPT_DIR / f"{m['name']}.pth", map_location="cpu"))
        model.eval()
        test = json.loads((IDS_DIR / f"{m['name']}_ids.json").read_text(encoding="utf-8"))["test"]
        held = [i for i in test if by_id[i]["folder"] in m["held_out_negative_cocci"]]
        ctrl_pos = [i for i in test if by_id[i]["gram"] == "positive" and by_id[i]["shape"] == "coccus"]
        ctrl_neg = [i for i in test if by_id[i]["gram"] == "negative" and by_id[i]["shape"] == "bacillus"]

        entry = {"held": held, "n_positive_cocci": len(ctrl_pos), "n_negative_bacilli": len(ctrl_neg),
                 "original": {}, "variants": {}}
        e0, w0 = err(model, held, lambda a: a)
        p0 = err(model, ctrl_pos, lambda a: a)[0]
        n0 = err(model, ctrl_neg, lambda a: a)[0]
        entry["original"] = {"error": e0, "wrong": w0.tolist(), "positive_cocci_error": p0, "negative_bacilli_error": n0}
        for name, fn in variants.items():
            e, w = err(model, held, fn)
            entry["variants"][name] = {
                "error": e, "wrong": w.tolist(),
                "positive_cocci_error": err(model, ctrl_pos, fn)[0],
                "negative_bacilli_error": err(model, ctrl_neg, fn)[0],
            }
        per_model[m["name"]] = entry
        print(f"  {m['name']}: 元 {e0:.1%} / " + " / ".join(
            f"{k} {v['error']:.1%}" for k, v in entry["variants"].items()), flush=True)

    def inc(a: float, b: float) -> float:
        """対照の悪化幅。対照が居ない(nan)ときは 0 とみなし、報告に n を残す。"""
        if np.isnan(a) or np.isnan(b):
            return 0.0
        return float(b - a)

    checks = {"sham_zoom_error_change": {n: per_model[n]["variants"]["zoom_1.0"]["error"] - per_model[n]["original"]["error"]
                                         for n in per_model},
              "saturation_target_median": target_sat}
    sham_ok = all(abs(v) < 0.05 for v in checks["sham_zoom_error_change"].values())

    primary = {}
    for ruler, names in (("b", ("B0", "B2")), ("c", ("C0", "C3"))):
        ids = sum((per_model[n]["held"] for n in names), [])
        groups = [by_id[i]["group_id"] for i in ids]
        w_base = np.concatenate([np.array(per_model[n]["original"]["wrong"]) for n in names])
        base = float(w_base.mean())

        def pack(key: str) -> dict:
            w = np.concatenate([np.array(per_model[n]["variants"][key]["wrong"]) for n in names])
            pc = [(per_model[n]["original"]["positive_cocci_error"],
                   per_model[n]["variants"][key]["positive_cocci_error"], per_model[n]["n_positive_cocci"]) for n in names]
            nb = [(per_model[n]["original"]["negative_bacilli_error"],
                   per_model[n]["variants"][key]["negative_bacilli_error"], per_model[n]["n_negative_bacilli"]) for n in names]
            def pooled(items):
                tot = sum(n for _, _, n in items)
                if tot == 0:
                    return 0.0, 0
                a = sum(o * n for o, _, n in items if n) / tot
                b = sum(v * n for _, v, n in items if n) / tot
                return inc(a, b), tot
            pc_inc, pc_n = pooled(pc)
            nb_inc, nb_n = pooled(nb)
            return {"error": float(w.mean()), "wrong": w,
                    "side_effect": {"positive_cocci_error_increase": pc_inc, "positive_cocci_n": pc_n,
                                    "negative_bacilli_error_increase": nb_inc, "negative_bacilli_n": nb_n}}

        zooms = {f"{s}": pack(f"zoom_{s}") for s in ZOOMS}
        sat = pack("saturation")
        best_zoom_key = min(zooms, key=lambda k: zooms[k]["error"])
        best_zoom = zooms[best_zoom_key]

        def qualifies(cand: dict) -> bool:
            side = cand["side_effect"]
            return (base - cand["error"] >= 0.50 and cand["error"] <= 0.40
                    and side["positive_cocci_error_increase"] < 0.20
                    and side["negative_bacilli_error_increase"] < 0.20)

        if not sham_ok:
            verdict = "判定しない"
        elif qualifies(best_zoom):
            verdict = "大きさが主因"
        elif qualifies(sat):
            verdict = "濃さが主因"
        elif (base - best_zoom["error"]) < 0.20 and (base - sat["error"]) < 0.20:
            verdict = "どちらでもない"
        else:
            verdict = "混合"

        best = best_zoom if best_zoom["error"] <= sat["error"] else sat
        point, lo, hi = bootstrap_drop(w_base, best["wrong"], groups)
        primary[ruler] = {
            "models": list(names), "n": len(ids), "error_original": base,
            "zoom": {k: {kk: vv for kk, vv in v.items() if kk != "wrong"} for k, v in zooms.items()},
            "saturation": {k: v for k, v in sat.items() if k != "wrong"},
            "best_variant": best_zoom_key if best is best_zoom else "saturation",
            "drop": {"point": point, "ci_low": lo, "ci_high": hi},
            "verdict": verdict,
        }
        print(f"  物差し {ruler.upper()}: 元 {base:.1%} / 拡大の最良 {best_zoom['error']:.1%}(s={best_zoom_key})"
              f" / 彩度 {sat['error']:.1%} → {verdict}", flush=True)

    # 観察: 形の規則が測る細長さと誤りの対応(判定には使わない)
    from ml.stage_b import shape_feature

    obs = {}
    for ruler, names in (("b", ("B0", "B2")), ("c", ("C0", "C3"))):
        ids = sum((per_model[n]["held"] for n in names), [])
        w = np.concatenate([np.array(per_model[n]["original"]["wrong"]) for n in names])
        el = np.array([shape_feature(image(i))["elongation"] or np.nan for i in ids])
        ok = ~np.isnan(el)
        obs[ruler] = {"n": int(ok.sum()),
                      "elongation_mean_wrong": float(el[ok & (w == 1)].mean()) if (ok & (w == 1)).any() else None,
                      "elongation_mean_right": float(el[ok & (w == 0)].mean()) if (ok & (w == 0)).any() else None,
                      "note": "形の規則が測る細長さ。判定には使わない"}

    doc = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rule": "SPEC §3.14(測る前に書いた)",
        "zooms": list(ZOOMS),
        "checks": checks,
        "checks_passed": sham_ok,
        "primary": primary,
        "observation_elongation": obs,
        "per_model": {n: {k: v for k, v in e.items() if k not in ("held",)} for n, e in per_model.items()},
    }
    for e in doc["per_model"].values():
        e["original"].pop("wrong", None)
        for v in e["variants"].values():
            v.pop("wrong", None)
    (REPORTS / "probe_negative_cocci.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {(REPORTS / 'probe_negative_cocci.json').relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("measure",))
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    return measure()


if __name__ == "__main__":
    raise SystemExit(main())
