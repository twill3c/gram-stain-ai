"""見ていない陰性球菌を外すとき、モデルはどこを見ているか(SPEC §3.20・loop_017)。

§3.13〜§3.19 は画像を作り替える実験で、手がかりを動かしたのか、見たことのない画像でモデルを揺さぶっただけなのかを
分けにくかった。ここでは**画像を作り替えずに**、Grad-CAM で判断が画素のどこに寄っているかを観察する。

  - モデル   : §3.12 で保存した 4 モデル(陰性球菌の分類群を学習で見ていない)
  - 画像     : 試験側の陰性球菌(全件)と、正しく球菌と答えた陽性球菌
  - 対象     : 予測クラス。重みを乱数化したモデルにも同じクラスを描かせる(Adebayo et al. 2018)
  - 層       : features[12](7x7・主)/ features[8](14x14)/ features[3](28x28)
  - 量       : 集中度比。染まった画素は 224px で彩度に画像ごとの大津の閾値(loop_008 の上位 10% も並べる)

判定規則は SPEC §3.20 に**測る前に**書いた。ここは当てはめるだけ。

使い方:
    python -m ml.gradcam_neg measure
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr

from ml.gradcam import concentration, grad_cam, group_bootstrap_mean, randomized_copy, stain_mask
from ml.mechanism import CKPT_DIR, IDS_DIR, MODELS, to_input
from ml.separate import stained_mask

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "dataset" / "metadata"
PROC = ROOT / "dataset" / "processed"
REPORTS = ROOT / "reports"
OVERLAY_DIR = REPORTS / "gradcam_neg_cocci"
LAYERS = {"features[12]": 12, "features[8]": 8, "features[3]": 3}
PRIMARY = "features[12]"
OVERLAY_LAYER = "features[3]"
SANITY_THRESHOLD = 0.5
SEED = 20260917


def overlay(rgb224: np.ndarray, cam224: np.ndarray) -> Image.Image:
    """地図を赤の濃さで重ねる(観察用)。"""
    c = cam224 / cam224.max() if cam224.max() > 0 else cam224
    base = rgb224.astype(np.float32)
    heat = np.zeros_like(base)
    heat[..., 0] = 255.0
    alpha = (0.6 * c)[..., None]
    return Image.fromarray(np.clip(base * (1 - alpha) + heat * alpha, 0, 255).astype(np.uint8))


def measure() -> int:
    from ml.stage_b import BACILLUS, COCCUS
    from ml.train import MEAN, STD, build_model

    rows = {json.loads(l)["image_id"]: json.loads(l)
            for l in (META / "prepared.jsonl").read_text(encoding="utf-8").splitlines()}
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    # 層ごと・物差しごと・群ごとに (集中度比, group) を集める
    recs: dict = {ln: {"b": {}, "c": {}} for ln in LAYERS}
    top10: dict = {ln: {"b": {}, "c": {}} for ln in LAYERS}
    rhos: dict[str, list[float]] = {ln: [] for ln in LAYERS}
    degenerate: dict = {ln: {} for ln in LAYERS}

    def add(store, layer, ruler, group_name, value, group_id):
        store[layer][ruler].setdefault(group_name, []).append((value, group_id))

    for k, m in enumerate(MODELS):
        ruler = "b" if m["splits_key"] == "ruler_b" else "c"
        trained = build_model()
        trained.load_state_dict(torch.load(CKPT_DIR / f"{m['name']}.pth", map_location="cpu"))
        trained.eval()
        random_model = randomized_copy(trained, SEED + k)
        test = json.loads((IDS_DIR / f"{m['name']}_ids.json").read_text(encoding="utf-8"))["test"]
        held = sorted(i for i in test if rows[i]["folder"] in m["held_out_negative_cocci"])
        pos = sorted(i for i in test if rows[i]["gram"] == "positive" and rows[i]["shape"] == "coccus")

        for group_name, ids in (("neg_cocci", held), ("pos_cocci", pos)):
            written = 0
            for image_id in ids:
                with Image.open(PROC / rows[image_id]["path"]) as im:
                    rgb = to_input(np.asarray(im.convert("RGB")))
                x = torch.from_numpy((((rgb.astype(np.float32) / 255.0) - MEAN) / STD).transpose(2, 0, 1)[None].copy())
                with torch.no_grad():
                    pred = int(trained(x).argmax(1))
                truth = BACILLUS if rows[image_id]["shape"] == "bacillus" else COCCUS
                if group_name == "pos_cocci" and pred != truth:
                    continue  # 正答だけを比べる相手にする
                mask_otsu = stained_mask(rgb)
                mask_top = stain_mask(rgb)
                for ln, idx in LAYERS.items():
                    cam_t = grad_cam(trained, trained.features[idx], x, pred, out_size=None)
                    cam_r = grad_cam(random_model, random_model.features[idx], x, pred, out_size=None)
                    if cam_t.std() > 0 and cam_r.std() > 0:
                        rho = spearmanr(cam_t.ravel(), cam_r.ravel()).statistic
                        if np.isfinite(rho):
                            rhos[ln].append(float(rho))
                    up = lambda c: F.interpolate(torch.from_numpy(c)[None, None], size=rgb.shape[:2],
                                                 mode="bilinear", align_corners=False)[0, 0].numpy()
                    ut, ur = up(cam_t), up(cam_r)
                    for who, cam in (("trained", ut), ("randomized", ur)):
                        if group_name == "pos_cocci" and who == "randomized":
                            continue
                        key = f"{group_name}_{who}"
                        v = concentration(cam, mask_otsu)
                        if v is None:
                            degenerate[ln][f"{ruler}:{key}"] = degenerate[ln].get(f"{ruler}:{key}", 0) + 1
                        else:
                            add(recs, ln, ruler, key, v, rows[image_id]["group_id"])
                        vt = concentration(cam, mask_top)
                        if vt is not None:
                            add(top10, ln, ruler, key, vt, rows[image_id]["group_id"])
                    if ln == OVERLAY_LAYER and written < 3:
                        overlay(rgb, ut).save(OVERLAY_DIR / f"{m['name']}_{group_name}_{image_id}.png", optimize=True)
                if written < 3:
                    written += 1
            print(f"  {m['name']} {group_name}: {len(ids)} 枚", flush=True)

    def summarise(items):
        if not items:
            return {"mean_ratio": None, "ci_low": None, "ci_high": None, "n_used": 0}
        v = np.array([a for a, _ in items])
        g = np.array([b for _, b in items])
        mean, lo, hi = group_bootstrap_mean(v, g)
        return {"mean_ratio": round(mean, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4), "n_used": len(v)}

    layers_doc = {}
    for ln in LAYERS:
        median_rho = float(np.median(rhos[ln])) if rhos[ln] else float("nan")
        sane = bool(median_rho < SANITY_THRESHOLD)
        rulers_doc = {}
        for ruler in ("b", "c"):
            nc = summarise(recs[ln][ruler].get("neg_cocci_trained", []))
            rnd = summarise(recs[ln][ruler].get("neg_cocci_randomized", []))
            pc = summarise(recs[ln][ruler].get("pos_cocci_trained", []))
            if not sane:
                verdict = {"location": "判定しない", "vs_correct": "判定しない"}
            else:
                if nc["ci_low"] > rnd["ci_high"]:
                    loc = "誤答の根拠は菌体の上に寄っている"
                elif nc["ci_high"] < rnd["ci_low"]:
                    loc = "誤答の根拠は菌体の外(背景)に寄っている"
                else:
                    loc = "区別できない"
                if pc["n_used"] == 0:
                    vs = "比べる正答が無い"
                else:
                    overlap = not (nc["ci_high"] < pc["ci_low"] or pc["ci_high"] < nc["ci_low"])
                    vs = "違いは見えない" if overlap else "見る場所が違う"
                verdict = {"location": loc, "vs_correct": vs}
            rulers_doc[ruler] = {
                "neg_cocci_trained": nc, "neg_cocci_randomized": rnd, "pos_cocci_trained": pc,
                "top10_reference": {key: summarise(val) for key, val in top10[ln][ruler].items()},
                "verdict": verdict,
            }
            print(f"  {ln} 物差し {ruler.upper()}: 陰性球菌 学習済み {nc['mean_ratio']} [{nc['ci_low']}, {nc['ci_high']}]"
                  f" / 乱数化 {rnd['mean_ratio']} [{rnd['ci_low']}, {rnd['ci_high']}]"
                  f" / 陽性球菌(正答) {pc['mean_ratio']} → {verdict}", flush=True)
        layers_doc[ln] = {"sanity": {"median_spearman": round(median_rho, 4), "n_pairs": len(rhos[ln]),
                                     "threshold": SANITY_THRESHOLD, "passed": sane},
                          "degenerate": degenerate[ln], "rulers": rulers_doc}
        print(f"  {ln}: 健全性 順位相関の中央値 {median_rho:.4f}({len(rhos[ln])} 対)→ {'通過' if sane else '不通過'}", flush=True)

    doc = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rule": "SPEC §3.20(測る前に書いた)",
        "primary_layer": PRIMARY,
        "stain_mask": "224px で彩度に画像ごとの大津の閾値(top10_reference は loop_008 の上位 10%)",
        "target_class": "学習済みモデルの予測クラス。乱数化モデルにも同じクラスを描かせる",
        "overlays": sorted(p.name for p in OVERLAY_DIR.glob("*.png")),
        "layers": layers_doc,
    }
    (REPORTS / "gradcam_negative_cocci.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {(REPORTS / 'gradcam_negative_cocci.json').relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("measure",))
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    return measure()


if __name__ == "__main__":
    raise SystemExit(main())
