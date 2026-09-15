"""陰性球菌で崩れる理由を測る(SPEC §3.12・loop_013)。

§3.10 で、形のモデルは学習で見ていない Gram 陰性球菌をほぼ全部桿菌と答えた。候補は二つ:
  (a) 色の近道 —— 学習に残る陰性球菌が 1 分類群しかなく「桃色なら桿菌」に寄っている
  (b) 見た目そのもの —— 双球菌の対や小ささが桿菌に近く見える

**形はそのままに色だけを変える操作**で切り分ける。色相だけを回し、彩度と明度はそのまま戻す。
形の規則の染色マスクは彩度から作るので、この操作は形の規則の特徴量を構造上変えない(T-283)。

判定規則は SPEC §3.12 に**測る前に**書いてある。ここは当てはめるだけ。

使い方:
    python -m ml.mechanism train     # 4 つの fold を学習し直して重みを保存(再開可能)
    python -m ml.mechanism measure   # 塗り替えて測り、判定する
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from ml.stage_b import BACILLUS, COCCUS, restrict_folds, shape_feature

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "dataset" / "metadata"
PROC = ROOT / "dataset" / "processed"
REPORTS = ROOT / "reports"
CKPT_DIR = ROOT / "ml" / "checkpoints" / "mechanism"
IDS_DIR = REPORTS / "mechanism"
N_BOOT = 2000

# 陰性球菌の分類群が試験側に入った fold(loop_013 で splits.json から数えた)
MODELS = [
    {"name": "B0", "splits_key": "ruler_b", "fold": 0, "held_out_negative_cocci": ["Neisseria gonorrhoeae"]},
    {"name": "B2", "splits_key": "ruler_b", "fold": 2, "held_out_negative_cocci": ["Veillonella"]},
    {"name": "C0", "splits_key": "ruler_c", "fold": 0, "held_out_negative_cocci": ["Neisseria gonorrhoeae"]},
    {"name": "C3", "splits_key": "ruler_c", "fold": 3, "held_out_negative_cocci": ["Veillonella"]},
]
SWEEP_CACHE_PREFIX = {"ruler_b": "SB", "ruler_c": "SC"}


# ------------------------------------------------------------------ 色空間


def rgb_to_hsv(a: np.ndarray) -> np.ndarray:
    """[0,1] の RGB → H(度)・S・V。"""
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1)
    mn = a.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 0
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = ((g - b)[rm] / d[rm]) % 6.0
    h[gm] = ((b - r)[gm] / d[gm]) + 2.0
    h[bm] = ((r - g)[bm] / d[bm]) + 4.0
    s = np.where(mx > 0, d / np.where(mx > 0, mx, 1.0), 0.0)
    return np.stack([(h * 60.0) % 360.0, s, mx], axis=-1)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h = (hsv[..., 0] % 360.0) / 60.0
    s, v = hsv[..., 1], hsv[..., 2]
    c = v * s
    x = c * (1.0 - np.abs(h % 2.0 - 1.0))
    m = v - c
    z = np.zeros_like(h)
    i = np.floor(h).astype(int) % 6
    choices = [
        np.stack([c, x, z], -1), np.stack([x, c, z], -1), np.stack([z, c, x], -1),
        np.stack([z, x, c], -1), np.stack([x, z, c], -1), np.stack([c, z, x], -1),
    ]
    out = np.zeros(hsv.shape, dtype=np.float64)
    for k in range(6):
        out[i == k] = choices[k][i == k]
    return out + m[..., None]


def rotate_hue(img: np.ndarray, delta_deg: float) -> np.ndarray:
    """色相だけを delta_deg 回す。彩度と明度はそのまま(T-281・T-282)。"""
    hsv = rgb_to_hsv(img.astype(np.float64) / 255.0)
    hsv[..., 0] = (hsv[..., 0] + delta_deg) % 360.0
    rgb = hsv_to_rgb(hsv)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def circular_mean_deg(values) -> float:
    """円周平均(T-284)。算術平均だと 350 度と 10 度の平均が 180 度になる。"""
    rad = np.deg2rad(np.asarray(values, dtype=np.float64))
    return float(np.rad2deg(math.atan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360.0)


def bootstrap_drop(wrong_before, wrong_after, groups, n_boot: int = N_BOOT, seed: int = 20260916):
    """誤り率の下がり幅(前 − 後)と、group 単位のブートストラップ 95% 区間(T-287)。"""
    wb = np.asarray(wrong_before, dtype=float)
    wa = np.asarray(wrong_after, dtype=float)
    g = np.asarray(groups)
    point = float(wb.mean() - wa.mean())
    uniq, inv = np.unique(g, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[k] for k in pick])
        stats.append(wb[idx].mean() - wa[idx].mean())
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return point, float(min(lo, point)), float(max(hi, point))


# ------------------------------------------------------------------ 素材


def load_rows() -> list[dict]:
    return [json.loads(line) for line in (META / "prepared.jsonl").read_text(encoding="utf-8").splitlines()]


def load_hue() -> dict[str, float]:
    feats = json.loads((META / "colour_features.json").read_text(encoding="utf-8"))
    return {k: v["hue_deg"] for k, v in feats.items()}


def target_hues(rows: list[dict], hue: dict[str, float]) -> tuple[float, float]:
    """紫 = Stage B の Gram 陽性画像の色相の円周平均、桃 = 陰性画像の円周平均。"""
    sb = [r for r in rows if r["stage_b"]]
    violet = circular_mean_deg([hue[r["image_id"]] for r in sb if r["gram"] == "positive"])
    pink = circular_mean_deg([hue[r["image_id"]] for r in sb if r["gram"] == "negative"])
    return violet, pink


def rb_ratio_of(rgb: np.ndarray) -> float:
    """ml/run_controls.colour_features の R/B 比を配列から取る(同じ定義)。"""
    from ml.run_controls import COLOUR_SIZE, STAIN_QUANTILE

    small = Image.fromarray(rgb).resize((COLOUR_SIZE, COLOUR_SIZE), Image.Resampling.BILINEAR)
    a = np.asarray(small, dtype=np.float32) / 255.0
    mx = a.max(2)
    mn = a.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    stained = a[sat >= np.quantile(sat, STAIN_QUANTILE)]
    if stained.size == 0:
        stained = a.reshape(-1, 3)
    return float(np.mean(stained[:, 0] / np.maximum(stained[:, 2], 1e-6)))


def to_input(rgb512: np.ndarray) -> np.ndarray:
    """学習時(ml/train.load_all)と同じ 224 への縮小。"""
    from ml.train import INPUT

    return np.asarray(Image.fromarray(rgb512).resize((INPUT, INPUT), Image.Resampling.BILINEAR))


# ------------------------------------------------------------------ 学習し直し


def train() -> int:
    import torch

    from ml.train import SEED, prepare, train_model

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    IDS_DIR.mkdir(parents=True, exist_ok=True)
    epochs = json.loads((REPORTS / "cnn_shape.json").read_text(encoding="utf-8"))["epoch_budget"]["chosen"]
    rows, splits, arr, index, meta = prepare("shape")
    for m in MODELS:
        ckpt = CKPT_DIR / f"{m['name']}.pth"
        ids_path = IDS_DIR / f"{m['name']}_ids.json"
        if ckpt.exists() and ids_path.exists():
            print(f"  {m['name']}: 保存済み", flush=True)
            continue
        fold = splits[m["splits_key"]]["folds"][m["fold"]]
        idx_tr = np.array([index[i] for i in fold["train"]])
        y_tr = np.array([meta[i]["y"] for i in fold["train"]])
        t0 = time.time()
        # 種は loop_011 の sweep と同じ(fold i は SEED + i)
        model, _ = train_model(arr, idx_tr, y_tr, epochs, SEED + m["fold"])
        torch.save(model.state_dict(), ckpt)
        ids_path.write_text(json.dumps({"train": fold["train"], "test": fold["test"], "epochs": epochs,
                                        "seed": SEED + m["fold"]}, ensure_ascii=False), encoding="utf-8")
        print(f"  {m['name']}: train {len(idx_tr)} / {epochs} エポック / {time.time()-t0:.0f}s → {ckpt.relative_to(ROOT)}", flush=True)
    return 0


# ------------------------------------------------------------------ 測る


def measure() -> int:
    import torch

    from ml.train import build_model, predict

    rows = load_rows()
    by_id = {r["image_id"]: r for r in rows}
    sb_ids = {r["image_id"] for r in rows if r["stage_b"]}
    splits = json.loads((META / "splits.json").read_text(encoding="utf-8"))
    hue = load_hue()
    colour = json.loads((META / "colour_features.json").read_text(encoding="utf-8"))
    hue_threshold = json.loads((REPORTS / "controls.json").read_text(encoding="utf-8"))["baselines"]["hue"]["oracle_threshold"]
    violet, pink = target_hues(rows, hue)
    print(f"  目標の色相: 紫 {violet:.1f} 度 / 桃 {pink:.1f} 度", flush=True)

    cache: dict[str, np.ndarray] = {}

    def image(i: str) -> np.ndarray:
        if i not in cache:
            with Image.open(PROC / by_id[i]["path"]) as im:
                cache[i] = np.asarray(im.convert("RGB"))
        return cache[i]

    def variants(ids: list[str], target: float | None) -> dict[str, np.ndarray]:
        out = {"original": np.stack([to_input(image(i)) for i in ids]),
               "sham": np.stack([to_input(rotate_hue(image(i), 0.0)) for i in ids])}
        if target is not None:
            out["recoloured"] = np.stack([to_input(rotate_hue(image(i), target - hue[i])) for i in ids])
        return out

    def y_of(ids):
        return np.array([BACILLUS if by_id[i]["shape"] == "bacillus" else COCCUS for i in ids])

    # ---- 確かめること M1・M2(モデルに依らない)
    nc_ids = sorted(i for i in sb_ids if by_id[i]["gram"] == "negative" and by_id[i]["shape"] == "coccus")
    m1_pos = 0
    m2_same = True
    helper_diff = 0.0
    for i in nc_ids:
        a = image(i)
        helper_diff = max(helper_diff, abs(rb_ratio_of(a) - colour[i]["rb_ratio"]))
        v = rotate_hue(a, violet - hue[i])
        m1_pos += int(rb_ratio_of(v) <= hue_threshold)  # R/B > 閾値 を陰性と答える規則
        f0, f1 = shape_feature(a), shape_feature(v)
        if f0["n_components"] != f1["n_components"] or (
            (f0["elongation"] is None) != (f1["elongation"] is None)
            or (f0["elongation"] is not None and abs(f0["elongation"] - f1["elongation"]) >= 1e-9)
        ):
            m2_same = False
    m1_rate = m1_pos / len(nc_ids)
    print(f"  M1 紫にした陰性球菌を色の規則が陽性と答える割合 {m1_rate:.1%}(R/B の再実装と保存値の最大差 {helper_diff:.2e})", flush=True)
    print(f"  M2 形の規則の特徴量が全件一致: {m2_same}", flush=True)

    # ---- モデルごと
    per_model = []
    wrong: dict[str, dict] = {}
    secondary_s1 = None
    secondary_s2 = {}
    for m in MODELS:
        model = build_model()
        model.load_state_dict(torch.load(CKPT_DIR / f"{m['name']}.pth", map_location="cpu"))
        model.eval()
        ids = json.loads((IDS_DIR / f"{m['name']}_ids.json").read_text(encoding="utf-8"))
        test = ids["test"]

        # S3 再現: 元の test 全体の予測を sweep の保存と比べる
        all_arr = np.stack([to_input(image(i)) for i in test])
        pred_all = predict(model, all_arr, np.arange(len(test)))
        sweep = json.loads((ROOT / "ml" / "checkpoints" / "fold_preds" /
                            f"{SWEEP_CACHE_PREFIX[m['splits_key']]}_{ids['epochs']}ep_{m['fold']:02d}.json").read_text(encoding="utf-8"))
        agreement = float(np.mean([int(p) == sweep[i] for i, p in zip(test, pred_all)]))

        held = [i for i in test if by_id[i]["folder"] in m["held_out_negative_cocci"]]
        v = variants(held, violet)
        preds = {k: predict(model, arr, np.arange(len(held))) for k, arr in v.items()}
        y = y_of(held)
        w = {k: (p != y).astype(int) for k, p in preds.items()}
        wrong[m["name"]] = {"ids": held, **w}
        sham_change = float(w["sham"].mean() - w["original"].mean())
        print(f"  {m['name']}: 陰性球菌 {len(held)} 枚 誤り 元 {w['original'].mean():.1%} / 偽 {w['sham'].mean():.1%} / 紫 {w['recoloured'].mean():.1%}"
              f"  (sweep の予測との一致 {agreement:.1%})", flush=True)

        # S1 特異性: 見ていない陰性桿菌を紫に
        nb = [i for i in test if by_id[i]["gram"] == "negative" and by_id[i]["shape"] == "bacillus"]
        if nb:
            vb = variants(nb, violet)
            yb = y_of(nb)
            eb = {k: float((predict(model, a, np.arange(len(nb))) != yb).mean()) for k, a in vb.items()}
            secondary_s1 = {"model": m["name"], "taxa": sorted({by_id[i]["folder"] for i in nb}), "n": len(nb),
                            "error_original": eb["original"], "error_violet": eb["recoloured"]}
        # S2 逆向き: 陽性球菌を桃に
        pc = [i for i in test if by_id[i]["gram"] == "positive" and by_id[i]["shape"] == "coccus"]
        if pc:
            vp = variants(pc, pink)
            yp = y_of(pc)
            ep = {k: float((predict(model, a, np.arange(len(pc))) != yp).mean()) for k, a in vp.items()}
            secondary_s2[m["name"]] = {"taxa": sorted({by_id[i]["folder"] for i in pc}), "n": len(pc),
                                       "error_original": ep["original"], "error_pink": ep["recoloured"]}

        per_model.append({**m, "checkpoint": str((CKPT_DIR / f"{m['name']}.pth").relative_to(ROOT)).replace("\\", "/"),
                          "ids_file": str((IDS_DIR / f"{m['name']}_ids.json").relative_to(ROOT)).replace("\\", "/"),
                          "n_held_out": len(held), "error_original": float(w["original"].mean()),
                          "error_sham": float(w["sham"].mean()), "error_violet": float(w["recoloured"].mean()),
                          "sham_error_change": sham_change, "sweep_prediction_agreement": agreement})

    checks = {
        "m1_hue_rule_positive_rate": m1_rate,
        "m1_hue_threshold": hue_threshold,
        "m1_rb_reimplementation_max_diff": helper_diff,
        "m2_shape_features_identical": m2_same,
        "m3_sham_error_change": {p["name"]: p["sham_error_change"] for p in per_model},
    }
    checks_ok = (m1_rate >= 0.90 and m2_same and all(abs(v) < 0.05 for v in checks["m3_sham_error_change"].values()))

    primary = {}
    for ruler, names in (("b", ("B0", "B2")), ("c", ("C0", "C3"))):
        ids = sum((wrong[n]["ids"] for n in names), [])
        wo = np.concatenate([wrong[n]["original"] for n in names])
        wv = np.concatenate([wrong[n]["recoloured"] for n in names])
        ws = np.concatenate([wrong[n]["sham"] for n in names])
        groups = [by_id[i]["group_id"] for i in ids]
        point, lo, hi = bootstrap_drop(wo, wv, groups)
        e_o, e_v = float(wo.mean()), float(wv.mean())
        drop = e_o - e_v
        if not checks_ok:
            verdict = "判定しない"
        elif drop >= 0.50 and e_v <= 0.40:
            verdict = "色が主因"
        elif drop < 0.20:
            verdict = "見た目が主因"
        else:
            verdict = "混合"
        per_taxon = {}
        for t in ("Neisseria gonorrhoeae", "Veillonella"):
            sel = np.array([by_id[i]["folder"] == t for i in ids])
            per_taxon[t] = {"n": int(sel.sum()), "error_original": float(wo[sel].mean()), "error_violet": float(wv[sel].mean())}
        primary[ruler] = {"models": list(names), "n": len(ids), "error_original": e_o, "error_sham": float(ws.mean()),
                          "error_violet": e_v, "drop": {"point": drop, "ci_low": lo, "ci_high": hi},
                          "verdict": verdict, "per_taxon": per_taxon}
        print(f"  物差し {ruler.upper()}: 誤り 元 {e_o:.1%} → 紫 {e_v:.1%}(下がり幅 {drop*100:.1f} ポイント [{lo*100:.1f}, {hi*100:.1f}])→ {verdict}", flush=True)

    doc = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rule": "SPEC §3.12(測る前に書いた)。M1〜M3 が通ってから、下がり幅 ≥ 50 ポイントかつ紫の誤り ≤ 40% で色が主因、"
                "下がり幅 < 20 ポイントで見た目が主因、それ以外は混合",
        "targets": {"violet_hue_deg": violet, "pink_hue_deg": pink},
        "models": per_model,
        "checks": checks,
        "checks_passed": checks_ok,
        "primary": primary,
        "secondary": {"s1_unseen_negative_bacilli_to_violet": secondary_s1, "s2_positive_cocci_to_pink": secondary_s2},
        "n_boot": N_BOOT,
    }
    (REPORTS / "mechanism_negative_cocci.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {(REPORTS / 'mechanism_negative_cocci.json').relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("train", "measure"))
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    return train() if args.command == "train" else measure()


if __name__ == "__main__":
    raise SystemExit(main())
