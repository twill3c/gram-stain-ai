"""Grad-CAM で、出荷モデルが菌体(染まった画素)を見ているかを測る(F-11 / G-10)。

## 何を問うか

本プロジェクトの中心の主張は「CNN の優位は分類群の見覚えである」だった(SPEC §3.5)。
見覚えの手がかりが菌体の形なのか、背景・染色のむら・ピントなのかは、成績の表からは分からない。
そこで、モデルの判断が画素のどこに寄っているかを Grad-CAM で測る。

## G-10 を測る前に引き直した理由

旧文言は「上位反応が視野円の外(黒隅)に無い」だった。だが前処理が視野円の中心 512 を
切り出しているので、**入力に黒隅は存在しない**。何を実装しても通るゲートは、
守られていないのに守られて見える。落ちうる形に引き直した(SPEC G-10)。

  (a) 健全性 —— 学習済みモデルと、重みを乱数化したモデルの CAM の順位相関の中央値が 0.5 未満。
      Adebayo et al. (2018) の model randomization test。地図が重みで変わらない手法は、
      モデルではなく入力の輪郭を写している
  (b) 本体 —— 染まった画素への CAM の集中度比について、学習済みモデルの 95% 区間の下限が
      乱数化モデルの区間の上限を上回る

**乱数化モデルを対照にするのは、畳み込みは学習しなくても濃い所に反応しうるからである。**
比が 1 を超えたこと自体は「学習の結果」の証拠にならない。建築由来の偏りを差し引く必要がある。

## 集中度比

  比 = (染まった画素に乗った CAM の割合) / (染まった画素の面積の割合)

一様な地図なら 1。染まった画素は色相ベースライン(reports/controls)と同じく
**彩度の上位 10%**。対照は 128 px で、こちらはモデルの入力と揃えて 224 px で測る(定義は同じ)。

使い方:
    python -m ml.gradcam
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "dataset" / "metadata"
PROC = ROOT / "dataset" / "processed"
REPORTS = ROOT / "reports"
JST = timezone(timedelta(hours=9))

STAIN_QUANTILE = 0.90
SANITY_THRESHOLD = 0.5
N_BOOT = 2000
SEED = 20260914


# ------------------------------------------------------------------ 手法


def grad_cam(model: torch.nn.Module, layer: torch.nn.Module, x: torch.Tensor,
             target_class: int, out_size: tuple[int, int] | None = (224, 224)) -> np.ndarray:
    """Grad-CAM(Selvaraju et al. 2017)。1 枚の地図を返す。

    alpha_k = 対象層の出力 A_k に対する dy_c の空間平均
    CAM     = ReLU( sum_k alpha_k * A_k )

    out_size が None なら対象層の解像度のまま返す(T-256 の手計算と突き合わせるため)。
    """
    store: dict[str, torch.Tensor] = {}

    def fwd_hook(_m, _inp, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    handle = layer.register_forward_hook(fwd_hook)
    try:
        model.eval()
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            logits = model(x)
            logits[0, target_class].backward()
    finally:
        handle.remove()

    a = store["a"].detach()[0]          # (K, h, w)
    g = store["g"].detach()[0]          # (K, h, w)
    alpha = g.mean(dim=(1, 2))          # (K,)
    cam = torch.relu((alpha[:, None, None] * a).sum(0))
    if out_size is not None:
        cam = F.interpolate(cam[None, None], size=out_size, mode="bilinear", align_corners=False)[0, 0]
    return cam.numpy()


def stain_mask(rgb224: np.ndarray) -> np.ndarray:
    """彩度の上位 STAIN_QUANTILE を『染まった画素』とする(対照と同じ定義)。"""
    a = rgb224.astype(np.float32) / 255.0
    mx = a.max(2)
    mn = a.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return sat >= np.quantile(sat, STAIN_QUANTILE)


def concentration(cam: np.ndarray, mask: np.ndarray) -> float | None:
    total = float(cam.sum())
    if total <= 1e-12:
        return None  # 全面 0。比が定義できない —— 数えて記録する(T-259)
    return float(cam[mask].sum() / total) / float(mask.mean())


def randomized_copy(model: torch.nn.Module, seed: int) -> torch.nn.Module:
    """重みをすべて初期化し直した複製。Adebayo et al. の independent randomization。"""
    import copy

    torch.manual_seed(seed)
    m = copy.deepcopy(model)
    for mod in m.modules():
        if hasattr(mod, "reset_parameters"):
            mod.reset_parameters()
    return m.eval()


def group_bootstrap_mean(values: np.ndarray, groups: np.ndarray, seed: int = SEED) -> tuple[float, float, float]:
    """group 単位のブートストラップで平均の 95% 区間。画像単位で取ると区間が狭くなる。"""
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    stats = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        stats[b] = values[idx].mean()
    return float(values.mean()), float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


# ------------------------------------------------------------------ 本体


def main() -> int:
    from ml.train import CKPT, INPUT, MEAN, STD, build_model

    torch.set_num_threads(3)
    REPORTS.mkdir(exist_ok=True)

    rows = {json.loads(l)["image_id"]: json.loads(l)
            for l in (META / "prepared.jsonl").read_text(encoding="utf-8").splitlines()}
    test_ids = json.loads((META / "splits.json").read_text(encoding="utf-8"))["ruler_a"]["test"]

    trained = build_model()
    trained.load_state_dict(torch.load(CKPT / "best_model.pth", map_location="cpu"))
    trained.eval()
    random_model = randomized_copy(trained, SEED)
    layer_t = trained.features[12]
    layer_r = random_model.features[12]

    rec_t, rec_r, rhos, groups, folders = [], [], [], [], []
    deg_t = deg_r = 0
    per_taxon: dict[str, list[float]] = {}

    for k, image_id in enumerate(test_ids, 1):
        r = rows[image_id]
        with Image.open(PROC / r["path"]) as im:
            rgb = np.asarray(im.convert("RGB").resize((INPUT, INPUT), Image.Resampling.BILINEAR))
        x = torch.from_numpy((((rgb.astype(np.float32) / 255.0) - MEAN) / STD).transpose(2, 0, 1)[None].copy())

        with torch.no_grad():
            target = int(trained(x).argmax(1))
        # 乱数化モデルにも**同じクラス**の地図を描かせる。比べる相手を揃える

        cam_t7 = grad_cam(trained, layer_t, x, target, out_size=None)
        cam_r7 = grad_cam(random_model, layer_r, x, target, out_size=None)
        up = lambda c: F.interpolate(torch.from_numpy(c)[None, None], size=(INPUT, INPUT),
                                     mode="bilinear", align_corners=False)[0, 0].numpy()
        mask = stain_mask(rgb)

        ct = concentration(up(cam_t7), mask)
        cr = concentration(up(cam_r7), mask)
        if ct is None:
            deg_t += 1
        else:
            rec_t.append((ct, r["group_id"]))
            per_taxon.setdefault(r["folder"], []).append(ct)
        if cr is None:
            deg_r += 1
        else:
            rec_r.append((cr, r["group_id"]))

        # 順位相関は 7x7 の生の地図で取る。引き伸ばすと補間が順位を作ってしまう
        if cam_t7.std() > 0 and cam_r7.std() > 0:
            rho = spearmanr(cam_t7.ravel(), cam_r7.ravel()).statistic
            if np.isfinite(rho):
                rhos.append(float(rho))

        if k % 50 == 0:
            print(f"  {k}/{len(test_ids)}", flush=True)

    def summarise(rec, deg):
        v = np.array([x for x, _ in rec])
        g = np.array([g for _, g in rec])
        mean, lo, hi = group_bootstrap_mean(v, g)
        return {"mean_ratio": round(mean, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                "median_ratio": round(float(np.median(v)), 4),
                "n_used": len(v), "n_degenerate": deg}

    t = summarise(rec_t, deg_t)
    rr = summarise(rec_r, deg_r)
    median_rho = float(np.median(rhos)) if rhos else float("nan")

    controls = json.loads((REPORTS / "controls.json").read_text(encoding="utf-8"))
    hard = [k for k, v in controls["baselines"]["hue"]["b"]["per_taxon_errors"].items() if v["error_rate"] >= 0.10]

    doc = {
        "generated_at": datetime.now(JST).isoformat(timespec="seconds"),
        "n_images": len(test_ids),
        "evaluated_on": "物差し A の test(出荷モデルが一度も学習していない 286 枚)",
        "target_layer": "features[12](576x7x7)",
        "target_class_rule": "学習済みモデルの予測クラス。乱数化モデルにも同じクラスを描かせる",
        "stain_quantile": STAIN_QUANTILE,
        "stain_mask_resolution": INPUT,
        "trained": t,
        "randomized": rr,
        "sanity": {"n_images": len(test_ids), "n_pairs_used": len(rhos),
                   "median_spearman_trained_vs_random": round(median_rho, 4),
                   "threshold": SANITY_THRESHOLD,
                   "source": "Adebayo et al., Sanity Checks for Saliency Maps, NeurIPS 2018"},
        "verdict": {
            "sanity_passed": bool(median_rho < SANITY_THRESHOLD),
            "looks_at_stain_beyond_architecture": bool(t["ci_low"] > rr["ci_high"]),
            "rule": "学習済みモデルの集中度比の 95% 区間の下限 > 乱数化モデルの区間の上限",
        },
        "per_taxon_trained": {k: {"n": len(v), "mean_ratio": round(float(np.mean(v)), 4),
                                  "hard_for_hue": k in hard}
                              for k, v in sorted(per_taxon.items())},
    }
    (REPORTS / "gradcam.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"学習済み   集中度比 {t['mean_ratio']} [{t['ci_low']}, {t['ci_high']}]  全面0 {deg_t}")
    print(f"乱数化     集中度比 {rr['mean_ratio']} [{rr['ci_low']}, {rr['ci_high']}]  全面0 {deg_r}")
    print(f"健全性     順位相関の中央値 {median_rho:.4f}(閾値 {SANITY_THRESHOLD}・使った対 {len(rhos)})")
    print(f"判定       健全性 {doc['verdict']['sanity_passed']} / 建築由来を超えて菌体を見ている "
          f"{doc['verdict']['looks_at_stain_beyond_architecture']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
