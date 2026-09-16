"""拡大の二つの効果を分ける(SPEC §3.16・loop_015)。

§3.15 の拡大(s = 3.0)は、見かけの大きさを 3 倍にすることと、視野に写る菌の数を約 1/9 にすることを
同時にしていた。これを二つの操作に分けて、保存済みの 4 モデル(§3.12)に当てる。

  - 数だけ     : 中心 (辺/s) 四方を**引き伸ばさずに**残し、周りを背景色で埋める
  - 大きさだけ : 染まった塊を一つずつ切り出し、**その重心の位置で** s 倍にして背景色のキャンバスへ貼り直す

大きさだけの操作は背景を平らな色に塗り直す。それだけで答えが変わりうるので、
**s = 1.0 で同じ経路を通す偽の貼り直し**で確かめてから判定する(§3.16 P1)。

使い方:
    python -m ml.separate measure
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from ml.mechanism import CKPT_DIR, IDS_DIR, MODELS, bootstrap_drop, rgb_to_hsv, to_input
from ml.stage_b import BACILLUS, COCCUS, EIGHT, otsu_threshold

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "dataset" / "metadata"
PROC = ROOT / "dataset" / "processed"
REPORTS = ROOT / "reports"
SCALES = (1.5, 2.0, 3.0)
JUDGE_SCALE = 3.0


# ------------------------------------------------------------------ 操作


def stained_mask(img: np.ndarray) -> np.ndarray:
    """染まった画素(彩度に画像ごとの大津の閾値)。形の規則と同じ決め方。"""
    s = rgb_to_hsv(img.astype(np.float64) / 255.0)[..., 1]
    return s > otsu_threshold(s)


def background_colour(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """染まっていない画素の RGB の中央値。"""
    if mask is None:
        mask = stained_mask(img)
    bg = img[~mask]
    if bg.size == 0:
        bg = img.reshape(-1, 3)
    return np.median(bg, axis=0).round().astype(np.uint8)


def count_only(img: np.ndarray, s: float) -> np.ndarray:
    """中心 (辺/s) 四方を等倍のまま残し、外を背景色で埋める。菌の大きさは変えず、数だけ約 1/s² にする。"""
    h, w = img.shape[:2]
    side = int(round(min(h, w) / s))
    top, left = (h - side) // 2, (w - side) // 2
    out = np.empty_like(img)
    out[:] = background_colour(img)
    out[top : top + side, left : left + side] = img[top : top + side, left : left + side]
    return out


def size_only(img: np.ndarray, s: float) -> tuple[np.ndarray, float]:
    """染まった塊を一つずつ、重心の位置で s 倍にして背景色のキャンバスへ貼り直す。

    返り値: (画像, 重なりの割合 = 二度以上塗られた画素 / 塗られた画素)
    s = 1.0 でも同じ経路を通す(背景を平らに塗り直すだけの、偽の操作になる)。
    """
    h, w = img.shape[:2]
    mask = stained_mask(img)
    out = np.empty_like(img)
    out[:] = background_colour(img, mask)
    painted = np.zeros((h, w), dtype=np.int32)
    labels, n = ndimage.label(mask, structure=EIGHT)
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        y0, x0 = sl[0].start, sl[1].start
        comp = labels[sl] == k
        patch = img[sl]
        ys, xs = np.nonzero(comp)
        cy, cx = ys.mean(), xs.mean()  # 塊の中での重心
        if s == 1.0:
            new_patch, new_mask = patch, comp
            ty, tx = y0, x0
        else:
            ph, pw = comp.shape
            nh, nw = max(1, int(round(ph * s))), max(1, int(round(pw * s)))
            new_patch = np.asarray(Image.fromarray(patch).resize((nw, nh), Image.Resampling.BILINEAR))
            new_mask = np.asarray(Image.fromarray((comp * 255).astype(np.uint8)).resize(
                (nw, nh), Image.Resampling.BILINEAR)) > 127
            # 重心を保つ: 元の画像での重心 (y0+cy) に、拡大後の重心 (cy+0.5)*s-0.5 を合わせる
            ty = int(round(y0 + cy - ((cy + 0.5) * s - 0.5)))
            tx = int(round(x0 + cx - ((cx + 0.5) * s - 0.5)))
        ph, pw = new_mask.shape
        # キャンバスからはみ出す分を切る
        sy0, sx0 = max(0, -ty), max(0, -tx)
        dy0, dx0 = max(0, ty), max(0, tx)
        dy1, dx1 = min(h, ty + ph), min(w, tx + pw)
        if dy1 <= dy0 or dx1 <= dx0:
            continue
        m = new_mask[sy0 : sy0 + (dy1 - dy0), sx0 : sx0 + (dx1 - dx0)]
        region = out[dy0:dy1, dx0:dx1]
        region[m] = new_patch[sy0 : sy0 + (dy1 - dy0), sx0 : sx0 + (dx1 - dx0)][m]
        painted[dy0:dy1, dx0:dx1][m] += 1
    total = int((painted > 0).sum())
    overlap = float((painted > 1).sum() / total) if total else 0.0
    return out, overlap


# ------------------------------------------------------------------ 測る


def measure() -> int:
    import torch

    from ml.train import build_model, predict

    rows = [json.loads(line) for line in (META / "prepared.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {r["image_id"]: r for r in rows}
    raw: dict[str, np.ndarray] = {}
    done: dict[tuple[str, str], np.ndarray] = {}
    overlaps: dict[str, list[float]] = {}

    def image(i: str) -> np.ndarray:
        if i not in raw:
            with Image.open(PROC / by_id[i]["path"]) as im:
                raw[i] = np.asarray(im.convert("RGB"))
        return raw[i]

    def transformed(i: str, variant: str) -> np.ndarray:
        """(画像, 操作) ごとに一度だけ作る。B0 と C0 は同じ Neisseria を持つので使い回す。"""
        key = (i, variant)
        if key not in done:
            a = image(i)
            if variant == "original":
                t = a
            elif variant.startswith("count_"):
                t = count_only(a, float(variant.split("_")[1]))
            else:
                t, ov = size_only(a, float(variant.split("_")[1]))
                overlaps.setdefault(variant, []).append(ov)
            done[key] = to_input(t)
        return done[key]

    def y_of(ids):
        return np.array([BACILLUS if by_id[i]["shape"] == "bacillus" else COCCUS for i in ids])

    def wrong(model, ids, variant) -> np.ndarray:
        if not ids:
            return np.zeros(0, dtype=int)
        arr = np.stack([transformed(i, variant) for i in ids])
        return (predict(model, arr, np.arange(len(ids))) != y_of(ids)).astype(int)

    held_variants = ["original", "size_1.0"] + [f"count_{s}" for s in SCALES] + [f"size_{s}" for s in SCALES]
    ctrl_variants = ["original", f"count_{JUDGE_SCALE}", f"size_{JUDGE_SCALE}"]

    per_model: dict[str, dict] = {}
    for m in MODELS:
        model = build_model()
        model.load_state_dict(torch.load(CKPT_DIR / f"{m['name']}.pth", map_location="cpu"))
        model.eval()
        test = json.loads((IDS_DIR / f"{m['name']}_ids.json").read_text(encoding="utf-8"))["test"]
        held = [i for i in test if by_id[i]["folder"] in m["held_out_negative_cocci"]]
        pc = [i for i in test if by_id[i]["gram"] == "positive" and by_id[i]["shape"] == "coccus"]
        nb = [i for i in test if by_id[i]["gram"] == "negative" and by_id[i]["shape"] == "bacillus"]
        e = {"held": held, "n_positive_cocci": len(pc), "n_negative_bacilli": len(nb),
             "held_wrong": {v: wrong(model, held, v) for v in held_variants},
             "pc_wrong": {v: wrong(model, pc, v) for v in ctrl_variants},
             "nb_wrong": {v: wrong(model, nb, v) for v in ctrl_variants}}
        per_model[m["name"]] = e
        print(f"  {m['name']}: " + " / ".join(f"{v} {e['held_wrong'][v].mean():.1%}" for v in held_variants), flush=True)

    p1 = {n: float(e["held_wrong"]["size_1.0"].mean() - e["held_wrong"]["original"].mean()) for n, e in per_model.items()}

    primary, p2 = {}, {}
    for ruler, names in (("b", ("B0", "B2")), ("c", ("C0", "C3"))):
        ids = sum((per_model[n]["held"] for n in names), [])
        groups = [by_id[i]["group_id"] for i in ids]

        def pool(kind: str, variant: str) -> np.ndarray:
            return np.concatenate([per_model[n][kind][variant] for n in names])

        base = pool("held_wrong", "original")
        entry = {"models": list(names), "n": len(ids), "error_original": float(base.mean())}
        for op in ("count", "size"):
            v = f"{op}_{JUDGE_SCALE}"
            w = pool("held_wrong", v)
            point, lo, hi = bootstrap_drop(base, w, groups)
            entry[f"{op}_only"] = {
                "error": float(w.mean()), "drop": point, "ci_low": lo, "ci_high": hi,
                "by_scale": {str(s): float(pool("held_wrong", f"{op}_{s}").mean()) for s in SCALES},
            }
            for kind, label in (("pc_wrong", "positive_cocci"), ("nb_wrong", "negative_bacilli")):
                o, t = pool(kind, "original"), pool(kind, v)
                if len(o):
                    p2[f"{ruler}:{op}_{JUDGE_SCALE}:{label}"] = float(t.mean() - o.mean())
        primary[ruler] = entry

    ok = all(abs(v) < 0.05 for v in p1.values()) and all(v < 0.20 for v in p2.values())
    for ruler, entry in primary.items():
        size_ok = entry["size_only"]["drop"] >= 0.15 and entry["size_only"]["ci_low"] > 0
        count_ok = entry["count_only"]["drop"] >= 0.15 and entry["count_only"]["ci_low"] > 0
        if not ok:
            verdict = "判定しない"
        elif size_ok and count_ok:
            verdict = "両方"
        elif size_ok:
            verdict = "大きさが効いている"
        elif count_ok:
            verdict = "数が効いている"
        else:
            verdict = "どちらでもない"
        entry["verdict"] = verdict
        print(f"  物差し {ruler.upper()}: 元 {entry['error_original']:.1%} / 数だけ {entry['count_only']['error']:.1%}"
              f"(下がり幅 {entry['count_only']['drop']*100:.1f} [{entry['count_only']['ci_low']*100:.1f}, {entry['count_only']['ci_high']*100:.1f}])"
              f" / 大きさだけ {entry['size_only']['error']:.1%}"
              f"(下がり幅 {entry['size_only']['drop']*100:.1f} [{entry['size_only']['ci_low']*100:.1f}, {entry['size_only']['ci_high']*100:.1f}])"
              f" → {verdict}", flush=True)

    probe = REPORTS / "probe_negative_cocci.json"
    reference = None
    if probe.exists():
        pr = json.loads(probe.read_text(encoding="utf-8"))["primary"]
        reference = {r: pr[r]["zoom"]["3.0"]["error"] for r in pr}

    doc = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rule": "SPEC §3.16(測る前に書いた)。P1 偽の貼り直し < 5pt・P2 正しく答えられている群の悪化 < 20pt が通ってから、"
                "s=3.0 の下がり幅 ≥ 15pt かつ区間の下限 > 0 で『効いている』",
        "checks": {"p1_sham_size_error_change": p1, "p2_side_effect_increase": p2},
        "checks_passed": ok,
        "primary": primary,
        "reference_zoom_3_error": reference,
        "observation_size_overlap_mean": {k: float(np.mean(v)) for k, v in overlaps.items()},
    }
    (REPORTS / "separate_negative_cocci.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {(REPORTS / 'separate_negative_cocci.json').relative_to(ROOT)}")
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
