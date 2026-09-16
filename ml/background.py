"""背景の効果を確かめる(SPEC §3.18・loop_016)。

§3.17 で事後に見つけた「染まっていない画素を平らな背景色で塗るだけで Neisseria の誤りが下がる」を、
再現するかから確かめ、再現したら効いている**場所**と**性質**を切り分ける。

  - 平ら          : 染まっていない画素を、染まっていない画素の RGB の中央値で塗る(§3.17 の偽の貼り直しと同じ定義)
  - 縁を残して平ら : 染まった画素からチェビシェフ距離 3 以内はそのまま残し、その外だけ平らに塗る
  - 並べ替え      : 染まっていない画素の値を、同じ画像の中で無作為に並べ替える(種は画像 ID から)

**どの操作も、染まった画素は一画素も変えない。** 染色マスクと背景色は ml/separate.py と同じ関数を使う ——
「再現するか」を、§3.17 と同じ操作で測るため。

使い方:
    python -m ml.background measure
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from ml.mechanism import CKPT_DIR, IDS_DIR, MODELS, bootstrap_drop, to_input
from ml.separate import background_colour, stained_mask
from ml.stage_b import BACILLUS, COCCUS

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "dataset" / "metadata"
PROC = ROOT / "dataset" / "processed"
REPORTS = ROOT / "reports"
RING = 3
NEG_COCCI_TAXA = ("Neisseria gonorrhoeae", "Veillonella")


# ------------------------------------------------------------------ 操作


def flat(img: np.ndarray) -> np.ndarray:
    """染まっていない画素を、染まっていない画素の RGB の中央値で塗る。"""
    m = stained_mask(img)
    out = img.copy()
    out[~m] = background_colour(img, m)
    return out


def flat_keep_ring(img: np.ndarray, ring: int = RING) -> np.ndarray:
    """染まった画素からチェビシェフ距離 ring 以内はそのまま残し、その外だけ平らに塗る。

    3x3 の正方形で ring 回膨張させると、チェビシェフ距離 ring 以内になる(T-297 は距離変換で照合する)。
    塗る色は「平ら」と同じ(染まっていない画素すべての中央値)。
    """
    m = stained_mask(img)
    keep = ndimage.binary_dilation(m, structure=np.ones((3, 3), dtype=bool), iterations=ring)
    out = img.copy()
    out[~keep] = background_colour(img, m)
    return out


def shuffle_background(img: np.ndarray, image_id: str) -> np.ndarray:
    """染まっていない画素の値を、同じ画像の中で無作為に並べ替える。種は画像 ID から決める(T-299)。"""
    m = stained_mask(img)
    seed = int(hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    flat_img = img.reshape(-1, 3)
    idx = np.flatnonzero(~m.ravel())
    out = img.copy().reshape(-1, 3)
    out[idx] = flat_img[idx][rng.permutation(len(idx))]
    return out.reshape(img.shape)


# ------------------------------------------------------------------ 測る


def measure() -> int:
    import torch

    from ml.train import build_model, predict

    rows = [json.loads(line) for line in (META / "prepared.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {r["image_id"]: r for r in rows}
    done: dict[tuple[str, str], np.ndarray] = {}

    def transformed(i: str, variant: str) -> np.ndarray:
        """(画像, 操作) ごとに一度だけ作る。B0 と C0 は同じ Neisseria を持つので使い回す。"""
        key = (i, variant)
        if key not in done:
            with Image.open(PROC / by_id[i]["path"]) as im:
                a = np.asarray(im.convert("RGB"))
            t = {"original": lambda: a, "flat": lambda: flat(a), "ring": lambda: flat_keep_ring(a),
                 "shuffle": lambda: shuffle_background(a, i)}[variant]()
            done[key] = to_input(t)
        return done[key]

    def y_of(ids):
        return np.array([BACILLUS if by_id[i]["shape"] == "bacillus" else COCCUS for i in ids])

    def wrong(model, ids, variant) -> np.ndarray:
        if not ids:
            return np.zeros(0, dtype=int)
        arr = np.stack([transformed(i, variant) for i in ids])
        return (predict(model, arr, np.arange(len(ids))) != y_of(ids)).astype(int)

    held_variants = ("original", "flat", "ring", "shuffle")
    per_model: dict[str, dict] = {}
    for m in MODELS:
        model = build_model()
        model.load_state_dict(torch.load(CKPT_DIR / f"{m['name']}.pth", map_location="cpu"))
        model.eval()
        test = json.loads((IDS_DIR / f"{m['name']}_ids.json").read_text(encoding="utf-8"))["test"]
        held = [i for i in test if by_id[i]["folder"] in m["held_out_negative_cocci"]]
        pc = [i for i in test if by_id[i]["gram"] == "positive" and by_id[i]["shape"] == "coccus"]
        nb = [i for i in test if by_id[i]["gram"] == "negative" and by_id[i]["shape"] == "bacillus"]
        e = {"held": held,
             "held_wrong": {v: wrong(model, held, v) for v in held_variants},
             "pc_wrong": {v: wrong(model, pc, v) for v in ("original", "flat")},
             "nb_wrong": {v: wrong(model, nb, v) for v in ("original", "flat")}}
        per_model[m["name"]] = e
        print(f"  {m['name']}: " + " / ".join(f"{v} {e['held_wrong'][v].mean():.1%}" for v in held_variants), flush=True)

    side: dict[str, float] = {}
    primary: dict[str, dict] = {}
    for ruler, names in (("b", ("B0", "B2")), ("c", ("C0", "C3"))):
        ids = sum((per_model[n]["held"] for n in names), [])
        groups = [by_id[i]["group_id"] for i in ids]

        def pool(kind: str, variant: str) -> np.ndarray:
            return np.concatenate([per_model[n][kind][variant] for n in names])

        base = pool("held_wrong", "original")
        entry: dict = {"models": list(names), "n": len(ids), "error_original": float(base.mean())}
        for op in ("flat", "ring", "shuffle"):
            w = pool("held_wrong", op)
            point, lo, hi = bootstrap_drop(base, w, groups)
            entry[op] = {"error": float(w.mean()), "drop": point, "ci_low": lo, "ci_high": hi}
        taxa = np.array([by_id[i]["folder"] for i in ids])
        entry["per_taxon"] = {
            t: {"n": int((taxa == t).sum()),
                **{f"error_{v}": float(pool('held_wrong', v)[taxa == t].mean()) for v in held_variants}}
            for t in NEG_COCCI_TAXA if (taxa == t).any()
        }
        for kind, label in (("pc_wrong", "positive_cocci"), ("nb_wrong", "negative_bacilli")):
            o, f = pool(kind, "original"), pool(kind, "flat")
            if len(o):
                side[f"{ruler}:{label}"] = float(f.mean() - o.mean())
        primary[ruler] = entry

    side_ok = all(v < 0.20 for v in side.values())
    for ruler, r in primary.items():
        d_flat = r["flat"]["drop"]
        replicated = d_flat >= 0.10 and r["flat"]["ci_low"] > 0
        if not side_ok:
            verdict = {"replication": "判定しない", "location": None, "nature": None}
        elif not replicated:
            verdict = {"replication": "再現しない", "location": None, "nature": None}
        else:
            verdict = {
                "replication": "再現した",
                "location": "菌体のすぐまわり(3 画素の縁)が効いている" if r["ring"]["drop"] <= d_flat / 2
                else "縁より外の背景が効いている",
                "nature": "背景の並び(模様)が効いている" if r["shuffle"]["drop"] >= d_flat / 2
                else "並びではなく、ばらつきが消えたこと(平らさ)が効いている",
            }
        r["verdict"] = verdict
        print(f"  物差し {ruler.upper()}: 元 {r['error_original']:.1%} / 平ら {r['flat']['error']:.1%}"
              f"(下がり幅 {d_flat*100:.1f} [{r['flat']['ci_low']*100:.1f}, {r['flat']['ci_high']*100:.1f}])"
              f" / 縁を残す {r['ring']['error']:.1%}(下がり幅 {r['ring']['drop']*100:.1f})"
              f" / 並べ替え {r['shuffle']['error']:.1%}(下がり幅 {r['shuffle']['drop']*100:.1f}) → {verdict}", flush=True)

    doc = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rule": "SPEC §3.18(測る前に書いた)",
        "ring_chebyshev": RING,
        "checks": {"side_effect_increase": side},
        "checks_passed": side_ok,
        "primary": primary,
        "per_model": {n: {v: float(e["held_wrong"][v].mean()) for v in held_variants} for n, e in per_model.items()},
    }
    (REPORTS / "background_negative_cocci.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {(REPORTS / 'background_negative_cocci.json').relative_to(ROOT)}")
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
