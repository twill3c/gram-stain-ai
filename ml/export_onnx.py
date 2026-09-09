"""出荷モデルを ONNX へ変換し、PyTorch と ONNX Runtime で照合する(G-01)。

## なぜ照合するのか

二実装照合は**非循環のオラクル**である。PyTorch と ONNX Runtime は同じ重みを別の実装で走らせる。
こちらが期待値を書いているのではなく、**片方が壊れれば一致しない**という構造で確かめている。
変換は静かに壊れる —— 演算子の解釈違い、精度の落ち、入力の並びの取り違え。
どれも例外を出さずに「それらしい数」を返すので、照合しなければ気づけない。

## exporter をどちらにするか

torch 2.14 では `torch.onnx.export` の `dynamo` 既定が True になった。
先行の transit-lens は `dynamo=False` + opset 17 で出しており、フリートでの動作実績がある。
**決め打たずに両方で出し、照合の一致度と配信サイズを実測してから選ぶ。**

## 配るもの

`public/models/` に置く。

  - `model.onnx`
  - `labels.json` — クラスの並び。**Web 側と Python 側で並びが食い違うと、静かに反転する**
  - `model_metadata.json` — 版・学習日・git commit・データセットの帰属(CC BY 4.0)、
    そして**三つの物差しの成績**。出荷モデルの test の数だけを配ると、
    画面がその数だけを示すことになる(T-243)

使い方:
    python -m ml.export_onnx
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from ml.metrics import NEGATIVE, POSITIVE, macro_f1
from ml.train import CKPT, INPUT, build_model, ids_to_arrays, prepare, predict

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "dataset" / "metadata"
REPORTS = ROOT / "reports"
PUBLIC = ROOT / "public" / "models"
JST = timezone(timedelta(hours=9))

MODEL_VERSION = "1.0.0"
OPSET = 17
# クラスの並び。**ここが唯一の出どころ**にする。
# Python 側の POSITIVE=0 / NEGATIVE=1 と、配る labels.json と、Web 側の表示が
# 別々に並びを持つと、どこかで反転しても例外は出ない
CLASSES = ["gram_positive", "gram_negative"]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 — 記録が取れないこと自体は致命ではない
        return "unknown"


def export(model: torch.nn.Module, path: Path, dynamo: bool) -> None:
    model.eval()
    dummy = torch.zeros(1, 3, INPUT, INPUT)
    kwargs = dict(
        input_names=["input"],
        output_names=["logits"],
        opset_version=OPSET,
        dynamo=dynamo,
    )
    if dynamo:
        kwargs["dynamic_shapes"] = {"input": {0: torch.export.Dim("batch")}}
    else:
        kwargs["dynamic_axes"] = {"input": {0: "batch"}, "logits": {0: "batch"}}
    torch.onnx.export(model, (dummy,), str(path), **kwargs)


def run_ort(path: Path, x: np.ndarray, batch: int = 32) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = []
    for s in range(0, len(x), batch):
        out.append(sess.run(None, {name: x[s : s + batch]})[0])
    return np.concatenate(out)


def torch_logits(model: torch.nn.Module, x: np.ndarray, batch: int = 32) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(x), batch):
            out.append(model(torch.from_numpy(x[s : s + batch])).numpy())
    return np.concatenate(out)


def signature(path: Path) -> dict:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    i = sess.get_inputs()[0]
    o = sess.get_outputs()[0]

    def shape(s):
        return [d if isinstance(d, int) else -1 for d in s]

    return {
        "input_names": [x.name for x in sess.get_inputs()],
        "output_names": [x.name for x in sess.get_outputs()],
        "input_shape": shape(i.shape),
        "output_shape": shape(o.shape),
        "dynamic_batch": not isinstance(i.shape[0], int),
        "opset": OPSET,
    }


def main() -> int:
    torch.set_num_threads(3)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    ckpt = CKPT / "best_model.pth"
    if not ckpt.exists():
        print(f"{ckpt} が無い。先に python -m ml.train --stage ship を実行する")
        return 1

    model = build_model()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    rows, splits, arr, index, meta = prepare()

    # 照合は**全画像**で行う。test だけで照合すると、
    # test に無い入力の並びで壊れていても気づけない
    from ml.train import MEAN, STD

    x_all = ((arr.astype(np.float32) / 255.0) - MEAN) / STD
    x_all = np.ascontiguousarray(x_all.transpose(0, 3, 1, 2))

    candidates = {}
    for dynamo in (False, True):
        tag = "dynamo" if dynamo else "torchscript"
        path = PUBLIC / f"model.{tag}.onnx"
        try:
            export(model, path, dynamo)
        except Exception as exc:  # noqa: BLE001 — どちらかが通ればよい
            print(f"  {tag}: 変換に失敗 ({type(exc).__name__}: {exc})")
            continue
        try:
            ref = torch_logits(model, x_all[:64])
            got = run_ort(path, x_all[:64])
            diff = float(np.abs(ref - got).max())
        except Exception as exc:  # noqa: BLE001
            print(f"  {tag}: 実行に失敗 ({type(exc).__name__}: {exc})")
            continue
        mb = path.stat().st_size / 1e6
        candidates[tag] = {"path": path, "megabytes": round(mb, 3), "sample_max_abs_diff": diff}
        print(f"  {tag}: {mb:.3f} MB / 標本 64 枚の最大絶対差 {diff:.3e}")

    if not candidates:
        print("どちらの exporter でも ONNX を作れなかった")
        return 1

    # 選定: まず一致度、次に配信サイズ。**先に基準を書いてから選ぶ**
    chosen_tag = min(candidates, key=lambda t: (candidates[t]["sample_max_abs_diff"] > 1e-4,
                                                candidates[t]["megabytes"]))
    chosen = candidates[chosen_tag]
    print(f"  → {chosen_tag} を採用({chosen['megabytes']} MB)")

    final = PUBLIC / "model.onnx"
    final.write_bytes(chosen["path"].read_bytes())
    for c in candidates.values():
        c["path"].unlink(missing_ok=True)

    # 全画像での照合(G-01)
    #
    # 測る量を二つに分ける(loop_005 で引き直した。SPEC §5.1):
    #   - 生 logits の絶対差は**有界でない量**なので、閾値がモデルの確信度に比例してしまう。
    #     測定値としては残すが、合否には使わない
    #   - 合否は argmax 一致率と softmax 確率の差で見る。確率は [0,1] に収まり、
    #     しかも**利用者が実際に見る量**である
    ref = torch_logits(model, x_all)
    got = run_ort(final, x_all)
    max_abs = float(np.abs(ref - got).max())
    mismatches = int((ref.argmax(1) != got.argmax(1)).sum())
    agreement = float((ref.argmax(1) == got.argmax(1)).mean())

    def softmax(a: np.ndarray) -> np.ndarray:
        e = np.exp(a - a.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    max_prob_diff = float(np.abs(softmax(ref) - softmax(got)).max())
    print(f"  全 {len(x_all)} 枚の照合: argmax 一致率 {agreement:.6f} / "
          f"確率の最大差 {max_prob_diff:.3e}(参考: 生 logits の最大絶対差 {max_abs:.3e})")

    # 出荷モデルの test を ONNX で測り直す(T-244)
    idx_te, y_te = ids_to_arrays(splits["ruler_a"]["test"], index, meta)
    x_te = x_all[idx_te]
    onnx_pred = run_ort(final, x_te).argmax(1)
    torch_pred = predict(model, arr, idx_te)
    shipped = {
        "n_images": len(idx_te),
        "argmax_agreement": float((onnx_pred == torch_pred).mean()),
        "mismatches": int((onnx_pred != torch_pred).sum()),
        "onnx_macro_f1": round(macro_f1(y_te, onnx_pred), 4),
        "torch_macro_f1": round(macro_f1(y_te, torch_pred), 4),
    }
    print(f"  出荷モデルの test を ONNX で測り直し: macro F1 {shipped['onnx_macro_f1']} "
          f"(PyTorch {shipped['torch_macro_f1']})")

    cnn = json.loads((REPORTS / "cnn.json").read_text(encoding="utf-8"))
    controls = json.loads((REPORTS / "controls.json").read_text(encoding="utf-8"))

    parity = {
        "generated_at": datetime.now(JST).isoformat(timespec="seconds"),
        "exporter": chosen_tag,
        "opset": OPSET,
        "onnx_megabytes": round(final.stat().st_size / 1e6, 3),
        "n_images": len(x_all),
        "argmax_agreement": agreement,
        "argmax_mismatches": mismatches,
        "max_prob_diff": max_prob_diff,
        "max_abs_diff_logits": max_abs,
        "signature": signature(final),
        "shipped_test": shipped,
        "candidates": {k: {"megabytes": v["megabytes"],
                           "sample_max_abs_diff": v["sample_max_abs_diff"]}
                       for k, v in candidates.items()},
        "gate": {
            "current": {"argmax_agreement": 1.0, "max_prob_diff": 1e-3,
                        "source": "SPEC 較正ゲート G-01(loop_005 で引き直し)"},
            "superseded": {
                "rule": "生 logits の max abs 差 < 1e-4",
                "measured": max_abs,
                "passed": max_abs < 1e-4,
                "why_replaced": "logits は有界でないため、絶対差の閾値がモデルの確信度に比例する。"
                                "また製品が表示するのは確率であって logits ではない。"
                                "**引き直した事実と、旧い書き方での結果を隠さない**(SPEC 5.1)",
            },
        },
    }
    (REPORTS / "onnx_parity.json").write_text(
        json.dumps(parity, ensure_ascii=False, indent=2), encoding="utf-8")

    (PUBLIC / "labels.json").write_text(json.dumps({
        "classes": CLASSES,
        "display_ja": {"gram_positive": "グラム陽性", "gram_negative": "グラム陰性"},
        "index": {c: i for i, c in enumerate(CLASSES)},
        "note": "並びは Python 側の POSITIVE=0 / NEGATIVE=1 と一致させてある。"
                "食い違うと例外を出さずに結果が反転する",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    labels_doc = json.loads((META / "labels.json").read_text(encoding="utf-8"))
    metadata = {
        "model_version": MODEL_VERSION,
        "architecture": cnn["model"]["architecture"],
        "input_size": cnn["model"]["input_size"],
        "preprocessing": {
            "field_crop": "視野円の内接正方形の中心 512x512 をリサイズなしで切り出したのち 224 へ縮小",
            "normalise": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "note": "四隅の黒(視野外)を入れない。黒の量は撮影ごとに変わるので近道になる",
        },
        "task": "gram-stain-image-classification",
        "classes": CLASSES,
        "dataset": "Bacteria Data for Machine Vision and Digital Biology",
        "dataset_doi": "10.17632/cvkgfzp7ck.1",
        "dataset_license": "CC BY 4.0",
        "attribution": "Jamshidi, Mohammad (Behdad); Sargolzaee, Saleh; Foorginezhad, Salimeh; "
                       "Moztarzadeh, Omid (2023), Bacteria Data for Machine Vision and Digital "
                       "Biology, Mendeley Data, V1, doi: 10.17632/cvkgfzp7ck.1. "
                       "Licensed under CC BY 4.0.",
        "label_authority": {
            "source": labels_doc["authority"]["primary"],
            "license": labels_doc["authority"]["primary_license"],
            "accessed_at": labels_doc["authority"]["accessed_at"],
        },
        "taxa_included": labels_doc["summary"]["taxa_included"],
        "trained_at": cnn["generated_at"],
        "git_commit": git_commit(),
        "epochs": cnn["epoch_budget"]["chosen"],
        "onnx": {"exporter": chosen_tag, "opset": OPSET,
                 "megabytes": parity["onnx_megabytes"],
                 "parity_argmax_agreement": agreement,
                 "parity_max_prob_diff": max_prob_diff,
                 "parity_max_abs_diff_logits": max_abs},
        "metrics": {
            "shipped_test": cnn["shipped_model"]["test"],
            "rulers": {r: {k: cnn["rulers"][r][k] for k in ("macro_f1", "ci_low", "ci_high")}
                       for r in ("a", "b", "c")},
            "hue_baseline": {r: {k: controls["baselines"]["hue"][r][k]
                                 for k in ("macro_f1", "ci_low", "ci_high")}
                             for r in ("a", "b", "c")},
            "label_permutation_control": cnn["permutation_control"]["a"]["macro_f1"],
            "ruler_meaning": {
                "a": "この 32 分類群のスライドを見分けられるか",
                "b": "学習で見ていない種の Gram 反応を当てられるか",
                "c": "学習で見ていない科の Gram 反応を当てられるか",
            },
            "caveat": "出荷モデルの test の数だけを示してはならない。"
                      "同じ骨格・同じ手続きで科をまるごと抜くと macro F1 は "
                      f"{cnn['rulers']['c']['macro_f1']} まで落ちる",
        },
        "not_a_medical_device": "教育・研究目的のデモである。医療機器ではなく、"
                                "出力は医学的診断を意味しない",
    }
    (PUBLIC / "model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n→ {(PUBLIC / 'model.onnx').relative_to(ROOT)} "
          f"({parity['onnx_megabytes']} MB)")
    print(f"→ {(REPORTS / 'onnx_parity.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
