/**
 * ONNX Runtime のセッション管理。
 *
 * 先行プロジェクト(manazashi-lab)が実測で踏んだ癖を、そのまま避けている。
 *
 * **入口は `onnxruntime-web/wasm`。** 既定の入口(`onnxruntime-web`)は WebGPU 込みの
 * jsep 版を取りに行く可能性があり、その wasm は倍近い。控えとして jsep 版も置く、という手は採らない ——
 * 置けば寄せ替えが壊れたときに黙って倍の量が配られる。置かなければ壊れたと分かる。
 *
 * **ORT は module 直下で import してはならない。** `"use client"` を付けても、
 * 静的書き出しでは一度サーバ側で prerender される。ORT の束は評価時に
 * `new URL(...)` を組み立てるが、Node には基準 URL が無く `ERR_INVALID_URL` で落ちる。
 * **ブラウザでしか動かないものは関数の中で取る。**
 *
 * **wasm は自分のオリジンから配る。** 既定は CDN を見に行くため、そのままだと
 * 閲覧時に外部へ通信する。これは無料枠の話であると同時に、
 * 「画像をサーバへ送らない」と言っている以上、通信そのものを減らす話でもある。
 */

import type * as OrtNS from "onnxruntime-web/wasm";

const BASE = "/models";
/** 実行系の入口。束に文字列としてそのまま残るよう、組み立てずに書く */
const ORT_ENTRY = "/ort/ort.wasm.bundle.min.mjs";

let ortPromise: Promise<typeof OrtNS> | null = null;

export function loadOrt(): Promise<typeof OrtNS> {
  if (!ortPromise) {
    // `webpackIgnore` でバンドラに触らせない。webpack に預けると、
    // 自オリジンへ梱包したのとは別に束の側にも wasm が複製される
    ortPromise = (import(/* webpackIgnore: true */ ORT_ENTRY) as Promise<typeof OrtNS>).then((ort) => {
      ort.env.wasm.wasmPaths = "/ort/";
      // 単スレッド固定。SharedArrayBuffer(COOP/COEP)を前提にしない ——
      // 静的ホスティングでヘッダを足せない場合でも同じ数が出る構成にしておく。
      // 速さより「どこでも同じ答え」を採る(G-02)
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = true;
      ort.env.logLevel = "error";
      return ort;
    });
    ortPromise.catch(() => {
      ortPromise = null;
    });
  }
  return ortPromise;
}

/**
 * 配るモデルは二つ(loop_012)。gram は Stage A、shape は Stage B(形)。
 * ファイルを分けてあるので、形を足しても Gram の配信物と照合結果は変わらない
 */
export type ModelName = "gram" | "shape";
const MODEL_FILE: Record<ModelName, string> = { gram: "model.onnx", shape: "model_shape.onnx" };

const sessionPromises: Partial<Record<ModelName, Promise<OrtNS.InferenceSession>>> = {};

/** モデルを取る。5.95 MB を無言で待たせない */
async function fetchModel(model: ModelName, onProgress?: (frac: number) => void): Promise<ArrayBuffer> {
  const res = await fetch(`${BASE}/${MODEL_FILE[model]}`);
  if (!res.ok) throw new Error(`モデルを取得できない (${res.status})`);
  const total = Number(res.headers.get("content-length") ?? 0);
  if (!res.body || !total || !onProgress) return res.arrayBuffer();

  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress(Math.min(1, got / total));
  }
  const out = new Uint8Array(got);
  let at = 0;
  for (const c of chunks) {
    out.set(c, at);
    at += c.length;
  }
  return out.buffer;
}

export function getSession(model: ModelName = "gram", onProgress?: (frac: number) => void): Promise<OrtNS.InferenceSession> {
  let p = sessionPromises[model];
  if (!p) {
    p = (async () => {
      const ort = await loadOrt();
      const buf = await fetchModel(model, onProgress);
      return ort.InferenceSession.create(buf, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
    })();
    sessionPromises[model] = p;
    p.catch(() => {
      delete sessionPromises[model];
    });
  }
  return p;
}

/** 1 枚推論する。入力は前処理済みの CHW float32。前処理は二つのモデルで共通 */
export async function infer(tensor: Float32Array, model: ModelName = "gram"): Promise<Float32Array> {
  const ort = await loadOrt();
  const session = await getSession(model);
  const input = new ort.Tensor("float32", tensor, [1, 3, 224, 224]);
  const out = await session.run({ input });
  const logits = out.logits.data as Float32Array;
  return logits;
}
