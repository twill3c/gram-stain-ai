"use client";

/**
 * 判定画面。
 *
 * **画像はサーバへ送らない**(SPEC F-08)。アップロード API を持たず、
 * 復号も前処理も推論もこのブラウザのメモリの中で完結する。
 *
 * 表示するのは softmax 値だが、**「診断確率」とは呼ばない**(AGENTS.md)。
 * 呼ぶのは「モデル出力スコア」である。閾値未満のときは確信度が低い旨を出す。
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { preprocess } from "@/lib/pipeline";
import { softmax } from "@/lib/preprocess";
import type { Image8 } from "@/lib/resize";
import { infer } from "@/lib/session";

const MAX_BYTES = 10 * 1024 * 1024;
const ACCEPT = ["image/jpeg", "image/png", "image/webp"];

/**
 * 確信度が低いと注記する閾値。
 *
 * **医学的な閾値ではない。** 検証データ(物差し A の val 290 枚)で、
 * このスコア未満に誤りが集まる位置を見て置いた値である。
 * 60% のような切りのよい数を根拠なく採らない(AGENTS.md)。
 */
const LOW_CONFIDENCE = 0.85;

interface Sample {
  id: string;
  file: string;
  folder: string;
  scientific_name: string;
  gram: string;
  shape: string | null;
  why: string;
}

interface SampleIndex {
  samples: Sample[];
  attribution: string;
  selection_rule: string;
}

interface Labels {
  classes: string[];
  display_ja: Record<string, string>;
}

type Status = "idle" | "loading-model" | "running" | "done" | "error";

async function decode(blob: Blob): Promise<Image8> {
  const bitmap = await createImageBitmap(blob);
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("canvas を作れない");
  // 復号だけを canvas にさせる。**縮小はさせない** ——
  // drawImage の補間法は仕様で定められておらず、ブラウザによって画素が変わる。
  // 学習時は PIL で縮小しているので、縮小は移植した resizePIL が行う
  ctx.drawImage(bitmap, 0, 0);
  const data = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  bitmap.close();
  return {
    data: new Uint8Array(data.data.buffer.slice(0)),
    width: data.width,
    height: data.height,
    channels: 4,
  };
}

export default function Classifier() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [attribution, setAttribution] = useState("");
  const [labels, setLabels] = useState<Labels | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [uploaded, setUploaded] = useState(false);
  const [scores, setScores] = useState<number[] | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);
  const objectUrl = useRef<string | null>(null);

  useEffect(() => {
    void (async () => {
      const [si, lb] = await Promise.all([
        fetch("/samples/index.json").then((r) => r.json() as Promise<SampleIndex>),
        fetch("/models/labels.json").then((r) => r.json() as Promise<Labels>),
      ]);
      setSamples(si.samples);
      setAttribution(si.attribution);
      setLabels(lb);
    })().catch(() => setError("サンプルの一覧を読み込めませんでした"));
  }, []);

  // ページ離脱・再読込で画像の状態を破棄する(SPEC 12)
  useEffect(() => {
    return () => {
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    };
  }, []);

  const run = useCallback(async (blob: Blob, url: string, fromUpload: boolean, id: string | null) => {
    setError(null);
    setScores(null);
    setElapsed(null);
    setSelected(id);
    setUploaded(fromUpload);
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    objectUrl.current = fromUpload ? url : null;
    setPreviewUrl(url);
    try {
      setStatus("loading-model");
      const img = await decode(blob);
      const { tensor } = preprocess(img);
      setStatus("running");
      const t0 = performance.now();
      const logits = await infer(tensor);
      setElapsed(performance.now() - t0);
      setScores(softmax(Array.from(logits)));
      setStatus("done");
    } catch (e) {
      setStatus("error");
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const onSample = useCallback(
    async (s: Sample) => {
      const res = await fetch(`/samples/${s.file}`);
      const blob = await res.blob();
      await run(blob, `/samples/${s.file}`, false, s.id);
    },
    [run],
  );

  const onFile = useCallback(
    async (file: File) => {
      if (!ACCEPT.includes(file.type)) {
        setError(`対応していない形式です(${file.type || "不明"})。JPEG / PNG / WebP を選んでください`);
        return;
      }
      if (file.size > MAX_BYTES) {
        setError(`ファイルが大きすぎます(${(file.size / 1e6).toFixed(1)} MB)。10 MB までです`);
        return;
      }
      await run(file, URL.createObjectURL(file), true, null);
    },
    [run],
  );

  const top = scores ? (scores[0] >= scores[1] ? 0 : 1) : null;
  const current = samples.find((s) => s.id === selected) ?? null;

  return (
    <section>
      <div className="card">
        <h2 style={{ marginTop: 0 }}>サンプルを試す</h2>
        <p className="muted" style={{ marginTop: 0 }}>
          グラム陽性 / 陰性、球菌 / 桿菌の両方を含めてあります。
          <strong>うまくいく例だけを並べていません</strong> ——
          色の規則が最もよく破れる <i>Listeria monocytogenes</i> も入れてあります。
        </p>
        <div className="samples">
          {samples.map((s) => (
            <button
              key={s.id}
              type="button"
              className={`sample-btn${selected === s.id ? " selected" : ""}`}
              onClick={() => void onSample(s)}
            >
              <img src={`/samples/${s.file}`} alt={s.folder} loading="lazy" />
              <span className="sample-cap">{s.folder}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="card">
        <h2 style={{ marginTop: 0 }}>手元の画像を試す</h2>
        <input
          type="file"
          accept={ACCEPT.join(",")}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void onFile(f);
          }}
        />
        <p className="muted">
          JPEG / PNG / WebP・10 MB まで。<strong>画像はサーバへ送られません</strong> ——
          アップロードの経路そのものがありません。復号も推論もこのブラウザの中で終わります。
        </p>
        <p className="muted">
          学習に使ったのは 100 倍対物で撮った視野の中心を切り出した画像です。
          倍率や写し方が違う画像では、当てになりません。
        </p>
      </div>

      {previewUrl && (
        <div className="card">
          <h2 style={{ marginTop: 0 }}>結果</h2>
          <img className="preview" src={previewUrl} alt="判定対象" />

          {current && (
            <p className="muted" style={{ marginTop: 10 }}>
              {/* data-taxon は実ブラウザ検品の目印。
                  曖昧な選択子(最初の <i> など)で拾うと、本文中の学名を掴んでしまう */}
              <i data-taxon>{current.scientific_name}</i> — {current.why}
            </p>
          )}
          {uploaded && (
            <p className="muted" style={{ marginTop: 10 }}>
              手元の画像です。中心の正方形を切り出してから 224×224 へ縮小しています。
            </p>
          )}

          {status === "loading-model" && (
            <p className="muted">モデルを読み込んでいます{progress ? `(${Math.round(progress * 100)}%)` : ""}…</p>
          )}
          {status === "running" && <p className="muted">推論中…</p>}
          {error && <p style={{ color: "var(--pink)" }}>{error}</p>}

          {scores && labels && (
            <>
              <div style={{ marginTop: 14 }}>
                {labels.classes.map((cls, i) => (
                  <div className="score-row" key={cls}>
                    <span className="score-label">{labels.display_ja[cls] ?? cls}</span>
                    <span className="score-bar">
                      <span
                        className={`score-fill ${i === 0 ? "positive" : "negative"}`}
                        style={{ width: `${(scores[i] * 100).toFixed(1)}%` }}
                      />
                    </span>
                    <span className="score-num">{(scores[i] * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
              <p className="muted" style={{ marginTop: 4 }}>
                これは<strong>モデル出力スコア</strong>であって、診断確率ではありません。
                {elapsed !== null && ` 推論 ${elapsed.toFixed(0)} ms。`}
              </p>
              {top !== null && scores[top] < LOW_CONFIDENCE && (
                <div className="low-confidence">
                  モデルの確信度が低い結果です。教育用の参考表示として扱ってください。
                  (この閾値 {Math.round(LOW_CONFIDENCE * 100)}% は検証データで置いた値で、
                  医学的な閾値ではありません)
                </div>
              )}
            </>
          )}
        </div>
      )}

      {attribution && (
        <p className="muted">
          サンプル画像の出所: {attribution}
        </p>
      )}
    </section>
  );
}
