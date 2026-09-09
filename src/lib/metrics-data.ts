/**
 * 画面に出す数の出どころ。**配信するメタデータ 1 か所から読む。**
 *
 * 画面に数を直接書くと、測り直したときに文書だけが古いまま残る。
 * ここで読むのは `public/models/model_metadata.json` —— 配信物そのものである。
 * 「画面に出ている数」と「配っているモデルに付いている数」が食い違わない。
 *
 * 静的書き出しなので、この読み込みはビルド時に一度だけ走る。
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";

export interface RulerScore {
  macro_f1: number;
  ci_low: number;
  ci_high: number;
}

export interface ModelMetadata {
  model_version: string;
  architecture: string;
  input_size: [number, number];
  classes: string[];
  dataset: string;
  dataset_doi: string;
  dataset_license: string;
  attribution: string;
  label_authority: { source: string; license: string; accessed_at: string };
  taxa_included: number;
  trained_at: string;
  git_commit: string;
  epochs: number;
  onnx: {
    exporter: string;
    opset: number;
    megabytes: number;
    parity_argmax_agreement: number;
    parity_max_prob_diff: number;
    parity_max_abs_diff_logits: number;
  };
  metrics: {
    shipped_test: { macro_f1: number; accuracy: number; n_images: number };
    rulers: Record<"a" | "b" | "c", RulerScore>;
    hue_baseline: Record<"a" | "b" | "c", RulerScore>;
    label_permutation_control: number;
    ruler_meaning: Record<"a" | "b" | "c", string>;
    caveat: string;
  };
  not_a_medical_device: string;
}

export function loadMetadata(): ModelMetadata {
  const p = join(process.cwd(), "public", "models", "model_metadata.json");
  return JSON.parse(readFileSync(p, "utf-8")) as ModelMetadata;
}

export const RULER_TITLE: Record<"a" | "b" | "c", string> = {
  a: "A 画像単位",
  b: "B 分類群ホールドアウト",
  c: "C 科ホールドアウト",
};

export function fmt(s: RulerScore): string {
  return `${s.macro_f1.toFixed(4)} [${s.ci_low.toFixed(4)}, ${s.ci_high.toFixed(4)}]`;
}
