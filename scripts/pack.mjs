// ONNX Runtime の実行系を node_modules から public/ort へ複製する。
//
// **複製であって変換ではない。** sha256 を突き合わせて、梱包器がなにも触っていないことを確かめる。
//
// wasm 専用の版だけを置く。WebGPU 込みの jsep 版は倍近くあり、控えとしても置かない ——
// 置けば寄せ替えが壊れたときに**黙って倍の量が配られる**。置かなければ壊れたと分かる。
//
// 実行: node scripts/pack.mjs

import { createHash } from "node:crypto";
import { copyFileSync, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const OUT = "public/ort";
const SRC = "node_modules/onnxruntime-web/dist";
// アプリからは import(/* webpackIgnore: true */ "/ort/ort.wasm.bundle.min.mjs") で実行時に取る
const FILES = [
  "ort.wasm.bundle.min.mjs",
  "ort-wasm-simd-threaded.wasm",
  "ort-wasm-simd-threaded.mjs",
];

const sha256 = (p) => createHash("sha256").update(readFileSync(p)).digest("hex");
const mb = (n) => (n / 1048576).toFixed(2);

mkdirSync(OUT, { recursive: true });

const manifest = { packed_at: new Date().toISOString(), files: {} };
let total = 0;

for (const f of FILES) {
  const src = join(SRC, f);
  if (!existsSync(src)) {
    throw new Error(`${src} が無い — onnxruntime-web の版でファイル名が変わった可能性がある`);
  }
  const dst = join(OUT, f);
  copyFileSync(src, dst);
  const a = sha256(src);
  const b = sha256(dst);
  if (a !== b) throw new Error(`${f} の複製が原本と一致しない`);
  const n = statSync(dst).size;
  total += n;
  manifest.files[f] = { bytes: n, sha256: a };
  console.log(`  ort/${f}  ${mb(n)} MB`);
}

const pkg = JSON.parse(readFileSync("node_modules/onnxruntime-web/package.json", "utf-8"));
manifest.onnxruntime_web_version = pkg.version;
writeFileSync(join(OUT, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");

console.log(`\n実行系 → ${OUT}  計 ${mb(total)} MB(生)  onnxruntime-web ${pkg.version}`);
