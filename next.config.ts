import type { NextConfig } from "next";

// 静的書き出しのみ。サーバ関数を一つも持たない(SPEC N-02)。
// モデルと ONNX Runtime の wasm は public/ から同一オリジンで配る。
//
// **next build はモデルを落としに行かない。** 行けば Vercel のビルドで外部へ取りに出ることになり、
// ゼロ課金の前提が静かに壊れる。配布物はリポジトリに置いてある(先例: manazashi-lab)。
const nextConfig: NextConfig = {
  output: "export",
  reactStrictMode: true,
  trailingSlash: true,
};

export default nextConfig;
