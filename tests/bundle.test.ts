/**
 * 出荷物そのものに対する検査(TEST_SPEC T-249〜T-254)。
 *
 * 期待値の出所: SPEC の N-01 / N-02 / N-08 と F-07。
 *
 * **ここで見るのはコードではなく `out/` の中身である。** 実装が正しくても
 * 梱包が壊れれば配られるものは変わる。先行プロジェクト(manazashi-lab)は、
 * バンドラに預けたせいで wasm が束の側にも複製され、13 MB が二重に入る事故を踏んでいる。
 *
 * `npx next build` を先に走らせておくこと。
 */

import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const OUT = join(process.cwd(), "out");
const has = existsSync(OUT);

function walk(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p, out);
    else out.push(p);
  }
  return out;
}

describe.skipIf(!has)("出荷物", () => {
  const files = has ? walk(OUT) : [];
  const rel = (p: string) => p.slice(OUT.length + 1).replaceAll("\\", "/");

  it("T-249 束の側に wasm が混入していない", () => {
    // webpackIgnore を外すと、自オリジンへ梱包したのとは別に束の側にも wasm が複製される。
    // 転送はされないが、置いてあること自体が「寄せ替えが壊れても気づけない」状態を作る
    const stray = files.filter((p) => rel(p).startsWith("_next/") && p.endsWith(".wasm"));
    expect(stray.map(rel)).toEqual([]);
  });

  it("T-250 実行系を自オリジンから名指ししている", () => {
    // 経路を組み立てずに定数で書いてあるので、束に文字列としてそのまま残る。
    // 残っていなければ寄せ替えが効いていない(既定の CDN を見に行く)
    const js = files.filter((p) => p.endsWith(".js"));
    const hit = js.some((p) => readFileSync(p, "utf-8").includes("/ort/ort.wasm.bundle.min.mjs"));
    expect(hit, "束の中に自オリジンの ORT 入口が見つからない").toBe(true);
  });

  it("T-251 サーバ関数を持たない", () => {
    // 静的書き出しなので、あってはならないものが無いことを確かめる
    for (const bad of ["_next/server", "api"]) {
      expect(existsSync(join(OUT, bad)), `${bad} がある`).toBe(false);
    }
    const html = files.filter((p) => p.endsWith(".html"));
    expect(html.length).toBeGreaterThanOrEqual(4);
  });

  it("T-252 全ページに医療用途でない旨が出ている", () => {
    const html = files.filter((p) => p.endsWith(".html") && !rel(p).startsWith("404"));
    expect(html.length).toBeGreaterThan(0);
    for (const p of html) {
      const s = readFileSync(p, "utf-8");
      expect(s.includes("医療用途ではありません"), `${rel(p)} に明示が無い`).toBe(true);
      expect(s.includes("医療機器ではなく"), `${rel(p)} に明示が無い`).toBe(true);
    }
  });

  it("T-253 フッタが規約どおり 5 項目ある", () => {
    const p = join(OUT, "index.html");
    const s = readFileSync(p, "utf-8");
    for (const item of ["MIT License", "GitHub", "測り方", "設計図", "App Menu"]) {
      expect(s.includes(item), `フッタに ${item} が無い`).toBe(true);
    }
  });

  it("T-254 配信物が揃っている", () => {
    for (const f of [
      "models/model.onnx",
      "models/labels.json",
      "models/model_metadata.json",
      "ort/ort.wasm.bundle.min.mjs",
      "ort/ort-wasm-simd-threaded.wasm",
      "samples/index.json",
    ]) {
      expect(existsSync(join(OUT, f)), `${f} が無い`).toBe(true);
    }
    const onnx = statSync(join(OUT, "models/model.onnx")).size / 1e6;
    expect(onnx, `model.onnx が ${onnx.toFixed(2)} MB`).toBeLessThanOrEqual(30);
  });

  it("T-255 画面に出る数が、配信するメタデータと同じ出どころから来ている", () => {
    // 数を画面に直接書くと、測り直したときに文書だけが古いまま残る。
    // 配信メタデータの値が、生成された HTML にそのまま現れることを確かめる
    const meta = JSON.parse(readFileSync(join(OUT, "models/model_metadata.json"), "utf-8"));
    const html = readFileSync(join(OUT, "monosashi/index.html"), "utf-8");
    for (const r of ["a", "b", "c"] as const) {
      const v = meta.metrics.rulers[r].macro_f1.toFixed(4);
      expect(html.includes(v), `物差し ${r} の ${v} が画面に無い`).toBe(true);
    }
  });

  it("T-277 形の配信物があり、二つの ONNX の合計が 30 MB 以内", () => {
    for (const f of ["models/model_shape.onnx", "models/labels_shape.json", "models/model_shape_metadata.json"]) {
      expect(existsSync(join(OUT, f)), `${f} が無い`).toBe(true);
    }
    const total = ["models/model.onnx", "models/model_shape.onnx"]
      .map((f) => statSync(join(OUT, f)).size / 1e6)
      .reduce((a, b) => a + b, 0);
    expect(total, `二つの ONNX の合計が ${total.toFixed(2)} MB`).toBeLessThanOrEqual(30);
  });

  it("T-278 三つの物差しのページの Stage B の数が、形の配信メタデータから来ている", () => {
    // 陰性球菌の崩れは、数だけ README に書いて画面に出さない、ということをしない
    const meta = JSON.parse(readFileSync(join(OUT, "models/model_shape_metadata.json"), "utf-8"));
    const html = readFileSync(join(OUT, "monosashi/index.html"), "utf-8");
    for (const r of ["a", "b", "c"] as const) {
      const v = meta.metrics.rulers[r].macro_f1.toFixed(4);
      expect(html.includes(v), `形の物差し ${r} の ${v} が画面に無い`).toBe(true);
    }
    const pct = `${Math.round(meta.negative_cocci.error_rate.c * 100)}%`;
    expect(html.includes(pct), `陰性球菌の C の誤り率 ${pct} が画面に無い`).toBe(true);
  });
});
