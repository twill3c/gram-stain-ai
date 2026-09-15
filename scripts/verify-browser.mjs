// 実ブラウザでの照合(SPEC 較正ゲート G-02)。
//
// **これは「検査が緑でも動かない」を捕まえるための工程である。**
// vitest の照合(tests/preprocess.test.ts)は Node の中で前処理だけを突き合わせている。
// 実ブラウザでは PNG の復号も ONNX Runtime Web も別の実装が担うので、そこは覆えない。
//
// ここでやるのは、**利用者と同じ経路**を Playwright に歩かせることである:
//   サンプルを押す → PNG をブラウザが復号 → 移植した resizePIL → ORT Web → softmax → 画面の%
// 出た%を、Python の ONNX Runtime が出した logits の softmax と突き合わせる。
//
// 画面は小数 1 桁で表示するので、許容差は 0.05 ポイント。粗く見えるが、
// 前処理やモデルが実際に食い違えば%はこの程度では済まない。
//
// 実行: npx next build && node scripts/verify-browser.mjs

import { createReadStream, existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize } from "node:path";
import { chromium } from "playwright";

const OUT = "out";
const PORT = 4319;

// 本番を検品するときは `node scripts/verify-browser.mjs https://gram-stain-ai.vercel.app`。
// **ローカルのビルドを検品しても、本番の検品にはならない。**
// 配信の設定・圧縮・ヘッダは本番にしか無く、そこで壊れることがある
const TARGET = process.argv[2] ?? null;
const BASE = TARGET ?? `http://localhost:${PORT}`;

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".png": "image/png",
  ".wasm": "application/wasm",
  ".svg": "image/svg+xml",
  ".txt": "text/plain",
};

function serve() {
  return new Promise((resolve) => {
    const server = createServer((req, res) => {
      const url = decodeURIComponent((req.url ?? "/").split("?")[0]);
      let p = normalize(join(OUT, url));
      if (!p.startsWith(normalize(OUT))) {
        res.writeHead(403).end();
        return;
      }
      if (existsSync(p) && readdirSync(OUT) && !extname(p)) p = join(p, "index.html");
      if (!existsSync(p)) {
        res.writeHead(404).end("not found");
        return;
      }
      res.writeHead(200, { "content-type": MIME[extname(p)] ?? "application/octet-stream" });
      createReadStream(p).pipe(res);
    });
    server.listen(PORT, () => resolve(server));
  });
}

function softmax(a) {
  const m = Math.max(...a);
  const e = a.map((x) => Math.exp(x - m));
  const s = e.reduce((x, y) => x + y, 0);
  return e.map((x) => x / s);
}

function readLogits(id, kind = "logits") {
  // kind: "logits"(Gram)/ "shape_logits"(形・loop_012)
  const buf = readFileSync(join("tests", "fixtures", "expect", `${id}.${kind}.f32`));
  return Array.from(new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4));
}

const TOLERANCE_POINTS = 0.05;

const server = TARGET ? null : await serve();
console.log(`検品先: ${BASE}${TARGET ? "(本番)" : "(ローカルの out/)"}`);
const browser = await chromium.launch();
const page = await browser.newPage();

const failures = [];
const results = [];
const timings = [];
let externalRequests = [];

// 外部への通信が一切無いことも同時に見る(SPEC N-02 / F-08)。
// wasm を CDN から取りに行っていたら、ここで露見する
page.on("request", (r) => {
  const u = r.url();
  if (!u.startsWith(BASE) && !u.startsWith("data:") && !u.startsWith("blob:")) {
    externalRequests.push(u);
  }
});

const samples = TARGET
  ? (await (await fetch(`${BASE}/samples/index.json`)).json()).samples
  : JSON.parse(readFileSync(join(OUT, "samples", "index.json"), "utf-8")).samples;

let firstMs = null;

// 形の注意書きに入っているはずの数(T-279)。**画面に直接書かれていないことを、配信物から確かめる**
const shapeMeta = TARGET
  ? await (await fetch(`${BASE}/models/model_shape_metadata.json`)).json()
  : JSON.parse(readFileSync(join(OUT, "models", "model_shape_metadata.json"), "utf-8"));
const caveatPct = `${Math.round(shapeMeta.negative_cocci.error_rate.c * 100)}%`;

try {
  await page.goto(`${BASE}/`, { waitUntil: "networkidle" });

  for (const s of samples) {
    const t0 = Date.now();
    const want = softmax(readLogits(s.id)).map((v) => v * 100);
    const wantShape = softmax(readLogits(s.id, "shape_logits")).map((v) => v * 100);

    await page.locator(`button.sample-btn:has(img[alt="${s.folder}"])`).click();
    // **押した直後は、直前のサンプルの得点がまだ DOM に残っている。**
    // 得点の出現だけを待つと、待ちが即座に成立して古い値を読む(loop_006 で踏んだ VERIF-FLAKE)。
    // 押したサンプルの学名が出ていることと、Gram と形の両方の得点が揃っていることを待つ。
    // loop_012 で形の塊が増えた。**Gram の得点だけで待つと、形の得点が前のサンプルのまま読まれる**
    await page.waitForFunction(
      (name) => {
        const shown = document.querySelector("i[data-taxon]")?.textContent?.trim();
        return shown === name
          && document.querySelectorAll('[data-model="gram"] .score-num').length >= 2
          && document.querySelectorAll('[data-model="shape"] .score-num').length >= 2
          && document.querySelector('[data-model="shape"]')?.getAttribute("data-for") === name;
      },
      s.scientific_name,
      { timeout: 120_000, polling: 50 },
    );

    // 1 枚目はモデルの取得と初期化を含む(N-01 の「モデルロード」)。
    // 2 枚目以降はセッションが使い回されるので、純粋な推論時間に近い
    const wall = Date.now() - t0;
    if (firstMs === null) firstMs = wall;

    const read = async (sel) => (await page.locator(sel).allTextContents())
      .map((t) => Number.parseFloat(t.replace("%", "")));
    const got = await read('[data-model="gram"] .score-num');
    const gotShape = await read('[data-model="shape"] .score-num');
    timings.push({ folder: s.folder, wall_ms: wall });

    const worst = Math.max(...got.map((g, i) => Math.abs(g - want[i])));
    const worstShape = Math.max(...gotShape.map((g, i) => Math.abs(g - wantShape[i])));
    const caveat = (await page.locator('[data-caveat="negative-cocci"]').allTextContents()).join(" ");
    const caveatOk = caveat.includes(caveatPct);
    const ok = worst <= TOLERANCE_POINTS && worstShape <= TOLERANCE_POINTS && caveatOk;
    results.push({
      folder: s.folder,
      want: want.map((v) => v.toFixed(1)), got, worst,
      want_shape: wantShape.map((v) => v.toFixed(1)), got_shape: gotShape, worst_shape: worstShape,
      caveat_shown: caveatOk,
    });
    if (worst > TOLERANCE_POINTS) failures.push(`${s.folder}: Gram 画面 ${got.join("/")} 対 Python ${want.map((v) => v.toFixed(1)).join("/")}`);
    if (worstShape > TOLERANCE_POINTS) failures.push(`${s.folder}: 形 画面 ${gotShape.join("/")} 対 Python ${wantShape.map((v) => v.toFixed(1)).join("/")}`);
    if (!caveatOk) failures.push(`${s.folder}: 陰性球菌の注意書き(${caveatPct} を含む)が出ていない`);
    console.log(
      `  ${ok ? "✓" : "✗"} ${s.folder.padEnd(30)} Gram ${got.map((v) => v.toFixed(1)).join(" / ")}(差 ${worst.toFixed(3)})  `
      + `形 ${gotShape.map((v) => v.toFixed(1)).join(" / ")}(差 ${worstShape.toFixed(3)})  注意書き ${caveatOk ? "有" : "無"}`,
    );
  }
} finally {
  await browser.close();
  if (server) server.close();
}

// N-01 の実測。**目標を書いておいて測らないのでは意味がない**
const rest = timings.slice(1).map((t) => t.wall_ms);
const restMax = rest.length ? Math.max(...rest) : 0;
console.log();
console.log(`1 枚目(モデル取得と初期化を含む): ${firstMs} ms  ——  目標 5000 ms 以内`);
console.log(`2 枚目以降の最大: ${restMax} ms(中央 ${rest.length ? rest.slice().sort((a, b) => a - b)[Math.floor(rest.length / 2)] : 0} ms)  ——  目標 2000 ms 以内`);
if (firstMs > 5000) failures.push(`1 枚目が ${firstMs} ms(目標 5000 ms)`);
if (restMax > 2000) failures.push(`2 枚目以降の最大が ${restMax} ms(目標 2000 ms)`);

writeFileSync(
  TARGET ? "reports/browser_verify_production.json" : "reports/browser_verify.json",
  JSON.stringify(
    {
      generated_at: new Date().toISOString(),
      target: BASE,
      tolerance_points: TOLERANCE_POINTS,
      samples: results,
      timings: { first_ms: firstMs, rest_max_ms: restMax, per_sample: timings },
      external_requests: [...new Set(externalRequests)],
      passed: failures.length === 0,
    },
    null,
    2,
  ) + "\n",
);

console.log();
if (externalRequests.length) {
  console.log(`!! 外部への通信が ${externalRequests.length} 件あった:`);
  for (const u of [...new Set(externalRequests)].slice(0, 5)) console.log(`   ${u}`);
  failures.push(`外部通信 ${externalRequests.length} 件`);
} else {
  console.log("外部への通信: 0 件(wasm もモデルも自オリジンから配られている)");
}

if (failures.length) {
  console.log(`\nG-02 不通過 — ${failures.length} 件`);
  for (const f of failures) console.log(`  ${f}`);
  process.exit(1);
}
console.log(`G-02 通過 — サンプル ${samples.length} 件すべてで、`
  + `実ブラウザの表示が Python の ONNX Runtime と ${TOLERANCE_POINTS} ポイント以内で一致`);
