// スマホでの押しやすさの検品。**実装より先に書いた**(loop_010)。
//
// なぜ要るのか: mobile:true の根拠を「横にはみ出さない」だけで測っていた。
// フリートの physics-puzzle-lab は、表示は崩れていないのにタッチで板を回せないまま
// mobile:true だった。表示が崩れないことと、指で操作できることは別の性質である。
//
// 見るもの(iPhone 13 相当・タッチ有効):
//   1. 操作の入口(サンプル・ファイル選択・見出しのナビ)のタップ領域が 44px 以上
//      —— Apple Human Interface Guidelines の最小タップ領域 44pt
//   2. サンプルをタップすると、**押したサンプルの**結果が出る
//      (直前の表示で待ちを成立させない —— この検品の前版はそこで誤った)
//   3. どのページも横にはみ出さない
// フリート共通の固定フッタのリンクは規約で決まった寸法なので、数えるが合否には入れない。
//
// 実行: node scripts/verify-mobile.mjs [https://gram-stain-ai.vercel.app]

import { createReadStream, existsSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize } from "node:path";
import { chromium, devices } from "playwright";

const TARGET = process.argv[2] ?? null;
const PORT = 4320;
const OUT = "out";
const MIN_TAP = 44;

const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript",
  ".css": "text/css", ".json": "application/json", ".png": "image/png", ".wasm": "application/wasm" };

function serve() {
  return new Promise((resolve) => {
    const server = createServer((req, res) => {
      let p = normalize(join(OUT, decodeURIComponent((req.url ?? "/").split("?")[0])));
      if (existsSync(p) && !extname(p)) p = join(p, "index.html");
      if (!existsSync(p)) { res.writeHead(404).end(); return; }
      res.writeHead(200, { "content-type": MIME[extname(p)] ?? "application/octet-stream" });
      createReadStream(p).pipe(res);
    });
    server.listen(PORT, () => resolve(server));
  });
}

const server = TARGET ? null : await serve();
const BASE = TARGET ?? `http://localhost:${PORT}`;
console.log(`検品先: ${BASE}${TARGET ? "(本番)" : "(ローカルの out/)"}`);

const browser = await chromium.launch();
const ctx = await browser.newContext({ ...devices["iPhone 13"] });
const page = await ctx.newPage();
const failures = [];

try {
  // 1. タップ領域
  await page.goto(`${BASE}/`, { waitUntil: "networkidle" });
  const targets = await page.evaluate(() => {
    const pick = (sel, label) => [...document.querySelectorAll(sel)].map((el) => {
      // ファイル選択は、見た目の入口がラベルに包まれていればラベルの矩形で測る
      const box = (el.closest("label") ?? el).getBoundingClientRect();
      return { label, text: (el.textContent || el.getAttribute("aria-label") || el.type || "").trim().slice(0, 24),
        w: Math.round(box.width), h: Math.round(box.height) };
    });
    return [
      ...pick('input[type="file"]', "ファイル選択"),
      ...pick(".masthead nav a", "ナビ"),
      ...pick("button.sample-btn", "サンプル"),
      ...pick(".site-footer a", "フッタ(参考)"),
    ];
  });
  for (const t of targets) {
    const small = t.h < MIN_TAP || t.w < MIN_TAP;
    const counted = t.label !== "フッタ(参考)";
    if (small && counted) failures.push(`${t.label}「${t.text}」が ${t.w}x${t.h}px`);
    if (small || t.label === "ファイル選択") {
      console.log(`${small ? (counted ? "✗" : "△") : "✓"} ${t.label.padEnd(8)} ${t.text.padEnd(24)} ${t.w}x${t.h}px`);
    }
  }
  const okCount = targets.filter((t) => t.label !== "フッタ(参考)" && t.h >= MIN_TAP && t.w >= MIN_TAP).length;
  console.log(`タップ領域 ${MIN_TAP}px 以上: ${okCount} / ${targets.filter((t) => t.label !== "フッタ(参考)").length}`);

  // 2. 押したサンプルの結果が出る
  const samples = TARGET
    ? (await (await fetch(`${BASE}/samples/index.json`)).json()).samples
    : JSON.parse((await import("node:fs")).readFileSync(join(OUT, "samples", "index.json"), "utf-8")).samples;
  for (const s of [samples[0], samples[samples.length - 1], samples[samples.length - 2]]) {
    const btn = page.locator(`button.sample-btn:has(img[alt="${s.folder}"])`);
    await btn.scrollIntoViewIfNeeded();
    await btn.tap();
    try {
      await page.waitForFunction((name) => document.querySelector("i[data-taxon]")?.textContent?.trim() === name
        && document.querySelectorAll(".score-num").length >= 2, s.scientific_name, { timeout: 60000, polling: 50 });
      console.log(`✓ tap ${s.folder}`);
    } catch {
      failures.push(`tap ${s.folder} で押したサンプルの結果が出ない`);
    }
  }

  // 3. 横はみ出し
  for (const p of ["/", "/gram/", "/monosashi/", "/model/"]) {
    await page.goto(`${BASE}${p}`, { waitUntil: "networkidle" });
    const over = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    if (over > 0) failures.push(`${p} が横に ${over}px はみ出す`);
  }
} finally {
  await browser.close();
  if (server) server.close();
}

if (failures.length) {
  console.log(`\nスマホ検品 不通過 — ${failures.length} 件`);
  for (const f of failures) console.log(`  ${f}`);
  process.exit(1);
}
console.log("\nスマホ検品 通過 — 操作の入口はすべて 44px 以上・押したサンプルの結果が出る・横はみ出し 0");
