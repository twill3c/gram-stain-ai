// フリート共通フッタの行き先(HC-098: 「どれかのリンクが github.com を向いている」では足りない)。
// 文言が規約で固定された項目は、**その項目の行き先**を見る。
// 実装より先に書いた(loop_010)。GitHub 公開前に置いた仮の https://github.com/ が残っていた。
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const layout = readFileSync(join(__dirname, "..", "src", "app", "layout.tsx"), "utf-8");
const REPO = "https://github.com/twill3c/gram-stain-ai";

function hrefOf(label: string): string | null {
  const m = layout.match(new RegExp(`<a href="([^"]+)"[^>]*>${label}</a>`));
  return m ? m[1] : null;
}

describe("フッタの行き先", () => {
  it("MIT License はこのリポジトリの LICENSE を指す", () => {
    expect(hrefOf("MIT License")).toBe(`${REPO}/blob/main/LICENSE`);
  });
  it("GitHub はこのリポジトリを指す", () => {
    expect(hrefOf("GitHub")).toBe(REPO);
  });
  it("App Menu は app-menu-amber を指す(app-menu.vercel.app は他者のサイト)", () => {
    expect(hrefOf("App Menu")).toBe("https://app-menu-amber.vercel.app/");
  });
  it("5 項目がこの並びで出る", () => {
    const order = ["MIT License", "© 2026 坂田哲朗", ">GitHub<", "Gram Stain AI の測り方", ">設計図<", ">App Menu<"];
    const at = order.map((s) => layout.indexOf(s));
    expect(at.every((i) => i >= 0)).toBe(true);
    expect([...at].sort((a, b) => a - b)).toEqual(at);
  });
});
