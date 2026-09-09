/**
 * ブラウザ側の前処理が Python 側と同じ数を作ることの照合(TEST_SPEC T-246〜T-248)。
 *
 * 期待値の出所:
 *   `tests/fixtures/expect/<id>.input.f32` — ml/build_samples.py が PIL + NumPy で作った
 *   224x224x3(CHW・float32 little endian)の正規化済みテンソル。2026-09-10 生成。
 *
 * **これは非循環のオラクルである。** 期待値をこちらで書いているのではなく、
 * 学習時に実際に使った経路(PIL)が出した数と突き合わせている。
 * 食い違えば、ブラウザは学習時と違う画像をモデルに見せていることになる。
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { preprocess } from "@/lib/pipeline";
import { softmax } from "@/lib/preprocess";
import type { Image8 } from "@/lib/resize";
import { decodePNG } from "./helpers/png";

const SAMPLES = join(process.cwd(), "public", "samples");
const EXPECT = join(process.cwd(), "tests", "fixtures", "expect");

interface SampleIndex {
  samples: { id: string; file: string; gram: string; folder: string }[];
}

const index: SampleIndex = JSON.parse(readFileSync(join(SAMPLES, "index.json"), "utf-8"));

function readF32(path: string): Float32Array {
  const buf = readFileSync(path);
  return new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
}

describe("前処理の二実装照合(G-02)", () => {
  it("T-246 サンプルが 1 枚以上ある", () => {
    expect(index.samples.length).toBeGreaterThan(0);
  });

  for (const s of index.samples) {
    it(`T-247 ${s.folder} のテンソルが Python と一致する`, () => {
      // 配布物と同じ PNG から始める。生の RGBA を期待値として持つと
      // 同じ中身が二重にリポジトリへ残る(10.5 MB)。復号は tests/helpers/png.ts が行う
      const png = decodePNG(readFileSync(join(SAMPLES, s.file)));
      const img: Image8 = {
        data: new Uint8Array(png.rgba.buffer.slice(0)),
        width: png.width,
        height: png.height,
        channels: 4,
      };
      const { tensor } = preprocess(img);
      const want = readF32(join(EXPECT, `${s.id}.input.f32`));

      expect(tensor.length).toBe(want.length);
      let max = 0;
      for (let i = 0; i < want.length; i++) {
        const d = Math.abs(tensor[i] - want[i]);
        if (d > max) max = d;
      }
      // 正規化後の値は概ね [-2.2, 2.7] に収まる。1e-5 は uint8 の 1 段(約 0.004)より
      // はるかに細かく、**同じ画素値から同じ式で計算していれば必ず満たす**
      expect(max, `最大差 ${max.toExponential(3)}`).toBeLessThan(1e-5);
    });
  }
});

describe("softmax", () => {
  it("T-248 合計が 1 になり、大小関係が logits と同じ", () => {
    for (const logits of [[4.14, -4.428], [-2.921, 3.041], [0, 0], [1e3, -1e3]]) {
      const p = softmax(logits);
      expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
      expect(p[0] > p[1]).toBe(logits[0] > logits[1]);
      for (const v of p) {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
      }
    }
  });

  it("T-248 大きな logits でも溢れない", () => {
    const p = softmax([1000, 999]);
    expect(Number.isFinite(p[0])).toBe(true);
    expect(p[0] + p[1]).toBeCloseTo(1, 12);
  });
});
