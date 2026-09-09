/**
 * 検査用の最小 PNG 復号器。
 *
 * なぜ持つのか: 期待値として生の RGBA を配ると 10.5 MB の**冗長な**バイナリが
 * リポジトリに残る(同じ中身の PNG は既に配布物として追跡している)。
 * ブラウザは PNG を自分で復号するので、検査側にも復号が要るだけである。
 *
 * 対応するのは PIL が既定で書く形だけ — 8 bit・RGBA(色型 6)または RGB(色型 2)・
 * 非インタレース。それ以外は落とす。**黙って別の絵を返さない**ことのほうが大事である。
 */

import { inflateSync } from "node:zlib";

export interface Decoded {
  rgba: Uint8ClampedArray;
  width: number;
  height: number;
}

function paeth(a: number, b: number, c: number): number {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

export function decodePNG(buf: Buffer): Decoded {
  const sig = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  for (let i = 0; i < 8; i++) {
    if (buf[i] !== sig[i]) throw new Error("PNG の署名が合わない");
  }

  let width = 0;
  let height = 0;
  let bitDepth = 0;
  let colourType = 0;
  const idat: Buffer[] = [];

  let at = 8;
  while (at < buf.length) {
    const len = buf.readUInt32BE(at);
    const type = buf.toString("ascii", at + 4, at + 8);
    const data = buf.subarray(at + 8, at + 8 + len);
    at += 12 + len;
    if (type === "IHDR") {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      bitDepth = data[8];
      colourType = data[9];
      if (bitDepth !== 8) throw new Error(`対応しない bit 深度: ${bitDepth}`);
      if (colourType !== 6 && colourType !== 2) {
        throw new Error(`対応しない色型: ${colourType}`);
      }
      if (data[12] !== 0) throw new Error("インタレースには対応しない");
    } else if (type === "IDAT") {
      idat.push(Buffer.from(data));
    } else if (type === "IEND") {
      break;
    }
  }

  const channels = colourType === 6 ? 4 : 3;
  const raw = inflateSync(Buffer.concat(idat));
  const stride = width * channels;
  const out = new Uint8ClampedArray(width * height * 4);
  const prev = new Uint8Array(stride);
  const line = new Uint8Array(stride);

  let p = 0;
  for (let y = 0; y < height; y++) {
    const filter = raw[p++];
    for (let i = 0; i < stride; i++) line[i] = raw[p + i];
    p += stride;

    for (let i = 0; i < stride; i++) {
      const a = i >= channels ? line[i - channels] : 0;
      const b = prev[i];
      const c = i >= channels ? prev[i - channels] : 0;
      switch (filter) {
        case 0: break;
        case 1: line[i] = (line[i] + a) & 0xff; break;
        case 2: line[i] = (line[i] + b) & 0xff; break;
        case 3: line[i] = (line[i] + ((a + b) >> 1)) & 0xff; break;
        case 4: line[i] = (line[i] + paeth(a, b, c)) & 0xff; break;
        default: throw new Error(`未知のフィルタ: ${filter}`);
      }
    }

    for (let x = 0; x < width; x++) {
      const s = x * channels;
      const d = (y * width + x) * 4;
      out[d] = line[s];
      out[d + 1] = line[s + 1];
      out[d + 2] = line[s + 2];
      out[d + 3] = channels === 4 ? line[s + 3] : 255;
    }
    prev.set(line);
  }

  return { rgba: out, width, height };
}
