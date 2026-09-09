/**
 * PIL(Pillow)の Image.resize を bit 一致で再現する。
 *
 * なぜ自前で書くか(SPEC N-03):
 * ブラウザの `ctx.drawImage` による縮小は、補間の方法が仕様で定められていない。
 * 実装依存であり、同じ画像でもブラウザによって画素が変わる。学習時の前処理は
 * PIL で行われているので、`drawImage` を推論経路に置くと **学習時と違う画像**を
 * モデルに見せることになる。それは F-04「前処理の罪」の展示物であって、
 * 出荷する推論経路の実装ではない。
 *
 * PIL は uint8 のまま **固定小数点**で畳み込む。浮動小数で素直に書くと
 * 1/255 ≒ 0.0039 のずれが出て、ゲート G-01 の 1e-5 を軽く超える。
 * ここでは PIL の Resample.c の手順(係数の事前計算 → 水平走査 → 垂直走査)を
 * そのまま写している。中間バッファが uint8 である点も含めて写すこと —
 * ここを float にすると一致しない。
 *
 * 手順の正しさは scripts/probe_pil_resample.py で Pillow 12.3.0 に対し
 * 実測検証済み(2026-08-29 / 2 フィルタ × 5 形状 = 10 ケースすべて bit 一致)。
 */

export type FilterName = "bilinear" | "bicubic" | "nearest";

/** PIL: Resample.c の PRECISION_BITS = 32 - 8 - 2 */
const PRECISION_BITS = 22;

/** PIL の bilinear。support = 1 */
function filterBilinear(x: number): number {
  const a = Math.abs(x);
  return a < 1 ? 1 - a : 0;
}

/** PIL の bicubic。a = -0.5(Catmull-Rom)。support = 2 */
function filterBicubic(x: number): number {
  const a = -0.5;
  const t = Math.abs(x);
  if (t < 1) return ((a + 2) * t - (a + 3)) * t * t + 1;
  if (t < 2) return (((t - 5) * t + 8) * t - 4) * a;
  return 0;
}

const FILTERS: Record<
  Exclude<FilterName, "nearest">,
  { fn: (x: number) => number; support: number }
> = {
  bilinear: { fn: filterBilinear, support: 1 },
  bicubic: { fn: filterBicubic, support: 2 },
};

interface Coeffs {
  /** 出力画素ごとの [入力開始位置, 使う画素数] */
  bounds: Int32Array;
  /** 固定小数点化した係数。ksize ごとに区切って詰めてある */
  kk: Int32Array;
  ksize: number;
}

/**
 * PIL の ImagingPrecomputeCoeffs + normalize_coeffs_8bpc。
 *
 * 注意: 開始位置の切り捨ては `Math.trunc`(0 方向)であって `Math.floor` ではない。
 * C の (int) キャストが 0 方向の切り捨てだからである。負値は直後に 0 へ丸め込まれるので
 * 結果は変わらないが、写し違いを残さないため trunc で書く。
 */
function precomputeCoeffs(
  inSize: number,
  outSize: number,
  filter: Exclude<FilterName, "nearest">,
): Coeffs {
  const { fn, support } = FILTERS[filter];
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const fsupport = support * filterscale;
  const ksize = Math.ceil(fsupport) * 2 + 1;

  const bounds = new Int32Array(outSize * 2);
  const kkFloat = new Float64Array(outSize * ksize);

  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const ss = 1 / filterscale;
    let xmin = Math.trunc(center - fsupport + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + fsupport + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;

    let ww = 0;
    for (let x = 0; x < xmax; x++) {
      const w = fn((x + xmin - center + 0.5) * ss);
      kkFloat[xx * ksize + x] = w;
      ww += w;
    }
    if (ww !== 0) {
      for (let x = 0; x < xmax; x++) kkFloat[xx * ksize + x] /= ww;
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }

  // 固定小数点へ。PIL は 0 から遠ざかる向きに丸める(負の係数は -0.5 を足す)
  const one = 1 << PRECISION_BITS;
  const kk = new Int32Array(outSize * ksize);
  for (let i = 0; i < kkFloat.length; i++) {
    const v = kkFloat[i];
    kk[i] = v < 0 ? Math.trunc(-0.5 + v * one) : Math.trunc(0.5 + v * one);
  }
  return { bounds, kk, ksize };
}

function clip8(v: number): number {
  const s = v >> PRECISION_BITS;
  return s < 0 ? 0 : s > 255 ? 255 : s;
}

/** 平坦な uint8 画像。data は行優先・チャネル詰め(HWC) */
export interface Image8 {
  data: Uint8Array;
  width: number;
  height: number;
  channels: number;
}

function resampleHorizontal(src: Image8, outW: number, filter: Exclude<FilterName, "nearest">): Image8 {
  const { width: w, height: h, channels: c, data } = src;
  const { bounds, kk, ksize } = precomputeCoeffs(w, outW, filter);
  const dst = new Uint8Array(outW * h * c);
  const init = 1 << (PRECISION_BITS - 1);

  for (let yy = 0; yy < h; yy++) {
    const srcRow = yy * w * c;
    const dstRow = yy * outW * c;
    for (let xx = 0; xx < outW; xx++) {
      const xmin = bounds[xx * 2];
      const xmax = bounds[xx * 2 + 1];
      const kOff = xx * ksize;
      for (let ch = 0; ch < c; ch++) {
        let ss = init;
        for (let x = 0; x < xmax; x++) {
          ss += data[srcRow + (x + xmin) * c + ch] * kk[kOff + x];
        }
        dst[dstRow + xx * c + ch] = clip8(ss);
      }
    }
  }
  return { data: dst, width: outW, height: h, channels: c };
}

function resampleVertical(src: Image8, outH: number, filter: Exclude<FilterName, "nearest">): Image8 {
  const { width: w, height: h, channels: c, data } = src;
  const { bounds, kk, ksize } = precomputeCoeffs(h, outH, filter);
  const dst = new Uint8Array(w * outH * c);
  const init = 1 << (PRECISION_BITS - 1);

  for (let yy = 0; yy < outH; yy++) {
    const ymin = bounds[yy * 2];
    const ymax = bounds[yy * 2 + 1];
    const kOff = yy * ksize;
    const dstRow = yy * w * c;
    for (let xx = 0; xx < w; xx++) {
      for (let ch = 0; ch < c; ch++) {
        let ss = init;
        for (let y = 0; y < ymax; y++) {
          ss += data[(y + ymin) * w * c + xx * c + ch] * kk[kOff + y];
        }
        dst[dstRow + xx * c + ch] = clip8(ss);
      }
    }
  }
  return { data: dst, width: w, height: outH, channels: c };
}

/** PIL の NEAREST。F-04 の展示用であって、出荷する推論経路では使わない */
function resampleNearest(src: Image8, outW: number, outH: number): Image8 {
  const { width: w, height: h, channels: c, data } = src;
  const dst = new Uint8Array(outW * outH * c);
  const sx = w / outW;
  const sy = h / outH;
  for (let yy = 0; yy < outH; yy++) {
    const y = Math.min(h - 1, Math.trunc((yy + 0.5) * sy));
    for (let xx = 0; xx < outW; xx++) {
      const x = Math.min(w - 1, Math.trunc((xx + 0.5) * sx));
      for (let ch = 0; ch < c; ch++) {
        dst[(yy * outW + xx) * c + ch] = data[(y * w + x) * c + ch];
      }
    }
  }
  return { data: dst, width: outW, height: outH, channels: c };
}

/**
 * PIL 互換のリサイズ。水平 → 垂直の 2 パスで、中間は uint8。
 * この「中間が uint8」は写し間違えやすいが、float にすると PIL と一致しなくなる。
 */
export function resizePIL(src: Image8, outW: number, outH: number, filter: FilterName): Image8 {
  if (outW <= 0 || outH <= 0) throw new Error(`出力寸法が不正: ${outW}x${outH}`);
  if (filter === "nearest") return resampleNearest(src, outW, outH);
  const tmp = resampleHorizontal(src, outW, filter);
  return resampleVertical(tmp, outH, filter);
}

/** PIL の center_crop 相当。HF の image_transforms.center_crop は左上寄せの丸めを使う */
export function centerCrop(src: Image8, cropW: number, cropH: number): Image8 {
  const { width: w, height: h, channels: c, data } = src;
  // HF: top = (h - crop_h) // 2, left = (w - crop_w) // 2(切り捨て)
  const top = Math.floor((h - cropH) / 2);
  const left = Math.floor((w - cropW) / 2);
  const dst = new Uint8Array(cropW * cropH * c);
  for (let y = 0; y < cropH; y++) {
    const sy = top + y;
    for (let x = 0; x < cropW; x++) {
      const sx = left + x;
      // 画像より大きく切る場合、HF はゼロ詰めする
      const inside = sy >= 0 && sy < h && sx >= 0 && sx < w;
      for (let ch = 0; ch < c; ch++) {
        dst[(y * cropW + x) * c + ch] = inside ? data[(sy * w + sx) * c + ch] : 0;
      }
    }
  }
  return { data: dst, width: cropW, height: cropH, channels: c };
}

/**
 * 比率を保ったまま短辺を shortestEdge に合わせたときの出力寸法。
 * HF の get_resize_output_image_size(default_to_square=False) と同じ丸め。
 */
export function shortestEdgeSize(w: number, h: number, shortestEdge: number): [number, number] {
  const short = Math.min(w, h);
  const long = Math.max(w, h);
  const newShort = shortestEdge;
  const newLong = Math.trunc((newShort * long) / short);
  return w <= h ? [newShort, newLong] : [newLong, newShort];
}
