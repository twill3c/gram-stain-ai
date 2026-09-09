/**
 * 画像 1 枚を、モデルが受け取る形まで運ぶ。
 *
 * **ここは Python 側と同じ数を作らなければならない。** 食い違っても例外は出ない ——
 * それらしい数が返ってきて、静かに間違える。検査は Python が書き出した
 * 期待値テンソルとの照合で行う(tests/preprocess.test.ts)。
 *
 * 学習時(ml/build_samples.py の normalise)は:
 *
 *   PIL で 224x224 へ双線形縮小 → 255 で割る → ImageNet の平均と標準偏差 → CHW
 *
 * `ctx.drawImage` による縮小は仕様で補間法が定められておらず、ブラウザによって画素が変わる。
 * だから縮小は PIL を移植した `resizePIL` で行う(src/lib/resize.ts)。
 */

import { INPUT_SIZE, toTensor } from "./preprocess";
import { centerCrop, resizePIL, type Image8 } from "./resize";

export { INPUT_SIZE };

/** RGBA の Image8 を作る */
export function imageFromRGBA(rgba: Uint8ClampedArray, width: number, height: number): Image8 {
  return { data: new Uint8Array(rgba.buffer.slice(0)), width, height, channels: 4 };
}

/**
 * 前処理を通してテンソルにする。
 *
 * 配るサンプルは視野円の内接正方形を切り出した正方画像なので、切り出しは素通りする。
 * 利用者が持ち込む画像は形がまちまちなので、**中心の正方形**を取ってから縮小する。
 * これは学習時と厳密には同じでない(学習時は視野円から切り出している)。
 * その限界は画面に書く。
 */
export function preprocess(src: Image8): { tensor: Float32Array; cropped: Image8 } {
  const side = Math.min(src.width, src.height);
  const square = side === src.width && side === src.height ? src : centerCrop(src, side, side);
  const small = resizePIL(square, INPUT_SIZE, INPUT_SIZE, "bilinear");

  // resizePIL は RGBA を返す。toTensor は RGBA を前提に RGB だけを読む
  const rgba = new Uint8ClampedArray(small.data.buffer, small.data.byteOffset, small.data.length);
  return { tensor: toTensor(rgba, INPUT_SIZE), cropped: square };
}
