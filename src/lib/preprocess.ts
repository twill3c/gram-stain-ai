/**
 * 前処理。**Python 側(ml/build_samples.py の normalise)と同じ数を作る**のが仕事である。
 *
 * ここが食い違うと、モデルは正しくても答えが変わる。しかも例外は出ない ——
 * それらしい数が返ってきて、静かに間違える。だから検査は
 * Python が書き出した期待値テンソルとの照合で行う(tests/preprocess.test.ts)。
 *
 * 学習時の前処理は次のとおりだった。
 *
 *   1. 配布画像(正方キャンバスに視野円が内接)から、中心 512x512 を**リサイズなしで**切り出す
 *   2. 224x224 へ双線形で縮小
 *   3. 255 で割り、ImageNet の平均と標準偏差で正規化
 *   4. CHW の順に並べ替える
 *
 * サンプル画像は 1 の済んだものを配っているので、ブラウザ側は 2 以降だけを行う。
 * 利用者が持ち込む画像には視野円の情報が無いので、**中心の正方形**を取ってから 2 へ進む。
 * これは学習時と厳密には同じでない —— その限界は画面に書く。
 */

export const INPUT_SIZE = 224;
export const MEAN = [0.485, 0.456, 0.406] as const;
export const STD = [0.229, 0.224, 0.225] as const;

/** 中心の正方形を取る。既に正方なら何もしない */
export function centreSquare(width: number, height: number) {
  const side = Math.min(width, height);
  return {
    sx: Math.floor((width - side) / 2),
    sy: Math.floor((height - side) / 2),
    side,
  };
}

/**
 * RGBA の画素配列(224x224)を、正規化済みの CHW float32 にする。
 *
 * **アルファは無視する。** 配布画像のアルファは視野円のマスクだが、
 * 学習時は PIL の convert("RGB") を通しており、これはアルファを合成せず捨てる。
 * ここで合成すると学習時と違う数になる。
 */
export function toTensor(rgba: Uint8ClampedArray, size = INPUT_SIZE): Float32Array {
  const n = size * size;
  if (rgba.length !== n * 4) {
    throw new Error(`画素数が合わない: ${rgba.length} バイト(期待 ${n * 4})`);
  }
  const out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      out[c * n + i] = (rgba[i * 4 + c] / 255 - MEAN[c]) / STD[c];
    }
  }
  return out;
}

/** logits を確率へ。表示に使う量はこちらであって logits ではない */
export function softmax(logits: ArrayLike<number>): number[] {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  const exp: number[] = [];
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - max);
    exp.push(e);
    sum += e;
  }
  return exp.map((e) => e / sum);
}
