import Link from "next/link";

import Disclaimer from "@/components/Disclaimer";
import { RULER_TITLE, fmt, loadMetadata } from "@/lib/metrics-data";

export const metadata = { title: "モデルとデータ — Gram Stain AI" };

export default function ModelPage() {
  const meta = loadMetadata();
  const m = meta.metrics;
  const rulers = ["a", "b", "c"] as const;

  return (
    <main>
      <h1>モデルとデータ</h1>
      <p className="lead">何をどこから取り、どう作り、どこまで確かめたか。</p>

      <Disclaimer />

      <h2>モデル</h2>
      <div className="wrap-x">
        <table>
          <tbody>
            <tr><th style={{ width: "12em" }}>版</th><td>{meta.model_version}</td></tr>
            <tr><th>骨格</th><td>{meta.architecture}(ImageNet 事前学習)</td></tr>
            <tr><th>入力</th><td>{meta.input_size[0]}×{meta.input_size[1]} RGB</td></tr>
            <tr><th>学習</th><td>{meta.epochs} エポック(予算は検証用データで決定・試験用は未使用)</td></tr>
            <tr><th>ONNX</th><td>{meta.onnx.megabytes} MB / opset {meta.onnx.opset} / {meta.onnx.exporter}</td></tr>
            <tr><th>学習日時</th><td>{meta.trained_at}</td></tr>
            <tr><th>コミット</th><td><code>{meta.git_commit.slice(0, 12)}</code></td></tr>
          </tbody>
        </table>
      </div>
      <p className="muted">
        当初は ResNet18 を標準候補にしていましたが、fp32 で 46.8 MB あり、
        量子化する前から配信の目標 30 MB を超えていました。目標を動かさず骨格を変えています。
      </p>

      <h2>二実装照合</h2>
      <p>
        変換は静かに壊れます —— 演算子の解釈違い、精度の落ち、入力の並びの取り違え。
        どれも例外を出さずに「それらしい数」を返します。
        だから PyTorch と ONNX Runtime に<strong>同じ重みを別実装で</strong>走らせて突き合わせています。
      </p>
      <div className="wrap-x">
        <table>
          <tbody>
            <tr><th style={{ width: "16em" }}>argmax 一致率</th><td className="num">{meta.onnx.parity_argmax_agreement.toFixed(6)}</td></tr>
            <tr><th>スコアの最大差</th><td className="num">{meta.onnx.parity_max_prob_diff.toExponential(3)}</td></tr>
            <tr><th>生 logits の最大絶対差(参考)</th><td className="num">{meta.onnx.parity_max_abs_diff_logits.toExponential(3)}</td></tr>
          </tbody>
        </table>
      </div>
      <p className="muted">
        このゲートは一度落ちて、引き直しました。当初は生 logits の絶対差で測っていましたが、
        logits は有界でないため閾値がモデルの確信度に比例してしまい、
        しかも画面が表示するのは確率であって logits ではありません。
        引き直した経緯と旧い書き方での結果はリポジトリに残してあります。
      </p>

      <h2>成績</h2>
      <div className="wrap-x">
        <table>
          <thead>
            <tr>
              <th />
              {rulers.map((r) => <th key={r}>{RULER_TITLE[r]}</th>)}
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>CNN</strong></td>
              {rulers.map((r) => <td className="num" key={r}>{fmt(m.rulers[r])}</td>)}
            </tr>
            <tr>
              <td>色相(学習なし)</td>
              {rulers.map((r) => <td className="num" key={r}>{fmt(m.hue_baseline[r])}</td>)}
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        読み方は <Link href="/monosashi/">三つの物差し</Link>にあります。
      </p>

      <h2>データセット</h2>
      <div className="wrap-x">
        <table>
          <tbody>
            <tr><th style={{ width: "12em" }}>名称</th><td>{meta.dataset}</td></tr>
            <tr>
              <th>DOI</th>
              <td>
                <a href={`https://doi.org/${meta.dataset_doi}`} target="_blank" rel="noreferrer">
                  {meta.dataset_doi}
                </a>
              </td>
            </tr>
            <tr><th>ライセンス</th><td>{meta.dataset_license}</td></tr>
            <tr><th>採録した分類群</th><td>{meta.taxa_included}</td></tr>
          </tbody>
        </table>
      </div>
      <p className="muted">{meta.attribution}</p>

      <h3>加工したこと</h3>
      <p>
        CC BY 4.0 は「変更したこと」の明示を求めます。次の加工を行っています。
      </p>
      <ul>
        <li>円形視野の内接正方形の中心 512×512 を、リサイズなしで切り出した(四隅の黒を入れない)</li>
        <li>同一内容の重複画像を取り除いた</li>
        <li>種名から、別の権威でグラム反応のラベルを付けた</li>
        <li><i>Candida albicans</i>(真菌)を対象から外した</li>
      </ul>

      <h2>ラベルの出どころ</h2>
      <p>
        グラム反応は画像からではなく<strong>種名から</strong>決まります。
        その対応表は、このプロジェクトの外にある権威に取っています ——
        自分で書いてしまうと、「モデルが当てた」ことの検算が自分の思い込みの検算になるからです。
      </p>
      <div className="wrap-x">
        <table>
          <tbody>
            <tr><th style={{ width: "12em" }}>権威</th><td>{meta.label_authority.source}</td></tr>
            <tr><th>ライセンス</th><td>{meta.label_authority.license}</td></tr>
            <tr><th>取得日時</th><td>{meta.label_authority.accessed_at}</td></tr>
          </tbody>
        </table>
      </div>

      <h2>プライバシー</h2>
      <ul>
        <li>アップロードの API を持たない</li>
        <li>画像をサーバへ保存しない</li>
        <li>解析ログへ画像の内容を送らない</li>
        <li>推論はブラウザのメモリ上で完結する</li>
        <li>ページを離れるか再読込すると画像の状態を破棄する</li>
      </ul>

      <h2>ライセンス</h2>
      <div className="wrap-x">
        <table>
          <tbody>
            <tr><th style={{ width: "12em" }}>ソースコード</th><td>MIT</td></tr>
            <tr><th>データセット</th><td>CC BY 4.0(画像・ラベル典拠とも)</td></tr>
            <tr><th>学習済みモデル</th><td>上記の重なり。帰属表示はデータセットに対しても必要</td></tr>
          </tbody>
        </table>
      </div>
    </main>
  );
}
