import Disclaimer from "@/components/Disclaimer";
import { RULER_TITLE, fmt, loadMetadata, loadShapeMetadata, pct } from "@/lib/metrics-data";

export const metadata = { title: "三つの物差し — Gram Stain AI" };

export default function MonosashiPage() {
  const meta = loadMetadata();
  const m = meta.metrics;
  const shape = loadShapeMetadata();
  const sm = shape.metrics;
  const nc = shape.negative_cocci;
  const rulers = ["a", "b", "c"] as const;

  return (
    <main>
      <h1>三つの物差し</h1>
      <p className="lead">
        同じモデル・同じ指標でも、データの割り方を変えると別の問いに答えることになります。
      </p>

      <Disclaimer />

      <h2>割り方が問いを決める</h2>
      <div className="wrap-x">
        <table>
          <thead>
            <tr>
              <th style={{ width: "12em" }}>物差し</th>
              <th>答えている問い</th>
            </tr>
          </thead>
          <tbody>
            {rulers.map((r) => (
              <tr key={r}>
                <td>{RULER_TITLE[r]}</td>
                <td>{m.ruler_meaning[r]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p>
        C を置いたのは実測の結果です。採録した {meta.taxa_included} 分類群のうち
        <strong>11 が Lactobacillaceae 科</strong>でした。種を 1 つ抜いても近縁の 10 が学習側に残るので、
        <strong>B は「未知の種」を測っているつもりで、近縁の見覚えを測りうる</strong>のです。
      </p>

      <h2>並べるとこうなる</h2>
      <div className="wrap-x">
        <table>
          <thead>
            <tr>
              <th />
              {rulers.map((r) => (
                <th key={r}>{RULER_TITLE[r]}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>CNN</strong>({meta.architecture})</td>
              {rulers.map((r) => (
                <td className="num" key={r}>{fmt(m.rulers[r])}</td>
              ))}
            </tr>
            <tr>
              <td>色相(紫か桃色か・<strong>学習なし</strong>)</td>
              {rulers.map((r) => (
                <td className="num" key={r}>{fmt(m.hue_baseline[r])}</td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
      <p className="muted">
        macro F1 と 95% 信頼区間。どの行も全画像を一度ずつ評価しています。
        信頼区間は group 単位のブートストラップで取っています ——
        画像単位で取ると、同じスライドの別視野が独立標本として数えられ、区間が実際より狭くなります。
      </p>

      <h2>読み方</h2>
      <p>
        CNN は {m.rulers.a.macro_f1.toFixed(4)} →{" "}
        {m.rulers.b.macro_f1.toFixed(4)} → <strong>{m.rulers.c.macro_f1.toFixed(4)}</strong> と落ちます。
        色の規則は {m.hue_baseline.a.macro_f1.toFixed(4)} →{" "}
        {m.hue_baseline.b.macro_f1.toFixed(4)} → {m.hue_baseline.c.macro_f1.toFixed(4)} としか動きません。
      </p>
      <p>
        色は種の同一性に依らないので、色の規則が動かないのは当然です。
        <strong>そして当然であることが重要です</strong> ——
        物差しを変えて大きく落ちるものがあれば、その落ちた分が
        <strong>「種の見覚え」だった</strong>ということになります。
      </p>

      <h2>合否の条件は、測る前に決めてある</h2>
      <p>
        数を見てから「効いた」と言える理由を探さないために、
        何をもって「CNN が効いた」と言うかを測定前に書いておきました。
      </p>
      <ul>
        <li>
          条件 1: 三つの物差しすべてで、色の規則を信頼区間の重なりなく上回る →{" "}
          <strong>不成立</strong>
        </li>
        <li>
          条件 2: 色の規則が落ちる分類群で明確に上回る → <strong>成立</strong>
          (<i>Listeria monocytogenes</i> 44%→33%、<i>Porphyromonas gingivalis</i> 37%→35%、
          <i>Staphylococcus epidermidis</i> 14%→2%)
        </li>
      </ul>
      <p>
        したがって結論は「何も足していない」ではありません。正確には次のとおりです。
      </p>
      <div className="card">
        <strong>
          CNN は色以外の手がかりを確かに少し使っているが、その手がかりは学習で見た分類群に
          強く依存する。全体では色の規則に勝てず、未知の科ではまったく歯が立たない。
        </strong>
      </div>

      <h2>学習系そのものを疑う</h2>
      <p>
        ラベルを無作為に置き換えて同じ手続きを回すと、macro F1 は{" "}
        <strong>{m.label_permutation_control.toFixed(4)}</strong> でした。
        信号を消せば成績も消えます。ここで成績が残るなら、
        それは入力の信号ではなく手続きのどこかにある漏れが出している数です。
      </p>

      <h2>出荷しているモデルの数</h2>
      <p>
        このアプリが実際に動かしているモデルの test macro F1 は{" "}
        <strong>{m.shipped_test.macro_f1}</strong>({m.shipped_test.n_images} 枚)です。
        正しい手順で出した数ではあります —— 学習用で作り、検証用で確認し、
        試験用には最後に一度だけ当てました。
      </p>
      <div className="disclaimer">
        <strong>それでも、この数だけを見てはいけません。</strong>
        {m.caveat}
      </div>

      <h2>形(球菌 / 桿菌)にも同じ物差しを当てる</h2>
      <p>
        形の典拠が一つに決まる {shape.stage_b.taxa} 分類群・{shape.stage_b.images.toLocaleString()} 枚で、
        同じ三つの物差しを当てました。対照は<strong>色を見ずに、染まった塊の細長さに閾値を一本引くだけ</strong>の規則で、
        CNN を回す前に測っています。
      </p>
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
              <td><strong>CNN</strong>(形)</td>
              {rulers.map((r) => <td className="num" key={r}>{fmt(sm.rulers[r])}</td>)}
            </tr>
            <tr>
              <td>形の規則(細長さ・<strong>学習なし</strong>)</td>
              {rulers.map((r) => <td className="num" key={r}>{fmt(sm.shape_rule_baseline[r])}</td>)}
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        測る前に決めた二条件(三つの物差しすべてで対照を区間の重なりなく上回る / 対照が外す分類群で誤りを減らす)は、
        <strong>{sm.verdict.condition_1 && sm.verdict.condition_2 ? "どちらも成立" : "成立していない"}</strong>しました。
        Gram の CNN と違い、科を抜いても崩れません。
      </p>
      <h3>それでも、ここでは外す</h3>
      <div className="wrap-x">
        <table>
          <thead>
            <tr>
              <th>Gram 陰性の球菌({nc.taxa.join("・")}・{nc.n_images} 枚)</th>
              {rulers.map((r) => <th key={r}>{RULER_TITLE[r]}</th>)}
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>CNN の誤り率</td>
              {rulers.map((r) => <td className="num" key={r}>{pct(nc.error_rate[r])}</td>)}
            </tr>
          </tbody>
        </table>
      </div>
      <div className="disclaimer">
        <strong>学習で見ていない分類群・科の Gram 陰性球菌を、形のモデルはほぼ全部桿菌と答えます。</strong>
        分類群を抜くと誤り {pct(nc.error_rate.b)}、科を抜くと {pct(nc.error_rate.c)}(学習しない形の規則は分類群を抜いても{" "}
        {pct(nc.shape_rule_error_rate_b)})。このデータの Gram 陰性球菌は {nc.taxa.length} 分類群しかありません。
        この崩れは判定の後で数え直して見つけたもので、合否の条件には入れていません。
        理由を一つ確かめました ——
        <strong>形はそのままに色相だけを紫へ回しても、誤りは変わりませんでした</strong>(形のモデルは色相で形を決めていません)。
        続けて大きさと濃さも測りました ——
        <strong>3 倍に拡大すると誤りは 92%→66%(分類群抜き)・98%→57%(科抜き)まで下がりますが、そこで止まります</strong>。
        彩度を陽性の水準に合わせても 17 / 6 ポイントしか動きません。
        色相でも、大きさだけでも、濃さだけでも説明できず、何を手がかりに桿菌と答えているかは特定していません。
      </div>
      <p className="muted">
        形のラベル置換の対照は {sm.label_permutation_control.observed.toFixed(4)} で、同じ予測を入れ替えた偶然水準
        (中央 {sm.label_permutation_control.null_median.toFixed(4)}・上側 {sm.label_permutation_control.null_q975.toFixed(4)})
        を上に外れていません。この対照は当初「多数派クラスと重なるか」で見ていましたが、
        多数派は全部を同じ答えにする予測の値で偶然水準ではないため、結果を見てから基準を引き直しました。
      </p>
    </main>
  );
}
