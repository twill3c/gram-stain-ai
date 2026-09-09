import Disclaimer from "@/components/Disclaimer";

export const metadata = { title: "グラム染色とは — Gram Stain AI" };

export default function GramPage() {
  return (
    <main>
      <h1>グラム染色とは</h1>
      <p className="lead">
        1884 年に Hans Christian Gram が考案した染め分け。
        いまでも細菌を最初に分ける手がかりとして使われています。
      </p>

      <Disclaimer />

      <h2>四つの工程</h2>
      <div className="wrap-x">
        <table>
          <thead>
            <tr>
              <th style={{ width: "6em" }}>工程</th>
              <th style={{ width: "9em" }}>使うもの</th>
              <th>起きること</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>1. 染色</td>
              <td>クリスタル紫</td>
              <td>すべての菌が紫に染まる。この段階では陽性も陰性も区別がつかない</td>
            </tr>
            <tr>
              <td>2. 媒染</td>
              <td>ヨウ素液</td>
              <td>色素とヨウ素が結びつき、細胞の中で大きな複合体になる</td>
            </tr>
            <tr>
              <td>3. 脱色</td>
              <td>アルコール</td>
              <td>
                <strong>ここが分かれ目。</strong>
                厚いペプチドグリカン層を持つ菌は複合体を抱え込んだままだが、
                層が薄く外膜を持つ菌は洗い流されて無色になる
              </td>
            </tr>
            <tr>
              <td>4. 後染色</td>
              <td>サフラニン</td>
              <td>無色になった菌が桃〜赤に染まる。紫のままの菌は色が変わらない</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>だから、これは色の検査である</h2>
      <p>
        結果として、<strong>グラム陽性は紫、グラム陰性は桃〜赤</strong>に見えます。
        分けているのは細胞壁の構造ですが、<strong>読み取っているのは色</strong>です。
      </p>
      <p>
        このことは、AI の成績を読むときに効いてきます。
        染まった画素の赤と青の比に閾値を一本引くだけの、学習を一切しない規則が
        macro F1 0.95 を超えます。だから
        <strong>「AI が 9 割当てた」と言う前に、色の規則がどこまで当てるかを測らなければなりません</strong>。
      </p>

      <h2>染め分けが破れるとき</h2>
      <p>
        工程 3 の脱色は、時間や菌の状態でぶれます。実測でも、色の規則の誤りは
        散らばらずに<strong>染色そのものが揺れる菌に集中しました</strong>。
      </p>
      <div className="wrap-x">
        <table>
          <thead>
            <tr>
              <th>分類群</th>
              <th style={{ width: "7em" }}>本来</th>
              <th>起きること</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><i>Listeria monocytogenes</i></td>
              <td>陽性</td>
              <td>脱色されやすく、桃色に見えることがある。色の規則の誤り率 44%</td>
            </tr>
            <tr>
              <td><i>Porphyromonas gingivalis</i></td>
              <td>陰性</td>
              <td>染まり方が揺れる。色の規則の誤り率 37%</td>
            </tr>
            <tr>
              <td><i>Acinetobacter baumannii</i></td>
              <td>陰性</td>
              <td>形が球菌とも桿菌とも言い切れない(coccobacillus)ため、形の分類から外した</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>形でも分ける</h2>
      <p>
        グラム反応と形(球菌 / 桿菌)を組み合わせると、四つの区分になります。
        ただし本アプリが学習しているのはグラム反応だけで、形の分類は
        <strong>典拠で形が一意に定まる 30 分類群</strong>を対象に次の版で扱います。
      </p>

      <h2>門からは決められない</h2>
      <p>
        「系統が分かれば染色も分かる」と考えたくなりますが、成り立ちません。
        <i>Veillonella</i> は Bacillota 門でありながらグラム陰性です。
        本プロジェクトが採録した他の Bacillota 19 分類群はすべて陽性なので、
        <strong>門で決め打つ実装はここだけ静かに間違えます</strong>。
        だからラベルは、種名ごとに外部の権威(BacDive)へ当たって取っています。
      </p>
    </main>
  );
}
