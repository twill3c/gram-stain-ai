import Link from "next/link";

import Classifier from "@/components/Classifier";
import Disclaimer from "@/components/Disclaimer";

export default function Home() {
  return (
    <main>
      <h1>Gram Stain AI</h1>
      <p className="lead">
        グラム染色した細菌の顕微鏡画像を、<strong>ブラウザの中だけ</strong>で分類します。
        画像はサーバへ送られません。
      </p>

      <Disclaimer />

      <div className="card">
        <p style={{ margin: 0 }}>
          このアプリが答えようとしているのは「何%当てられるか」ではなく、
          <strong>その%が何を測っているのか</strong>です。
        </p>
        <p style={{ marginBottom: 0 }}>
          同じモデル・同じ指標でも、データの割り方を変えると別の問いに答えることになります。
          このモデルは、学習で見た分類群のスライドなら macro F1 <strong>0.9731</strong> で見分けますが、
          <strong>科をまるごと抜くと 0.6942 まで落ちます</strong>。
          一方、学習を一切しない「紫か桃色か」だけの規則は 0.9714 → 0.9601 としか動きません。
          {" "}
          <Link href="/monosashi/">三つの物差し</Link>で詳しく説明しています。
        </p>
      </div>

      <Classifier />
    </main>
  );
}
