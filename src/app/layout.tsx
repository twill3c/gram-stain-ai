import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Gram Stain AI — グラム染色像をブラウザの中だけで分類する",
  description:
    "公開データ(CC BY 4.0)で学習したモデルを ONNX Runtime Web で走らせ、"
    + "グラム染色像を分類する教育・研究用デモ。画像はサーバへ送られません。"
    + "精度を一つの数で出さず、三つの物差しと非学習の対照を並べます。",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ja">
      <body>
        <header className="masthead">
          <div className="masthead-inner">
            <strong style={{ fontSize: "0.95rem" }}>
              <Link href="/" style={{ color: "inherit", textDecoration: "none" }}>
                Gram Stain AI
              </Link>
            </strong>
            <nav>
              <Link href="/">判定してみる</Link>
              <Link href="/gram/">グラム染色とは</Link>
              <Link href="/model/">モデルとデータ</Link>
              <Link href="/monosashi/">三つの物差し</Link>
            </nav>
          </div>
        </header>
        {children}
        {/* fleet: fixed footer */}
        <footer className="site-footer">
          {/* 行き先は項目ごとに tests/footer.test.ts が見る(HC-098) */}
          <span><a href="https://github.com/twill3c/gram-stain-ai/blob/main/LICENSE" target="_blank" rel="noreferrer">MIT License</a> © 2026 坂田哲朗</span>
          <span className="sep">・</span>
          <a href="https://github.com/twill3c/gram-stain-ai" target="_blank" rel="noreferrer">GitHub</a>
          <span className="sep">・</span>
          {/* 歩き方・設計図はフリート規約どおりアーティファクト(loop_010 で作成) */}
          <a href="https://claude.ai/code/artifact/15456057-6554-490e-a1ae-d0e7a40674ae" target="_blank" rel="noreferrer">Gram Stain AI の測り方</a>
          <span className="sep">・</span>
          <a href="https://claude.ai/code/artifact/975607d2-0c20-4390-9b63-9c8deca7460b" target="_blank" rel="noreferrer">設計図</a>
          <span className="sep">・</span>
          {/* app-menu の本番は app-menu-amber.vercel.app。
              app-menu.vercel.app は他者の別サイトなので、そちらを指さない */}
          <a href="https://app-menu-amber.vercel.app/" target="_blank" rel="noreferrer">App Menu</a>
        </footer>
      </body>
    </html>
  );
}
