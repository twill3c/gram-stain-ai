/**
 * 医療用途でない旨の明示(SPEC F-07)。**全ページに出す。**
 *
 * 「トップにだけ書いてある」では足りない。利用者はどのページから来るか分からないし、
 * 結果を見せる画面にこそ要る。
 */
export default function Disclaimer() {
  return (
    <div className="disclaimer" role="note">
      <strong>医療用途ではありません。</strong>
      Gram Stain AI は教育・研究目的の AI デモです。本システムは医療機器ではなく、
      出力は医学的診断を意味しません。診断・治療・患者管理には使用しないでください。
      実患者の検体画像をアップロードしないでください。
    </div>
  );
}
