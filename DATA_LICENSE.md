# DATA_LICENSE.md — データの出所と帰属

このリポジトリでは、**三つのライセンスを混ぜない**(SPEC N-04)。

| 対象 | ライセンス | 文書 |
|---|---|---|
| ソースコード(`ml/` `src/` `tests/` `harness/`) | MIT | [LICENSE](LICENSE) |
| データセット(画像・ラベル典拠) | 各出所の条件に従う | 本書 |
| 学習済みモデル(`public/models/*.onnx`) | 下記「学習済みモデル」参照 | 本書 |

「公開されている」ことと「自由に再配布できる」ことは同義ではない。
本書は、**何をどこから取り、どう加工し、何を再配布しているか**を出所ごとに書く。

---

## 1. 画像データ — Bacteria Data for Machine Vision and Digital Biology

| 項目 | 内容 |
|---|---|
| 提供元 | Mendeley Data |
| データページ | https://data.mendeley.com/datasets/cvkgfzp7ck/1 |
| DOI | [10.17632/cvkgfzp7ck.1](https://doi.org/10.17632/cvkgfzp7ck.1) |
| 版 | version 1(公開日 2023-09-03) |
| ライセンス | **CC BY 4.0** — https://creativecommons.org/licenses/by/4.0/ |
| ライセンスの確認方法 | 公式 API レコード `https://data.mendeley.com/public-api/datasets/cvkgfzp7ck` の `data_licence.short_name` フィールドを実測(2026-09-08)。第三者ミラーの表示は根拠にしていない |
| 実測した配布物 | zip 34 本 / 3,597,653,809 バイト。うち 1 本は同一内容の重複(`Lactobacillus rhamnosus-…-001-5FKFIs.zip`) |

### 帰属表示(CC BY 4.0 の要求)

> Jamshidi, Mohammad (Behdad); Sargolzaee, Saleh; Foorginezhad, Salimeh; Moztarzadeh, Omid (2023),
> "Bacteria Data for Machine Vision and Digital Biology", Mendeley Data, V1,
> doi: [10.17632/cvkgfzp7ck.1](https://doi.org/10.17632/cvkgfzp7ck.1).
> Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

### 加工の内容(CC BY 4.0 は「変更したこと」の明示を求める)

本プロジェクトは原データに次の加工を行っている。加工物は原データではない。

- 円形視野の内接正方形を切り出し、視野外の黒い隅を除いた
- 学習用に画素寸法を揃えた
- 同一 SHA-256 の重複ファイルを取り除いた
- 種名から Gram 反応・形態のラベルを別の権威(下記 2)で付与した
- `Candida albicans`(真菌)を学習・評価の対象から外した

加工の実装は `ml/` 配下にあり、加工前後の対応は `dataset/metadata/` に記録している。

### 上流データについて

このデータセットの説明によれば、内容は次の二つの結合である。

1. Jamshidi et al., *Metaverse and Microorganism Digital Twins: A Deep Transfer Learning Approach*,
   Applied Soft Computing (2023)
2. **DIBaS**(Digital Images of Bacteria Species) — Zieliński B, Plichta A, Misztal K, Spurek P,
   Brzychczy-Włoch M, Ochońska D, *Deep learning approach to bacterial colony classification*,
   PLoS ONE 12(9): e0184554 (2017).
   doi: [10.1371/journal.pone.0184554](https://doi.org/10.1371/journal.pone.0184554)

**DIBaS 原データそのものは、本リポジトリでは扱わない**(SPEC §7 スコープ外)。
第三者の GitHub / Kaggle ミラーのライセンス表示だけを根拠に、原データの再配布条件を判断しない。
本プロジェクトが依拠しているのは、上記 Mendeley レコードに明示された CC BY 4.0 だけである。

---

## 2. ラベルの典拠 — BacDive

Gram 反応と細胞形態は、画像からではなく**種名から**決まる。その対応表は外部権威に取っている
(SPEC §2.1・較正ゲート G-07)。

| 項目 | 内容 |
|---|---|
| 提供元 | BacDive — Leibniz Institute DSMZ |
| API | https://api.bacdive.dsmz.de/ (v2) |
| ライセンス | **CC BY 4.0** |
| 取得日 | 2026-09-08 |
| 取得範囲 | 33 分類群 × 各最大 100 株。株ごとの `Morphology → cell morphology → gram stain / cell shape` と LPSN 分類 |
| 生の回答 | `dataset/metadata/authority_bacdive.json`(加工前) |
| 採用後 | `dataset/metadata/labels.json`(採用規則を明記して適用) |

### 引用

> Schober I. et al., *BacDive in 2025: the core database for prokaryotic strain data*,
> Nucleic Acids Research 53(D1): D748,
> doi: [10.1093/nar/gkae959](https://doi.org/10.1093/nar/gkae959).
> Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

## 3. 分類学上の判定 — NCBI Taxonomy

`Candida albicans` を細菌の分類から外した根拠。

| 項目 | 内容 |
|---|---|
| 提供元 | NCBI Taxonomy(National Center for Biotechnology Information) |
| 参照 | taxid 5476 — https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=5476 |
| 取得日 | 2026-09-08 |
| 取得内容 | Lineage: cellular organisms; Eukaryota; Opisthokonta; **Fungi**; Dikarya; Ascomycota; saccharomyceta; Saccharomycotina; Pichiomycetes; Serinales; Debaryomycetaceae; Candida |

NCBI のデータは米国政府の著作物として公開されており、利用に制限は課されていない。
引用の際は NCBI を出所として示す。

---

## 4. リポジトリに入れているもの / 入れていないもの

**入れていない**(`.gitignore`):

- `dataset/raw/` — 配布元の zip(3.6 GB)
- `dataset/processed/` — 加工後の画像
- 利用者がブラウザで読み込んだ画像(そもそもサーバへ送られない — SPEC F-08)

**入れている**:

- `dataset/metadata/` — 取得記録・SHA-256・権威の生の回答・ラベル表
- `ml/` — 取得・加工・学習・変換のコード
- `reports/` — 評価結果
- 学習済み ONNX モデルと、再配布条件を確認したサンプル画像

原画像を再配布しないのは、CC BY 4.0 が禁じているからではない(禁じていない)。
**3.6 GB をリポジトリに置く必要が無く、置けば取得元の版と食い違ったときに気づけなくなる**からである。
取得は `ml/download_dataset.py` が公式 API から行い、SHA-256 を照合できる。

---

## 5. 学習済みモデル

配布する ONNX モデルは、次の三つの条件が重なった成果物である。

1. 上記 CC BY 4.0 の画像から学習した
2. ImageNet 事前学習の重み(torchvision 配布)を初期値にしている
3. 学習・変換に用いたライブラリの条件

このため、モデルの配布条件はコード(MIT)と同一ではない。
**帰属表示は画像データセットに対しても必要である** —— モデルはその派生物として扱う。
実際の配布条件は [docs/MODEL.md](docs/MODEL.md) に、学習に使った版・日付・評価結果、
そのまま使える帰属表示の文面とともに記載してある。
