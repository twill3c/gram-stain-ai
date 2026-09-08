# DATASET.md — 採録した分類群とラベルの典拠

<!-- ml/build_docs.py が生成。手で編集しない。生成日時 2026-09-08T14:52:53+09:00 -->

ラベル(Gram 反応・形態)は画像からではなく種名から決まる。
その対応表は外部権威に取っている(SPEC §2.1・較正ゲート G-07)。

- 権威: **BacDive (Leibniz Institute DSMZ)**(CC BY 4.0)
- 取得日時: 2026-09-08T09:45:26+09:00
- 引用: Schober et al., BacDive in 2025: the core database for prokaryotic strain data, Nucleic Acids Research 53(D1):D748, https://doi.org/10.1093/nar/gkae959

## 採用規則

- Gram: 権威記録 1 件以上・多数派が過半数で採用。一致率 100% を tier A、未満を tier B
- 形態: 正規化後の一致率 90% 以上のときだけ採用
- 形態の語の正規化:

| 権威が使う語 | 正規化後 |
|---|---|
| `coccus-shaped` | 球菌 |
| `oval-shaped` | 球菌 |
| `ovoid-shaped` | 球菌 |
| `rod-shaped` | 桿菌 |
| `curved-shaped` | 桿菌 |

**この正規化は結果を変える。** 正規化しないと `Veillonella`(53.3%)と
`Neisseria gonorrhoeae`(87.5%)が一致率不足で落ち、Stage B から Gram 陰性球菌が
全部消えて 4 クラスが 3 クラスになる(検査 T-111)。

## 採録の要約

| 項目 | 実測 |
|---|---|
| 配布元の分類群 | 33 |
| 採用 | 32 |
| Gram 陽性 / 陰性 | 23 / 9 |
| 一致率 100% でない Gram ラベル | `Bifidobacterium spp`, `Fusobacterium` |
| Stage B 対象 | 30 |
| Stage B 対象外 | `Acinetobacter baumannii`, `Porphyromonas gingivalis` |
| 科の数 | 19 |
| 配布 zip | 34 本 / 3,597,653,809 バイト |
| 同一内容の重複 zip | 1 組 |

## 配布物の実測(展開せず zip の中身を数えた)

| 項目 | 実測 |
|---|---|
| 走査した zip | 34 本 |
| フォルダ(分類群) | 33 |
| 配布元の記載枚数 | 2,722 |
| **実測枚数** | **2,094** |
| 一意な画像(SHA-256) | 2,028 |
| バイト完全一致の重複 | 66 組 / 余分な複製 66 枚 |
| 画像形式 | 全て正方: True / モード: RGBA |
| キャンバス幅 | 739〜1532 px(100 通り) |

**記載 2,722 枚に対し実測 2,094 枚(差 628 枚)。** 記載が結合前後のどちらを指すのか判別できないため、
本プロジェクトは実測値のみを使う。

### キャンバス寸法は分類群の指紋になるか(G-11 の材料)

- 寸法が 1 通りしかないフォルダ: 0 個
- 単一のフォルダにしか現れない寸法: 35 通り
- その寸法を持つ画像: 88 枚(4.2%)

寸法だけで分類群が特定できる画像は少数だが、**少数でもゼロではない**。
画素を一切見ない分類器の成績を G-11 で実測してから、CNN の成績を語る。

## 前処理と分割(loop_002)

| 項目 | 値 |
|---|---|
| 切り出し | 視野円の内接正方形の中心 512x512 px(リサイズなし) |
| 視野円の前提に反した画像 | 0 枚 |
| 学習に使う画像 | 1,966 枚(除外分類群の 62 枚を除く) |
| group | 929 個(2 枚以上を含むもの 388 個・最大 12 枚) |
| スライドの代理で結合したセル | 388(同一 分類群+寸法・12 枚以下) |
| 知覚ハッシュで結合した対 | 3(ハミング距離 ≤ 45) |
| **group を作れなかった画像** | **503 枚(25.6%)** |

物差し A の枚数: {'train': 1390, 'val': 290, 'test': 286}

物差し B は分類群を抜く 8 fold、物差し C は科を抜く 6 fold。
各 fold の test には Gram 陽性と陰性の両方が入るようにしてある ——
片方しか無い fold で macro F1 を取ると、存在しないクラスの F1 が 0 として混ざり、
成績が fold の作り方の産物になる。

**限界:** 撮影セッション ID は配布物に無い。同一 (分類群, キャンバス寸法) をスライドの代理として使っているが、これは代理であって本物ではない。1225px のような大きなセルは group を作れないまま残り、その分だけ物差し A は甘い

閾値をどう決めたかは [`reports/group_calibration/README.md`](../reports/group_calibration/README.md) にある。

## 科ごとの分類群数(物差し C が切る単位)

| 科 | 分類群数 |
|---|---|
| Lactobacillaceae | 11 |
| Staphylococcaceae | 3 |
| Enterococcaceae | 2 |
| Moraxellaceae | 1 |
| Actinomycetaceae | 1 |
| Bacteroidaceae | 1 |
| Bifidobacteriaceae | 1 |
| Clostridiaceae | 1 |
| Enterobacteriaceae | 1 |
| Fusobacteriaceae | 1 |
| Listeriaceae | 1 |
| Micrococcaceae | 1 |
| Neisseriaceae | 1 |
| Porphyromonadaceae | 1 |
| Propionibacteriaceae | 1 |
| Morganellaceae | 1 |
| Pseudomonadaceae | 1 |
| Streptococcaceae | 1 |
| Veillonellaceae | 1 |

最大の科は **Lactobacillaceae** の 11 分類群。
種を 1 つ抜いても近縁が学習側に残るため、物差し B は「未知の種」ではなく
近縁の見覚えを測りうる。物差し C を科の単位に置いたのはこのためである。

## 分類群ごとの採録内容

| フォルダ名 | 現行の学名 | Gram | 一致率 | 形態 | 一致率 | 科 | Stage B |
|---|---|---|---|---|---|---|---|
| Acinetobacter baumannii | Acinetobacter baumannii | 陰性 | 100.0% | — | 58.8% | Moraxellaceae | × |
| Actinomyces israelii | Actinomyces israelii | 陽性 | 100.0% | 桿菌 | 100.0% | Actinomycetaceae | ○ |
| Bacteroides fragilis | Bacteroides fragilis | 陰性 | 100.0% | 桿菌 | 100.0% | Bacteroidaceae | ○ |
| Bifidobacterium spp | **Bifidobacterium** | 陽性 | 97.1% | 桿菌 | 100.0% | Bifidobacteriaceae | ○ |
| Clostridium perfringens | Clostridium perfringens | 陽性 | 100.0% | 桿菌 | 100.0% | Clostridiaceae | ○ |
| Enterococcus faecalis | Enterococcus faecalis | 陽性 | 100.0% | 球菌 | 100.0% | Enterococcaceae | ○ |
| Enterococcus faecium | Enterococcus faecium | 陽性 | 100.0% | 球菌 | 100.0% | Enterococcaceae | ○ |
| Escherichia coli | Escherichia coli | 陰性 | 100.0% | 桿菌 | 100.0% | Enterobacteriaceae | ○ |
| Fusobacterium | Fusobacterium | 陰性 | 87.5% | 桿菌 | 100.0% | Fusobacteriaceae | ○ |
| Lactobacillus casei | **Lacticaseibacillus casei** | 陽性 | 100.0% | 桿菌 | 100.0% | Lactobacillaceae | ○ |
| Lactobacillus crispatus | Lactobacillus crispatus | 陽性 | 100.0% | 桿菌 | 100.0% | Lactobacillaceae | ○ |
| Lactobacillus delbrueckii | Lactobacillus delbrueckii | 陽性 | 100.0% | 桿菌 | 100.0% | Lactobacillaceae | ○ |
| Lactobacillus gasseri | Lactobacillus gasseri | 陽性 | 100.0% | 桿菌 | 96.0% | Lactobacillaceae | ○ |
| Lactobacillus jensenii | Lactobacillus jensenii | 陽性 | 100.0% | 桿菌 | 92.3% | Lactobacillaceae | ○ |
| Lactobacillus johnsonii | Lactobacillus johnsonii | 陽性 | 100.0% | 桿菌 | 100.0% | Lactobacillaceae | ○ |
| Lactobacillus paracasei | **Lacticaseibacillus paracasei** | 陽性 | 100.0% | 桿菌 | 100.0% | Lactobacillaceae | ○ |
| Lactobacillus plantarum | **Lactiplantibacillus plantarum** | 陽性 | 100.0% | 桿菌 | 95.8% | Lactobacillaceae | ○ |
| Lactobacillus reuteri | **Limosilactobacillus reuteri** | 陽性 | 100.0% | 桿菌 | 90.9% | Lactobacillaceae | ○ |
| Lactobacillus rhamnosus | **Lacticaseibacillus rhamnosus** | 陽性 | 100.0% | 桿菌 | 100.0% | Lactobacillaceae | ○ |
| Lactobacillus salivarius | **Ligilactobacillus salivarius** | 陽性 | 100.0% | 桿菌 | 100.0% | Lactobacillaceae | ○ |
| Listeria monocytogenes | Listeria monocytogenes | 陽性 | 100.0% | 桿菌 | 100.0% | Listeriaceae | ○ |
| Micrococcus spp | **Micrococcus** | 陽性 | 100.0% | 球菌 | 100.0% | Micrococcaceae | ○ |
| Neisseria gonorrhoeae | Neisseria gonorrhoeae | 陰性 | 100.0% | 球菌 | 100.0% | Neisseriaceae | ○ |
| Porphyromonas gingivalis | Porphyromonas gingivalis | 陰性 | 100.0% | — | 0.0% | Porphyromonadaceae | × |
| Propionibacterium acnes | **Cutibacterium acnes** | 陽性 | 100.0% | 桿菌 | 97.0% | Propionibacteriaceae | ○ |
| Proteus | Proteus | 陰性 | 100.0% | 桿菌 | 100.0% | Morganellaceae | ○ |
| Pseudomonas aeruginosa | Pseudomonas aeruginosa | 陰性 | 100.0% | 桿菌 | 100.0% | Pseudomonadaceae | ○ |
| Staphylococcus aureus | Staphylococcus aureus | 陽性 | 100.0% | 球菌 | 100.0% | Staphylococcaceae | ○ |
| Staphylococcus epidermidis | Staphylococcus epidermidis | 陽性 | 100.0% | 球菌 | 100.0% | Staphylococcaceae | ○ |
| Staphylococcus saprophyticus | Staphylococcus saprophyticus | 陽性 | 100.0% | 球菌 | 100.0% | Staphylococcaceae | ○ |
| Streptococcus agalactiae | Streptococcus agalactiae | 陽性 | 100.0% | 球菌 | 100.0% | Streptococcaceae | ○ |
| Veillonella | Veillonella | 陰性 | 100.0% | 球菌 | 93.3% | Veillonellaceae | ○ |

太字は、フォルダ名が旧名で現行の学名と違うもの。
`Lactobacillus` 属は 2020 年に分割され、11 フォルダは現在 5 属にまたがる。
**フォルダ名をそのまま属として切ると、物差し C が意図した単位で切れない**(検査 T-107)。

## 除外した分類群

### Candida albicans

- 理由: 細菌ではない(酵母・真菌)。Gram 陽性菌として扱うと、Gram 陽性クラスに真菌の見た目が混ざる
- 典拠: NCBI Taxonomy taxid 5476 — https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=5476(取得日 2026-09-08)
- 権威の回答: Lineage: cellular organisms; Eukaryota; Opisthokonta; Fungi; Dikarya; Ascomycota; saccharomyceta; Saccharomycotina; Pichiomycetes; Serinales; Debaryomycetaceae; Candida
- 備考: 細菌の権威である BacDive に該当分類群が無かったことも、独立の傍証になっている

## 種水準の典拠が無かったもの

### Porphyromonas gingivalis

- 補いの水準: **genus**(BacDive — 属 Porphyromonas の株記録)
- 権威の回答: 属 43 株のうち細胞形態の記録がある 3 株はすべて gram=negative / rod-shaped (BacDive 133139 P. loveana / 12513 P. bennonis / 12505 P. asaccharolytica)
- 注意: P. gingivalis そのものには BacDive に Gram の記録が無い。属水準の合議で補っており、種水準の典拠ではない。形態は権威の記述が short rod / coccobacilli と揺れるため Stage B の対象外とする

## 門と Gram の対応(近道が成立しないことの実測)

| 門 | Gram | 分類群数 |
|---|---|---|
| Actinomycetota | 陽性 | 4 |
| Bacillota | 陰性 | 1 |
| Bacillota | 陽性 | 19 |
| Bacteroidota | 陰性 | 2 |
| Fusobacteriota | 陰性 | 1 |
| Pseudomonadota | 陰性 | 5 |

**門から Gram は決められない。** 陽性と陰性が同居する門: Bacillota。
採用した他の Bacillota はすべて陽性なので、門で決め打つ実装はここだけ静かに間違える(検査 T-108)。

