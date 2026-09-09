# MODEL.md — 配布する学習済みモデル

`public/models/model.onnx` の素性と配布条件。

数はすべて `public/models/model_metadata.json`(配信物そのもの)から転記している。
食い違いがあればメタデータが正しい。

---

## 1. 素性

| 項目 | 値 |
|---|---|
| 版 | 1.0.0 |
| 骨格 | MobileNetV3-Small(torchvision・ImageNet 事前学習 `IMAGENET1K_V1`) |
| パラメータ | 1,519,906 |
| 入力 | 224×224 RGB(CHW・ImageNet 正規化) |
| 出力 | logits 2 個(`gram_positive`, `gram_negative` の順) |
| ファイル | ONNX opset 17・torchscript exporter・**5.95 MB** |
| 学習 | 3 エポック(予算は物差し A の検証用データで決定・試験用は不使用) |

### なぜ ResNet18 ではないのか

SPEC は当初 ResNet18 を標準候補にしていたが、**実測で外した**。

| 骨格 | fp32 サイズ | CPU 学習速度 |
|---|---|---|
| ResNet18 | 46.8 MB | 17.4 枚/秒 |
| EfficientNet-B0 | 21.2 MB | 16.4 枚/秒 |
| **MobileNetV3-Small(採用)** | 10.2 MB(2 クラス化後 6.08 MB) | 60.1 枚/秒 |

ResNet18 は**量子化する前から**配信目標 30 MB を超える。目標を動かさず骨格を変えた。

---

## 2. 成績

| | A 画像単位 | B 分類群ホールドアウト | C 科ホールドアウト |
|---|---|---|---|
| **本モデル** | 0.9731 [0.9580, 0.9858] | 0.9276 [0.9085, 0.9443] | **0.6942** [0.6577, 0.7329] |
| 色相(**学習なし**の対照) | 0.9714 [0.9593, 0.9824] | 0.9524 [0.9363, 0.9673] | 0.9601 [0.9457, 0.9736] |
| 多数派クラス | 0.4171 | 同左 | 同左 |
| ラベル置換(学習系の対照) | 0.4356 [0.4074, 0.4625] | — | — |

macro F1 と 95% 信頼区間。全 1,966 枚を一度ずつ評価。区間は group 単位のブートストラップ 2,000 回。

**このモデルを使う人が知っておくべきこと。**

- 学習で見ていない**科**の菌に対しては、**学習を一切しない色の規則より大きく劣る**
- 全体でも色の規則を上回っていない。上回るのは、色の規則が破れる分類群に限られる
  (`Listeria monocytogenes` 44%→33%、`Porphyromonas gingivalis` 37%→35%、
  `Staphylococcus epidermidis` 14%→2%)

出荷モデル単体の試験成績は macro F1 **1.0**(286 枚)だが、
**この数だけを引用してはならない**。上の表と必ず並べること。

---

## 3. 二実装照合

| 量 | 実測 |
|---|---|
| argmax 一致率 | 1.000000(全 1,966 枚) |
| softmax 確率の最大差 | 1.125e-05 |
| 生 logits の最大絶対差(参考) | 1.0133e-04 |

PyTorch と ONNX Runtime に同じ重みを別実装で走らせた結果。
実ブラウザ(ONNX Runtime Web)でも、画面の表示が Python 側と 0.05 ポイント以内で一致する。

---

## 4. 前処理(モデルを使う側が揃えるもの)

揃えないと、モデルは正しくても答えが変わる。**しかも例外は出ない。**

1. 視野円の内接正方形の中心 **512×512 をリサイズなしで**切り出す(四隅の黒を入れない)
2. **PIL の双線形**で 224×224 へ縮小する
3. 255 で割り、`mean = [0.485, 0.456, 0.406]` / `std = [0.229, 0.224, 0.225]` で正規化
4. CHW の順に並べ替える

**2 を「適当な縮小」で代替しない。** ブラウザの `ctx.drawImage` は補間法が仕様で定められておらず、
実装によって画素が変わる。本リポジトリは PIL の縮小を TypeScript へ移植して使っている
(`src/lib/resize.ts`)。

---

## 5. 学習に使ったデータ

| 項目 | 値 |
|---|---|
| データセット | Bacteria Data for Machine Vision and Digital Biology |
| DOI | [10.17632/cvkgfzp7ck.1](https://doi.org/10.17632/cvkgfzp7ck.1) |
| ライセンス | CC BY 4.0 |
| 採録した分類群 | 32(配布元の 33 から `Candida albicans` を除外) |
| 画像 | 1,966 枚(重複を畳んだ後) |
| ラベルの典拠 | BacDive(Leibniz Institute DSMZ・CC BY 4.0) |

詳細は [DATASET.md](DATASET.md) と [../DATA_LICENSE.md](../DATA_LICENSE.md)。

---

## 6. 配布条件

**このモデルの配布条件は、ソースコード(MIT)と同一ではない。**
次の三つが重なった成果物である。

| 由来 | 条件 |
|---|---|
| 学習データ(上記) | **CC BY 4.0** — 帰属表示が必要 |
| ImageNet 事前学習の重み | torchvision の配布物。BSD-3-Clause(torchvision 本体)。重み自体の再利用は上流の条件に従う |
| 学習・変換に用いたライブラリ | PyTorch(BSD-3-Clause)・ONNX(Apache-2.0)・ONNX Runtime(MIT) |

したがって、**本モデルを再配布・利用する際は CC BY 4.0 の帰属表示を行うこと。**
モデルは CC BY 4.0 のデータの派生物として扱う。

### 帰属表示(そのまま使えます)

> This model was trained on "Bacteria Data for Machine Vision and Digital Biology"
> by Jamshidi, Mohammad (Behdad); Sargolzaee, Saleh; Foorginezhad, Salimeh; Moztarzadeh, Omid
> (2023), Mendeley Data, V1, doi: 10.17632/cvkgfzp7ck.1, licensed under CC BY 4.0.
> Labels were derived from BacDive (Leibniz Institute DSMZ), licensed under CC BY 4.0.
> The images were modified (cropped to the inscribed square of the microscope field,
> deduplicated, and relabelled by Gram reaction).

---

## 7. 使ってはならない用途

- **医学的診断。** 本モデルは医療機器ではなく、出力は診断を意味しない
- 感度・特異度を臨床診断指標として提示すること
- 学習で見ていない科の菌に対する判定を、根拠として用いること(§2 のとおり大きく劣る)
- 100 倍対物の視野以外(倍率・染色手順の異なる画像)への適用。前処理の前提が崩れる
