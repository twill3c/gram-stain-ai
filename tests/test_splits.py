"""前処理と分割に対する検査(TEST_SPEC T-201〜T-203・T-217〜T-221)。

期待値の出所:
  - 視野円の幾何: 2026-09-08 実測(33 分類群 102 枚の標本で、直径 = キャンバス幅・
    中心ずれ 0.00px・面積が理論円の 0.9999〜1.0001 倍)。全画像に対しては前処理側が assert する
  - 重複の実在: 2026-09-08 実測(バイト完全一致 66 組。うち 6 組は同一 zip 内)
  - 分割の不変量: SPEC §6.1

**この検査は実装より先に書いた。** 走らせて赤であることを確かめてから前処理を書く。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "dataset" / "metadata"

SPLIT_NAMES = ("train", "val", "test")
RAW = ROOT / "dataset" / "raw"


def _require(name: str) -> Path:
    """派生物を要求する。

    **素材がある環境で派生物が無いのは失敗であって skip ではない。**
    skip にすると、生成器が黙って出力しなくなったときに検査が緑のまま通る。
    素材(dataset/raw)を持たない環境(取り立ての clone など)だけ skip する。
    """
    path = META / name
    if path.exists():
        return path
    if not list(RAW.glob("*.zip")):
        pytest.skip(f"dataset/raw に素材が無い環境。先に python ml/download_dataset.py を実行する")
    pytest.fail(f"{name} が無い。素材はあるので生成できるはず: python ml/prepare_dataset.py")


def _load(name: str) -> dict:
    return json.loads(_require(name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def prepared() -> list[dict]:
    """前処理後の画像台帳。1 枚 1 行。"""
    path = _require("prepared.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def splits() -> dict:
    return _load("splits.json")


# ---------------------------------------------------------------- 前処理


@pytest.mark.integration
def test_t217_crop_is_inside_the_field_circle(prepared: list[dict]) -> None:
    """T-217 / SPEC §6.2 — 切り出しが視野円の内側に収まっている。

    黒い四隅は分類情報ではないのに、撮影ごとに違う量で入ってくる。
    切り出しが円からはみ出していれば、はみ出した分だけ近道の材料が増える。
    """
    assert prepared, "前処理された画像が 1 枚も無い"
    bad = [
        r["source_member"]
        for r in prepared
        if r["crop_size"] > r["source_width"] / (2**0.5)
    ]
    assert bad == [], f"内接正方形からはみ出す切り出し: {bad[:5]}(計 {len(bad)} 枚)"


@pytest.mark.integration
def test_t218_all_crops_share_one_pixel_size(prepared: list[dict]) -> None:
    """T-218 / SPEC §6.2 — 切り出しは全画像で同じ画素寸法(画像ごとにリサイズしない)。

    寸法の違う画像を一律に縮めると、縮小率が画像ごとに変わり、菌の見かけの大きさが変わる。
    球菌と桿菌を見分ける課題では、見かけの大きさは情報である。
    """
    sizes = Counter(r["crop_size"] for r in prepared)
    assert len(sizes) == 1, f"切り出し寸法が複数ある: {dict(sizes)}"


@pytest.mark.integration
def test_t219_prepared_images_are_deduplicated(prepared: list[dict]) -> None:
    """T-219 / G-03 — 前処理後の台帳にバイト完全一致の重複が残っていない。

    2026-09-08 実測で 66 組の完全一致がある(60 組は重複 zip 由来、6 組は同一 zip 内)。
    前処理はこれを 1 枚に畳んでいなければならない。
    """
    digests = [r["source_sha256"] for r in prepared]
    dups = [d for d, n in Counter(digests).items() if n > 1]
    assert dups == [], f"前処理後にも同一 SHA-256 が残っている: {len(dups)} 組"


# ---------------------------------------------------------------- group


@pytest.mark.integration
def test_t220_near_duplicates_share_a_group(prepared: list[dict]) -> None:
    """T-220 / G-04 — 知覚ハッシュの近傍が同じ group にまとめられている。

    同一視野の近接ショットが多数ある(loop_001 の目視で確認)。
    バイト一致しないので SHA-256 では畳めない。まとめ損ねると、
    ほぼ同じ画像が train と test に分かれて入る。
    """
    assert all("group_id" in r for r in prepared), "group_id を持たない行がある"
    # まとめが 1 枚ずつに割れていたら、まとめていないのと同じ
    groups = Counter(r["group_id"] for r in prepared)
    multi = sum(1 for n in groups.values() if n > 1)
    assert multi > 0, (
        "2 枚以上を含む group が一つも無い。近接ショットが実在するのにまとめられていない"
    )


@pytest.mark.integration
def test_t221_group_ids_do_not_cross_taxa(prepared: list[dict]) -> None:
    """T-221 / G-04 — 一つの group が二つの分類群にまたがらない。

    またがるなら、まとめすぎている(別の菌を同じ視野と見なした)。
    """
    by_group: dict[str, set[str]] = {}
    for r in prepared:
        by_group.setdefault(r["group_id"], set()).add(r["folder"])
    crossing = {g: sorted(f) for g, f in by_group.items() if len(f) > 1}
    assert crossing == {}, f"分類群をまたぐ group: {list(crossing.items())[:3]}"


# ---------------------------------------------------------------- 分割


@pytest.mark.integration
def test_t201_no_identical_image_crosses_a_split(splits: dict, prepared: list[dict]) -> None:
    """T-201 / G-03 — 分割をまたぐ SHA-256 完全重複が 0 件。"""
    by_id = {r["image_id"]: r for r in prepared}
    a = splits["ruler_a"]
    seen: dict[str, str] = {}
    for name in SPLIT_NAMES:
        for image_id in a[name]:
            digest = by_id[image_id]["source_sha256"]
            if digest in seen and seen[digest] != name:
                pytest.fail(f"同一画像が {seen[digest]} と {name} の両方にある: {image_id}")
            seen[digest] = name


@pytest.mark.integration
def test_t202_no_group_crosses_a_split(splits: dict, prepared: list[dict]) -> None:
    """T-202 / T-203 / G-04 — 同一 group ID が二つの分割に現れない。

    近接ショットをまとめた group が分割をまたぐと、
    「見たことのある視野」を test で当てることになる。
    """
    by_id = {r["image_id"]: r for r in prepared}
    a = splits["ruler_a"]
    seen: dict[str, str] = {}
    for name in SPLIT_NAMES:
        for image_id in a[name]:
            g = by_id[image_id]["group_id"]
            if g in seen and seen[g] != name:
                pytest.fail(f"group {g} が {seen[g]} と {name} にまたがっている")
            seen[g] = name


@pytest.mark.integration
def test_t222_ruler_a_is_stratified_over_taxa(splits: dict, prepared: list[dict]) -> None:
    """T-222 / F-09 — 物差し A では、各分類群が train / val / test すべてに現れる。

    A は「この分類群のスライドを見分けられるか」を測る物差しなので、
    分類群が抜けている分割では測っている対象が変わってしまう。
    """
    by_id = {r["image_id"]: r for r in prepared}
    a = splits["ruler_a"]
    per_split = {
        name: {by_id[i]["folder"] for i in a[name]} for name in SPLIT_NAMES
    }
    all_taxa = {r["folder"] for r in prepared}
    for name in SPLIT_NAMES:
        missing = sorted(all_taxa - per_split[name])
        assert missing == [], f"物差し A の {name} に現れない分類群: {missing}"


@pytest.mark.integration
def test_t223_ruler_b_holds_out_whole_taxa(splits: dict, prepared: list[dict]) -> None:
    """T-223 / F-09 — 物差し B の各 fold で、test の分類群が train に一切現れない。

    ここが漏れていると、B は「未知の種」を測っていない。
    """
    by_id = {r["image_id"]: r for r in prepared}
    folds = splits["ruler_b"]["folds"]
    assert folds, "物差し B の fold が無い"
    for i, fold in enumerate(folds):
        train_taxa = {by_id[x]["folder"] for x in fold["train"]}
        test_taxa = {by_id[x]["folder"] for x in fold["test"]}
        overlap = sorted(train_taxa & test_taxa)
        assert overlap == [], f"物差し B fold {i}: train と test に共通の分類群 {overlap}"


@pytest.mark.integration
def test_t224_ruler_c_holds_out_whole_families(splits: dict, prepared: list[dict]) -> None:
    """T-224 / F-09 — 物差し C の各 fold で、test の科が train に一切現れない。

    32 分類群のうち 11 が Lactobacillaceae 科である(loop_001 実測)。
    科でまとめて抜かないと、近縁が train に残って「未知の科」を測れない。
    """
    by_id = {r["image_id"]: r for r in prepared}
    folds = splits["ruler_c"]["folds"]
    assert folds, "物差し C の fold が無い"
    for i, fold in enumerate(folds):
        train_fam = {by_id[x]["family"] for x in fold["train"]}
        test_fam = {by_id[x]["family"] for x in fold["test"]}
        overlap = sorted(train_fam & test_fam)
        assert overlap == [], f"物差し C fold {i}: train と test に共通の科 {overlap}"


@pytest.mark.integration
def test_t225_every_fold_has_both_gram_classes_in_test(splits: dict, prepared: list[dict]) -> None:
    """T-225 / F-09 — ホールドアウトの各 fold の test に、Gram 陽性と陰性の両方がある。

    片方しか無い fold で macro F1 を計算すると、
    存在しないクラスの F1 が 0 として混ざり、成績が測り方の産物になる。
    **fold の作り方が指標を壊しうる**ので、fold を作る側で保証する。
    """
    by_id = {r["image_id"]: r for r in prepared}
    for ruler in ("ruler_b", "ruler_c"):
        for i, fold in enumerate(splits[ruler]["folds"]):
            grams = {by_id[x]["gram"] for x in fold["test"]}
            assert grams == {"positive", "negative"}, (
                f"{ruler} fold {i} の test にある Gram: {sorted(grams)}。両方が要る"
            )


@pytest.mark.integration
def test_t226_all_images_are_used_exactly_once_per_ruler(splits: dict, prepared: list[dict]) -> None:
    """T-226 / F-09 — 各物差しで、全画像がちょうど一度ずつ使われている。

    取りこぼしがあると、三つの物差しが違う母集団を測ることになり、比べられなくなる。
    """
    all_ids = {r["image_id"] for r in prepared}

    a = splits["ruler_a"]
    used = [i for name in SPLIT_NAMES for i in a[name]]
    assert len(used) == len(set(used)), "物差し A に重複した image_id がある"
    assert set(used) == all_ids, (
        f"物差し A の被覆が全画像と違う(欠け {len(all_ids - set(used))} / "
        f"余り {len(set(used) - all_ids)})"
    )

    for ruler in ("ruler_b", "ruler_c"):
        test_ids = [i for fold in splits[ruler]["folds"] for i in fold["test"]]
        assert len(test_ids) == len(set(test_ids)), f"{ruler} で同じ画像が複数の fold の test にある"
        assert set(test_ids) == all_ids, (
            f"{ruler} の test の和集合が全画像と違う"
            f"(欠け {len(all_ids - set(test_ids))} / 余り {len(set(test_ids) - all_ids)})"
        )
