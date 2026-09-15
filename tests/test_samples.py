"""画面のサンプルに対する検査(TEST_SPEC T-280)。

期待値の出所: dataset/metadata/splits.json の物差し A(出荷モデルは train で学習し val で確認した)。

**実装より先に書く。** loop_012 で、10 枚のうち 5 枚が train・1 枚が val だったのを見つけた。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_t280_samples_are_never_seen_by_the_shipped_models() -> None:
    """T-280 / G-08 — 画面のサンプルは、出荷モデルが学習にも選定にも使っていない画像だけ。

    学習に使った画像を「試してみる」例として出すと、画面の答えはモデルの見覚えになる。
    Gram と形の出荷モデルは、どちらも物差し A の train で学習し val で確認している
    (形は Stage B の対象外を抜いただけで、同じ分割)。だから test からだけ選ぶ。
    """
    splits = ROOT / "dataset" / "metadata" / "splits.json"
    if not splits.exists():
        pytest.skip("分割の記録が無い環境")
    a = json.loads(splits.read_text(encoding="utf-8"))["ruler_a"]
    index = json.loads((ROOT / "public" / "samples" / "index.json").read_text(encoding="utf-8"))
    prepared = {json.loads(line)["image_id"]: json.loads(line)
                for line in (ROOT / "dataset" / "metadata" / "prepared.jsonl").read_text(encoding="utf-8").splitlines()}

    seen = set(a["train"]) | set(a["val"])
    test = set(a["test"])
    groups = []
    for s in index["samples"]:
        assert s["id"] not in seen, f"{s['folder']} のサンプル {s['id']} は出荷モデルが学習か選定で見た画像"
        assert s["id"] in test, f"{s['folder']} のサンプル {s['id']} が test に無い"
        groups.append(prepared[s["id"]]["group_id"])
    assert len(groups) == len(set(groups)), "同じ group(同じスライド)から二枚選んでいる"
    assert len(index["samples"]) == 10
