"""CI の依存が、検査が実行時に読み込むものを網羅していることの検査(TEST_SPEC T-260)。

なぜ要るのか(loop_009):
  loop_007 で CI の依存を「検査が実際に import するもの」に絞ったが、その実測は
  grep で**行頭の import 文だけ**を拾っていた。loop_008 で足した T-256 は関数の中で
  `import torch` しており、網を抜けた。手元の venv には torch があるので緑に見え、
  CI は GitHub に上がっておらず一度も走っていなかったので、誰も気づけなかった。
  clone した空のディレクトリで CI と同じ依存だけを入れて再現して、初めて露見した。

**計器が測る対象を覆っていなかった。** だから依存の網羅そのものを検査にする。
ast で数えるので、関数の中の import も、検査が読み込む ml モジュールの
モジュール水準の import も落ちない。

期待値の出所: `.github/workflows/ci.yml` の pip install 行(実物)と、tests/ と ml/ の構文木(実物)。
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# import 名 → pip の配布名。載っていない名前は import 名と同じとみなす
PIP_NAME = {"sklearn": "scikit-learn", "PIL": "pillow", "onnxruntime": "onnxruntime"}
STDLIB = set(sys.stdlib_module_names)
LOCAL = {"ml", "tests"}


def _imports(tree: ast.AST, module_level_only: bool) -> set[str]:
    """ast から最上位のモジュール名を集める。"""
    nodes = tree.body if module_level_only else list(ast.walk(tree))
    out: set[str] = set()
    for n in nodes:
        if isinstance(n, ast.Import):
            out |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            out.add(n.module.split(".")[0])
    return out


def _ml_modules_used_by_tests(tree: ast.AST) -> set[str]:
    mods: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("ml."):
            mods.add(n.module.split(".")[1])
        elif isinstance(n, ast.Import):
            mods |= {a.name.split(".")[1] for a in n.names if a.name.startswith("ml.")}
    return mods


def required_distributions() -> set[str]:
    needed: set[str] = set()
    ml_mods: set[str] = set()
    for f in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        # 検査ファイルは関数の中の import も数える —— ここを行頭だけにしたのが loop_009 の穴
        needed |= _imports(tree, module_level_only=False)
        ml_mods |= _ml_modules_used_by_tests(tree)
    for m in sorted(ml_mods):
        tree = ast.parse((ROOT / "ml" / f"{m}.py").read_text(encoding="utf-8"))
        # ml モジュールは import した瞬間に走るモジュール水準だけを数える。
        # 関数の中(main など)は検査から呼ばれないので、数えると依存を過大に見積もる
        needed |= _imports(tree, module_level_only=True)
    names = {x for x in needed if x not in STDLIB and x not in LOCAL and x != "__future__"}
    return {PIP_NAME.get(x, x).lower() for x in names}


def installed_in_ci() -> set[str]:
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    pkgs: set[str] = set()
    for line in text.splitlines():
        m = re.search(r"pip install (.+)$", line)
        if not m:
            continue
        for tok in m.group(1).split():
            if tok.startswith("-") or tok.startswith("http") or tok in ("pip",):
                continue
            pkgs.add(re.split(r"[<>=\[]", tok)[0].lower())
    return pkgs


@pytest.mark.unit
def test_t260_ci_installs_everything_the_tests_import() -> None:
    """T-260 / N-06 — CI の pip install が、検査の実行時に要る配布物をすべて入れている。"""
    need = required_distributions()
    have = installed_in_ci()
    missing = sorted(need - have)
    assert missing == [], (
        f"CI に入っていない依存: {missing}(検査が要るもの {sorted(need)} / CI {sorted(have)})"
    )


@pytest.mark.unit
def test_t261_the_scan_sees_function_local_imports() -> None:
    """T-261 — 走査器の陽性対照: 関数の中の import を見落とさない。

    loop_009 の穴は、計器が行頭の import しか見ていなかったことだった。
    計器そのものが関数内の import を拾えることを、合成の入力で確かめる。
    """
    src = "import os\n\ndef f():\n    import torch\n    from sklearn.metrics import f1_score\n"
    tree = ast.parse(src)
    assert _imports(tree, module_level_only=False) >= {"os", "torch", "sklearn"}
    assert "torch" not in _imports(tree, module_level_only=True), "モジュール水準の走査が関数内まで拾っている"
