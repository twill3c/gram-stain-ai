"""指標と信頼区間。

自前で書くものは、権威ある実装(scikit-learn)と突き合わせて検算する(T-227)。
指標が違っていると、そのあとの比較がすべて意味を失う。
"""

from __future__ import annotations

import numpy as np

# クラスの符号。0 = Gram 陽性、1 = Gram 陰性。
# 数の大小に意味は無いが、どこかで決めておかないと混線する
POSITIVE = 0
NEGATIVE = 1
CLASSES = (POSITIVE, NEGATIVE)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """クラスごとの F1 の単純平均。

    予測にも正解にも現れないクラスの F1 は 0 とする(scikit-learn の zero_division=0 に合わせる)。
    **この扱いが効いてくる場面がある** —— 片方のクラスしか無い fold で macro F1 を取ると、
    存在しないクラスの F1 が 0 として混ざり、成績が fold の作り方の産物になる。
    だから fold を作る側で両クラスの存在を保証している(T-225)。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = []
    for c in CLASSES:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return float(np.mean(scores))


def bootstrap_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    seed: int = 20260908,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """group 単位のブートストラップで macro F1 の点推定と 95% 区間を返す。

    **画像単位で再標本してはならない。** 同じスライドの別視野は独立ではないので、
    画像を独立標本として数えると**区間が実際より狭くなる**。
    狭い区間は「差がある」と言いやすくする方向に効くので、
    間違える向きが都合のいい方に揃っている。

    再標本は group を復元抽出し、選ばれた group に属する画像をすべて採る。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)

    point = macro_f1(y_true, y_pred)

    uniq, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)

    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[i] for i in pick])
        stats[b] = macro_f1(y_true[idx], y_pred[idx])

    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return point, lo, hi


def fit_threshold(x: np.ndarray, y: np.ndarray, n_candidates: int = 300) -> float:
    """1 次元の特徴量に閾値を一本引く。x > threshold を NEGATIVE と予測する。

    候補は与えられた x の分位から取る。**この関数に渡してよいのは train だけ**である
    (T-231)。test を渡せば、それは対照ではなく test に当てた上限になる。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    candidates = np.quantile(x, np.linspace(0.01, 0.99, n_candidates))
    best_score, best_t = -1.0, float(np.median(x))
    for t in candidates:
        score = macro_f1(y, (x > t).astype(int))
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t
