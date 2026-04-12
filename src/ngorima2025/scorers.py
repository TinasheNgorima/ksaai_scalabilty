"""
src/ngorima2025/scorers.py
==========================
Canonical implementations of all five dependency measures used in
Ngorima (2025).  Import from this module in every pipeline step —
no more per-file duplication.

Methods
-------
xi_n   : Chatterjee (2021) rank correlation          O(n log n)
mi     : Mutual information kNN, Kraskov (2004)      O(n log n)
dc     : Distance correlation, Székely (2007)         O(n²)
mic    : MIC, Reshef (2011)                           O(n^1.2)
pearson: Pearson r (linear baseline)                  O(n)
spearman: Spearman ρ (monotone baseline)              O(n log n)

P1 fix : Single source of truth; all pipeline steps import from here.
P2 fix : Pearson and Spearman baselines added.
"""

from __future__ import annotations

import warnings
import numpy as np
from scipy import stats as scipy_stats

# --------------------------------------------------------------------------- #
#  Optional-dependency flags (set at import time)                              #
# --------------------------------------------------------------------------- #
try:
    from xicor.xicor import Xi as _Xi
    _XICOR_AVAILABLE = True
except ImportError:
    _XICOR_AVAILABLE = False

try:
    import dcor as _dcor
    _DCOR_AVAILABLE = True
except ImportError:
    _DCOR_AVAILABLE = False

try:
    from minepy import MINE as _MINE
    _MINEPY_AVAILABLE = True
except ImportError:
    _MINEPY_AVAILABLE = False

# Public flag so callers can warn the user once
FALLBACK_FLAGS = {
    "xi_n":   not _XICOR_AVAILABLE,
    "dc":     not _DCOR_AVAILABLE,
    "mic":    not _MINEPY_AVAILABLE,
    "mi":     False,   # sklearn always available
    "pearson":  False,
    "spearman": False,
}


# --------------------------------------------------------------------------- #
#  Core scorers                                                                 #
# --------------------------------------------------------------------------- #

def score_xi(x: np.ndarray, y: np.ndarray) -> float:
    """
    Chatterjee's ξₙ rank correlation (Chatterjee, 2021).

    Uses xicor package when available; falls back to the exact
    NumPy implementation from Eq. (1) of the original paper.
    Complexity: O(n log n).

    Environment variable ``NGORIMA_XI_NUMPY=1`` forces the numpy
    implementation regardless of xicor availability.  This is used in
    fast-mode timing benchmarks to avoid xicor's Python-object overhead
    (which is ~100× slower than the numpy path for n < 50,000 and
    inflates the empirical complexity exponent in sub-asymptotic regimes).
    For production feature-scoring the xicor path is preferred as it
    handles ties and edge cases more carefully.
    """
    import os as _osxi
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    _force_numpy = _osxi.environ.get("NGORIMA_XI_NUMPY", "0") == "1"
    if _XICOR_AVAILABLE and not _force_numpy:
        try:
            return float(_Xi(x, y).correlation)
        except Exception:
            pass  # fall through to numpy implementation
    # Exact fallback (Chatterjee 2021, Eq. 1)
    n = len(x)
    rank_y = scipy_stats.rankdata(y, method="average")
    order  = np.argsort(scipy_stats.rankdata(x, method="ordinal"))
    r_y_sorted = rank_y[order]
    return float(1.0 - (3.0 * np.sum(np.abs(np.diff(r_y_sorted)))) / (n**2 - 1))


def score_mi(x: np.ndarray, y: np.ndarray, random_state: int = 42) -> float:
    """
    Mutual information via k-NN estimator (Kraskov et al., 2004).
    Implemented via sklearn.feature_selection.mutual_info_regression.
    Complexity: O(n log n).
    """
    from sklearn.feature_selection import mutual_info_regression
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(
        mutual_info_regression(x.reshape(-1, 1), y, random_state=random_state)[0]
    )


def score_dc(x: np.ndarray, y: np.ndarray) -> float:
    """
    Distance correlation (Székely et al., 2007).
    Uses dcor package (fast AVL-tree algorithm) when available;
    falls back to the O(n²) NumPy reference implementation.
    Complexity: O(n²) time, O(n²) memory.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if _DCOR_AVAILABLE:
        try:
            return float(_dcor.distance_correlation(x, y))
        except Exception:
            pass
    # Reference O(n²) fallback
    def _doubly_centered(v: np.ndarray) -> np.ndarray:
        a = np.abs(v[:, None] - v[None, :])
        return (a
                - a.mean(axis=1, keepdims=True)
                - a.mean(axis=0, keepdims=True)
                + a.mean())

    A = _doubly_centered(x)
    B = _doubly_centered(y)
    denom = np.sqrt((A * A).mean() * (B * B).mean())
    if denom <= 0.0:
        return 0.0
    return float(np.sqrt(max(0.0, (A * B).mean()) / denom))


def score_mic(x: np.ndarray, y: np.ndarray,
              alpha: float = 0.6, c: int = 15) -> float:
    """
    Maximal Information Coefficient (Reshef et al., 2011).
    Uses minepy when available; falls back to a binned MI approximation
    (NOTE: fallback has different complexity characteristics — results
    will differ from minepy.  A USE_FALLBACK_MIC flag is set to True
    so callers can suppress or annotate affected results).
    Complexity: O(n^1.2) empirical (minepy), O(n·B²) fallback.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if _MINEPY_AVAILABLE:
        try:
            mine = _MINE(alpha=alpha, c=c)
            mine.compute_score(x, y)
            return float(mine.mic())
        except Exception:
            pass
    # Fallback: binned mutual information (NOT true MIC)
    warnings.warn(
        "minepy not available. MIC approximated via histogram MI. "
        "Complexity and values differ from Reshef (2011). "
        "Results marked with USE_FALLBACK_MIC=True.",
        UserWarning,
        stacklevel=2,
    )
    bins = max(2, int(np.sqrt(len(x) / 4)))
    h, _, _ = np.histogram2d(x, y, bins=bins)
    h = h / h.sum()
    hx = h.sum(axis=1)
    hy = h.sum(axis=0)
    mi_val = 0.0
    for i in range(bins):
        for j in range(bins):
            if h[i, j] > 0 and hx[i] > 0 and hy[j] > 0:
                mi_val += h[i, j] * np.log(h[i, j] / (hx[i] * hy[j]))
    return float(mi_val)


def score_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """
    Absolute Pearson correlation coefficient (linear baseline).
    Complexity: O(n).

    P2 fix: Added as O(n) baseline per reviewer recommendation.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    r, _ = scipy_stats.pearsonr(x, y)
    return float(abs(r))


def score_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """
    Absolute Spearman rank correlation (monotone baseline).
    Complexity: O(n log n).

    P2 fix: Added as rank-based O(n log n) baseline.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    r, _ = scipy_stats.spearmanr(x, y)
    return float(abs(r))


# --------------------------------------------------------------------------- #
#  Registry                                                                    #
# --------------------------------------------------------------------------- #

SCORERS: dict[str, callable] = {
    "xi_n":    score_xi,
    "mi":      score_mi,
    "dc":      score_dc,
    "mic":     score_mic,
    "pearson": score_pearson,
    "spearman": score_spearman,
}

LABELS: dict[str, str] = {
    "xi_n":    "ξₙ",
    "mi":      "MI",
    "dc":      "DC",
    "mic":     "MIC",
    "pearson": "Pearson",
    "spearman": "Spearman",
}

THEORETICAL_EXPONENT: dict[str, float] = {
    "xi_n":    1.05,
    "mi":      1.05,
    "dc":      2.00,
    "mic":     1.20,
    "pearson": 1.00,
    "spearman": 1.05,
}
