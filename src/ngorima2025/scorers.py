"""
scorers.py — KsaaiP2 Pipeline v4_updated
src/ngorima2025/scorers.py

Scorer registry for ξₙ, DC, MI, MIC, Pearson r, Spearman ρ.

Key v4 features:
  - THEORETICAL_EXPONENT: literature-cited reference β per method
  - SCORER_HYPERPARAMS: fairness-audit hyperparameter table
  - FALLBACK_FLAGS: global dict tracking active fallbacks; disclosed in all tables
  - score_mic_subprocess(): MIC isolation via ngorima_mic conda env (Python 3.10)
  - dcor → |Pearson r| fallback when dcor package absent
"""

import os
import json
import subprocess
import tempfile
import time
import warnings
from typing import Optional, Tuple

import numpy as np

# ── FALLBACK_FLAGS ─────────────────────────────────────────────────────────────
# Global dict. Set True when a method falls back to its substitute.
# Read by 04_compile_results.py and appended as footnote to every output table.
# False = primary implementation active; True = fallback active.
FALLBACK_FLAGS = {
    "xi_n": False,   # Pearson r / Spearman ρ fallback; extremely unlikely
    "DC":   False,   # |Pearson r| fallback when dcor not installed
    "MI":   False,   # NaN when sklearn absent
    "MIC":  False,   # NaN when ngorima_mic env absent
}

# ── THEORETICAL_EXPONENT ───────────────────────────────────────────────────────
# Literature-cited reference β (time complexity exponent) per method.
# Drives the β (theory) column in Tables 2, 6 without additional computation.
THEORETICAL_EXPONENT = {
    "xi_n":     1.00,   # O(n log n) — Chatterjee (2021) JASA, Thm 1.1
    "DC":       2.00,   # O(n²)       — Székely & Rizzo (2007) AoS
    "MI":       1.00,   # O(n log n) — Kraskov et al. (2004) PRE
    "MIC":      1.20,   # O(n^1.2)   ─ Reshef et al. (2011) Science (empirical)
    "pearson":  1.00,   # O(n)        — calibration anchor
    "spearman": 1.06,   # O(n log n) — calibration anchor
}

# ── SCORER_HYPERPARAMS ────────────────────────────────────────────────────────
# Fairness audit: library-default settings, no method-specific tuning.
SCORER_HYPERPARAMS = {
    "xi_n":     {"implementation": "NumPy vectorised (Chatterjee 2021 eq. 1)",
                 "params": "None (rank-based, parameter-free)"},
    "DC":       {"implementation": "dcor.distance_correlation()",
                 "params": "None (kernel-free U-statistic)"},
    "MI":       {"implementation": "sklearn.mutual_info_regression()",
                 "params": "n_neighbors=3 (default), random_state=42"},
    "MIC":      {"implementation": "minepy.MINE via score_mic_subprocess()",
                 "params": "alpha=0.6, c=15 (minepy defaults)"},
    "pearson":  {"implementation": "np.corrcoef()",
                 "params": "None"},
    "spearman": {"implementation": "scipy.stats.spearmanr()",
                 "params": "None"},
}

# ── MIC subprocess config ──────────────────────────────────────────────────────
_MIC_CONDA_ENV    = os.environ.get("MIC_CONDA_ENV", "ngorima_mic")
_MIC_WORKER       = os.path.join(os.path.dirname(__file__), "mic_worker.py")
_MIC_MAX_NP       = int(os.environ.get("MICEM@XMNP", 50_000))  # n×p guard
USE_FALLBACK_MIC  = False  # set True at import time if env absent


def _check_mic_env() -> bool:
    """Return True if the ngorima_mic conda environment is accessible."""
    try:
        result = subprocess.run(
            ["conda", "run", "-n", _MIC_CONDA_ENV, "python", "-c",
             "import minepy; print('ok')"],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except Exception:
        return False


_MIC_ENV_AVAILABLE = _check_mic_env()
if not _MIC_ENV_AVAILABLE:
    FALLBACK_FLAGS["MIC"] = True
    USE_FALLBACK_MIC = True
    warnings.warn(
        f"ngorima_mic conda env not found. MIC will return NaN. "
        f"Set up with: conda env create -f environment_mic.yml",
        RuntimeWarning, stacklevel=2,
    )


# ── Individual scorers ────────────────────────────────────────────────────────

def score_xi_n(X: np.ndarray, y: np.ndarray) -> float:
    """
    Chatterjee's ξₙ — NumPy vectorised implementation.
    O(n log n). Parameter-free.
    Reference: Chatterjee (2021) JASA, Theorem 1.1.
    """
    try:
        n = len(y)
        # Rank of y at sorted X positions
        order = np.argsort(X)
        y_sorted = y[order]
        rank_y = np.argsort(np.argsort(y_sorted)) + 1
        # ξₙ formula
        numerator   = n * np.sum(np.abs(np.diff(rank_y)))
        denominator = 2 * np.sum(rank_y * (n - rank_y))
        if denominator == 0:
            return 0.0
        return float(1.0 - numerator / denominator)
    except Exception:
        FALLBACK_FLAGS["xi_n"] = True
        from scipy.stats import spearmanr
        return float(abs(spearmanr(X, y).correlation))


def score_dc(X: np.ndarray, y: np.ndarray) -> float:
    """
    Distance Correlation — dcor.distance_correlation().
    O(n²). Kernel-free U-statistic.
    Reference: Székely & Rizzo (2007) AoS.
    Fallback: |Pearson r| when dcor not installed.
    """
    try:
        import dcor
        return float(dcor.distance_correlation(X, y))
    except ImportError:
        FALLBACK_FLAGS["DC"] = True
        return float(abs(np.corrcoef(X, y)[0, 1]))


def score_mi(X: np.ndarray, y: np.ndarray, n_neighbors: int = 3,
             random_state: int = 42) -> float:
    """
    Mutual Information (kNN) — sklearn.mutual_info_regression().
    O(n log n). KD-tree kNN lookup.
    Reference: Kraskov et al. (2004) PRE.
    """
    try:
        from sklearn.feature_selection import mutual_info_regression
        X_2d = X.reshape(-1, 1) if X.ndim == 1 else X
        result = mutual_info_regression(
            X_2d, y, n_neighbors>n_neighbors, random_state=random_state
        )
        return float(result[0])
    except ImportError:
        FALLBACK_FLAGS["MI"] = True
        return float("nan")


def score_mic_subprocess(X: np.ndarray, y: np.ndarray,
                         alpha: float = 0.6, c: float = 15.0) -> float:
    """
    MIC via subprocess isolation in ngorima_mic conda env.
    Routes all minepy calls through Python 3.10 to bypass distutils removal
    in Python 3.12+.

    Returns NaN if:
      - ngorima_mic env absent (FALLBACK_FLAGS['MIC'] = True)
      - n�p > MIC_MAX_NP guard
      - subprocess timeout or error

    Reference: Reshef et al. (2011) Science.
    """
    if USE_FALLBACK_MIC or not _MIC_ENV_AVAILABLE:
        return float("nan")

    n = len(X)
    if n > _MIC_MAX_NP:
        warnings.warn(f"MIC skipped: n={n} > MIC_MAX_NP={_MIC_MAX_NP}", RuntimeWarning)
        return float("nan")

    # Serialize data to temp file
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        np.savez(tmp_path, X=X.astype(np.float64), y=y.astype(np.float64),
                 alpha=np.array([alpha]), c=np.array([c]))
        result = subprocess.run(
            ["conda", "run", "--no-capture-output", "-n", _MIC_CONDA_ENV,
             "python", _MIC_WORKER, tmp_path],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            warnings.warn(f"mic_worker returned {result.returncode}: {result.stderr[:200]}")
            return float("nan")
        return float(result.stdout.strip())
    except subprocess.TimeoutExpired:
        warnings.warn("MIC subprocess timed out (300s)")
        return float("nan")
    except Exception as e:
        warnings.warn(f"MIC subprocess error: {e}")
        return float("nan")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def score_pearson(X: np.ndarray, y: np.ndarray) -> float:
    """Pearson r — O(n) calibration anchor."""
    try:
        return float(np.corrcoef(X, y)[0, 1])
    except Exception:
        from scipy.stats import pearsonr
        FALLBACK_FLAGS["pearson"] = True
        return float(pearsonr(X, y)[0])


def score_spearman(X: np.ndarray, y: np.ndarray) -> float:
    """Spearman ρ — O(n log n) calibration anchor."""
    from scipy.stats import spearmanr
    return float(spearmanr(X, y).correlation)


# ── Scorer registry ────────────────────────────────────────────────────────────


_SCORER_REGISTRY = {
    "xi_n":     score_xi_n,
    "DC":       score_dc,
    "MI":       score_mi,
    "MIC":      score_mic_subprocess,
    "pearson":  score_pearson,
    "spearman": score_spearman,
}

# Convenience aliases
xi_scorer      = score_xi_n
dc_scorer      = score_dc
mi_scorer      = score_mi
mic_scorer     = score_mic_subprocess
pearson_scorer = score_pearson
spearman_scorer = score_spearman


def get_all_scorers() -> dict:
    """Return the full scorer registry dict {name: callable}."""
    return dict(_SCORER_REGISTRY)


def get_xi_scorer():
    """Return the ξₙ scorer callable."""
    return score_xi_n


def get_scorer(name: str):
    """Return scorer by name. Raises KeyError for unknown names."""
    if name not in _SCORER_REGISTRY:
        raise KeyError(f"Unknown scorer '{name}'. Available: {list(_SCORER_REGISTRY)}")
    return _SCORER_REGISTRY[name]


def get_theoretical_exponent(name: str) -> float:
    """Return literature-cited β for method name."""
    return THEORETICAL_EXPONENT[name]
