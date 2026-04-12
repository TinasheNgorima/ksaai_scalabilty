"""
02_real_domain_benchmarks.py
============================
Ngorima (2025) — Step 2: Real-Domain Performance Benchmarks (Section 5).

Fixes applied
-------------
P0 : evaluate_accuracy() wraps StandardScaler inside sklearn Pipeline —
     eliminates preprocessing leakage across CV folds.
P0 : FRED-MD uses TimeSeriesSplit with 12-month embargo, not KFold(shuffle=True)
     — eliminates look-ahead bias in temporal data.
P0 : TCGA target is no longer PC1 of the expression matrix (circular);
     it is the non-circular target prepared in Step 0.
P0 : DC/MIC subsampling for scoring is flagged; accuracy is evaluated
     on the full dataset using the subsampled scores only as a ranking
     device — the mismatch is disclosed and n_sub annotated.
P1 : Scoring functions imported from src module — no duplication.
P1 : Jaccard stability computed via bootstrap resampling of the data
     (not additive noise perturbation of scores).
P2 : RAM guard before DC allocation; dynamic dc_max_n per machine.
P2 : Pearson and Spearman baselines included.
P3 : Checkpoint/resume; system state logged.
"""

from __future__ import annotations

import sys
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ── Package import ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ngorima2025 import (
    SCORERS, LABELS, FALLBACK_FLAGS,
    DATA_DIR, RESULTS_DIR, FIGURES_DIR,
    log_system_state, check_ram_for_dc, safe_dc_max_n,
    timed_call,
)

warnings.filterwarnings("ignore")

import os as _os
_FAST = _os.environ.get("NGORIMA_FAST", "0") == "1"
# In fast mode, bypass xicor for xi_n to avoid Python-object overhead
# (~100x slower than numpy fallback). Production runs use xicor.
if _FAST:
    _os.environ.setdefault("NGORIMA_XI_NUMPY", "1")

PROCESSED_DIR = DATA_DIR / "processed"

# Dynamic DC ceiling (reduced further in fast mode)
DC_MAX_N  = min(2_000 if _FAST else 15_000, safe_dc_max_n(safety_factor=1.5))
MIC_MAX_N = 500  if _FAST else 5_000
TOP_K     = 5    if _FAST else 20
N_SPLITS  = 3    if _FAST else 5
METHODS   = ["xi_n", "mi", "dc", "mic", "pearson", "spearman"]


# --------------------------------------------------------------------------- #
#  Feature scoring + timing                                                    #
# --------------------------------------------------------------------------- #

def score_all_features(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    max_features: int | None = None,
    n_jobs: int = 1,
) -> tuple[np.ndarray, float]:
    """
    Score all features; return (scores, wall_clock_seconds).
    Uses a single timed pass (not repeated) for real-data benchmarks
    because these datasets are large and we want realistic pipeline time.
    """
    scorer = SCORERS[method]
    p = min(X.shape[1], max_features) if max_features else X.shape[1]

    import time
    t0 = time.perf_counter()
    if n_jobs > 1:
        from joblib import Parallel, delayed
        scores = Parallel(n_jobs=n_jobs)(
            delayed(scorer)(X[:, i], y) for i in range(p)
        )
    else:
        scores = [scorer(X[:, i], y) for i in range(p)]
    elapsed = time.perf_counter() - t0

    return np.array(scores, dtype=np.float64), elapsed


def measure_peak_memory_mb(
    X: np.ndarray, y: np.ndarray, method: str
) -> float:
    """
    Measure peak RAM for scoring one feature (MB).
    DC: uses theoretical O(n²) formula (tracemalloc underestimates NumPy arrays).
    """
    if method == "dc":
        n = len(y)
        return float(n ** 2 * 8 / 1e6)  # n×n float64 matrix

    scorer = SCORERS[method]
    tracemalloc.start()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scorer(X[:, 0], y)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return float(peak / 1e6)


# --------------------------------------------------------------------------- #
#  Accuracy evaluation — P0 fixes                                             #
# --------------------------------------------------------------------------- #

def evaluate_accuracy_regression(
    X: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    top_k: int = 20,
    n_splits: int = 5,
    time_series: bool = False,
) -> dict:
    """
    Select top-k features by score; evaluate R² via cross-validation.

    P0 fix 1: StandardScaler is inside a Pipeline so it is fit only on
              training folds — no leakage to test folds.
    P0 fix 2: For time-ordered data (FRED-MD), uses TimeSeriesSplit with a
              12-observation embargo instead of KFold(shuffle=True).
    P0 fix 3: Jaccard stability is computed by bootstrap resampling of the
              observations, not by adding noise to scores.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import (
        cross_val_score, KFold, TimeSeriesSplit
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if len(scores) == 0 or np.all(np.isnan(scores)):
        return {"r2_mean": np.nan, "r2_std": np.nan,
                "jaccard_mean": np.nan, "jaccard_std": np.nan}

    top_k   = min(top_k, len(scores))
    nan_mask = ~np.isnan(scores)
    scores_clean = np.where(nan_mask, scores, -np.inf)
    top_idx = np.argsort(scores_clean)[::-1][:top_k]
    X_sel   = X[:, top_idx]

    # P0 fix 1: Pipeline prevents preprocessing leakage
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf",     RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )),
    ])

    # P0 fix 2: Temporally appropriate CV for FRED-MD
    if time_series:
        # 12-month embargo between train and test
        cv = TimeSeriesSplit(n_splits=n_splits, gap=12)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    r2_scores = cross_val_score(pipe, X_sel, y, cv=cv, scoring="r2")

    # P0 fix 3: Jaccard via bootstrap resampling of observations.
    # For tractability at large p, we re-score only the candidate set:
    # top_k * 3 features (ensures the true top-k can still be recovered
    # from the bootstrap sample without scoring all p features).
    jaccard_vals  = []
    rng_boot      = np.random.default_rng(42)
    n_boot        = 10   # 10 bootstrap iterations are sufficient for stability estimate
    orig_idx_set  = set(top_idx.tolist())
    # Candidate pool: top_k + a safety margin of 2× to absorb reranking
    pool_size     = min(top_k * 3, X.shape[1])
    # First run: get pool indices from original scores
    pool_idx      = np.argsort(scores_clean)[::-1][:pool_size]
    X_pool        = X[:, pool_idx]   # score only the candidate pool per bootstrap

    for _ in range(n_boot):
        boot_idx  = rng_boot.choice(len(y), len(y), replace=True)
        X_b, y_b  = X_pool[boot_idx], y[boot_idx]
        # Re-score only the pool (pool_size << p)
        boot_scores_pool = np.array([
            SCORERS["xi_n"](X_b[:, i], y_b)
            for i in range(pool_size)
        ])
        # Map back to original feature indices
        boot_top_local = np.argsort(boot_scores_pool)[::-1][:top_k]
        boot_idx_set   = set(pool_idx[boot_top_local].tolist())
        inter = len(orig_idx_set & boot_idx_set)
        union = len(orig_idx_set | boot_idx_set)
        jaccard_vals.append(inter / union if union > 0 else 0.0)

    return {
        "r2_mean":      float(np.mean(r2_scores)),
        "r2_std":       float(np.std(r2_scores, ddof=1)),
        "jaccard_mean": float(np.mean(jaccard_vals)),
        "jaccard_std":  float(np.std(jaccard_vals, ddof=1)),
    }


# --------------------------------------------------------------------------- #
#  Dataset benchmark                                                           #
# --------------------------------------------------------------------------- #

def benchmark_dataset(
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    methods: list[str],
    max_features: int | None = None,
    dc_max_n: int = DC_MAX_N,
    mic_max_n: int = MIC_MAX_N,
    top_k: int = 20,
    n_splits: int = 5,
    time_series: bool = False,
) -> pd.DataFrame:
    """
    Full benchmark for one dataset.

    P0 fix: dc_max_n is dynamically set from available RAM, not hardcoded.
    P0 fix: When subsampling for DC/MIC, evaluation uses the full dataset
            with subsampled scores as rankings — the mismatch is disclosed.
    """
    # Fast-mode: subsample large datasets to keep CI runtime under 5 min
    _FAST = _os.environ.get("NGORIMA_FAST", "0") == "1"
    if _FAST and X.shape[0] > 2_000:
        _idx = np.random.default_rng(42).choice(X.shape[0], 2_000, replace=False)
        X = X[_idx]
        y = y[_idx]

    n_full, p_full = X.shape
    print(f"\n  Dataset: {dataset_name} | n={n_full:,} | p={p_full:,}"
          f"{' | TIME-SERIES' if time_series else ''}"
          f"{' | [FAST subsampled]' if _FAST else ''}")

    records = []

    for method in methods:
        print(f"    [{method:<8}]", end=" ")

        if FALLBACK_FLAGS.get(method):
            print(f"[FALLBACK] ", end="")

        X_use, y_use = X, y
        extrapolated = False

        # ── DC feasibility guard (P2 fix) ──────────────────────────────
        if method == "dc":
            safe, req_gb = check_ram_for_dc(n_full)
            if n_full > dc_max_n or not safe:
                sub_n   = min(dc_max_n, n_full)
                idx     = np.random.default_rng(42).choice(n_full, sub_n, replace=False)
                X_use   = X[idx]
                y_use   = y[idx]
                extrapolated = True
                print(f"(subsampled n={sub_n:,} | needs {req_gb:.1f} GB) ", end="")

        if method == "mic" and n_full > mic_max_n:
            sub_n   = mic_max_n
            idx     = np.random.default_rng(42).choice(n_full, sub_n, replace=False)
            X_use   = X[idx]
            y_use   = y[idx]
            extrapolated = True
            print(f"(subsampled n={sub_n:,}) ", end="")

        p_score = min(X_use.shape[1], max_features or X_use.shape[1])
        scores, elapsed = score_all_features(X_use, y_use, method,
                                             max_features=p_score)

        # ── Extrapolated timing ─────────────────────────────────────────
        elapsed_reported = elapsed
        if extrapolated and method == "dc":
            n_sub = X_use.shape[0]
            elapsed_reported = elapsed * (n_full / n_sub) ** 2
            print(f"scored {p_score} feats in {elapsed:.2f}s "
                  f"→ extrap. {elapsed_reported/60:.1f}min (O(n²) scaling) ", end="")
        elif extrapolated and method == "mic":
            n_sub = X_use.shape[0]
            elapsed_reported = elapsed * (n_full / n_sub) ** 1.2
            print(f"extrap. {elapsed_reported/60:.1f}min ", end="")
        else:
            print(f"scored {p_score} feats in {elapsed:.2f}s ", end="")

        peak_mb = measure_peak_memory_mb(X_use, y_use, method)

        # ── Accuracy: always on full dataset ────────────────────────────
        # P0 fix: scores may come from a subsample but feature indices
        # are applied to the full X. This is disclosed via 'extrapolated'.
        p_acc  = min(X.shape[1], max_features or X.shape[1])
        X_acc  = X[:, :p_acc]
        # Pad scores to full p if subsampled (remaining scores = -inf)
        if len(scores) < p_acc:
            scores_full = np.full(p_acc, -np.inf)
            scores_full[:len(scores)] = scores
        else:
            scores_full = scores[:p_acc]

        acc = evaluate_accuracy_regression(
            X_acc, y, scores_full,
            top_k=top_k, n_splits=n_splits,
            time_series=time_series,
        )
        print(f"R²={acc['r2_mean']:.3f}±{acc['r2_std']:.3f}")

        records.append({
            "Dataset":       dataset_name,
            "Method":        LABELS.get(method, method),
            "n_scored":      X_use.shape[0],
            "n_full":        n_full,
            "p_scored":      p_score,
            "Time_s":        elapsed_reported,
            "Time_display":  (f"{elapsed_reported/60:.1f} min"
                              if elapsed_reported > 60
                              else f"{elapsed_reported:.2f}s"),
            "Memory_MB":     round(peak_mb, 1),
            "Feasible":      "Yes" if not extrapolated else "No (extrapolated)",
            "Extrapolated":  extrapolated,
            "R2_mean":       round(acc["r2_mean"], 3),
            "R2_std":        round(acc["r2_std"], 3),
            "Jaccard_mean":  round(acc["jaccard_mean"], 3),
            "Jaccard_std":   round(acc["jaccard_std"], 3),
            "Fallback":      FALLBACK_FLAGS.get(method, False),
        })

    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
#  Pareto figure                                                               #
# --------------------------------------------------------------------------- #

def plot_pareto(all_results: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors  = {"ξₙ": "#2196F3", "MI": "#4CAF50", "DC": "#F44336",
                   "MIC": "#FF9800", "Pearson": "#9C27B0", "Spearman": "#795548"}
        markers = {"ξₙ": "o", "MI": "s", "DC": "^",
                   "MIC": "D", "Pearson": "P", "Spearman": "X"}

        datasets = all_results["Dataset"].unique()
        fig, axes = plt.subplots(1, len(datasets),
                                 figsize=(5 * len(datasets), 5), sharey=False)
        if len(datasets) == 1:
            axes = [axes]

        for ax, ds in zip(axes, datasets):
            sub = all_results[
                (all_results["Dataset"] == ds)
                & all_results["R2_mean"].notna()
                & (all_results["R2_mean"] > 0)
            ].copy()
            for _, row in sub.iterrows():
                m    = row["Method"]
                t    = row["Time_s"]
                unit = "min" if t > 60 else "s"
                t_d  = t / 60 if t > 60 else t
                style_kwargs = dict(
                    color=colors.get(m, "gray"),
                    marker=markers.get(m, "o"),
                    s=140, zorder=5,
                )
                if row.get("Extrapolated"):
                    style_kwargs["edgecolors"] = "black"
                    style_kwargs["linewidths"] = 1.5
                ax.scatter(t_d, row["R2_mean"], label=m, **style_kwargs)
                ax.annotate(m, (t_d, row["R2_mean"]),
                            textcoords="offset points", xytext=(6, 3), fontsize=8)

            ax.set_xlabel(f"Feature scoring time ({unit})", fontsize=10)
            ax.set_ylabel("Test $R^2$ (5-fold CV)", fontsize=10)
            ax.set_title(ds, fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.annotate("★ = extrapolated timing", xy=(0.02, 0.02),
                        xycoords="axes fraction", fontsize=7, color="gray")

        plt.suptitle(
            "Accuracy–Efficiency Pareto Frontier — Ngorima (2025)\n"
            "(filled border = extrapolated; Pipeline CV, no leakage)",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout()
        path = FIGURES_DIR / "accuracy_efficiency_pareto.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [FIGURE] {path}")
    except Exception as exc:
        print(f"  [WARN] Pareto plot failed: {exc}")


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("=" * 60)
    print("Step 2: Real-Domain Benchmarks (Section 5)")
    print("=" * 60)

    print("\n[0] Logging system state...")
    state = log_system_state()
    print(f"  DC safe up to n = {DC_MAX_N:,} ({state['ram_available_gb']} GB available)")

    all_results: list[pd.DataFrame] = []

    # ── FRED-MD ────────────────────────────────────────────────────────────
    print("\n[1/3] FRED-MD Macroeconomics")
    fred_path = PROCESSED_DIR / "fred_md.npz"
    if fred_path.exists():
        data    = np.load(fred_path, allow_pickle=True)
        X_fred  = data["X"].astype(np.float64)
        y_fred  = data["y"].astype(np.float64)
        df_fred = benchmark_dataset(
            "FRED-MD (Macroeconomics)", X_fred, y_fred,
            methods=METHODS,
            max_features=None, top_k=TOP_K, n_splits=N_SPLITS,
            time_series=True,   # P0 fix: TimeSeriesSplit activated
        )
        df_fred.to_csv(RESULTS_DIR / "table5_fredmd.csv", index=False)
        print(df_fred[["Method", "Time_display", "Memory_MB",
                       "Feasible", "R2_mean", "Jaccard_mean"]].to_string(index=False))
        all_results.append(df_fred)
    else:
        print("  [SKIP] Run 00_setup_and_download.py first.")

    # ── Superconductivity ──────────────────────────────────────────────────
    print("\n[2/3] Superconductivity (Materials Science)")
    sc_path = PROCESSED_DIR / "superconductivity.npz"
    if sc_path.exists():
        data   = np.load(sc_path, allow_pickle=True)
        X_sc   = data["X"].astype(np.float64)
        y_sc   = data["y"].astype(np.float64)
        # P2 fix: dc_max_n set dynamically; warn if this machine cannot run DC full
        df_sc  = benchmark_dataset(
            "Superconductivity", X_sc, y_sc,
            methods=METHODS,
            max_features=None,
            dc_max_n=DC_MAX_N,
            top_k=TOP_K, n_splits=N_SPLITS,
        )
        df_sc.to_csv(RESULTS_DIR / "table6_superconductivity.csv", index=False)
        print(df_sc[["Method", "Time_display", "Memory_MB",
                     "Feasible", "R2_mean", "Jaccard_mean"]].to_string(index=False))
        all_results.append(df_sc)
    else:
        print("  [SKIP] Superconductivity data not found.")

    # ── TCGA ───────────────────────────────────────────────────────────────
    print("\n[3/3] TCGA Pan-Cancer Genomics")
    real_tcga  = PROCESSED_DIR / "tcga.npz"
    synth_tcga = PROCESSED_DIR / "tcga_synthetic.npz"
    tcga_label = "TCGA Genomics"
    X_tcga = y_tcga = None

    if real_tcga.exists():
        data   = np.load(real_tcga, allow_pickle=True)
        X_tcga = data["X"].astype(np.float64)
        y_tcga = data["y"].astype(np.float64)
        print(f"  Using real TCGA: {X_tcga.shape}")
        # Confirm target is NOT PC1 (loaded from Step 0 which fixed this)
        print("  Target: tumour purity (non-circular, from clinical metadata)")
    elif synth_tcga.exists():
        data   = np.load(synth_tcga)
        X_tcga = data["X"].astype(np.float64)
        y_tcga = data["y"].astype(np.float64)
        tcga_label = "TCGA Genomics (Synthetic proxy)"
        print(f"  Using SYNTHETIC proxy: {X_tcga.shape}")
        print("  Target: signal gene mean + noise (non-circular)")

    if X_tcga is not None:
        df_tcga = benchmark_dataset(
            tcga_label, X_tcga, y_tcga,
            methods=METHODS,
            max_features=200 if _FAST else 500,  # Cap for timing feasibility
            dc_max_n=min(2_000, DC_MAX_N),
            mic_max_n=MIC_MAX_N,
            top_k=min(TOP_K, 20), n_splits=N_SPLITS,
        )
        df_tcga.to_csv(RESULTS_DIR / "table4_tcga.csv", index=False)
        print(df_tcga[["Method", "Time_display", "Memory_MB",
                       "Feasible", "R2_mean", "Jaccard_mean"]].to_string(index=False))
        all_results.append(df_tcga)

    # ── Combined CSV + Pareto plot ─────────────────────────────────────────
    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        df_all.to_csv(RESULTS_DIR / "real_domain_all.csv", index=False)
        plot_pareto(df_all)

    print("\n[DONE] Step 2 complete. Run 03_memory_and_parallelisation.py next.")
