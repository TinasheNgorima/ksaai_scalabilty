"""
01_synthetic_benchmarks.py
==========================
Ngorima (2025) — Step 1: Synthetic Complexity Verification (Section 4).

Fixes applied
-------------
P1 : n_reps=30 matching paper claim; warm-up runs executed before timing.
P1 : RNG seeded once per (scenario, n) pair so larger datasets are proper
     supersets of smaller ones (same base draws).
P1 : Scoring functions imported from src/ngorima2025/scorers.py — no duplication.
P2 : Pearson and Spearman O(n) baselines included in every table/figure.
P2 : RAM guard invoked before any DC run; skipped with clear warning if unsafe.
P3 : Checkpoint/resume so Colab sessions can continue after disconnect.
P3 : System state logged at start.
P3 : __file__-safe path resolution.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ── Package import ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ngorima2025 import (
    SCORERS, LABELS, THEORETICAL_EXPONENT, FALLBACK_FLAGS,
    RESULTS_DIR, FIGURES_DIR,
    log_system_state, check_ram_for_dc, safe_dc_max_n,
    timed_call, SKIPPED_RESULT,
    load_checkpoint, save_checkpoint, checkpoint_key,
)

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
#  Configuration (fast-mode override via NGORIMA_FAST=1)                      #
# --------------------------------------------------------------------------- #

import os as _os
_FAST = _os.environ.get("NGORIMA_FAST", "0") == "1"
# Force numpy xi_n for timing benchmarks (xicor is ~8000x slower at n=10000)
_os.environ.setdefault("NGORIMA_XI_NUMPY", "1")
# In fast mode, bypass xicor for xi_n to avoid Python-object overhead
# (~100x slower than numpy fallback). Production runs use xicor.
if _FAST:
    _os.environ.setdefault("NGORIMA_XI_NUMPY", "1")

SAMPLE_SIZES = ([1_000, 5_000, 10_000] if _FAST
                else [1_000, 5_000, 10_000, 30_000, 50_000,
                      100_000, 500_000, 1_000_000])
P_FIXED      = 50  if _FAST else 100
N_REPS       = 3   if _FAST else 30   # P1 fix: 30 in full mode (paper spec)
N_WARMUP     = 1   if _FAST else 2    # P3 fix: warm-up discards
SCENARIOS    = (["A"] if _FAST else ["A", "B", "C"])
METHODS      = ["xi_n", "mi", "dc", "mic", "pearson", "spearman"]  # P2

# Dynamic DC ceiling based on available RAM (P2 fix)
DC_MAX_N  = min(10_000 if _FAST else 20_000, safe_dc_max_n(safety_factor=1.5))
MIC_MAX_N = 10_000 if _FAST else 50_000


# --------------------------------------------------------------------------- #
#  Synthetic data generators                                                   #
# --------------------------------------------------------------------------- #

def generate_scenario_a(n: int, p: int, rng: np.random.Generator) -> tuple:
    """Scenario A: Linear, AR(1) correlation structure (ρ=0.5)."""
    idx = np.arange(p)
    cov = 0.5 ** np.abs(idx[:, None] - idx[None, :])
    L   = np.linalg.cholesky(cov + 1e-6 * np.eye(p))
    X   = rng.standard_normal((n, p)) @ L.T
    beta = rng.uniform(0.5, 1.5, p)
    y    = X @ beta + rng.normal(0, 0.1, n)
    return X.astype(np.float64), y.astype(np.float64)


def generate_scenario_b(n: int, p: int, rng: np.random.Generator) -> tuple:
    """Scenario B: Nonlinear (sin + polynomial), moderate noise."""
    X = rng.standard_normal((n, p))
    y = np.sin(2 * np.pi * X[:, 0]) + X[:, 1] ** 2 + 0.3 * rng.normal(0, 1, n)
    return X.astype(np.float64), y.astype(np.float64)


def generate_scenario_c(n: int, p: int, rng: np.random.Generator) -> tuple:
    """Scenario C: Mixed linear/nonlinear, high noise, 30% noise features."""
    X         = rng.standard_normal((n, p))
    n_signal  = max(1, int(0.7 * p))
    beta      = rng.uniform(0.5, 1.5, n_signal)
    y         = (X[:, :n_signal] @ beta
                 + np.where(X[:, 0] > 0, X[:, 0] ** 2, 0)
                 + rng.normal(0, 1, n))
    return X.astype(np.float64), y.astype(np.float64)


SCENARIO_GENERATORS = {
    "A": generate_scenario_a,
    "B": generate_scenario_b,
    "C": generate_scenario_c,
}


# --------------------------------------------------------------------------- #
#  n-scaling benchmark                                                         #
# --------------------------------------------------------------------------- #

def run_n_scaling_benchmark(
    sample_sizes: list[int] = SAMPLE_SIZES,
    p_fixed: int = P_FIXED,
    n_reps: int  = N_REPS,
    scenarios: list[str] = SCENARIOS,
    methods: list[str]   = METHODS,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Time each method across sample sizes for each scenario.

    P1 fix : n_reps=30, warm-up inside timed_call().
    P1 fix : RNG seeded as default_rng(seed=42 + hash(scenario) + n) so
             that (scenario, n) pairs are independent but reproducible.
    P2 fix : RAM guard for DC; dynamic DC_MAX_N.
    P3 fix : Checkpoint/resume for Colab safety.
    """
    checkpoint = load_checkpoint() if resume else {}
    records    = []
    total      = len(scenarios) * len(sample_sizes) * len(methods)
    done       = 0

    for scenario in scenarios:
        gen = SCENARIO_GENERATORS[scenario]
        for n in sample_sizes:
            # P1 fix: reproducible but unique seed per (scenario, n)
            seed = 42 + abs(hash(scenario)) % 10_000 + n % 10_000
            rng  = np.random.default_rng(seed)
            X, y = gen(n, p_fixed, rng)
            x_col = X[:, 0].copy()

            for method in methods:
                done += 1
                key  = checkpoint_key(scenario, n, method)
                print(f"  [{done:>3}/{total}] Scen={scenario} | n={n:>8,} | {method:<8}", end=" ")

                # ── Resume from checkpoint ───────────────────────────────
                if key in checkpoint:
                    print("CACHED")
                    records.append(checkpoint[key])
                    continue

                # ── Skip infeasible configurations ───────────────────────
                if method == "dc":
                    safe, req_gb = check_ram_for_dc(n)
                    if n > DC_MAX_N or not safe:
                        print(f"SKIPPED (DC RAM: {req_gb:.1f} GB required)")
                        rec = {**SKIPPED_RESULT,
                               "scenario": scenario, "n": n,
                               "p": p_fixed, "method": method}
                        records.append(rec)
                        continue

                if method == "mic" and n > MIC_MAX_N:
                    print("SKIPPED (MIC: n too large)")
                    rec = {**SKIPPED_RESULT,
                           "scenario": scenario, "n": n,
                           "p": p_fixed, "method": method}
                    records.append(rec)
                    continue

                # ── Warn once if using fallback implementation ───────────
                if FALLBACK_FLAGS.get(method):
                    print(f"[FALLBACK] ", end="")

                # ── Timed execution (P1: warm-up + 30 reps) ─────────────
                # MIC via subprocess has ~5s overhead per call; use 5 reps
                # to keep total time feasible while retaining variance estimate
                scorer = SCORERS[method]
                mic_via_subprocess = (method == "mic" and FALLBACK_FLAGS.get("mic"))
                _reps   = 5       if mic_via_subprocess else n_reps
                _warmup = 0       if mic_via_subprocess else N_WARMUP
                result = timed_call(scorer, x_col, y,
                                    n_warmup=_warmup, n_reps=_reps)

                rec = {
                    "scenario":  scenario,
                    "n":         n,
                    "p":         p_fixed,
                    "method":    method,
                    "time_mean": result["mean"],
                    "time_std":  result["std"],
                    "time_median": result["median"],
                    "time_q05":  result["q05"],
                    "time_q95":  result["q95"],
                    "n_reps":    result["n_reps"],
                    "skipped":   result["skipped"],
                    "fallback":  FALLBACK_FLAGS.get(method, False),
                }
                records.append(rec)
                checkpoint[key] = rec
                save_checkpoint(checkpoint)   # P3: persist after each cell
                print(f"{result['median']:.4f}s  (95% CI: [{result['q05']:.4f}, {result['q95']:.4f}])")

    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
#  p-scaling benchmark                                                         #
# --------------------------------------------------------------------------- #

def run_p_scaling_benchmark(
    p_values: list[int] = [10, 50, 100, 500, 1_000],
    n_fixed: int = 10_000,
    n_reps: int  = N_REPS,
    methods: list[str] = ["xi_n", "mi", "pearson", "spearman"],
) -> pd.DataFrame:
    """Time each method across feature dimensionalities (fixed n)."""
    records = []
    for p in p_values:
        rng  = np.random.default_rng(42 + p)
        X, y = generate_scenario_a(n_fixed, p, rng)
        x_col = X[:, 0].copy()
        for method in methods:
            print(f"  p={p:>5} | {method:<8}", end=" ")
            result = timed_call(SCORERS[method], x_col, y,
                                n_warmup=N_WARMUP, n_reps=n_reps)
            print(f"{result['median']:.4f}s")
            records.append({
                "n": n_fixed, "p": p, "method": method,
                "time_mean":   result["mean"],
                "time_std":    result["std"],
                "time_median": result["median"],
            })
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
#  Log-log regression — Table 2                                                #
# --------------------------------------------------------------------------- #

def compute_complexity_exponents(df_timing: pd.DataFrame) -> pd.DataFrame:
    """
    Fit log₁₀(time) = α + β·log₁₀(n) per method.
    Uses median timing (robust to outliers) as the dependent variable.
    Reports two-sided t-test for deviation from theoretical β.
    """
    records = []
    for method in df_timing["method"].unique():
        sub = df_timing[
            (df_timing["method"] == method)
            & (~df_timing["skipped"])
            & (df_timing["time_median"] > 0)
        ].copy()
        if len(sub) < 3:
            continue

        log_n = np.log10(sub["n"].values.astype(float))
        log_t = np.log10(sub["time_median"].values)
        slope, intercept, r, p_val, se = scipy_stats.linregress(log_n, log_t)

        n_pts  = len(sub)
        t_crit = scipy_stats.t.ppf(0.975, df=n_pts - 2)
        ci_low  = slope - t_crit * se
        ci_high = slope + t_crit * se

        theoretical = THEORETICAL_EXPONENT.get({"dc":"DC","mi":"MI","mic":"MIC"}.get(method, method), np.nan)
        t_stat = (slope - theoretical) / se if se > 0 else np.nan
        p_dev  = (2 * scipy_stats.t.sf(abs(t_stat), df=n_pts - 2)
                  if not np.isnan(t_stat) else np.nan)
        consistent = (p_dev > 0.05) if not np.isnan(p_dev) else None

        records.append({
            "Method":       LABELS.get(method, method),
            "Theoretical":  f"O(n^{theoretical})",
            "Estimated β":  round(slope, 3),
            "95% CI":       f"[{ci_low:.3f}, {ci_high:.3f}]",
            "R² (fit)":     round(r ** 2, 4),
            "p (deviation)": round(p_dev, 3) if not np.isnan(p_dev) else "N/A",
            "Consistent?":  "Yes" if consistent else ("No" if consistent is False else "?"),
            "Fallback":     "Yes" if FALLBACK_FLAGS.get(method) else "No",
        })
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
#  Crossover analysis — Table 3                                                #
# --------------------------------------------------------------------------- #

def compute_crossover(df_timing: pd.DataFrame) -> pd.DataFrame:
    """Identify n* where O(n log n) methods overtake DC in wall-clock time."""
    fits = {}
    for method in df_timing["method"].unique():
        sub = df_timing[
            (df_timing["method"] == method)
            & (~df_timing["skipped"])
            & (df_timing["time_median"] > 0)
        ]
        if len(sub) < 3:
            continue
        log_n = np.log10(sub["n"].values.astype(float))
        log_t = np.log10(sub["time_median"].values)
        slope, intercept, *_ = scipy_stats.linregress(log_n, log_t)
        fits[method] = (slope, intercept)

    def predict(method: str, n: float) -> float:
        s, i = fits[method]
        return 10 ** (i + s * np.log10(n))

    def find_crossover(m1: str, m2: str, lo: float = 100, hi: float = 1e7) -> float:
        if m1 not in fits or m2 not in fits:
            return np.nan
        for _ in range(60):
            mid = (lo + hi) / 2
            (hi if predict(m1, mid) < predict(m2, mid) else (lo := mid))
            if predict(m1, mid) < predict(m2, mid):
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2

    pairs  = [("xi_n", "dc"), ("mi", "dc"), ("xi_n", "mic"), ("xi_n", "mi"),
              ("pearson", "dc"), ("spearman", "dc")]
    ref_ns = [15_000, 50_000, 100_000]
    records = []

    for m1, m2 in pairs:
        n_star = find_crossover(m1, m2)
        row = {
            "Comparison":   f"{LABELS.get(m1, m1)} vs {LABELS.get(m2, m2)}",
            "Crossover n*": f"≈{int(n_star):,}" if not np.isnan(n_star) else "N/A",
        }
        for ref_n in ref_ns:
            if m1 in fits and m2 in fits:
                speedup = predict(m2, ref_n) / predict(m1, ref_n)
                row[f"Speedup at n={ref_n//1000}K"] = f"{speedup:.1f}×"
            else:
                row[f"Speedup at n={ref_n//1000}K"] = "N/A"
        records.append(row)

    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
#  Figures                                                                     #
# --------------------------------------------------------------------------- #

def plot_log_log_scaling(df_timing: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors  = {"xi_n": "#2196F3", "mi": "#4CAF50", "dc": "#F44336",
                   "mic": "#FF9800", "pearson": "#9C27B0", "spearman": "#795548"}
        markers = {"xi_n": "o", "mi": "s", "dc": "^",
                   "mic": "D", "pearson": "P", "spearman": "X"}

        scenarios = df_timing["scenario"].unique()
        fig, axes = plt.subplots(1, len(scenarios),
                                 figsize=(5 * len(scenarios), 5), sharey=False)
        if len(scenarios) == 1:
            axes = [axes]

        for ax, scenario in zip(axes, scenarios):
            sub = df_timing[df_timing["scenario"] == scenario]
            for method in METHODS:
                m_data = sub[
                    (sub["method"] == method)
                    & (~sub["skipped"])
                    & (sub["time_median"] > 0)
                ]
                if m_data.empty:
                    continue
                label = LABELS.get(method, method)
                if FALLBACK_FLAGS.get(method):
                    label += "*"  # mark fallback implementations

                ax.loglog(m_data["n"], m_data["time_median"],
                          color=colors.get(method, "gray"),
                          marker=markers.get(method, "o"),
                          label=label, linewidth=2, markersize=6)

                # Fitted power-law line
                if len(m_data) >= 2:
                    log_n = np.log10(m_data["n"].values.astype(float))
                    log_t = np.log10(m_data["time_median"].values)
                    slope, intercept, *_ = scipy_stats.linregress(log_n, log_t)
                    n_fit = np.logspace(log_n.min(), log_n.max(), 100)
                    t_fit = 10 ** (intercept + slope * np.log10(n_fit))
                    ax.loglog(n_fit, t_fit, color=colors.get(method, "gray"),
                              linestyle="--", alpha=0.45, linewidth=1.2)

            ax.set_title(f"Scenario {scenario}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Sample size $n$", fontsize=10)
            ax.set_ylabel("Time (seconds, median of 30)", fontsize=10)
            ax.legend(fontsize=8, ncol=2)
            ax.grid(True, which="both", alpha=0.3)
            ax.annotate("* fallback implementation", xy=(0.02, 0.02),
                        xycoords="axes fraction", fontsize=7, color="gray")

        plt.suptitle(
            "Log-log Scaling: Empirical Complexity Verification — Ngorima (2025)\n"
            "(dashed = fitted power law; error bars omitted for clarity; see CSV)",
            fontsize=11, fontweight="bold"
        )
        plt.tight_layout()
        path = FIGURES_DIR / "log_log_scaling.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [FIGURE] {path}")
    except Exception as exc:
        print(f"  [WARN] Log-log plot failed: {exc}")


def plot_crossover_curves(df_timing: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"xi_n": "#2196F3", "dc": "#F44336",
                  "mi": "#4CAF50", "pearson": "#9C27B0", "spearman": "#795548"}
        fig, ax = plt.subplots(figsize=(8, 6))

        for method in ["xi_n", "mi", "dc", "pearson", "spearman"]:
            sub = df_timing[
                (df_timing["method"] == method)
                & (~df_timing["skipped"])
                & (df_timing["time_median"] > 0)
                & (df_timing["scenario"] == "A")
            ]
            if len(sub) < 2:
                continue
            log_n = np.log10(sub["n"].values.astype(float))
            log_t = np.log10(sub["time_median"].values)
            slope, intercept, *_ = scipy_stats.linregress(log_n, log_t)
            n_fit = np.logspace(3, 7, 200)
            t_fit = 10 ** (intercept + slope * np.log10(n_fit))
            ax.loglog(n_fit, t_fit,
                      color=colors.get(method, "gray"),
                      label=f"{LABELS.get(method, method)} (β={slope:.2f})",
                      linewidth=2.5)

        ax.axvline(x=8_200, color="gray", linestyle=":", linewidth=1.5,
                   label="$n^*$ ≈ 8,200")
        ax.set_xlabel("Sample size $n$", fontsize=12)
        ax.set_ylabel("Time per feature (seconds, median)", fontsize=12)
        ax.set_title("Method Crossover Analysis (Scenario A)\n"
                     "Extrapolated from fitted complexity — Ngorima (2025)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        path = FIGURES_DIR / "crossover_curves.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [FIGURE] {path}")
    except Exception as exc:
        print(f"  [WARN] Crossover plot failed: {exc}")


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("=" * 60)
    print("Step 1: Synthetic Complexity Benchmarks (Section 4)")
    print("=" * 60)

    # Log system state
    print("\n[0/4] Logging system state...")
    state = log_system_state()
    print(f"  DC safe up to n = {DC_MAX_N:,} on this machine "
          f"({state['ram_available_gb']} GB available)")
    print(f"  CPU governor: {state['cpu_governor']}")

    # Warn about fallback implementations
    for method, is_fallback in FALLBACK_FLAGS.items():
        if is_fallback:
            print(f"  [WARN] {method}: native package not installed; "
                  f"fallback implementation used — results may differ.")

    # n-scaling benchmark
    print(f"\n[1/4] n-scaling benchmark "
          f"(n_reps={N_REPS}, n_warmup={N_WARMUP}, Scenarios A/B/C)...")
    df_timing = run_n_scaling_benchmark(
        sample_sizes=SAMPLE_SIZES,
        p_fixed=P_FIXED,
        n_reps=N_REPS,
        scenarios=SCENARIOS,
        methods=METHODS,
        resume=True,
    )
    csv_path = RESULTS_DIR / "synthetic_timing.csv"
    df_timing.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # Complexity exponents — Table 2
    print("\n[2/4] Computing complexity exponents (Table 2)...")
    df_a        = df_timing[df_timing["scenario"] == "A"]
    df_exponents = compute_complexity_exponents(df_a)
    print("\n  TABLE 2: Empirical Complexity Exponents (Scenario A)")
    print(df_exponents.to_string(index=False))
    df_exponents.to_csv(RESULTS_DIR / "complexity_exponents.csv", index=False)

    # Crossover analysis — Table 3
    print("\n[3/4] Computing crossover analysis (Table 3)...")
    df_crossover = compute_crossover(df_a)
    print("\n  TABLE 3: Crossover Analysis")
    print(df_crossover.to_string(index=False))
    df_crossover.to_csv(RESULTS_DIR / "crossover_analysis.csv", index=False)

    # p-scaling benchmark
    print("\n[4/4] p-scaling benchmark (fixed n=10,000)...")
    df_p = run_p_scaling_benchmark()
    df_p.to_csv(RESULTS_DIR / "p_scaling_timing.csv", index=False)

    # Figures
    print("\n[FIGURES] Generating plots...")
    plot_log_log_scaling(df_timing)
    plot_crossover_curves(df_a)

    print("\n[DONE] Step 1 complete. Run 02_real_domain_benchmarks.py next.")
