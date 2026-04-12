"""
03_memory_and_parallelisation.py
================================
Ngorima (2025) — Step 3: Memory Profiling and Parallelisation (Section 6).

Fixes applied
-------------
P1 : Scoring functions from shared module — no duplication.
P2 : RAM guard before DC allocations at each n; skips unsafe configurations
     with a clear log entry rather than crashing.
P2 : System state logged at start.
P3 : warm-up runs before parallelisation timing.
P3 : __file__-safe path resolution.
"""

from __future__ import annotations

import multiprocessing
import sys
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── Package import ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ngorima2025 import (
    SCORERS, LABELS, FALLBACK_FLAGS,
    RESULTS_DIR, FIGURES_DIR,
    log_system_state, check_ram_for_dc, safe_dc_max_n,
    timed_call, SKIPPED_RESULT,
)

warnings.filterwarnings("ignore")

import os as _os
_FAST    = _os.environ.get("NGORIMA_FAST", "0") == "1"
if _FAST:
    _os.environ.setdefault("NGORIMA_XI_NUMPY", "1")
    # Also use smaller n for parallelisation test in fast mode
    _PAR_N = 2_000
else:
    _PAR_N = 15_542
DC_MAX_N = min(5_000 if _FAST else 25_000, safe_dc_max_n(safety_factor=1.5))
MEM_SIZES = [1_000, 5_000, 10_000] if _FAST else [1_000, 5_000, 10_000, 25_000, 50_000]
PAR_REPS  = 2 if _FAST else 5
METHODS  = ["xi_n", "mi", "dc", "mic", "pearson", "spearman"]


# --------------------------------------------------------------------------- #
#  Memory profiling — Table 7                                                  #
# --------------------------------------------------------------------------- #

def measure_memory_mb(method: str, n: int, p: int = 100) -> float:
    """
    Measure peak RAM for scoring one feature pair (n observations).

    DC: theoretical O(n²) formula — tracemalloc underestimates NumPy
        array allocation because it tracks Python objects, not raw
        C-level buffers. The formula n²×8 bytes is the ground truth.

    P2 fix: Returns np.nan with a warning for unsafe DC configurations
            rather than attempting the allocation and crashing.
    """
    rng = np.random.default_rng(42)
    X   = rng.standard_normal((n, p))
    y   = X[:, 0] + rng.normal(0, 0.1, n)

    if method == "dc":
        safe, req_gb = check_ram_for_dc(n)
        if not safe:
            print(f"      [SKIP DC] n={n:,} requires {req_gb:.1f} GB — unsafe on this machine")
            return np.nan
        return float(n ** 2 * 8 / 1e6)

    scorer = SCORERS[method]
    tracemalloc.start()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scorer(X[:, 0], y)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return float(peak / 1e6)


def run_memory_benchmark(
    sample_sizes: list[int] = [1_000, 5_000, 10_000, 25_000, 50_000],
    p_fixed: int = 100,
) -> pd.DataFrame:
    """Table 7: Peak memory consumption vs n (p=100 fixed)."""
    records = []
    print(f"\n  Memory profiling (p={p_fixed} fixed, DC ceiling = n ≤ {DC_MAX_N:,})...")

    for n in sample_sizes:
        row: dict = {"n": n}
        for method in METHODS:
            print(f"    n={n:>6,} | {method:<8}", end=" ... ")
            mb = measure_memory_mb(method, n, p_fixed)
            label = LABELS.get(method, method)
            row[f"{label}_MB"] = round(mb, 1) if not np.isnan(mb) else "N/A"
            status = f"{mb:.1f} MB" if not np.isnan(mb) else "SKIPPED"
            print(status)

        # DC / ξₙ ratio (paper Table 7)
        xi_mb = row.get("ξₙ_MB", None)
        dc_mb = row.get("DC_MB", None)
        if isinstance(xi_mb, float) and isinstance(dc_mb, float) and xi_mb > 0:
            row["DC/ξₙ_ratio"] = f"{dc_mb/xi_mb:.1f}×"
        else:
            row["DC/ξₙ_ratio"] = "N/A"
        records.append(row)

    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
#  Parallelisation benchmark — Table 8                                         #
# --------------------------------------------------------------------------- #

def time_parallel_scoring(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    n_cores: int,
    n_reps: int = 5,
    n_warmup: int = 1,
) -> float:
    """
    Time parallel feature scoring across p features using n_cores.

    P3 fix: n_warmup warm-up passes before measured timing.
    """
    from joblib import Parallel, delayed
    scorer = SCORERS[method]
    p      = X.shape[1]

    def _run() -> None:
        if n_cores == 1:
            [scorer(X[:, i], y) for i in range(p)]
        else:
            Parallel(n_jobs=n_cores)(
                delayed(scorer)(X[:, i], y) for i in range(p)
            )

    # Warm-up
    for _ in range(n_warmup):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _run()

    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _run()
        times.append(time.perf_counter() - t0)

    return float(np.median(times))


def run_parallelisation_benchmark(
    n: int = 15_542,
    p: int = 81,
    core_counts: list[int] | None = None,
    n_reps: int = 5,
) -> pd.DataFrame:
    """
    Table 8: Speedup S(k) = T(1) / T(k) across core counts.

    P2 fix: DC is capped at dc_p features to prevent OOM when each
            worker holds an n×n matrix. Warning is emitted.
    """
    if core_counts is None:
        max_cores   = multiprocessing.cpu_count()
        core_counts = sorted(set([1, 2, 4, min(8, max_cores), max_cores]))

    safe_dc, dc_req_gb = check_ram_for_dc(n, safety_factor=float(max(core_counts)))
    if not safe_dc:
        # Restrict DC feature count so each worker's matrix fits
        dc_p = min(p, 10)
        print(f"  [WARN] DC parallelisation: {max(core_counts)} workers × "
              f"{dc_req_gb:.1f} GB/worker may exceed RAM. "
              f"Limiting DC to {dc_p} features.")
    else:
        dc_p = p

    rng = np.random.default_rng(42)
    X   = rng.standard_normal((n, p))
    y   = X[:, 0] + X[:, 1] ** 2 + rng.normal(0, 0.3, n)

    print(f"\n  Parallelisation benchmark: n={n:,}, p={p}")
    print(f"  Core counts tested: {core_counts}")

    # Single-core baselines (with warm-up)
    baselines: dict[str, float] = {}
    print("\n  Single-core baselines...")
    for method in METHODS:
        p_use = dc_p if method == "dc" else p
        X_use = X[:, :p_use]
        print(f"    {method:<8} (p={p_use}) ... ", end="")
        t1 = time_parallel_scoring(method, X_use, y, n_cores=1,
                                    n_reps=n_reps, n_warmup=2)
        baselines[method] = t1
        print(f"{t1:.3f}s")

    records = []
    for k in core_counts:
        row = {"Cores (k)": k, "Ideal S(k)": f"{k:.1f}×"}
        for method in METHODS:
            p_use = dc_p if method == "dc" else p
            X_use = X[:, :p_use]
            if k == 1:
                t_k = baselines[method]
            else:
                print(f"    {method:<8} | cores={k} ... ", end="")
                t_k = time_parallel_scoring(method, X_use, y, n_cores=k,
                                             n_reps=n_reps, n_warmup=1)
                print(f"{t_k:.3f}s")

            speedup = baselines[method] / t_k if t_k > 0 else np.nan
            row[f"{LABELS.get(method, method)} S(k)"] = (
                f"{speedup:.2f}×" if not np.isnan(speedup) else "N/A"
            )
        records.append(row)

    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
#  Figures                                                                     #
# --------------------------------------------------------------------------- #

def plot_memory_scaling(df_memory: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        color_map = {
            "ξₙ_MB": "#2196F3", "MI_MB": "#4CAF50",
            "DC_MB": "#F44336", "MIC_MB": "#FF9800",
            "Pearson_MB": "#9C27B0", "Spearman_MB": "#795548",
        }
        label_map = {k: k.replace("_MB", "") for k in color_map}

        fig, ax = plt.subplots(figsize=(8, 6))
        for col, color in color_map.items():
            if col not in df_memory.columns:
                continue
            vals = pd.to_numeric(df_memory[col], errors="coerce")
            valid = df_memory.loc[vals.notna(), "n"]
            ax.plot(valid, vals.dropna(),
                    color=color, marker="o",
                    label=label_map[col], linewidth=2.5, markersize=7)

        ax.set_xlabel("Sample size $n$", fontsize=12)
        ax.set_ylabel("Peak Memory (MB)", fontsize=12)
        ax.set_title(
            "Peak Memory Consumption vs Sample Size ($p=100$)\n"
            "DC: theoretical $O(n^2)$; others: measured via tracemalloc — Ngorima (2025)",
            fontsize=11, fontweight="bold",
        )
        ax.set_yscale("log")
        ax.legend(fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        path = FIGURES_DIR / "memory_scaling.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [FIGURE] {path}")
    except Exception as exc:
        print(f"  [WARN] Memory plot failed: {exc}")


def plot_parallelisation(df_par: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        color_map = {
            "ξₙ S(k)": "#2196F3", "MI S(k)": "#4CAF50",
            "DC S(k)": "#F44336",  "MIC S(k)": "#FF9800",
            "Pearson S(k)": "#9C27B0", "Spearman S(k)": "#795548",
        }
        fig, ax = plt.subplots(figsize=(8, 6))
        cores = df_par["Cores (k)"].values.astype(float)

        # Ideal line
        ax.plot(cores, cores, "k--", label="Ideal (linear)", linewidth=1.5, alpha=0.5)

        for col, color in color_map.items():
            if col not in df_par.columns:
                continue
            vals = (df_par[col].str.replace("×", "", regex=False)
                                .replace("N/A", "nan")
                                .astype(float).values)
            ax.plot(cores, vals, color=color, marker="o",
                    label=col.replace(" S(k)", ""), linewidth=2.5, markersize=8)

        ax.set_xlabel("Number of cores $k$", fontsize=12)
        ax.set_ylabel("Speedup $S(k) = T(1)/T(k)$", fontsize=12)
        ax.set_title(
            "Parallelisation Speedup (median of 5 reps, 1 warm-up)\n"
            f"$n=15{','}542$, $p=81$ — Ngorima (2025)",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = FIGURES_DIR / "parallelisation_speedup.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [FIGURE] {path}")
    except Exception as exc:
        print(f"  [WARN] Parallelisation plot failed: {exc}")


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("=" * 60)
    print("Step 3: Memory Profiling & Parallelisation (Section 6)")
    print("=" * 60)

    print("\n[0] Logging system state...")
    state = log_system_state()
    print(f"  RAM available: {state['ram_available_gb']} GB | "
          f"DC safe to n ≤ {DC_MAX_N:,}")

    # Table 7: Memory
    print("\n[1/2] Memory Profiling (Table 7)...")
    df_memory = run_memory_benchmark(sample_sizes=MEM_SIZES)
    print("\n  TABLE 7: Peak Memory Consumption (p=100)")
    print(df_memory.to_string(index=False))
    df_memory.to_csv(RESULTS_DIR / "table7_memory_profile.csv", index=False)
    plot_memory_scaling(df_memory)

    # Table 8: Parallelisation
    print("\n[2/2] Parallelisation Benchmark (Table 8)...")
    df_par = run_parallelisation_benchmark(n=_PAR_N, p=81, n_reps=PAR_REPS)
    print("\n  TABLE 8: Parallelisation Speedup")
    print(df_par.to_string(index=False))
    df_par.to_csv(RESULTS_DIR / "table8_parallelisation.csv", index=False)
    plot_parallelisation(df_par)

    print("\n[DONE] Step 3 complete. Run 04_compile_results.py next.")
