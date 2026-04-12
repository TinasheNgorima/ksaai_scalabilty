"""
src/ngorima2025/utils.py
========================
Shared utilities: path resolution, timing, system-state logging,
RAM guard, and checkpoint/resume.

P1 fix : Centralised warm-up logic; mean-of-30 enforced.
P2 fix : RAM guard before DC allocation.
P2 fix : System-state logging (CPU freq, available RAM, Python version).
P2 fix : __file__ / Colab-safe path resolution.
P3 fix : Checkpoint / resume for long Colab sessions.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import psutil

logger = logging.getLogger("ngorima2025")

# --------------------------------------------------------------------------- #
#  Colab-safe project root resolution                                          #
# --------------------------------------------------------------------------- #

def get_project_root() -> Path:
    """
    Return the project root directory in a way that works in:
      - Normal script execution  (uses __file__ of calling module)
      - Jupyter / Google Colab   (falls back to cwd)
      - PyInstaller bundles       (uses sys.executable directory)

    P2 fix: __file__ is undefined in Colab cells; this guard prevents NameError.
    """
    # Walk up from this file's location to find the directory
    # that contains 'src/' or 'results/' or 'data/'
    try:
        candidate = Path(__file__).resolve().parent
        for _ in range(5):
            if (candidate / "results").exists() or (candidate / "data").exists():
                return candidate
            candidate = candidate.parent
    except NameError:
        pass  # __file__ not defined (Colab / interactive)

    # Fallback: current working directory
    cwd = Path(os.getcwd())
    # If running from inside src/ngorima2025/, go up two levels
    if cwd.name == "ngorima2025" and cwd.parent.name == "src":
        return cwd.parent.parent
    return cwd


PROJECT_ROOT = get_project_root()
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"
FIGURES_DIR  = PROJECT_ROOT / "figures"
REPORT_DIR   = PROJECT_ROOT / "report"

for _d in [DATA_DIR / "raw", DATA_DIR / "processed",
           RESULTS_DIR, FIGURES_DIR, REPORT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
#  System-state logging                                                        #
# --------------------------------------------------------------------------- #

def log_system_state(out_path: Optional[Path] = None) -> dict:
    """
    Record CPU, RAM, and software versions at experiment start.
    Saved to results/system_state.json.

    P2 fix: Without this, timing results cannot be compared across machines.
    """
    state: dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform":  platform.platform(),
        "python":    sys.version,
        "cpu_count_logical":  psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb":     round(psutil.virtual_memory().total / 1e9, 2),
        "ram_available_gb": round(psutil.virtual_memory().available / 1e9, 2),
        "packages": {},
    }

    # CPU frequency (may not be available on all systems)
    try:
        freq = psutil.cpu_freq()
        state["cpu_freq_current_mhz"] = round(freq.current, 1) if freq else None
        state["cpu_freq_max_mhz"]     = round(freq.max, 1) if freq else None
    except Exception:
        state["cpu_freq_current_mhz"] = None

    # CPU scaling governor (Linux only)
    gov_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    state["cpu_governor"] = gov_path.read_text().strip() if gov_path.exists() else "unknown"

    # Key package versions
    for pkg in ["numpy", "scipy", "sklearn", "pandas",
                "xicor", "dcor", "minepy", "joblib"]:
        try:
            mod = __import__(pkg)
            state["packages"][pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            state["packages"][pkg] = "not installed"

    out_path = out_path or (RESULTS_DIR / "system_state.json")
    out_path.write_text(json.dumps(state, indent=2))
    logger.info("System state saved to %s", out_path)
    return state


# --------------------------------------------------------------------------- #
#  RAM guard                                                                   #
# --------------------------------------------------------------------------- #

def check_ram_for_dc(n: int, safety_factor: float = 1.5) -> tuple[bool, float]:
    """
    Check whether allocating the DC distance matrix (n×n float64) is safe.

    Returns (safe: bool, required_gb: float).
    Raises MemoryError with a descriptive message if unsafe and raise_on_fail=True.

    P2 fix: Prevents silent OOM crashes on machines with ≤ 8 GB RAM.
    """
    required_bytes = n * n * 8 * safety_factor  # float64 + overhead
    required_gb    = required_bytes / 1e9
    available_gb   = psutil.virtual_memory().available / 1e9
    return (available_gb >= required_gb), required_gb


def safe_dc_max_n(safety_factor: float = 1.5) -> int:
    """
    Compute the maximum n for which DC is safe on this machine.
    Uses available (not total) RAM to account for OS and other processes.
    """
    available_bytes = psutil.virtual_memory().available
    max_n = int(np.sqrt(available_bytes / (8 * safety_factor)))
    return max_n


# --------------------------------------------------------------------------- #
#  Warm-up + timed execution                                                   #
# --------------------------------------------------------------------------- #

def timed_call(
    fn: Callable,
    *args,
    n_warmup: int = 2,
    n_reps: int = 30,
    **kwargs,
) -> dict:
    """
    Execute fn(*args, **kwargs) with warm-up discards and repeated timing.

    Parameters
    ----------
    fn      : callable to time
    n_warmup: number of warm-up calls (discarded) — P3 fix
    n_reps  : number of measured repetitions (paper specifies 30) — P1 fix
    Returns dict with keys: mean, std, median, q05, q95, n_reps
    """
    # Warm-up runs (populate caches, trigger JIT compilation)
    for _ in range(n_warmup):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn(*args, **kwargs)

    times: list[float] = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)

    arr = np.array(times)
    return {
        "mean":   float(arr.mean()),
        "std":    float(arr.std(ddof=1)),
        "median": float(np.median(arr)),
        "q05":    float(np.percentile(arr, 5)),
        "q95":    float(np.percentile(arr, 95)),
        "n_reps": n_reps,
        "skipped": False,
    }


SKIPPED_RESULT = {
    "mean": np.nan, "std": np.nan, "median": np.nan,
    "q05": np.nan, "q95": np.nan, "n_reps": 0, "skipped": True,
}


# --------------------------------------------------------------------------- #
#  Checkpoint / resume (Colab session safety)                                  #
# --------------------------------------------------------------------------- #

CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"


def load_checkpoint() -> dict:
    """Load existing checkpoint or return empty dict."""
    if CHECKPOINT_PATH.exists():
        try:
            return json.loads(CHECKPOINT_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_checkpoint(state: dict) -> None:
    """Persist checkpoint atomically (write-rename pattern)."""
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(CHECKPOINT_PATH)


def checkpoint_key(scenario: str, n: int, method: str) -> str:
    return f"{scenario}_{n}_{method}"
