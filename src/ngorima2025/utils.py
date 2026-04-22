"""
utils.py — KsaaiP2 Pipeline v4_updated
src/ngorima2025/utils.py

Timing harness, statistical inference utilities, and hardware fingerprinting.

Key functions:
  - timed_call(): perf_counter timing with warm-up and N_REPS
  - bootstrap_median_ratio_ci(): 95% CI on ratio of median runtimes (RQ3)
  - wilcoxon_timing_test(): paired Wilcoxon signed-rank test (RQ3)
  - measure_mic_spawn_overhead(): fixed MIC subprocess cost in ms (RQ5)
  - log_hardware_fingerprint(): CPU/RAM/OS metadata for system_state.json
"""

import os
import gc
import json
import time
import platform
import subprocess
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# ── Constants (overridable via config.yaml / env) 
N_REPS      = int(os.environ.get("NGORIMA_N_REPS",     30))
N_REPS_MIC  = int(os.environ.get("NGORIMA_N_REPS_MIC",  5))
N_WARMUP    = int(os.environ.get("NGORIMA_N_WARMUP",    2))
FAST_MODE   = os.environ.get("NGORIMA_FAST", "0") == "1"


def timed_call(
    fn: Callable,
    *args,
    n_reps: int = N_REPS,
    n_warmup: int = N_WARMUP,
    gc_collect: bool = True,
    **kwargs,
) -> Tuple[float, float, float, List[float]]:
    """
    Time fn(*args, **kwargs) with warm-up and repeated measurements.
    Uses time.perf_counter() — monotonic, sub-microsecond resolution.
    Returns: (median_s, p5_s, p95_s, all_times)
    """
    for _ in range(n_warmup):
        fn(*args, **kwargs)
    times = []
    for _ in range(n_reps):
        if gc_collect:
            gc.collect()
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    return (float(np.median(times)), float(np.percentile(times, 5)),
            float(np.percentile(times, 95)), times.tolist())


def cov_of_times(times: List[float]) -> float:
    arr = np.array(times)
    if np.mean(arr) == 0: return float("nan")
    cov = np.std(arr) / np.mean(arr)
    if cov > 0.10:
        warnings.warn(f"MIC timing CoV = {cov:.3f} > 0.10", UserWarning, stacklevel=2)
    return float(cov)


def dc_ram_required_gb(n: int) -> float:
    return (n ** 2 * 8) / (1024 ** 3)


def available_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError: return float("inf")


def dc_feasible(n: int, safety_factor: float = 1.1) -> bool:
    required = dc_ram_required_gb(n) * safety_factor
    available = available_ram_gb()
    if required > available:
        warnings.warn(f"DC at n={n}: requires {required:.1f} GB, available {available:.1f} GB — skipping", ResourceWarning, stacklevel=2)
        return False
    return True


def save_checkpoint(path: str, state: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp,"w") as f: fjson.dump(state, f, indent=2)
    os.replace(tmp, path)


def load_checkpoint(path: str) -> Optional[dict]:
    if not os.path.exists(path): return None
    with open(path) as f: return json.load(f)


def bootstrap_median_ratio_ci(
    times_a: List[float], times_b: List[float],
    n_bootstrap: int = 10_000, conf: float = 0.95, seed: int = 42,
) -> Tuple[float, float, float]:
    """95% CI on ratio of median runtimes (RQ3)."""
    rng = np.random.default_rng(seed)
    a, b = np.array(times_a), np.array(times_b)
    point_ratio = np.median(a) / np.median(b)
    ratios = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        ra = rng.choice(a, size=len(a), replace=True)
        rb = rng.choice(b, size=len(b), replace=True)
        med_b = np.median(rb)
        ratios[i] = np.median(ra) / med_b if med_b > 0 else float("nan")
    alpha = 1 - conf
    return (float(point_ratio),
            float(np.nanpercentile(ratios, 100*alpha/2)),
            float(np.nanpercentile(ratios, 100*(1-alpha/2))))


def wilcoxon_timing_test(
    times_a: List[float], times_b: List[float], alternative: str = "two-sided",
) -> Tuple[float, float]:
    a, b = np.array(times_a), np.array(times_b)
    if len(a) != len(b): raise ValueError(&Paired vectors must have equal length")
    result = stats.wilcoxon(a, b, alternative=alternative)
    return float(result.statistic), float(result.pvalue)


def measure_mic_spawn_overhead(
    conda_env: str = "ngorima_mic", n_reps: int = 10, conf: float = 0.95,
) -> dict:
    times_ms = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        try:
            subprocess.run(["conda","run","--no-capture-output","-n",conda_env,"Python","-c","pass"], capture_output=True, timeout=60)
        except Exception: pass
        times_ms.append((time.perf_counter()-t0)*1000)
    arr = np.array(times_ms)
    alpha = 1 - conf
    return {"median_ms":float(np.median(arr)),"ci_lo_ms":float(np.percentile(arr,100*alpha/2)),"ci_hi_ms":float(np.percentile(arr,100*(1-alpha/2))),"n_reps":n_reps,"conda_env":conda_env}


def log_hardware_fingerprint() -> dict:
    info = {"python_version":platform.python_version(),"platform":platform.platform(),"processor":platform.processor(),"machine":platform.machine()}
    try:
        import psutil
        info["physical_cores"]=psutil.cpu_count(logical=False)
        info["logical_cores"]=psutil.cpu_count(logical=True)
        info["ram_total_gb"]=round(psutil.virtual_memory().total/1024**3,2)
        freq=psutil.cpu_freq(); info["cpu_freq_max_mhz"]=round(freq.max,1) if freq else None
    except ImportError: info["physical_cores"]=os.cpu_count()
    try:
        gov=subprocess.check_output(["cat","/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"],text=True,timeout=5).strip()
        info["cpu_governor"]=gov
    except Exception: info["cpu_governor"]="unknown"
    try:
        import numpy as np
        blas_str=str(getattr(np.__config__,"blas_opt_info",""))
        info["blas_backend"]="MKL" if "mkl" in blas_str.lower() else "OpenBLAS/other"
    except Exception: info["blas_backend"]="unknown"
    for pkg in ["numpy","scipy","sklearn","dcor","minepy","yaml","joblib"]:
        try:
            import importlib; mod=importlib.import_module(pkg.replace("-","_").split(".")[0])
            info[f"pkg_{pkg}"]=getattr(mod,"__version__","unknown")
        except ImportError: info[f"pkg_{pkg}"]="not installed"
    return info
