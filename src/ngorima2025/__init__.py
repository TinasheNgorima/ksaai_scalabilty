"""
ngorima2025 — KsaaiP2 feature selection complexity benchmarking package.
v4_updated · April 2026 · Sohar University
"""
from .scorers import (
    get_all_scorers,
    get_xi_scorer,
    get_scorer,
    get_theoretical_exponent,
    FALLBACK_FLAGS,
    THEORETICAL_EXPONENT,
    SCORER_HYPERPARAMS,
    score_xi_n,
    score_dc,
    score_mi,
    score_mic_subprocess,
    score_pearson,
    score_spearman,
    xi_scorer,
    dc_scorer,
    mi_scorer,
    mic_scorer,
    pearson_scorer,
    spearman_scorer,
    USE_FALLBACK_MIC,
)
from .utils import (
    timed_call,
    cov_of_times,
    bootstrap_median_ratio_ci,
    wilcoxon_timing_test,
    measure_mic_spawn_overhead,
    log_hardware_fingerprint,
    dc_feasible,
    dc_ram_required_gb,
    available_ram_gb,
    save_checkpoint,
    load_checkpoint,
    N_REPS,
    N_REPS_MIC,
    N_WARMUP,
    FAST_MODE,
)

__version__ = "4.0.0"
__author__  = "Tinashe Ngorima"

__all__ = [
    # Scorers
    "get_all_scorers", "get_xi_scorer", "get_scorer", "get_theoretical_exponent",
    "FALLBACK_FLAGS", "THEORETICAL_EXPONENT", "SCORER_HYPERPARAMS",
    "score_xi_n", "score_dc", "score_mi", "score_mic_subprocess",
    "score_pearson", "score_spearman",
    "xi_scorer", "dc_scorer", "mi_scorer", "mic_scorer",
    "pearson_scorer", "spearman_scorer",
    "USE_FALLBACK_MIC",
    # Utils
    "timed_call", "cov_of_times",
    "bootstrap_median_ratio_ci", "wilcoxon_timing_test",
    "measure_mic_spawn_overhead", "log_hardware_fingerprint",
    "dc_feasible", "dc_ram_required_gb", "available_ram_gb",
    "save_checkpoint", "load_checkpoint",
    "N_REPS", "N_REPS_MIC", "N_WARMUP", "FAST_MODE",
]


# ── v1 compatibility shims (required by 01_, 02_, 03_ pipeline scripts) ────
from pathlib import Path as _Path
import os as _os, time as _time, gc as _gc

RESULTS_DIR = _Path('results')
FIGURES_DIR = _Path('figures')
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

SCORERS = {
    'xi_n':    score_xi_n,
    'dc':      score_dc,
    'mi':      score_mi,
    'mic':     score_mic_subprocess,
    'pearson': score_pearson,
    'spearman': score_spearman,
}

LABELS = {
    'xi_n': 'xi_n', 'dc': 'DC', 'mi': 'MI',
    'mic': 'MIC', 'pearson': 'Pearson r', 'spearman': 'Spearman rho',
}

SKIPPED_RESULT = {
    'mean': None, 'std': None, 'median': None,
    'q05': None, 'q95': None, 'n_reps': 0, 'skipped': True,
}

_CKPT_PATH = _Path('results/checkpoint.json')

def save_checkpoint(state: dict) -> None:
    import json
    _CKPT_PATH.parent.mkdir(exist_ok=True)
    tmp = str(_CKPT_PATH) + '.tmp'
    with open(tmp, 'w') as f: json.dump(state, f, indent=2)
    _os.replace(tmp, str(_CKPT_PATH))

def load_checkpoint() -> dict:
    import json
    if not _CKPT_PATH.exists(): return {}
    with open(_CKPT_PATH) as f: return json.load(f)

def checkpoint_key(*args) -> str:
    return '__'.join(str(a) for a in args)

def timed_call(fn, *args, n_warmup=2, n_reps=30, **kwargs) -> dict:
    import numpy as _np
    for _ in range(n_warmup):
        fn(*args, **kwargs)
    times = []
    for _ in range(n_reps):
        _gc.collect()
        t0 = _time.perf_counter()
        fn(*args, **kwargs)
        times.append(_time.perf_counter() - t0)
    arr = _np.array(times)
    return {
        'mean': float(_np.mean(arr)), 'std': float(_np.std(arr)),
        'median': float(_np.median(arr)),
        'q05': float(_np.percentile(arr, 5)),
        'q95': float(_np.percentile(arr, 95)),
        'n_reps': n_reps, 'skipped': False,
    }

def check_ram_for_dc(n: int, safety_factor: float = 1.1):
    req_gb = dc_ram_required_gb(n) * safety_factor
    avail  = available_ram_gb()
    return (req_gb <= avail), req_gb

def safe_dc_max_n(safety_factor: float = 1.5) -> int:
    import numpy as _np
    avail = available_ram_gb()
    max_n = int((avail / (safety_factor * 8)) ** 0.5 * 1024**3 ** 0.5)
    return max(1000, min(max_n, 100_000))

def log_system_state() -> dict:
    info = log_hardware_fingerprint()
    info['ram_available_gb'] = round(available_ram_gb(), 2)
    info.setdefault('cpu_governor', 'unknown')
    import json, datetime
    info['generated_at'] = datetime.datetime.utcnow().isoformat() + 'Z'
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / 'system_state.json', 'w') as f:
        json.dump(info, f, indent=2)
    return info
