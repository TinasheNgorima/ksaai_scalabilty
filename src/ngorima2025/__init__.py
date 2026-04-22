"""
ngorima2025 — KsaaiP2 feature selection complexity benchmarking package.
v4_updated · April 2026 · Sohar University
"""
from .scorers import (
    get_all_scorers, get_xi_scorer, get_scorer, get_theoretical_exponent,
    FALLBACK_FLAGS, THEORETICAL_EXPONENT, SCORER_HYPERPARAMS,
    score_xi_n, score_dc, score_mi, score_mic_subprocess, score_pearson, score_spearman,
    xi_scorer, dc_scorer, mi_scorer, mic_scorer, pearson_scorer, spearman_scorer, USE_FALLBACK_MIC,
)
from .utils import (
    timed_call, cov_of_times, bootstrap_median_ratio_ci, wilcoxon_timing_test,
    measure_mic_spawn_overhead, log_hardware_fingerprint,
    dc_feasible, dc_ram_required_gb, available_ram_gb,
    save_checkpoint, load_checkpoint, N_REPS, N_REPS_MIC, N_WARMUP, FAST_MODE,
)
__version__ = "4.0.0"
__author__  = "Tinashe Ngorima"
__all__ = [
    "get_all_scorers","get_xi_scorer","get_scorer","get_theoretical_exponent",
    "FALLBACK_FLAGS","THEORETICAL_EXPONENT","SCORER_HYPERPARAMS",
    "score_xi_n","score_dc","score_mi","score_mic_subprocess",
    "score_pearson","score_spearman",
    "xi_scorer","dc_scorer","mi_scorer","mic_scorer",
    "pearson_scorer","spearman_scorer","USE_FALLBACK_MIC",
    "timed_call","cov_of_times","bootstrap_median_ratio_ci","wilcoxon_timing_test",
    "measure_mic_spawn_overhead","log_hardware_fingerprint",
    "dc_feasible","dc_ram_required_gb","available_ram_gb",
    "save_checkpoint","load_checkpoint","N_REPS","N_REPS_MIC","N_WARMUP","FAST_MODE",
]
