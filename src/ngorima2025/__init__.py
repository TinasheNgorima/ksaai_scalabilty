"""
ngorima2025 — Computational Complexity of Correlation-Based Feature Selection
=============================================================================
Ngorima, T. (2025). Sohar University.
"""
from .scorers import SCORERS, LABELS, THEORETICAL_EXPONENT, FALLBACK_FLAGS
from .utils   import (
    PROJECT_ROOT, DATA_DIR, RESULTS_DIR, FIGURES_DIR, REPORT_DIR,
    log_system_state, check_ram_for_dc, safe_dc_max_n,
    timed_call, SKIPPED_RESULT,
    load_checkpoint, save_checkpoint, checkpoint_key,
)

__version__ = "1.0.0"
__all__ = [
    "SCORERS", "LABELS", "THEORETICAL_EXPONENT", "FALLBACK_FLAGS",
    "PROJECT_ROOT", "DATA_DIR", "RESULTS_DIR", "FIGURES_DIR", "REPORT_DIR",
    "log_system_state", "check_ram_for_dc", "safe_dc_max_n",
    "timed_call", "SKIPPED_RESULT",
    "load_checkpoint", "save_checkpoint", "checkpoint_key",
]
