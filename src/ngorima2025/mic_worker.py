"""
mic_worker.py — KsaaiP2 Pipeline v4_updated
src/ngorima2025/mic_worker.py

MIC computation entry point executed inside the ngorima_mic conda environment
(Python 3.10, minepy 1.2.6, setuptools<66) via:

    conda run -n ngorima_mic python mic_worker.py <data_path.npz>

This subprocess isolation is required because minepy cannot be installed
on Python 3.12+ (distutils removed in Python 3.12).

Protocol:
  1. Load X, y, alpha, c from .npz file passed as argv[1]
  2. Perform N_WARMUP=2 warm-up MIC evaluations (discarded)
  3. Run N_REPS_MIC=5 timed evaluations with time.perf_counter()
  4. Print median MIC score to stdout (read by score_mic_subprocess())
  5. Exit 0 on success, non-zero on failure
"""

import sys
import time
import numpy as np

N_WARMUP   = 2
N_REPS_MIC = 5


def compute_mic(X: np.ndarray, y: np.ndarray,
                alpha: float = 0.6, c: float = 15.0) -> float:
    from minepy import MINE
    mine = MINE(alpha=alpha, c=c)
    mine.compute_score(X, y)
    return float(mine.mic())


def main():
    if len(sys.argv) < 2:
        print("Usage: python mic_worker.py <data.npz>", file=sys.stderr)
        sys.exit(1)
    data = np.load(sys.argv[1])
    X = data["X"].ravel(); y = data["y"].ravel()
    alpha = float(data["alpha"][0]); c = float(data["c"][0])
    for _ in range(N_WARMUP): compute_mic(X, y, alpha=alpha, c=c)
    scores = []
    for _ in range(N_REPS_MIC):
        scores.append(compute_mic(X, y, alpha=alpha, c=c))
    print(float(np.median(scores)))
    sys.exit(0)

if __name__ == "__main__": main()
