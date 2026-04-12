"""
tests/conftest.py
=================
Shared pytest fixtures for Ngorima (2025) test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# --------------------------------------------------------------------------- #
#  Data fixtures                                                               #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def small_iid(rng) -> tuple[np.ndarray, np.ndarray]:
    """Small i.i.d. dataset (n=300, p=20) for fast unit tests."""
    X = rng.standard_normal((300, 20))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.3, 300)
    return X.astype(np.float64), y.astype(np.float64)


@pytest.fixture(scope="session")
def small_timeseries(rng) -> tuple[np.ndarray, np.ndarray]:
    """Small time-ordered dataset (n=120, p=10) for temporal CV tests."""
    n = 120
    t = np.arange(n)
    X = np.column_stack([
        np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.1, n),
        rng.standard_normal((n, 9)),
    ])
    y = X[:, 0] + 0.3 * X[:, 1] + rng.normal(0, 0.2, n)
    return X.astype(np.float64), y.astype(np.float64)


@pytest.fixture(scope="session")
def correlated_pair(rng) -> tuple[np.ndarray, np.ndarray]:
    """Strongly correlated (x, y) pair for scorer sanity checks."""
    x = rng.standard_normal(500)
    y = x + rng.normal(0, 0.2, 500)
    return x.astype(np.float64), y.astype(np.float64)


@pytest.fixture(scope="session")
def noise_pair(rng) -> tuple[np.ndarray, np.ndarray]:
    """Independent (x, y) pair — scores should be near zero."""
    x = rng.standard_normal(500)
    y = rng.standard_normal(500)
    return x.astype(np.float64), y.astype(np.float64)
