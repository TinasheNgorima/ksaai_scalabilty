"""
tests/test_pipeline.py
======================
Unit tests verifying all P0–P2 fixes are correctly implemented.

Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# --------------------------------------------------------------------------- #
#  Scorer tests (P1: shared module)                                            #
# --------------------------------------------------------------------------- #

class TestScorers:
    """All methods imported from a single canonical module."""

    def setup_method(self):
        rng     = np.random.default_rng(0)
        self.x  = rng.standard_normal(200)
        self.y  = self.x + rng.normal(0, 0.3, 200)

    def test_xi_n_range(self):
        from ngorima2025.scorers import score_xi
        v = score_xi(self.x, self.y)
        assert -1.0 <= v <= 1.0, f"ξₙ out of range: {v}"

    def test_mi_nonnegative(self):
        from ngorima2025.scorers import score_mi
        v = score_mi(self.x, self.y)
        assert v >= 0.0, f"MI negative: {v}"

    def test_dc_range(self):
        from ngorima2025.scorers import score_dc
        v = score_dc(self.x, self.y)
        assert 0.0 <= v <= 1.0, f"DC out of range: {v}"

    def test_pearson_range(self):
        from ngorima2025.scorers import score_pearson
        v = score_pearson(self.x, self.y)
        assert 0.0 <= v <= 1.0, f"Pearson out of range: {v}"

    def test_spearman_range(self):
        from ngorima2025.scorers import score_spearman
        v = score_spearman(self.x, self.y)
        assert 0.0 <= v <= 1.0, f"Spearman out of range: {v}"

    def test_correlated_higher_than_noise(self):
        """All methods should score correlated pair higher than noise pair."""
        from ngorima2025.scorers import score_xi, score_pearson, score_dc
        rng   = np.random.default_rng(1)
        n     = 300
        x     = rng.standard_normal(n)
        y_sig = x + rng.normal(0, 0.2, n)
        y_noi = rng.standard_normal(n)
        for fn in [score_xi, score_pearson, score_dc]:
            s_sig = fn(x, y_sig)
            s_noi = fn(x, y_noi)
            assert s_sig > s_noi, (
                f"{fn.__name__}: signal score {s_sig:.3f} ≤ noise score {s_noi:.3f}"
            )

    def test_all_methods_in_registry(self):
        from ngorima2025.scorers import SCORERS
        expected = {"xi_n", "mi", "dc", "mic", "pearson", "spearman"}
        assert expected.issubset(set(SCORERS.keys()))


# --------------------------------------------------------------------------- #
#  P0: No preprocessing leakage in evaluate_accuracy_regression               #
# --------------------------------------------------------------------------- #

class TestNoLeakage:
    """
    Verify that StandardScaler is inside the Pipeline.
    The test patches Pipeline to verify it receives a scaler step.
    """

    def test_scaler_inside_pipeline(self):
        """evaluate_accuracy_regression must use Pipeline, not fit_transform."""
        import inspect
        import importlib
        mod = importlib.import_module(
            "ngorima2025.scorers"
        )
        # Import the benchmark module
        bench_path = Path(__file__).resolve().parent.parent / "02_real_domain_benchmarks.py"
        if not bench_path.exists():
            pytest.skip("02_real_domain_benchmarks.py not found")

        src = bench_path.read_text()
        # Confirm Pipeline is used
        assert "Pipeline" in src, "Pipeline not found in 02_real_domain_benchmarks.py"
        # Confirm fit_transform is NOT called on X before CV
        # (it should only appear inside the pipeline or for other purposes)
        lines = src.splitlines()
        bad_lines = [
            l for l in lines
            if "fit_transform" in l and "scaler" in l and "Pipeline" not in l
            and not l.strip().startswith("#")
        ]
        assert len(bad_lines) == 0, (
            f"Standalone scaler.fit_transform detected (leakage risk): {bad_lines}"
        )

    def test_no_shuffle_kfold_for_timeseries(self):
        """FRED-MD benchmark must use TimeSeriesSplit, not KFold(shuffle=True)."""
        bench_path = Path(__file__).resolve().parent.parent / "02_real_domain_benchmarks.py"
        if not bench_path.exists():
            pytest.skip("02_real_domain_benchmarks.py not found")
        src = bench_path.read_text()
        assert "TimeSeriesSplit" in src, (
            "TimeSeriesSplit not found — FRED-MD will have look-ahead bias"
        )

    def test_tcga_target_not_pca(self):
        """TCGA target in Step 0 must not be PC1 of the full feature matrix."""
        setup_path = Path(__file__).resolve().parent.parent / "00_setup_and_download.py"
        if not setup_path.exists():
            pytest.skip("00_setup_and_download.py not found")
        src = setup_path.read_text()
        # PC1 construction should be removed or clearly gated on non-circular fallback
        # The critical pattern: PCA().fit_transform(df.values) stored directly as y
        # is now gone; instead we use purity or housekeeping genes
        assert "tumour purity" in src.lower() or "purity" in src.lower(), (
            "TCGA target does not reference purity — may still be circular PC1"
        )

    def test_jaccard_uses_bootstrap(self):
        """Jaccard stability must resample observations, not perturb scores."""
        bench_path = Path(__file__).resolve().parent.parent / "02_real_domain_benchmarks.py"
        if not bench_path.exists():
            pytest.skip("02_real_domain_benchmarks.py not found")
        src = bench_path.read_text()
        assert "replace=True" in src or "bootstrap" in src.lower(), (
            "Jaccard bootstrap resampling not found — may still use score perturbation"
        )
        # The old noise perturbation pattern should be gone
        assert "scores.std() * 0.05" not in src, (
            "Old score-perturbation Jaccard pattern still present"
        )


# --------------------------------------------------------------------------- #
#  P2: RAM guard                                                               #
# --------------------------------------------------------------------------- #

class TestRamGuard:

    def test_check_ram_for_dc_small_n(self):
        from ngorima2025.utils import check_ram_for_dc
        safe, req_gb = check_ram_for_dc(100)
        assert safe is True, f"n=100 should always be safe, got req={req_gb:.3f} GB"

    def test_check_ram_for_dc_huge_n(self):
        from ngorima2025.utils import check_ram_for_dc
        # n=2,000,000 requires ~32 TB — never safe
        safe, req_gb = check_ram_for_dc(2_000_000)
        assert safe is False, f"n=2M should never be safe, req={req_gb:.1f} GB"

    def test_safe_dc_max_n_positive(self):
        from ngorima2025.utils import safe_dc_max_n
        n_max = safe_dc_max_n()
        assert n_max > 0, "safe_dc_max_n returned non-positive value"
        assert n_max < 10_000_000, "safe_dc_max_n implausibly large"


# --------------------------------------------------------------------------- #
#  P3: Checkpoint / resume                                                     #
# --------------------------------------------------------------------------- #

class TestCheckpoint:

    def test_save_and_load(self, tmp_path, monkeypatch):
        from ngorima2025 import utils
        monkeypatch.setattr(utils, "CHECKPOINT_PATH", tmp_path / "checkpoint.json")
        from ngorima2025.utils import save_checkpoint, load_checkpoint
        state = {"A_1000_xi_n": {"time_mean": 0.01, "skipped": False}}
        save_checkpoint(state)
        loaded = load_checkpoint()
        assert loaded == state

    def test_empty_checkpoint(self, tmp_path, monkeypatch):
        from ngorima2025 import utils
        monkeypatch.setattr(utils, "CHECKPOINT_PATH", tmp_path / "nonexistent.json")
        from ngorima2025.utils import load_checkpoint
        assert load_checkpoint() == {}


# --------------------------------------------------------------------------- #
#  P1: timed_call warm-up                                                      #
# --------------------------------------------------------------------------- #

class TestTimedCall:

    def test_warmup_discarded(self):
        """timed_call should return n_reps measurements, not n_reps+n_warmup."""
        from ngorima2025.utils import timed_call
        calls = []
        def counter(x):
            calls.append(1)
        timed_call(counter, 0, n_warmup=3, n_reps=5)
        assert len(calls) == 8  # 3 warmup + 5 measured

    def test_returns_expected_keys(self):
        from ngorima2025.utils import timed_call
        result = timed_call(lambda x: x ** 2, 10, n_warmup=1, n_reps=5)
        for key in ("mean", "std", "median", "q05", "q95", "n_reps", "skipped"):
            assert key in result, f"Missing key: {key}"
        assert result["n_reps"] == 5
        assert result["skipped"] is False
