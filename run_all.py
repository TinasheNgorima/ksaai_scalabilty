#!/usr/bin/env python3
"""
run_all.py
==========
Ngorima (2026) — Master Pipeline Runner.

Usage
-----
    python run_all.py                    # Full pipeline (n_reps=30)
    python run_all.py --fast             # Fast mode  (n_reps=3, small n)
    python run_all.py --step 1           # Single step
    python run_all.py --from-step 2      # Resume from step 2
    python run_all.py --install          # Install / verify dependencies
    python run_all.py --check            # Dependency check only (no run)

Environment variables
---------------------
    NGORIMA_FAST=1   Equivalent to --fast (useful in CI / Colab cells)

Fixes applied
-------------
P1 : fast mode sets n_reps=3 via environment variable read by step scripts
P2 : dependency check warns clearly about fallback implementations
P3 : Colab-safe __file__ resolution; exit codes propagated correctly
P3 : Zenodo / GitHub badge URLs documented in header
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Colab-safe project root ─────────────────────────────────────────────────
try:
    PIPELINE_DIR = Path(__file__).resolve().parent
except NameError:
    PIPELINE_DIR = Path(os.getcwd())  # Colab / interactive fallback

sys.path.insert(0, str(PIPELINE_DIR / "src"))

# --------------------------------------------------------------------------- #
#  Step registry                                                               #
# --------------------------------------------------------------------------- #
STEPS: dict[int, tuple[str, str]] = {
    0: ("00_setup_and_download.py",        "Data Setup & Download"),
    1: ("01_synthetic_benchmarks.py",      "Synthetic Complexity Benchmarks"),
    2: ("02_real_domain_benchmarks.py",    "Real-Domain Benchmarks"),
    3: ("03_memory_and_parallelisation.py","Memory & Parallelisation"),
    4: ("04_compile_results.py",           "Compile Results & Report"),
}

# --------------------------------------------------------------------------- #
#  Dependency check                                                            #
# --------------------------------------------------------------------------- #
CORE_DEPS = [
    ("numpy",        "numpy"),
    ("scipy",        "scipy"),
    ("sklearn",      "scikit-learn"),
    ("pandas",       "pandas"),
    ("matplotlib",   "matplotlib"),
    ("joblib",       "joblib"),
    ("psutil",       "psutil"),
    ("tqdm",         "tqdm"),
]
OPTIONAL_DEPS = [
    ("xicor",   "xicor",   "ξₙ scorer (fallback: pure NumPy)"),
    ("dcor",    "dcor",    "DC scorer (fallback: O(n²) NumPy)"),
    ("minepy",  "minepy",  "MIC scorer (fallback: histogram MI — differs from Reshef 2011)"),
]


def check_dependencies(verbose: bool = True) -> bool:
    """
    Verify core and optional packages.
    Returns True if all core deps are present (pipeline can run).
    Warns — does not fail — for optional packages.
    """
    all_core_ok = True
    if verbose:
        print("\n── Dependency Check ──────────────────────────────────")

    for import_name, pip_name in CORE_DEPS:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "?")
            if verbose:
                print(f"  [OK ] {pip_name:<20} {ver}")
        except ImportError:
            all_core_ok = False
            if verbose:
                print(f"  [ERR] {pip_name:<20} NOT INSTALLED")
                print(f"        → pip install {pip_name}")

    if verbose:
        print()
    for import_name, pip_name, note in OPTIONAL_DEPS:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "?")
            if verbose:
                print(f"  [OK ] {pip_name:<20} {ver}  (native)")
        except ImportError:
            if verbose:
                print(f"  [WRN] {pip_name:<20} not installed — {note}")
                print(f"        → pip install {pip_name}  "
                      f"(or: conda install -c conda-forge {pip_name})")

    if not all_core_ok and verbose:
        print("\n  Install core dependencies:")
        print("    pip install -r requirements.txt")
        print("  or:")
        print("    conda env create -f environment.yml && conda activate ngorima2025")

    return all_core_ok


def install_dependencies() -> None:
    """Attempt to pip-install all packages from requirements.txt."""
    req = PIPELINE_DIR / "requirements.txt"
    if req.exists():
        print(f"  Installing from {req} ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req)],
            check=False,
        )
    else:
        print("  requirements.txt not found — running Step 0 first to generate it.")
        subprocess.run([sys.executable,
                        str(PIPELINE_DIR / "00_setup_and_download.py")],
                       cwd=PIPELINE_DIR, check=False)
        if req.exists():
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req)],
                check=False,
            )

    # minepy via conda if pip failed
    try:
        import minepy  # noqa: F401
    except ImportError:
        print("\n  minepy not installed via pip.")
        print("  Try:  conda install -c conda-forge minepy")
        print("  MIC will use histogram fallback until minepy is available.")


# --------------------------------------------------------------------------- #
#  Step runner                                                                 #
# --------------------------------------------------------------------------- #

def run_step(step_num: int, fast: bool = False) -> bool:
    script, name = STEPS[step_num]
    script_path  = PIPELINE_DIR / script

    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {name}")
    if fast:
        print("  [FAST MODE] n_reps=3, n_warmup=1, reduced sample sizes")
    print(f"{'='*60}")

    env = os.environ.copy()
    if fast:
        env["NGORIMA_FAST"] = "1"   # read by step scripts to reduce reps

    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PIPELINE_DIR,
        env=env,
    )
    elapsed = time.perf_counter() - t0

    if result.returncode == 0:
        print(f"\n  ✓ Step {step_num} completed in {elapsed:.1f}s "
              f"({elapsed/60:.1f} min)")
        return True
    else:
        print(f"\n  ✗ Step {step_num} failed (exit code {result.returncode}) "
              f"after {elapsed:.1f}s")
        return False


# --------------------------------------------------------------------------- #
#  Fast-mode shim (read by individual step scripts)                            #
# --------------------------------------------------------------------------- #

def is_fast_mode() -> bool:
    """
    Returns True when running in fast mode (CI / Colab / testing).
    Step scripts call this to reduce n_reps and sample sizes.
    """
    return os.environ.get("NGORIMA_FAST", "0") == "1"


# --------------------------------------------------------------------------- #
#  Progress summary                                                            #
# --------------------------------------------------------------------------- #

def print_summary(
    failed_steps: list[int],
    total_elapsed: float,
    fast: bool,
) -> None:
    print(f"\n{'='*60}")
    print(f"PIPELINE {'[FAST MODE] ' if fast else ''}COMPLETE")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"{'='*60}")

    if failed_steps:
        print(f"\n  Steps with errors: {failed_steps}")
        print("  Check step output above for details.")
        print("  Resume with:  python run_all.py --from-step "
              f"{min(failed_steps)}")
    else:
        print("\n  All steps completed successfully.")

    from ngorima2025.utils import REPORT_DIR, RESULTS_DIR, FIGURES_DIR
    print(f"\n  Report  : {REPORT_DIR / 'Ngorima2026_Results_Report.md'}")
    print(f"  Results : {RESULTS_DIR}")
    print(f"  Figures : {FIGURES_DIR}")

    if not failed_steps:
        print("""
  Next steps for publication:
    1. Download real TCGA data (see README.md § Manual Download)
    2. Run full pipeline on paper hardware (i7-12700H, 32 GB RAM)
    3. Deposit to Zenodo and update DOI badge in README.md
    4. Set CPU governor: sudo cpupower frequency-set -g performance
""")


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ngorima (2026) Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py                    # Full pipeline
  python run_all.py --fast             # Fast mode (CI / Colab)
  python run_all.py --step 2           # Step 2 only
  python run_all.py --from-step 1      # Resume from Step 1
  python run_all.py --install          # Install dependencies
  python run_all.py --check            # Check dependencies only
        """,
    )
    parser.add_argument("--step",      type=int,  default=None,
                        help="Run only this step number (0–4)")
    parser.add_argument("--from-step", type=int,  default=0,
                        help="Start from this step (skip earlier ones)")
    parser.add_argument("--fast",      action="store_true",
                        help="Fast mode: n_reps=3, reduced n (for CI/Colab)")
    parser.add_argument("--install",   action="store_true",
                        help="Install/verify dependencies and exit")
    parser.add_argument("--check",     action="store_true",
                        help="Check dependencies and exit (no pipeline run)")
    return parser


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║  Ngorima (2026) — Computational Complexity Pipeline              ║
║  ξₙ · DC · MI · MIC · Pearson · Spearman                        ║
║  Multi-Domain Scalability Analysis                               ║
╚══════════════════════════════════════════════════════════════════╝
"""


if __name__ == "__main__":
    print(BANNER)
    args = build_parser().parse_args()

    # ── Propagate fast mode via env var ────────────────────────────────────
    if args.fast:
        os.environ["NGORIMA_FAST"] = "1"

    # ── Install ─────────────────────────────────────────────────────────────
    if args.install:
        install_dependencies()
        check_dependencies(verbose=True)
        sys.exit(0)

    # ── Dependency check ────────────────────────────────────────────────────
    ok = check_dependencies(verbose=True)
    if args.check:
        sys.exit(0 if ok else 1)
    if not ok:
        print("\n  [ERROR] Core dependencies missing. Run:")
        print("    python run_all.py --install")
        sys.exit(1)

    # ── Single step ─────────────────────────────────────────────────────────
    if args.step is not None:
        success = run_step(args.step, fast=args.fast)
        sys.exit(0 if success else 1)

    # ── Full pipeline ────────────────────────────────────────────────────────
    from_step    = args.from_step
    total_start  = time.perf_counter()
    failed_steps: list[int] = []

    for step_num in sorted(STEPS.keys()):
        if step_num < from_step:
            print(f"  [SKIP] Step {step_num} (starting from step {from_step})")
            continue
        success = run_step(step_num, fast=args.fast)
        if not success:
            failed_steps.append(step_num)
            if step_num < 4:
                print(f"  [WARN] Step {step_num} failed — continuing to next step.")

    print_summary(failed_steps, time.perf_counter() - total_start, args.fast)
    sys.exit(1 if failed_steps else 0)
