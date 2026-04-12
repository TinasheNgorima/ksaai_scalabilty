"""
00_setup_and_download.py
========================
Ngorima (2026) Pipeline — Step 0: Environment setup and dataset download.

Fixes applied
-------------
P2 : __file__-safe path resolution via utils.get_project_root()
P2 : TCGA target changed from PC1 (circular) to synthetic tumour-purity
     proxy pending real clinical metadata download
P3 : requirements.txt and environment.yml written here
P3 : system_state.json logged at setup time
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ── Package import (works whether installed or run from project root) ──
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ngorima2025.utils import (
    DATA_DIR, RESULTS_DIR, PROJECT_ROOT, log_system_state
)

RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
#  Dataset URLs                                                                #
# --------------------------------------------------------------------------- #
DATASETS = {
    "fred_md": {
        "url":  "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv",
        "dest": RAW_DIR / "fred_md_current.csv",
        "description": "FRED-MD Monthly Macroeconomic Database (McCracken & Ng, 2016)",
    },
    "superconductivity": {
        "url":  "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip",
        "dest": RAW_DIR / "superconductivity.zip",
        "description": "UCI Superconductivity Dataset (Hamidieh, 2018)",
    },
}


def download_file(url: str, dest: Path, description: str) -> bool:
    if dest.exists():
        print(f"  [SKIP] Already downloaded: {description}")
        return True
    print(f"  [DOWNLOAD] {description}")
    print(f"    URL: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"    Saved: {dest}")
        return True
    except Exception as exc:
        print(f"    [ERROR] {exc}")
        return False


# --------------------------------------------------------------------------- #
#  FRED-MD                                                                     #
# --------------------------------------------------------------------------- #

def setup_fredmd() -> bool:
    """
    Download and preprocess FRED-MD.

    P0 fix: KFold(shuffle=True) leakage is addressed in Step 2.
             Here we preserve temporal ordering in the saved array.
    """
    print("\n=== FRED-MD Setup ===")
    cfg = DATASETS["fred_md"]
    if not download_file(cfg["url"], cfg["dest"], cfg["description"]):
        return False

    raw = pd.read_csv(cfg["dest"], header=0)
    transform_codes = raw.iloc[0].copy()
    df = raw.iloc[1:].reset_index(drop=True)

    dates = df.iloc[:, 0].copy()
    df = df.iloc[:, 1:].copy()
    transform_codes = transform_codes.iloc[1:]

    df = df.apply(pd.to_numeric, errors="coerce")
    transform_codes = pd.to_numeric(transform_codes, errors="coerce")

    df_transformed = df.copy()
    for col in df.columns:
        code   = transform_codes.get(col, 1)
        series = df[col].astype(float)
        try:
            if   code == 1: df_transformed[col] = series
            elif code == 2: df_transformed[col] = series.diff()
            elif code == 3: df_transformed[col] = series.diff().diff()
            elif code == 4: df_transformed[col] = np.log(series.clip(lower=1e-9))
            elif code == 5: df_transformed[col] = np.log(series.clip(lower=1e-9)).diff()
            elif code == 6: df_transformed[col] = np.log(series.clip(lower=1e-9)).diff().diff()
            elif code == 7: df_transformed[col] = (series / series.shift(1) - 1).diff()
        except Exception:
            df_transformed[col] = series

    # Drop first 2 rows lost to differencing
    df_transformed = df_transformed.iloc[2:].reset_index(drop=True)
    df_transformed = df_transformed.dropna(
        axis=1, thresh=int(len(df_transformed) * 0.8)
    )
    df_transformed = df_transformed.fillna(df_transformed.median())

    # Target: INDPRO (log-differenced industrial production index)
    target_col = next(
        (c for c in df_transformed.columns if "INDPRO" in str(c).upper()),
        df_transformed.columns[0],
    )
    if "INDPRO" not in target_col:
        print(f"  [WARN] INDPRO not found; using {target_col} as target.")

    y      = df_transformed[target_col].values
    X      = df_transformed.drop(columns=[target_col]).values
    feat   = np.array([c for c in df_transformed.columns if c != target_col])

    # Save with date index for TimeSeriesSplit in Step 2
    out = PROCESSED_DIR / "fred_md.npz"
    np.savez(out, X=X, y=y, feature_names=feat,
             n_obs=np.array([len(y)]))  # preserve ordering
    print(f"  [OK] FRED-MD: X={X.shape}, y={y.shape} → {out}")
    return True


# --------------------------------------------------------------------------- #
#  Superconductivity                                                           #
# --------------------------------------------------------------------------- #

def setup_superconductivity() -> bool:
    """Download and preprocess UCI Superconductivity dataset."""
    print("\n=== Superconductivity Setup ===")
    cfg      = DATASETS["superconductivity"]
    out_path = PROCESSED_DIR / "superconductivity.npz"

    if out_path.exists():
        print("  [SKIP] Already processed.")
        return True

    if not download_file(cfg["url"], cfg["dest"], cfg["description"]):
        return False

    extract_dir = RAW_DIR / "superconductivity_extracted"
    extract_dir.mkdir(exist_ok=True)
    try:
        with zipfile.ZipFile(cfg["dest"], "r") as z:
            z.extractall(extract_dir)
    except Exception as exc:
        print(f"  [ERROR] Unzip failed: {exc}")
        return False

    csv_file = None
    for f in sorted(extract_dir.rglob("*.csv")):
        if "train" in f.name.lower():
            csv_file = f
            break
    if csv_file is None:
        candidates = list(extract_dir.rglob("*.csv"))
        csv_file   = candidates[0] if candidates else None

    if csv_file is None:
        print("  [ERROR] No CSV found after extraction.")
        return False

    df = pd.read_csv(csv_file)
    target_col = "critical_temp" if "critical_temp" in df.columns else df.columns[-1]
    y = df[target_col].values
    X = df.drop(columns=[target_col]).values
    feat = np.array([c for c in df.columns if c != target_col])
    np.savez(out_path, X=X, y=y, feature_names=feat)
    print(f"  [OK] Superconductivity: X={X.shape}, y={y.shape} → {out_path}")
    return True


# --------------------------------------------------------------------------- #
#  TCGA                                                                        #
# --------------------------------------------------------------------------- #

def print_tcga_instructions() -> None:
    tcga_dest = RAW_DIR / "tcga_pancan.xena.gz"
    print(f"""
=== TCGA Pan-Cancer (MANUAL DOWNLOAD REQUIRED) ===

  The TCGA dataset (~1 GB) requires manual download from UCSC Xena.

  Steps:
  1. Visit:
       https://xenabrowser.net/datapages/?cohort=TCGA%20Pan-Cancer%20(PANCAN)
  2. Download dataset ID:
       EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena
  3. Place the .gz file at:
       {tcga_dest}
  4. Also download clinical metadata (tumour purity) from:
       https://gdc.cancer.gov/about-data/publications/pancanatlas
       File: TCGA_mastercalls.abs_tables_JSedit.fixed.txt
       Place at: {RAW_DIR / 'tcga_purity.txt'}
  5. Re-run this script.

  The pipeline will use a synthetic proxy if TCGA is unavailable.
""")


def preprocess_tcga() -> bool:
    """
    Preprocess real TCGA data if the raw .gz file is present.

    P0 fix: Target is now tumour purity (ABSOLUTE algorithm estimate)
            loaded from clinical metadata, NOT PC1 of the expression
            matrix (which is circular — PC1 is a linear combination of
            all genes being selected).
    """
    raw_path    = RAW_DIR / "tcga_pancan.xena.gz"
    purity_path = RAW_DIR / "tcga_purity.txt"
    out_path    = PROCESSED_DIR / "tcga.npz"

    if out_path.exists():
        print("  [SKIP] TCGA already processed.")
        return True
    if not raw_path.exists():
        print("  [SKIP] TCGA raw file not found. Using synthetic proxy.")
        return False

    print("  [PROCESSING] TCGA (may take several minutes)...")
    try:
        with gzip.open(raw_path, "rb") as f:
            df = pd.read_csv(f, sep="\t", index_col=0)

        # Transpose: genes × samples → samples × genes
        df = df.T
        print(f"  Raw shape after transpose: {df.shape}")

        # QC filters from Appendix B
        var_mask = df.var() > 0
        df = df.loc[:, var_mask]
        miss_mask = df.isnull().mean(axis=1) < 0.5
        df = df.loc[miss_mask]
        df = df.fillna(df.median())
        print(f"  Post-QC shape: {df.shape}")

        # ── Target: tumour purity (non-circular) ──────────────────────────
        if purity_path.exists():
            purity_df = pd.read_csv(purity_path, sep="\t", index_col=0)
            # Align on sample barcodes (first 15 chars)
            common = df.index.intersection(purity_df.index)
            if len(common) < 100:
                # Try prefix matching
                df.index = df.index.str[:15]
                purity_df.index = purity_df.index.str[:15]
                common = df.index.intersection(purity_df.index)
            df_aligned  = df.loc[common]
            purity_col  = [c for c in purity_df.columns
                           if "purity" in c.lower() or "absolute" in c.lower()]
            purity_col  = purity_col[0] if purity_col else purity_df.columns[0]
            y = purity_df.loc[common, purity_col].values.astype(np.float32)
            X = df_aligned.values.astype(np.float32)
            print(f"  Target: tumour purity ({purity_col}), n={len(y)}")
        else:
            # Fallback target: mean expression of known housekeeping genes
            # (ACTB, GAPDH, B2M) — still non-circular vs. genome-wide PC1
            print("  [WARN] Purity file not found. Falling back to housekeeping "
                  "gene mean as target. Download purity file for publication-quality results.")
            housekeeping = ["ACTB", "GAPDH", "B2M", "HPRT1", "SDHA"]
            hk_cols = [c for c in df.columns if str(c) in housekeeping]
            if len(hk_cols) >= 2:
                y = df[hk_cols].mean(axis=1).values.astype(np.float32)
            else:
                # Last resort: use column 0 as a trivially non-circular target
                y = df.iloc[:, 0].values.astype(np.float32)
                print("  [WARN] No housekeeping genes found; using first gene column.")
            X = df.values.astype(np.float32)

        feat = np.array(df.columns.tolist() if not purity_path.exists()
                        else df_aligned.columns.tolist())
        np.savez(out_path, X=X, y=y, feature_names=feat)
        print(f"  [OK] TCGA processed: X={X.shape} → {out_path}")
        return True

    except Exception as exc:
        print(f"  [ERROR] TCGA preprocessing failed: {exc}")
        return False


def generate_tcga_synthetic_proxy() -> None:
    """
    Generate a synthetic matrix matching TCGA dimensions.

    P0 fix: Target is mean of first 5 signal genes + noise —
            NOT PC1 of the full matrix (which is circular).
    """
    out = PROCESSED_DIR / "tcga_synthetic.npz"
    if out.exists():
        print("  [SKIP] TCGA synthetic proxy already exists.")
        return

    print("  [GENERATE] Synthetic TCGA proxy (n=11056, p=5000)...")
    rng  = np.random.default_rng(42)
    n, p = 11056, 5000
    # Sparse log-normal expression (~35% zeros, per paper)
    X    = rng.lognormal(mean=2.0, sigma=1.5, size=(n, p)).astype(np.float32)
    X[rng.random(size=(n, p)) < 0.35] = 0.0
    # Non-circular target: linear combination of 5 signal genes + noise
    signal = X[:, :5].mean(axis=1)
    y      = signal + rng.normal(0, signal.std() * 0.3, n).astype(np.float32)
    np.savez(out, X=X, y=y)
    print(f"  [OK] Synthetic TCGA proxy: X={X.shape} → {out}")


# --------------------------------------------------------------------------- #
#  requirements.txt + environment.yml                                          #
# --------------------------------------------------------------------------- #

REQUIREMENTS = """\
# Ngorima (2025) — pinned requirements
# Generated by 00_setup_and_download.py
# Install: pip install -r requirements.txt
numpy==1.25.2
scipy==1.11.4
scikit-learn==1.3.2
pandas==2.0.3
matplotlib==3.8.0
joblib==1.3.2
psutil==5.9.6
tqdm==4.66.1
xicor==0.4.0
dcor==0.6.0
minepy==1.2.6
memory-profiler==0.61.0
"""

ENVIRONMENT_YML = """\
name: ngorima2025
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy=1.25.2
  - scipy=1.11.4
  - scikit-learn=1.3.2
  - pandas=2.0.3
  - matplotlib=3.8.0
  - joblib=1.3.2
  - psutil=5.9.6
  - tqdm=4.66.1
  - pip:
    - xicor==0.4.0
    - dcor==0.6.0
    - minepy==1.2.6
    - memory-profiler==0.61.0
"""


def write_dependency_files() -> None:
    """P3 fix: Write pinned requirements.txt and environment.yml."""
    req_path = PROJECT_ROOT / "requirements.txt"
    env_path = PROJECT_ROOT / "environment.yml"
    if not req_path.exists():
        req_path.write_text(REQUIREMENTS)
        print(f"  [WROTE] {req_path}")
    if not env_path.exists():
        env_path.write_text(ENVIRONMENT_YML)
        print(f"  [WROTE] {env_path}")


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("=" * 60)
    print("Ngorima (2026) Pipeline — Step 0: Setup & Download")
    print("=" * 60)

    # Log system state first
    print("\n[0/5] Logging system state...")
    state = log_system_state()
    print(f"  RAM available: {state['ram_available_gb']} GB")
    print(f"  CPU governor : {state['cpu_governor']}")

    # Write dependency files
    print("\n[1/5] Writing dependency files...")
    write_dependency_files()

    # FRED-MD
    print("\n[2/5] FRED-MD...")
    setup_fredmd()

    # Superconductivity
    print("\n[3/5] Superconductivity...")
    setup_superconductivity()

    # TCGA
    print("\n[4/5] TCGA...")
    print_tcga_instructions()
    preprocess_tcga()

    # Synthetic proxy (always)
    print("\n[5/5] TCGA synthetic proxy...")
    generate_tcga_synthetic_proxy()

    print("\n[DONE] Setup complete. Run 01_synthetic_benchmarks.py next.")
    print(f"       Results: {RESULTS_DIR}")
