# Ngorima (2026) — Computational Complexity Pipeline

[![Pipeline Smoke Test](https://github.com/TinasheNgorima/ksaai_scalabilty/actions/workflows/ci.yml/badge.svg)](https://github.com/TinasheNgorima/ksaai_scalabilty/actions/workflows/ci.yml)
[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

> **Add Zenodo DOI after first release deposit.**

Automated reproduction pipeline for:

> Ngorima, T. (2026). *Computational Complexity of Correlation-Based Feature
> Selection: A Multi-Domain Scalability Analysis of ξₙ, DC, MI, and MIC.*
> General Foundation Program, Sohar University, Oman.

---

## Methodological Integrity

| Issue | Status | Fix location |
|-------|--------|--------------|
| Preprocessing leakage (StandardScaler before CV) | ✅ Fixed | `02_real_domain_benchmarks.py` |
| Temporal leakage in FRED-MD (`KFold(shuffle=True)`) | ✅ Fixed | `02_real_domain_benchmarks.py` |
| Circular TCGA target (PC1 of feature matrix) | ✅ Fixed | `00_setup_and_download.py` |
| Jaccard via score perturbation (not bootstrap) | ✅ Fixed | `02_real_domain_benchmarks.py` |
| `n_reps=10` vs paper's stated 30 | ✅ Fixed | `01_synthetic_benchmarks.py` |
| Scorer code triplicated across steps | ✅ Fixed | `src/ngorima2025/scorers.py` |
| No warm-up before timing | ✅ Fixed | `src/ngorima2025/utils.py` |
| No RAM guard for DC | ✅ Fixed | `src/ngorima2025/utils.py` |
| Pearson/Spearman baselines absent | ✅ Fixed | All steps |
| `__file__` failure in Colab | ✅ Fixed | `src/ngorima2025/utils.py` |
| No pinned `requirements.txt` | ✅ Fixed | `requirements.txt` / `environment.yml` |
| No CI/CD | ✅ Fixed | `.github/workflows/ci.yml` |
| No system metadata logging | ✅ Fixed | `results/system_state.json` |
| No checkpoint/resume | ✅ Fixed | `src/ngorima2025/utils.py` |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/TinasheNgorima/ksaai_scalabilty.git
cd ngorima2025-complexity

# 2. Install (pip)
pip install -e ".[full]"          # includes xicor, dcor, minepy

# 3. Install (conda — recommended for minepy)
conda env create -f environment.yml
conda activate ngorima2025

# 4. Download data and run full pipeline
python 00_setup_and_download.py   # auto-downloads FRED-MD + Superconductivity
python run_all.py                 # runs steps 1–4
```

---

---

## Pipeline Steps

| Step | Script | Outputs | Paper Section |
|------|--------|---------|---------------|
| 0 | `00_setup_and_download.py` | `data/`, `requirements.txt` | — |
| 1 | `01_synthetic_benchmarks.py` | Tables 2, 3 + Figures 1, 2 | §4 |
| 2 | `02_real_domain_benchmarks.py` | Tables 4–6 + Figure 5 | §5 |
| 3 | `03_memory_and_parallelisation.py` | Tables 7, 8 + Figures 3, 4 | §6 |
| 4 | `04_compile_results.py` | Full Markdown report | — |

```bash
# Run individual steps
python run_all.py --step 1       # Step 1 only
python run_all.py --from-step 2  # Resume from Step 2
python run_all.py --fast         # Fast mode (n_reps=3, small n)
```

---

## Datasets

### Automatic Download (Step 0)
| Dataset | Source | n | p |
|---------|--------|---|---|
| FRED-MD | St. Louis Fed | ≈780 | 128 |
| Superconductivity | Matasov & Krasavina (2020), `sc_mean.csv` | 15,542 | 81 |

### Manual Download Required
**TCGA Pan-Cancer RNA-seq** (~1 GB):
1. Visit: <https://xenabrowser.net/datapages/?cohort=TCGA%20Pan-Cancer%20(PANCAN)>
2. Download: `EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz`
3. Save to: `data/raw/tcga_pancan.xena.gz`
4. Download tumour purity file from: <https://gdc.cancer.gov/about-data/publications/pancanatlas>
   Save to: `data/raw/tcga_purity.txt`
5. Re-run `00_setup_and_download.py`

> Without TCGA, a synthetic proxy (n=11,056, p=5,000) with matched
> sparsity is used automatically.

---

## Target Variables

| Dataset | Target | Construction |
|---------|--------|-------------|
| Superconductivity | `critical_temp` | Matasov & Krasavina (2020) |
| FRED-MD | INDPRO (log-diff) | McCracken-Ng code 5 |
| TCGA | Tumour purity | ABSOLUTE algorithm (clinical metadata) |
| TCGA (synthetic) | Signal gene mean + noise | Non-circular by construction |

> **Why not PC1?** Using the first principal component of the full
> expression matrix as a regression target is circular: PC1 is a linear
> combination of all genes being ranked, so any gene with a non-negligible
> loading will exhibit inflated correlation with the target. This was
> corrected in the current version.

---

## Cross-Validation Protocol

| Dataset | CV Scheme | Rationale |
|---------|-----------|-----------|
| Superconductivity | `KFold(n_splits=5, shuffle=True)` | i.i.d. observations |
| TCGA | `KFold(n_splits=5, shuffle=True)` | Cross-sectional samples |
| FRED-MD | `TimeSeriesSplit(n_splits=5, gap=12)` | Temporal autocorrelation |

All scalers are fitted **inside** `sklearn.Pipeline` — no leakage to test folds.

---

## Dependencies

```bash
# Minimum (fallbacks active for xicor/dcor/minepy)
pip install numpy>=1.25 scipy>=1.11 scikit-learn>=1.3 pandas>=2.0 \
    matplotlib>=3.8 joblib>=1.3 psutil>=5.9 tqdm>=4.66

# Full (native implementations — required for publication)
pip install xicor>=0.4.0 dcor>=0.6.0 minepy>=1.2.6

# minepy (requires C compiler; use conda if pip fails)
conda install -c conda-forge minepy
```

---

## Hardware Recommendations

| Requirement | Minimum | Paper hardware |
|-------------|---------|---------------|
| RAM | 8 GB | 32 GB DDR5 |
| CPU | 4 cores | i7-12700H (14 cores) |
| OS | Ubuntu 20.04+ | Ubuntu 22.04 LTS |
| Python | 3.9 | 3.11 |

> **CPU governor:** For reproducible timing, set to `performance`:
> ```bash
> sudo cpupower frequency-set -g performance
> ```

DC memory ceiling is **computed dynamically** from available RAM at runtime —
configurations that exceed available memory are skipped with a warning rather
than crashing.

---

## Output Files

```
results/
  system_state.json            # Hardware + software metadata (logged at start)
  checkpoint.json              # Resume state for long runs / Colab sessions
  complexity_exponents.csv     # Table 2
  crossover_analysis.csv       # Table 3
  table4_tcga.csv              # Table 4
  table5_fredmd.csv            # Table 5
  table6_superconductivity.csv # Table 6
  table7_memory_profile.csv    # Table 7
  table8_parallelisation.csv   # Table 8
  synthetic_timing.csv         # Full timing data (median + CI)
  real_domain_all.csv          # Combined real-domain results

figures/
  log_log_scaling.png          # Figure 1 (fallback methods annotated)
  crossover_curves.png         # Figure 2 (includes Pearson/Spearman baselines)
  memory_scaling.png           # Figure 3
  parallelisation_speedup.png  # Figure 4 (includes warm-up)
  accuracy_efficiency_pareto.png # Figure 5 (Pipeline CV, no leakage)

report/
  Ngorima2026_Results_Report.md  # Full compiled report with open-science checklist
```

---

## Tests

```bash
pytest tests/ -v               # All unit tests
pytest tests/ -k "leakage"     # P0 leakage tests only
pytest tests/ -k "ram"         # P2 RAM guard tests only
```

---

## Estimated Runtimes

| Step | Fast (`NGORIMA_FAST=1`) | Full (`n_reps=30`) |
|------|-------------------------|---------------------|
| Step 0 (setup) | 1–2 min | 1–2 min |
| Step 1 (synthetic) | 10–20 min | 3–6 hours |
| Step 2 (real domains) | 15–40 min | 2–4 hours |
| Step 3 (memory/parallel) | 5–15 min | 30–60 min |

---

## Citation

```bibtex
@techreport{ngorima2025complexity,
  author      = {Ngorima, Tinashe},
  title       = {Computational Complexity of Correlation-Based Feature Selection:
                 A Multi-Domain Scalability Analysis of $\xi_n$, DC, MI, and MIC},
  institution = {Sohar University},
  year        = {2026},
  note        = {Code: \url{https://github.com/TinasheNgorima/ksaai_scalabilty}
                 DOI: 10.5281/zenodo.XXXXXXX}
}
```
