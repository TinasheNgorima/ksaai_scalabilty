# Ngorima (2026) — Computational Complexity Pipeline

Benchmarking ξₙ, DC, MI, and MIC feature selection methods for computational complexity and scalability across TCGA Pan-Cancer genomics, FRED-MD macroeconomics, and Matasov & Krasavina (2020) superconductivity (`sc_mean.csv`) — with log-log complexity verification, RAM profiling, and parallelisation analysis.

Automated reproduction pipeline for:

> *Computational Complexity of Correlation-Based Feature Selection: A Multi-Domain Scalability Analysis of ξₙ, DC, MI, and MIC*  
> Tinashe Ngorima, Sohar University, 2026

---

## Quick Start

```bash
# 1. Install dependencies
python run_all.py --install

# 2. Run full pipeline
python run_all.py

# 3. View report
cat report/Ngorima2026_Results_Report.md
```

---

## Pipeline Steps

| Step | Script | Output | Paper Section |
|------|--------|--------|---------------|
| 0 | `00_setup_and_download.py` | `data/` | — |
| 1 | `01_synthetic_benchmarks.py` | Tables 2, 3 + Figures 1, 2 | §4 |
| 2 | `02_real_domain_benchmarks.py` | Tables 4, 5, 6 + Figure 5 | §5 |
| 3 | `03_memory_and_parallelisation.py` | Tables 7, 8 + Figures 3, 4 | §6 |
| 4 | `04_compile_results.py` | Full Markdown report | — |

Run individual steps with `--step N`, e.g.:
```bash
python run_all.py --step 1   # Only synthetic benchmarks
python run_all.py --from-step 2  # Skip setup, run from Step 2
```

---

## Datasets

### Automatic Download
- **FRED-MD** — Federal Reserve Bank of St. Louis (auto-downloaded in Step 0)
- **Superconductivity** — Matasov & Krasavina (2020), derived from Hamidieh (2018). File `sc_mean.csv`, publicly available at https://github.com/matasovav/DATA_SC/blob/master/sc_mean.csv. Place at `data/raw/sc_mean.csv` before running Step 0.

### Manual Download Required
- **TCGA Pan-Cancer RNA-seq** (~1 GB):
  1. Visit: https://xenabrowser.net/datapages/?cohort=TCGA%20Pan-Cancer%20(PANCAN)
  2. Download: `EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz`
  3. Save to: `data/raw/tcga_pancan.xena.gz`
  4. Re-run `00_setup_and_download.py`

  If not downloaded, a synthetic proxy with matched dimensions (n=11,056, p=5,000) is used for timing benchmarks.

---

## Dependencies

```bash
pip install numpy scipy scikit-learn pandas matplotlib joblib
pip install xicor dcor minepy  # Feature selection methods
```

**Notes:**
- `minepy` (MIC) may need: `conda install -c conda-forge minepy`
- Fallback implementations exist for `xicor`, `dcor`, `minepy` if not installed
- Python 3.9+ required

---

## Hardware

Paper used: Intel Core i7-12700H (14 cores, 2.3 GHz), 32 GB DDR5 RAM, Ubuntu 22.04 LTS.

Parallelisation experiments (Step 3) auto-detect available cores.  
For paper-quality results, run on hardware with ≥16 GB RAM.

---

## Output Files

```
results/
  complexity_exponents.csv     # Table 2
  crossover_analysis.csv       # Table 3
  table4_tcga.csv              # Table 4
  table5_fredmd.csv            # Table 5
  table6_superconductivity.csv # Table 6
  table7_memory_profile.csv    # Table 7
  table8_parallelisation.csv   # Table 8
  synthetic_timing.csv         # Full timing data (all scenarios)
  real_domain_all.csv          # Combined real-domain results

figures/
  log_log_scaling.png          # Figure 1
  crossover_curves.png         # Figure 2
  memory_scaling.png           # Figure 3
  parallelisation_speedup.png  # Figure 4
  accuracy_efficiency_pareto.png # Figure 5

report/
  Ngorima2025_Results_Report.md  # Full compiled report
```

---

## Estimated Runtimes

| Step | Fast mode | Full (n_reps=30) |
|------|-----------|------------------|
| Step 0 (setup) | 1–2 min | 1–2 min |
| Step 1 (synthetic) | 15–30 min | 3–6 hours |
| Step 2 (real domains) | 20–60 min | 2–4 hours |
| Step 3 (memory/parallel) | 10–20 min | 30–60 min |

DC and MIC at large n are automatically skipped/subsampled to maintain feasibility.

---

## Citation

If using this pipeline, please cite the original paper and data sources:

```
Ngorima, T. (2026). Computational Complexity of Correlation-Based Feature Selection:
A Multi-Domain Scalability Analysis of ξₙ, DC, MI, and MIC. Sohar University.

Matasov, A. & Krasavina, V. (2020). Visualization of superconducting materials.
SN Applied Sciences, 2(8), 1463.

Hamidieh, K. (2018). A data-driven statistical model for predicting the critical
temperature of a superconductor. Computational Materials Science, 154, 346–354.
```

---

## License

See `LICENSE.txt` for full terms.
