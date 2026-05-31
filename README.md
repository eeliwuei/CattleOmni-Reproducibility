# Decomposing evaluation artefacts in cattle genomic prediction

Reproducibility repository for the manuscript *"Decomposing evaluation artefacts in
cattle genomic prediction: baseline tuning, relatedness, aggregation, and leakage."*

## Overview
High-capacity models are increasingly compared with GBLUP under random cross-validation,
but an apparent advantage can reflect several **evaluation artefacts** rather than transferable
marker signal. This work treats apparent model advantage as a *compound* quantity and
decomposes it, on a Holstein SNP panel (n=1092) and a Japanese Black relationship-matrix
dataset (n=9850), into five components measured by separate prespecified contrasts:
(1) baseline tuning, (2) relatedness, (3) aggregation/calibration, (4) leakage, and
(5) sample size. Two further steps are corollaries, not axes: an evidence-aware abstention
analysis and a genome-build/coordinate audit (claim control).

## Repository status
**This repository is currently private and under manuscript review/preparation.**
Sensitive raw genomic and farm-level data are **not** included; server paths, hostnames,
and credentials have been removed from all scripts.

## Paper-to-code map
| Paper section | Experiment | Code | Output |
|---|---|---|---|
| Sec 2 Methods (splits) | GRM + validation-regime construction | `src/data/build_holstein_splits.py` | split definitions |
| Sec 2 Methods (h2) | REML heritability | `src/evaluation/h2_reml.py` | `results/metrics/` |
| Sec 3.2 R2 | Clean random-CV benchmark (baseline tuning) | `src/models/run_holstein_benchmark.py` | `results/tables/` |
| Sec 3.3 R3 | Relationship-aware validation (relatedness) | `src/models/run_holstein_benchmark.py` | `results/tables/` |
| Sec 3.4 R4 | Aggregation / calibration | `src/evaluation/within_cluster_repro.py`, `wc_verify_all5.py`, `lco_verify.py` | `results/tables/` |
| Sec 3.5 R5 | Controlled leakage counterexample | `src/models/run_holstein_benchmark.py` (leaky pipeline) | `results/tables/` |
| Sec 3.6 R6 | snp3 sample-size confounding (matched-N) | `src/models/run_budget_sweep_N400.py`, `src/evaluation/budget_aggregate.py` | `results/tables/` |
| Sec 3.7 R7 | Evidence-aware abstention | `src/evaluation/a7_risk_coverage.py`, `src/visualization/a7_plot.py` | `results/figures/` |
| Sec 3.8 / Suppl | Genome-build & coordinate audit | `src/audit/a6_stability.py` | `results/a6_audit/` |
| Suppl (Bayesian) | BayesB/BRR robustness | `src/models/bglr_*.R`, `src/data/bglr_prep2.py`, `src/evaluation/bayesb_aggregate.py` | Suppl Table S2 |

## Quick start
```
conda env create -f environment.yml
conda activate cattleomni
cp .env.example .env            # set CATTLEOMNI_DATA_ROOT to your processed-data location
python src/data/build_holstein_splits.py
python src/models/run_holstein_benchmark.py
python src/evaluation/holstein_postprocess.py
```
Real data are required to reproduce numbers (see **Data availability**); the scripts are the
exact, sanitised analysis code and document the method end-to-end.

## Data availability
The Holstein genotype/phenotype data and the Japanese Black relationship matrices are **not**
redistributed here; they are available from the corresponding author on reasonable request,
subject to the data providers' agreements. See `data/README.md` for the expected layout and
field descriptions, and `docs/data_availability.md` for access terms. No raw data, server
paths, or credentials are stored in this repository.

## Reproducibility
`docs/experiment_index.md` maps every experiment to its script, config, and output;
`docs/reproducibility_checklist.md` lists seeds, environment, and the table/figure regeneration
order. Fold-level predictions are saved and all tables are re-derived from them.

## Citation
See `CITATION.cff`.
