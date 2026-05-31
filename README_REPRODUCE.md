# Reproduction guide

**Paper:** *Decomposing evaluation artefacts in cattle genomic prediction.*

## 1. What this repository contains
Sanitised analysis code, figure/table generation scripts, **derived** result tables (no raw
genotype/phenotype), the artefact-decomposition audit files, and reproduction documentation. Every
number in the manuscript is traceable through `revision_audit/claim_to_evidence_table.csv` (and the
per-analysis notes) to a result file and the script that produced it.

## 2. Raw data NOT included (and why)
- **Holstein** raw genotype (1092 × 164,312 dosages) and phenotype records are **not redistributed**
  because of data-use restrictions; available from the corresponding author on reasonable request,
  subject to the data providers' agreements.
- **Japanese Black (snp3)** additive/dominance relationship matrices are subject to their original
  access conditions; no individual genotypes are distributed.
- No `.npy`/`.vcf`/`.bed`/`.bim`/`.fam` or per-animal source files are committed (see `.gitignore`).
- **Holstein phenotype provenance:** phenotypes were available only as **pre-adjusted records**; the
  fixed-effect correction model is **not recoverable** from the supplied files. They are treated as
  the dataset-defined response; no fixed effects are re-estimated in this pipeline (no fold-level
  leakage), and the provider-side correction model is an acknowledged provenance limitation.

## 3. Environment
Python 3.10 (NumPy, SciPy, scikit-learn, PyTorch, Matplotlib); R 4.1 + BGLR for the Bayesian-ridge
robustness check. See `environment.yml` / `requirements.txt`. Global fixed seed **20260529**.

## 4. Reproduce tables/figures from the provided derived outputs
The committed CSV/JSON under `results/` already contain the locked numbers; the table/figure
scripts read them directly (no raw data needed):
- `results/leakage_3arm/` → matched three-arm leakage positive control (Table 4, Figure 4)
- `results/grm_sensitivity/` → GRM allele-frequency sensitivity (Suppl. Table)
- `results/clustering_distance_sensitivity/` → clustering-distance sensitivity (Suppl. Table)

## 5. Rerun analyses (requires restricted raw inputs placed under `data/processed/`, see `data/README.md`)
| Analysis | Script | Output |
|---|---|---|
| Main benchmark (R2/R3) | `src/models/run_holstein_benchmark.py` | `results/tables/` |
| **Matched 3-arm leakage (R5)** | `src/models/leak_matched.py` → `src/evaluation/leak_matched_stats.py` | `results/leakage_3arm/` |
| **GRM fold-internal sensitivity** | `src/audit/grm_foldint_sens.py` | `results/grm_sensitivity/` |
| **Clustering-distance sensitivity** | `src/audit/clust_dist_sens.py` | `results/clustering_distance_sensitivity/` |
| Within-cluster / aggregation (R4) | `src/evaluation/within_cluster_repro.py`, `wc_verify_all5.py` | `results/tables/` |
| Heritability (REML) | `src/evaluation/h2_reml.py` | `results/metrics/` |
| Build/coordinate audit (R8) | `src/audit/a6_stability.py` | `results/a6_audit/` |

## 6. Matched 3-arm leakage — how to read it
Three arms share identical outer folds, models, *k*, scaling, training schedule and evaluation:
`clean_var` (train-fold variance top-k, unsupervised), `clean_sup` (train-fold |corr(X,y)| top-k,
supervised but fold-internal), `leaky_sup` (global |corr(X,y)| top-k, cross-split). **Pure
cross-split leakage = leaky_sup − clean_sup** (selector type held fixed). See
`revision_audit/leakage_counterexample_notes.md`.

## 7. Expected key outputs (locked)
- Random-CV ΔPearson (MLP − tuned GBLUP) = +0.0296; relationship-aware +0.0134 / +0.0143 / +0.0039 (n.s.).
- Pure cross-split leakage +0.18 to +0.29; total positive-control inflation +0.20 to +0.38.
- GRM global-vs-fold-internal random-CV GBLUP difference ≈ −0.001 (conclusions unchanged).
- Clustering-distance within-cluster Δ invariant across three distances.

## 8. Statistical pairing
See `revision_audit/statistical_inference_table.md` (pairing unit per analysis; effect size +
bootstrap 95% CI are primary; p reported as p<0.001 in main text, exact in audit).

## 9. Contact
Corresponding authors: Xiangzi Li (lxz@ybu.edu.cn), Changguo Yan (ycg@ybu.edu.cn).
