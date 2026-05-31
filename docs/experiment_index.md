# Experiment index

Each experiment lists purpose, input, script, config, output, and the manuscript section.
Real inputs live under `data/processed/` (not distributed); see `data/README.md`.

## Phase 2 -- Clean benchmark & relationship-aware validation (R2, R3)
- Purpose: measure the apparent flexible-vs-GBLUP advantage under random CV and test whether it
  survives relationship-aware validation (group k-fold, leave-cluster-out, genetic-distance budget).
- Input: Holstein genotype matrix + phenotypes + VanRaden GRM (data/processed/holstein/).
- Script: `src/data/build_holstein_splits.py`, `src/models/run_holstein_benchmark.py`.
- Config: `configs/default.yaml`.
- Output: `results/tables/` (per-fold metrics, per-sample predictions).
- Manuscript: Sec 3.2 (R2), Sec 3.3 (R3); Table 2, Figure 2.

## Phase 3 -- Aggregation/calibration & leakage counterexample (R4, R5)
- Purpose: show pooled vs within-cluster summaries move the comparison; quantify the maximal
  inflation of a single cross-split supervised feature-selection step (clean vs leaky pipeline).
- Script: `src/evaluation/within_cluster_repro.py`, `wc_verify_all5.py`, `lco_verify.py`;
  leaky pipeline in `src/models/run_holstein_benchmark.py`.
- Output: `results/tables/`.
- Manuscript: Sec 3.4 (R4), Sec 3.5 (R5); Tables 3 + Suppl S2-S4, Figures 3-4.

## Phase 4 -- Sample-size confounding & abstention (R6, R7)
- Purpose: matched-N budget on snp3 shows an apparent validation ladder is a sample-size effect;
  evidence-aware abstention (max GRM to the training fold) improves retained-set skill.
- Script: `src/models/run_budget_sweep_N400.py`, `src/evaluation/budget_aggregate.py`,
  `src/evaluation/a7_risk_coverage.py`, `src/visualization/a7_plot.py`.
- Output: `results/tables/`, `results/figures/`.
- Manuscript: Sec 3.6 (R6), Sec 3.7 (R7); Tables 4-5, Figures 5-6.

## A6 audit -- Genome build / coordinate mismatch (claim control)
- Purpose: detect a genome-build/coordinate-mapping mismatch before any position-based biological claim.
- Scientific risk addressed: a build/map mismatch (Btau_4.6.1 vs ARS-UCD1.2) and a mis-mapped marker
  could have produced an unsupported DGAT1-level annotation.
- Why it matters: **the evidence system prevented a potentially incorrect biological claim caused by
  build/map/gene-annotation mismatch.**
- Script: `src/audit/a6_stability.py`.
- Output: `results/a6_audit/`.
- Manuscript: Sec 3.8 + Supplement. See `docs/a6_audit.md`.
