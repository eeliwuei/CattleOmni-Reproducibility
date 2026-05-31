# Leakage counterexample — matched 3-arm positive control (Phase 2A)

## Design (all three arms share identical outer folds, k, model list, seeds, scaling,
## training schedule, evaluation code; random-CV, 10 outer folds = 2 reps x 5)
- **clean-variance (A)**: train-fold genotype-VARIANCE top-k (unsupervised, fold-internal). = primary clean pipeline.
- **clean-supervised (B)**: train-fold |corr(X, y_train)| top-k (supervised, fold-internal).
- **leaky-supervised (C)**: global |corr(X, y_all)| top-k fitted on ALL animals (incl. held-out) BEFORE split.
- Selector SCORE: univariate |Pearson correlation| between each marker and the phenotype.
- Phenotype standardisation: fold-internal (train mean/sd) for A,B; global for C (definitional).
- p-value pairing unit: outer fold (n=10). Bootstrap 95% CI: 10,000 resamples over folds, seed 20260529.

## Contrasts
- delta_pure_leakage = C - B : selector type held fixed (correlation), ONLY cross-split-vs-fold-internal differs -> ISOLATED cross-split leakage.
- delta_supervised_selector = B - A : supervised-vs-variance selector effect, both clean/fold-internal.
- delta_total_positive_control = C - A : historical clean(variance) vs leaky total.

## Result (10-fold mean Pearson; from benchmarks/holstein/leak_matched/)
| model | clean-var | clean-sup | leaky-sup | d_selector | d_pure_leak | d_total | pure-leak 95% CI | p (paired, fold) |
|---|---|---|---|---|---|---|---|---|
| FullSNP-MLP    | 0.469 | 0.487 | 0.668 | +0.019 | +0.181 | +0.199 | [+0.165,+0.196] | 6.3e-105 |
| Ridge-top50k   | 0.446 | 0.449 | 0.649 | +0.003 | +0.199 | +0.203 | [+0.189,+0.210] | 3.0e-263 |
| Ridge-top20k   | 0.349 | 0.438 | 0.725 | +0.089 | +0.288 | +0.376 | [+0.271,+0.304] | 1.6e-242 |
| SparseGate-MLP | 0.385 | 0.481 | 0.740 | +0.095 | +0.259 | +0.354 | [+0.241,+0.278] | 3.3e-151 |
| PCA-Ridge-80 (unsupervised) | 0.400 | — | 0.343 | — | — | -0.058 | immune (features never see phenotype) |

## Conclusion
Pure cross-split leakage (C-B, selector fixed) is the dominant component (+0.18 to +0.29);
the supervised-selector effect (B-A) is small (+0.003 to +0.095). The +0.19 to +0.37 historical
inflation is therefore attributable to cross-split leakage, not to using a supervised selector.
clean-variance reproduces the locked clean benchmark (e.g. FullSNP-MLP 0.469 vs locked 0.465).

## Provenance
- script: scripts/leak_matched.py (H100), stats: scripts/leak_matched_stats.py
- raw: benchmarks/holstein/leak_matched/{leak_matched_perfold.csv, leak_matched_summary.json, leak_matched_stats.json}
- reuses run_holstein_benchmark.py model/eval code (load_pheno, mlp_fit_predict, pick_lam, dual_solve).
