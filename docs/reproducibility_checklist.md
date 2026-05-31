# Reproducibility checklist
- Fixed global seed: 20260529 (split construction and model training).
- Environment: Python 3.10 (NumPy, SciPy, scikit-learn, PyTorch, Matplotlib); R 4.1 + BGLR for
  the Bayesian-ridge robustness check.
- All preprocessing (allele frequencies, feature selection, PCA bases, standardisation, penalties)
  is fitted inside the training fold only.
- Two clustering definitions: a 10-cluster leave-cluster-out map (within-cluster analyses, Suppl
  Tables S2-S4) and a 40-cluster map (group-CV fold construction only) -- not interchangeable.
- Tables/figures are regenerated from saved fold-level predictions via `src/evaluation/` and
  `src/visualization/`.
