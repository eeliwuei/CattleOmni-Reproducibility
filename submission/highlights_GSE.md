# Highlights

- Apparent neural-vs-GBLUP advantage in cattle genomic prediction is decomposed into five
  evaluation artefacts: baseline tuning, relatedness, aggregation/calibration, leakage, sample size.
- A properly tuned GBLUP narrows the random-CV edge to +0.0296 Pearson, and the edge is not stable
  under relationship-aware validation (group k-fold, leave-cluster-out, genetic-distance budget).
- A matched three-arm positive control isolates pure cross-split supervised leakage of +0.18 to
  +0.29 Pearson—larger than the clean model-class difference—separately from selector-type effects.
- An apparent validation ladder on an independent Japanese Black dataset is a sample-size artefact
  that vanishes at matched training size, not a relatedness effect.
- The framework is a reproducible, claim-controlled benchmark: every reported number maps to a
  deposited result and script, with no gene/locus/pathway claims.

# Significance summary
Benchmarks that compare flexible models with GBLUP under random cross-validation can mistake
evaluation artefacts for genuine model advantage. By decomposing and separately measuring baseline
tuning, relatedness, aggregation/calibration, leakage, and sample size on cattle data—and by
isolating pure cross-split leakage with a matched positive control—this study provides a
trustworthy, transferable template for evaluating genomic-prediction models before any claim of
model superiority is made.
