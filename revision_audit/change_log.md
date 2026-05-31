# Change log (GSE revision)
- Matched three-arm leakage positive control added (clean-variance / clean-supervised / leaky-supervised); isolates pure cross-split leakage from selector-type effect. Table 4 + Figure 4 replaced.
- GRM allele-frequency sensitivity (global vs fold-internal) added; conclusions unchanged.
- Clustering-distance sensitivity (max(G)-G / Euclidean / 1-normG) added; within-cluster Delta invariant.
- Execution-level hyperparameter table + validation-regime/pairing-unit table added.
- Abstract leakage wording corrected: pure leakage (+0.18..+0.29) vs total inflation (+0.20..+0.38) separated.
- Main-text p-values reported as p<0.001 (exact in audit); effect size + bootstrap CI primary.
- Title shortened; figure arrows removed; Table 2 width fixed (tabularx).
