# Statistical inference — pairing units (no animal-level pseudo-replication)
| Analysis | Test | Pairing unit | n units | CI / bootstrap |
|---|---|---|---|---|
| Random-CV MLP vs tuned GBLUP (R2) | paired t / Wilcoxon | outer fold | 50 | Fisher-z / fold bootstrap |
| group10 / group5 / LCO Δ (R3) | paired t / Wilcoxon | group fold (10/5) / held-out cluster (10) | 10/5/10 | fold/cluster bootstrap |
| Within-cluster Δ (R4) | paired, sample-weighted | (fold × cluster) residual pool | — | fold bootstrap |
| Leakage pure/total (R5) | paired t / Wilcoxon | outer fold | 10 | 10,000× fold bootstrap (seed 20260529) |
| snp3 matched-N (R6) | paired | matched split repeat | per regime | matched-repeat |
| Abstention (R7) | retained-set vs random-abstention band | coverage grid; random repeats | — | random-abstention repeats |
Effect size + bootstrap 95% CI are primary; p-values reported as p<0.001 in main text, exact in audit.
