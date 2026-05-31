# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""A7 risk-coverage figure: accuracy on retained animals vs coverage, evidence-ranked abstention (solid) vs
random abstention (dashed). Demonstrates evidence-aware selective prediction (Trait Potential)."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = "benchmarks/holstein"
D = {}
for d in csv.DictReader(open(f"{OUT}/A7_RISK_COVERAGE.csv")):
    m = d["model"]; D.setdefault(m, {"cov": [], "r2": [], "r2r": [], "pe": [], "per": []})
    D[m]["cov"].append(float(d["coverage"]) * 100); D[m]["r2"].append(float(d["r2_vs_mean"]))
    D[m]["r2r"].append(float(d["r2_random"])); D[m]["pe"].append(float(d["pearson"])); D[m]["per"].append(float(d["pearson_random"]))
nice = {"GBLUP_full": "GBLUP", "FullSNP_MLP_top50k": "FullSNP-MLP"}
col = {"GBLUP_full": "#1f77b4", "FullSNP_MLP_top50k": "#d62728"}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
for m in D:
    c = col.get(m, "#333"); lab = nice.get(m, m)
    ax[0].plot(D[m]["cov"], D[m]["r2"], "-o", color=c, label=f"{lab} (evidence)")
    ax[0].plot(D[m]["cov"], D[m]["r2r"], "--", color=c, alpha=0.55, label=f"{lab} (random abstain)")
    ax[1].plot(D[m]["cov"], D[m]["pe"], "-o", color=c, label=f"{lab} (evidence)")
    ax[1].plot(D[m]["cov"], D[m]["per"], "--", color=c, alpha=0.55, label=f"{lab} (random abstain)")
for a, t, yl in [(ax[0], "Skill on retained (R$^2$-vs-mean)", "R$^2$-vs-mean"), (ax[1], "Correlation on retained (Pearson)", "Pearson r")]:
    a.set_xlabel("Coverage (% animals predicted, abstain rest)"); a.set_ylabel(yl); a.set_title(t)
    a.invert_xaxis(); a.grid(alpha=0.3); a.legend(fontsize=7.5, loc="best")
fig.suptitle("A7 Trait Potential: evidence-aware selective prediction (Holstein, random CV)\n"
             "Evidence = max genomic relationship to reference; abstaining on low-evidence animals raises skill on the rest",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUT}/fig_A7_risk_coverage.png", dpi=140)
print("saved fig_A7_risk_coverage.png")
