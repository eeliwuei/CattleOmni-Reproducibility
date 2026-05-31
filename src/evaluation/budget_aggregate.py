# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""Aggregate N=150 budget sweep -> per (model,kind,tau): mean Pearson, R2-vs-mean, var(yhat) + bootstrap CI.
Decisive test: does GBLUP fall BELOW markers SMOOTHLY along tau (crossover -> reversal real) or stay
competitive (no crossover -> conservative framework)? Primary metric = R2-vs-mean (robust). matched-N=150."""
import csv, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "benchmarks/holstein"
TAUS = [("t200", 0.200), ("t150", 0.150), ("t100", 0.100), ("t070", 0.070), ("t050", 0.050)]
TORDER = [t[0] for t in TAUS]; TVAL = dict(TAUS)
MODELS = ["GBLUP_full", "PCARidge_PCA80", "Ridge_top50k", "FullSNP_MLP_top50k", "SparseGate_MLP_top20k"]

data = {}  # (model,kind,tau)-> dict(pearson=[],r2mean=[],var=[])
for d in csv.DictReader(open(f"{B}/budget_metrics.csv")):
    if d["pearson"] in ("ERR", "", "nan"): continue
    k = (d["model"], d["kind"], d["tau_name"]); data.setdefault(k, dict(p=[], r=[], v=[]))
    data[k]["p"].append(float(d["pearson"])); data[k]["r"].append(float(d["r2_vs_mean"])); data[k]["v"].append(float(d["var_yhat"]))

def ci(v):
    v = np.asarray(v, float)
    if len(v) < 2: return (float(v.mean()) if len(v) else np.nan, np.nan, np.nan)
    rng = np.random.default_rng(20260529)
    bs = [rng.choice(v, len(v), replace=True).mean() for _ in range(2000)]
    return (float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))

with open(f"{B}/budget_summary.csv", "w") as f:
    f.write("model,kind,tau_name,tau,pearson_mean,pearson_lo,pearson_hi,r2mean_mean,r2mean_lo,r2mean_hi,var_mean,n\n")
    for model in MODELS:
        for kind in ("matched", "natural"):
            for tn in TORDER:
                k = (model, kind, tn)
                if k not in data: continue
                pp = ci(data[k]["p"]); rr = ci(data[k]["r"]); vv = ci(data[k]["v"])
                f.write(f"{model},{kind},{tn},{TVAL[tn]},{pp[0]:.4f},{pp[1]:.4f},{pp[2]:.4f},{rr[0]:.4f},{rr[1]:.4f},{rr[2]:.4f},{vv[0]:.1f},{len(data[k]['p'])}\n")

def series(model, kind, metric):
    out = []
    for tn in TORDER:
        k = (model, kind, tn)
        out.append(ci(data[k][metric]) if k in data else (np.nan, np.nan, np.nan))
    return out

xt = [TVAL[tn] for tn in TORDER]
for kind in ("matched", "natural"):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for mi, metric, ylab, title in [(0, "p", "Pearson r", "Pearson"), (1, "r", "R²-vs-train-mean (skill)", "R²-vs-mean (robust)")]:
        for model in MODELS:
            s = series(model, kind, metric); ys = [a[0] for a in s]; lo = [a[1] for a in s]; hi = [a[2] for a in s]
            ax[mi].plot(xt, ys, marker="o", label=model, lw=1.8); ax[mi].fill_between(xt, lo, hi, alpha=0.13)
        ax[mi].invert_xaxis(); ax[mi].set_xlabel("max train-test relatedness τ (stricter →)"); ax[mi].set_ylabel(ylab)
        ax[mi].set_title(title); ax[mi].grid(alpha=0.3); ax[mi].axhline(0, color="k", lw=0.6)
    ax[0].legend(fontsize=7)
    fig.suptitle(f"Holstein genetic-distance budget sweep — {kind}-N (N=150 fixed, 8 resamples, 95% bootstrap CI)" if kind=="matched" else f"budget sweep — natural (train shrinks with τ)")
    plt.tight_layout(); plt.savefig(f"{B}/fig_budget_crossover_{kind}.png", dpi=150); plt.close()

# verdict: at matched, does GBLUP fall below markers in R2-vs-mean smoothly?
print(f"{'tau':6s}" + "".join(f"{m.split('_')[0][:8]:>10s}" for m in MODELS) + "   <- matched R2-vs-mean")
gblup_below = 0
for tn in TORDER:
    vals = {m: (ci(data[(m,'matched',tn)]["r"])[0] if (m,'matched',tn) in data else np.nan) for m in MODELS}
    marker_med = np.nanmedian([vals[m] for m in ["Ridge_top50k","PCARidge_PCA80","FullSNP_MLP_top50k"]])
    below = vals["GBLUP_full"] < marker_med - 0.03
    gblup_below += below
    print(f"{tn:6s}" + "".join(f"{vals[m]:>10.3f}" for m in MODELS) + f"   GBLUP<markers:{below}")
print(f"\nGBLUP below markers (R2-vs-mean, matched) in {gblup_below}/{len(TORDER)} tau levels")
print("CROSSOVER smooth along tau" if gblup_below >= 3 else "NO smooth crossover -> conservative framework confirmed")
