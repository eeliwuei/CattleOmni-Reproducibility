# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""lco-reversal verification (uses EXISTING strict per-sample data). Tests whether 'GBLUP collapses /
marker wins at leave-cluster-out' is REAL (mechanism: GBLUP -> constant mean prediction, var->0,
R2-vs-train-mean->0) or an ARTIFACT (Pearson distortion on tiny clusters)."""
import csv, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "benchmarks/holstein"
SP = "splits/holstein"
PHE = "data/holstein_phenotype_table.csv"
TRAIT = "milk_le_sum_305"; MIN = 10
MODELS = ["GBLUP_full", "PCARidge_PCA80", "Ridge_top50k", "FullSNP_MLP_top50k", "SparseGate_MLP_top20k"]

y_for_id = {}
for d in csv.DictReader(open(PHE)): y_for_id[int(d["ID"])] = float(d[TRAIT])
def train_mean(fold):
    lines = open(f"{SP}/leave_cluster_out/{fold}_train.csv").read().splitlines()[1:]
    v = [y_for_id[int(float(l.split(',')[0]))] for l in lines if l and int(float(l.split(',')[0])) in y_for_id]
    return float(np.mean(v))
def pear(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 1e-9 and np.std(b) > 1e-9 else float("nan")

rows = {}
for d in csv.DictReader(open(f"{B}/holstein_per_sample.csv")):
    if d["family"] != "leave_cluster_out": continue
    rows.setdefault((d["model"], d["fold"]), []).append((float(d["y_true_raw"]), float(d["y_pred_raw"])))
clusters = sorted(set(f for (m, f) in rows))
tmean = {f: train_mean(f) for f in clusters}

summ = {}
for model in MODELS:
    perc = []; pyt = []; pyp = []; ss_res = 0.0; ss_base = 0.0
    for f in clusters:
        rs = rows.get((model, f), [])
        if not rs: continue
        yt = np.array([a for a, b in rs]); yp = np.array([b for a, b in rs])
        perc.append((f, len(rs), pear(yt, yp), float(np.var(yp))))
        if len(rs) >= MIN:
            pyt += list(yt); pyp += list(yp)
            ss_res += float(np.sum((yt - yp) ** 2)); ss_base += float(np.sum((yt - tmean[f]) ** 2))
    big = [t for t in perc if t[1] >= MIN]
    wsum = sum(n for f, n, p, v in big if not np.isnan(p))
    summ[model] = dict(
        pooled=pear(pyt, pyp),
        wt=(sum(n * p for f, n, p, v in big if not np.isnan(p)) / wsum if wsum else float("nan")),
        unwt=float(np.nanmean([p for f, n, p, v in perc])),
        r2mean=(1 - ss_res / ss_base if ss_base > 0 else float("nan")),
        var=float(np.var(pyp)), perc=perc, nbig=len(big))

print(f"lco clusters total={len(clusters)}, n_test>=10 kept={summ['GBLUP_full']['nbig']}")
print(f"{'model':24s} pooled  wt-mean unwt-mean  R2vsMean   var(yhat)")
for m in MODELS:
    s = summ[m]; print(f"{m:24s} {s['pooled']:.3f}  {s['wt']:.3f}   {s['unwt']:.3f}    {s['r2mean']:+.3f}   {s['var']:.1f}")

# per-cluster decomposition GBLUP vs FullSNP_MLP
gb = {f: (n, p, v) for f, n, p, v in summ["GBLUP_full"]["perc"]}
ml = {f: (n, p, v) for f, n, p, v in summ["FullSNP_MLP_top50k"]["perc"]}
big_clusters = [f for f in clusters if gb.get(f, (0,))[0] >= MIN]
gb_wins = sum(1 for f in big_clusters if gb[f][1] > ml[f][1])
ml_wins = sum(1 for f in big_clusters if ml[f][1] > gb[f][1])
print(f"\nper-cluster (n>=10, {len(big_clusters)} clusters): MLP>GBLUP in {ml_wins}, GBLUP>MLP in {gb_wins}")
print(f"{'cluster':12s} n_test  GBLUP_r  MLP_r   GBLUP_var  MLP_var")
for f in big_clusters:
    print(f"{f:12s} {gb[f][0]:5d}  {gb[f][1]:+.3f}  {ml[f][1]:+.3f}   {gb[f][2]:9.1f} {ml[f][2]:9.1f}")

# ---- decision (preliminary, lco-only; budget continuous sweep pending) ----
G = summ["GBLUP_full"]; markers = [summ[m] for m in ["Ridge_top50k", "PCARidge_PCA80", "FullSNP_MLP_top50k"]]
collapse_var = G["var"] < 0.2 * np.median([m["var"] for m in markers])
collapse_r2 = G["r2mean"] < 0.05
marker_skill = np.median([m["r2mean"] for m in markers]) > 0.05
pooled_holds = (np.median([m["pooled"] for m in markers]) - G["pooled"]) > 0.05
whole_cluster = ml_wins >= 0.6 * len(big_clusters)
real = collapse_var and collapse_r2 and marker_skill and pooled_holds and whole_cluster
verdict = "REVERSAL_LIKELY_REAL" if real else "REVERSAL_NOT_CONFIRMED_BY_LCO_ALONE"

L = ["# LCO reversal — verification (lco data; budget continuous sweep pending)\n",
     f"lco clusters n_test>=10: {summ['GBLUP_full']['nbig']} / {len(clusters)}\n",
     "## pooled / weighted / unweighted Pearson + R2-vs-train-mean + var(yhat)",
     "| model | pooled | wt-mean | unwt-mean | R2-vs-mean | var(yhat) |", "|---|---|---|---|---|---|"]
for m in MODELS:
    s = summ[m]; L.append(f"| {m} | {s['pooled']:.3f} | {s['wt']:.3f} | {s['unwt']:.3f} | {s['r2mean']:+.3f} | {s['var']:.1f} |")
L += [f"\n## per-cluster (n>=10): MLP>GBLUP in {ml_wins}/{len(big_clusters)}, GBLUP>MLP in {gb_wins}",
      "\n## decision tests",
      f"- GBLUP var collapse (< 20% of marker median var): {collapse_var} (GBLUP var={G['var']:.1f})",
      f"- GBLUP R2-vs-mean ~0 (<0.05): {collapse_r2} (={G['r2mean']:+.3f})",
      f"- markers retain skill (median R2-vs-mean>0.05): {marker_skill} (={np.median([m['r2mean'] for m in markers]):+.3f})",
      f"- reversal holds in POOLED (markers - GBLUP > 0.05): {pooled_holds}",
      f"- whole-cluster (MLP wins >=60% big clusters): {whole_cluster}",
      f"\n## VERDICT (lco-only, preliminary): **{verdict}**",
      "Final framework decision also requires the matched-N budget continuous sweep (smooth crossover vs endpoint jump)."]
open(f"{B}/LCO_REVERSAL_DECISION.md", "w").write("\n".join(L) + "\n")
print(f"\nVERDICT (lco-only): {verdict}")

# figures
plt.figure(figsize=(8.5, 5))
xb = np.arange(len(big_clusters)); w = 0.4
plt.bar(xb - w/2, [gb[f][1] for f in big_clusters], w, label="GBLUP", color="#4C72B0")
plt.bar(xb + w/2, [ml[f][1] for f in big_clusters], w, label="FullSNP_MLP", color="#C44E52")
plt.xticks(xb, [f"{f}\nn={gb[f][0]}" for f in big_clusters], fontsize=7)
plt.ylabel("Pearson r (within cluster)"); plt.axhline(0, color="k", lw=0.8)
plt.title("fig_percluster_decomp: leave-cluster-out, GBLUP vs MLP per held-out cluster")
plt.legend(); plt.grid(alpha=0.3, axis="y"); plt.tight_layout(); plt.savefig(f"{B}/fig_percluster_decomp.png", dpi=150); plt.close()

plt.figure(figsize=(7.5, 5))
xv = np.arange(len(MODELS))
plt.bar(xv, [summ[m]["var"] for m in MODELS], color=["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3"])
plt.xticks(xv, [m.replace("_full","").replace("_PCA80","").replace("_top50k","").replace("_top20k","") for m in MODELS], fontsize=8)
plt.ylabel("var(y_pred) pooled over lco"); plt.title("fig_gblup_variance_collapse: prediction variance at leave-cluster-out\n(GBLUP→0 = collapse to mean)")
plt.grid(alpha=0.3, axis="y"); plt.tight_layout(); plt.savefig(f"{B}/fig_gblup_variance_collapse.png", dpi=150); plt.close()
print("figures: fig_percluster_decomp.png fig_gblup_variance_collapse.png")
