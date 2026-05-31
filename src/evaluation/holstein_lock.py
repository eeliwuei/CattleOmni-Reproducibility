# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""Lock-down A4 Holstein results (paper Step 1-3):
- re-derive REAL y_true per sample (predictions stored 0 placeholder) from train-fold mean/std
- unify leave-cluster-out metric = POOLED prediction Pearson (exclude clusters n_test<10)
- build enriched per-sample file; regenerate paper figures Fig2 (stringency), Fig3 (Delta MLP-GBLUP), Fig4 (leak counterexample)
Pure numpy + matplotlib (Agg). Runs on H100.
"""
import os, csv, json, math
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

B   = "benchmarks/holstein"
SP  = "splits/holstein"
PHE = "data/holstein_phenotype_table.csv"
MET = f"{B}/fold_metrics_full.csv"
PRED= f"{B}/fold_predictions_full.csv"
TRAIT = "milk_le_sum_305"
MIN_NTEST = 10
MODELS = ["GBLUP_full","PCARidge_PCA80","Ridge_top50k","Ridge_top20k","FullSNP_MLP_top50k","SparseGate_MLP_top20k"]
LADDER = ["random","group10","group5","leave_cluster_out"]

def log(m): print(m, flush=True)

# ---- phenotype: sample_id -> raw milk_le_sum_305 ----
y_for_id = {}
with open(PHE, newline="") as f:
    for d in csv.DictReader(f):
        y_for_id[int(d["ID"])] = float(d[TRAIT])

def train_csv(family, fold):
    if family == "random": return f"{SP}/random/{fold}_train.csv"
    if family == "group5": return f"{SP}/relationship_group/group_5fold/{fold}_train.csv"
    if family == "group10": return f"{SP}/relationship_group/group_10fold/{fold}_train.csv"
    if family == "leave_cluster_out": return f"{SP}/leave_cluster_out/{fold}_train.csv"
    if family == "budget_natural": return f"{SP}/relatedness_budget/natural/{fold.replace('natural_','')}/train.csv"
    if family == "budget_matched_size": return f"{SP}/relatedness_budget/matched_size/{fold.replace('matched_size_','')}/train.csv"
    return None

def read_ids(path):
    out = []
    with open(path, newline="") as f:
        next(f, None)
        for ln in f:
            ln = ln.strip()
            if ln: out.append(int(float(ln.split(",")[0])))
    return out

# cache train mean/std per (family,fold)
_stat = {}
def fold_stat(family, fold):
    k = (family, fold)
    if k not in _stat:
        p = train_csv(family, fold)
        ids = read_ids(p) if p and os.path.exists(p) else []
        ys = np.array([y_for_id[i] for i in ids if i in y_for_id], float)
        _stat[k] = (ys.mean(), ys.std()) if len(ys) else (0.0, 1.0)
    return _stat[k]

# ---- load predictions (strict only for per-sample; y_true_std stored as 0 -> re-derive) ----
rows = []
with open(PRED, newline="") as f:
    for d in csv.DictReader(line.replace("\x00","") for line in f):
        if d.get("mode") != "strict": continue
        sid = int(float(d["sample_id"]))
        if sid not in y_for_id: continue
        mu, sd = fold_stat(d["family"], d["fold"])
        yraw = y_for_id[sid]
        rows.append(dict(model=d["model"], family=d["family"], fold=d["fold"], sample_id=sid,
                         y_true_raw=yraw, y_pred_raw=float(d["y_pred"]) * sd + mu,
                         y_true_std=(yraw - mu) / sd, y_pred_std=float(d["y_pred"])))
log(f"loaded {len(rows)} strict per-sample predictions; y_true re-derived")

# enriched per-sample file
with open(f"{B}/holstein_per_sample.csv", "w") as f:
    f.write("model,family,fold,sample_id,y_true_raw,y_pred_raw,y_true_std,y_pred_std\n")
    for r in rows:
        f.write(f"{r['model']},{r['family']},{r['fold']},{r['sample_id']},{r['y_true_raw']:.4f},{r['y_pred_raw']:.4f},{r['y_true_std']:.5f},{r['y_pred_std']:.5f}\n")

def pear(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9: return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

# ---- fold-level Pearson per (model,family,fold) from re-derived per-sample (sanity vs fold_metrics) ----
byff = {}
for r in rows:
    byff.setdefault((r["model"], r["family"], r["fold"]), []).append(r)

# sanity check vs fold_metrics
fm = {}
with open(MET, newline="") as f:
    for d in csv.DictReader(line.replace("\x00","") for line in f):
        if d.get("mode")=="strict" and d.get("pearson") not in ("ERR","","nan",None):
            fm[(d["model"], d["family"], d["fold"])] = float(d["pearson"])
disc = []
for k, rs in byff.items():
    p_rederived = pear([x["y_true_std"] for x in rs], [x["y_pred_std"] for x in rs])
    if k in fm and not math.isnan(p_rederived):
        disc.append(abs(p_rederived - fm[k]))
log(f"SANITY re-derived vs fold_metrics Pearson: max abs diff = {max(disc):.4f}, mean = {np.mean(disc):.5f} (should be ~0)")

# ---- per-family metric per model ----
def family_pearson(model, family):
    """random/group: mean of per-fold Pearson (balanced folds). lco: POOLED across samples (exclude n_test<10 clusters)."""
    folds = {fold: rs for (m, fam, fold), rs in byff.items() if m == model and fam == family}
    if not folds: return (float("nan"), float("nan"), 0)
    if family == "leave_cluster_out":
        pooled = []
        nkept = 0
        for fold, rs in folds.items():
            if len(rs) >= MIN_NTEST:
                pooled += rs; nkept += 1
        p = pear([x["y_true_raw"] for x in pooled], [x["y_pred_raw"] for x in pooled])
        return (p, float("nan"), nkept)  # pooled, no std, n_clusters_kept
    pf = [pear([x["y_true_std"] for x in rs], [x["y_pred_std"] for x in rs]) for rs in folds.values()]
    pf = [x for x in pf if not math.isnan(x)]
    return (float(np.mean(pf)), float(np.std(pf)), len(pf))

summary = {}
for model in MODELS:
    for fam in LADDER:
        summary[(model, fam)] = family_pearson(model, fam)

# write unified summary
with open(f"{B}/holstein_summary_v2.csv", "w") as f:
    f.write("model,family,metric_type,pearson,std,n\n")
    for model in MODELS:
        for fam in LADDER:
            mt = "pooled_lco_nt>=10" if fam == "leave_cluster_out" else "fold_mean"
            p, s, n = summary[(model, fam)]
            f.write(f"{model},{fam},{mt},{p:.4f},{s if not math.isnan(s) else ''},{n}\n")

# ---- leak (random) from fold_metrics historical vs strict ----
def mode_random_mean(mode, model):
    vals = []
    with open(MET, newline="") as f:
        for d in csv.DictReader(line.replace("\x00","") for line in f):
            if d.get("mode")==mode and d.get("model")==model and d.get("family")=="random" and d.get("pearson") not in ("ERR","","nan",None):
                vals.append(float(d["pearson"]))
    return float(np.mean(vals)) if vals else float("nan")

# ================= FIGURES =================
xlab = ["random\n(rel~0.9)", "group10", "group5", "leave-cluster-out\n(pooled, nt>=10)"]
x = np.arange(len(LADDER))

# Fig2: stringency ladder
plt.figure(figsize=(8.5,5.2))
for model in MODELS:
    ys = [summary[(model,fam)][0] for fam in LADDER]
    es = [summary[(model,fam)][1] if not math.isnan(summary[(model,fam)][1]) else 0 for fam in LADDER]
    plt.errorbar(x, ys, yerr=es, marker="o", capsize=3, label=model, lw=1.8)
plt.xticks(x, xlab); plt.ylabel("Pearson r"); plt.ylim(0,0.55)
plt.title("Fig2  Holstein milk: prediction vs validation stringency (STRICT, no leakage)")
plt.legend(fontsize=8, ncol=2); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f"{B}/fig2_stringency_v2.png", dpi=150); plt.close()

# Fig3: Delta = MLP - GBLUP across stringency
plt.figure(figsize=(8.5,5.2))
dys = [summary[("FullSNP_MLP_top50k",fam)][0] - summary[("GBLUP_full",fam)][0] for fam in LADDER]
plt.axhline(0, color="k", lw=1)
plt.plot(x, dys, marker="o", color="crimson", lw=2.5, ms=9)
for xi, dv in zip(x, dys):
    plt.annotate(f"{dv:+.3f}", (xi, dv), textcoords="offset points", xytext=(0,10), ha="center", fontsize=10)
plt.xticks(x, xlab); plt.ylabel("ΔPearson  (FullSNP_MLP − GBLUP)")
plt.title("Fig3  Deep-model edge over GBLUP collapses under relatedness-aware validation")
plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(f"{B}/fig3_delta_collapse.png", dpi=150); plt.close()

# Fig4: leak counterexample (random CV, clean vs leaky)
lm = ["FullSNP_MLP_top50k","Ridge_top50k","Ridge_top20k","SparseGate_MLP_top20k","PCARidge_PCA80"]
clean = [mode_random_mean("strict", m) for m in lm]
leaky = [mode_random_mean("historical", m) for m in lm]
xb = np.arange(len(lm)); w = 0.38
plt.figure(figsize=(9,5.2))
plt.bar(xb-w/2, clean, w, label="clean (train-fold variance selection)", color="#4C72B0")
plt.bar(xb+w/2, leaky, w, label="deliberately leaky (global supervised selection)", color="#C44E52")
for xi, c, l in zip(xb, clean, leaky):
    plt.annotate(f"+{l-c:.2f}", (xi, max(c,l)+0.01), ha="center", fontsize=9)
plt.xticks(xb, [m.replace("_top","\ntop").replace("_PCA80","\nPCA80").replace("_full","") for m in lm], fontsize=8)
plt.ylabel("Pearson r (random CV)"); plt.ylim(0,0.8)
plt.title("Fig4  Leaky supervised feature selection inflates apparent performance (PCA does not)")
plt.legend(fontsize=9); plt.grid(alpha=0.3, axis="y"); plt.tight_layout()
plt.savefig(f"{B}/fig4_leak_counterexample.png", dpi=150); plt.close()

# ---- report ----
L = ["# A4 Holstein — locked results (paper Step 1-3)\n",
     f"lco metric = POOLED prediction Pearson across all held-out samples; clusters with n_test<{MIN_NTEST} EXCLUDED. random/group = mean per-fold Pearson (balanced folds).\n",
     "\n## Pearson by stringency (STRICT)\n", "| model | random | group10 | group5 | lco(pooled) |", "|---|---|---|---|---|"]
for model in MODELS:
    L.append("| " + model + " | " + " | ".join(f"{summary[(model,fam)][0]:.3f}" for fam in LADDER) + " |")
L += ["\n## ΔPearson (MLP − GBLUP)\n"] + [f"- {fam}: {summary[('FullSNP_MLP_top50k',fam)][0]-summary[('GBLUP_full',fam)][0]:+.3f}" for fam in LADDER]
L += ["\n## Leak counterexample (random CV)\n", "| model | clean | leaky | inflation |", "|---|---|---|---|"]
for m in lm:
    c=mode_random_mean('strict',m); l=mode_random_mean('historical',m); L.append(f"| {m} | {c:.3f} | {l:.3f} | {l-c:+.3f} |")
open(f"{B}/HOLSTEIN_LOCKED_REPORT.md","w").write("\n".join(L)+"\n")

log("DONE. Figures: fig2_stringency_v2.png fig3_delta_collapse.png fig4_leak_counterexample.png")
for model in ["FullSNP_MLP_top50k","GBLUP_full"]:
    log(f"  {model}: " + ", ".join(f"{fam}={summary[(model,fam)][0]:.3f}" for fam in LADDER))
log("  Delta(MLP-GBLUP): " + ", ".join(f"{fam}={summary[('FullSNP_MLP_top50k',fam)][0]-summary[('GBLUP_full',fam)][0]:+.3f}" for fam in LADDER))
