# CattleOmni reproducibility repo -- sanitised analysis script (paths repo-relative; no server paths/credentials/raw data).
#!/usr/bin/env python3
"""Matched leakage positive-control: isolate CROSS-SPLIT supervised leakage by holding the
selector type fixed (univariate |correlation| with y) and varying ONLY whether the ranking is
fitted inside the training fold or globally before the split.

Three arms, identical outer folds / k / model / scaling / eval:
  clean_var  : train-fold VARIANCE top-k        (unsupervised, fold-internal)  [historical clean]
  clean_sup  : train-fold |corr(X,y)| top-k     (supervised,   fold-internal)  [matched clean]
  leaky_sup  : global    |corr(X,y)| top-k      (supervised,   cross-split)    [leaky]
clean_sup vs leaky_sup differ in EXACTLY one respect: where the supervised ranking is fitted.

Reuses the locked benchmark pipeline (run_holstein_benchmark). Random CV folds only.
Outputs per-fold Pearson per arm per model + paired summary. NOTHING fabricated.
"""
import os, sys, glob, csv, time
import numpy as np
sys.path.insert(0, "src")
import importlib.util
spec = importlib.util.spec_from_file_location("H", "src/run_holstein_benchmark.py")
H = importlib.util.module_from_spec(spec)
# prevent its __main__ from running
_argv = sys.argv; sys.argv = ["H"]
spec.loader.exec_module(H)
sys.argv = _argv

SP = "splits/holstein/random"
OUT = "results/leak_matched"
os.makedirs(OUT, exist_ok=True)
TOP50K, TOP20K = H.TOP50K, H.TOP20K
MODELS = ["FullSNP_MLP_top50k", "Ridge_top50k", "Ridge_top20k", "SparseGate_MLP_top20k", "PCARidge_PCA80"]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# ---- data (identical loading to the benchmark) ----
M = np.load(H.GENO).astype(np.float64)
row_for_id, ids = H.load_alignment()
n = len(ids)
y_all = H.load_pheno(row_for_id, n)   # exact benchmark phenotype vector (trait via H.TRAIT)
TRAIT = H.TRAIT
log(f"M{M.shape} n={n} trait={TRAIT} y_nan={np.isnan(y_all).sum()}")

def corr_topk(X, y, k):
    yc = y - y.mean(); Xc = X - X.mean(0)
    num = Xc.T @ yc
    den = np.sqrt((Xc**2).sum(0)) * (np.sqrt((yc**2).sum())+1e-12) + 1e-12
    return np.argpartition(-np.abs(num/den), k)[:k]
def var_topk(X, k):
    return np.argpartition(-X.var(0), k)[:k]

def predict(name, idx_tr_arm, tr, te, ytr):
    """fit/predict one model given a marker index set (idx) chosen per arm."""
    k = TOP50K if "50k" in name else (TOP20K if "20k" in name else None)
    if name == "PCARidge_PCA80":
        # PCA arms: clean=train PCA; leaky=global PCA (handled by caller via idx flag)
        return None  # handled separately
    idx = idx_tr_arm
    mu = M[np.ix_(tr, idx)].mean(0); sd = M[np.ix_(tr, idx)].std(0) + 1e-8
    Xtr = (M[np.ix_(tr, idx)] - mu) / sd
    Xte = (M[np.ix_(te, idx)] - mu) / sd
    if name.startswith("Ridge"):
        K = Xtr @ Xtr.T; lam = H.pick_lam(K, ytr); return H.dual_solve(K, ytr, Xte @ Xtr.T, lam)
    if name == "FullSNP_MLP_top50k":
        return H.mlp_fit_predict(Xtr, ytr, Xte, gate=False)
    if name == "SparseGate_MLP_top20k":
        return H.mlp_fit_predict(Xtr, ytr, Xte, gate=True, l1=1e-3)

def pear(a, b): return float(np.corrcoef(a, b)[0, 1])

# precompute GLOBAL supervised indices once (leaky arm)
ystd_all = (y_all - y_all.mean()) / (y_all.std() + 1e-9)
g_idx50 = corr_topk(M, ystd_all, TOP50K)
g_idx20 = corr_topk(M, ystd_all, TOP20K)
log("global supervised idx precomputed (leaky arm)")

folds = sorted(glob.glob(f"{SP}/*_train.csv"))
NREP = int(os.environ.get("NFOLDS", "10"))   # default first 10 random folds (2 reps x 5)
folds = folds[:NREP]
log(f"{len(folds)} random folds")

rd = lambda p: [int(float(l.split(",")[0])) for l in open(p).read().splitlines()[1:] if l.strip()]
rows = []
for fi, trp in enumerate(folds):
    tep = trp.replace("_train.csv", "_test.csv")
    tr = np.array(sorted(row_for_id[i] for i in rd(trp) if i in row_for_id))
    te = np.array(sorted(row_for_id[i] for i in rd(tep) if i in row_for_id))
    ym, ys = y_all[tr].mean(), y_all[tr].std()
    ytr = (y_all[tr]-ym)/ys; yte = (y_all[te]-ym)/ys
    for name in MODELS:
        if name == "PCARidge_PCA80":
            continue  # unsupervised; immune; report separately if needed
        k = TOP50K if "50k" in name else TOP20K
        # arm indices
        idx_var = var_topk(M[tr], k)
        idx_csup = corr_topk(M[tr], ytr, k)          # supervised, fold-internal
        idx_leak = g_idx50 if k == TOP50K else g_idx20  # supervised, global
        for arm, idx in [("clean_var", idx_var), ("clean_sup", idx_csup), ("leaky_sup", idx_leak)]:
            p = pear(yte, predict(name, idx, tr, te, ytr))
            rows.append((name, arm, fi, p))
    log(f"fold {fi} done")

with open(f"{OUT}/leak_matched_perfold.csv", "w") as f:
    f.write("model,arm,fold,pearson\n")
    for r in rows: f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.5f}\n")

# summary
print("\n=== MATCHED LEAKAGE (mean Pearson over folds) ===")
print(f"{'model':24s} {'clean_var':>10s} {'clean_sup':>10s} {'leaky_sup':>10s}  {'leak(sup)':>10s} {'sel(sup-var)':>12s}")
agg = {}
for name in MODELS:
    if name == "PCARidge_PCA80": continue
    d = {}
    for arm in ("clean_var","clean_sup","leaky_sup"):
        vs = [p for (m,a,f,p) in rows if m==name and a==arm]
        d[arm] = np.mean(vs) if vs else float("nan")
    leak_pure = d["leaky_sup"] - d["clean_sup"]      # PURE cross-split leakage (selector held fixed)
    sel_effect = d["clean_sup"] - d["clean_var"]     # selector-type effect (sup vs var), both clean
    agg[name] = dict(d, leak_pure=leak_pure, sel_effect=sel_effect)
    print(f"{name:24s} {d['clean_var']:10.4f} {d['clean_sup']:10.4f} {d['leaky_sup']:10.4f}  {leak_pure:+10.4f} {sel_effect:+12.4f}")
import json
json.dump(agg, open(f"{OUT}/leak_matched_summary.json","w"), indent=2)
print(f"\nsaved {OUT}/leak_matched_perfold.csv + summary.json")
