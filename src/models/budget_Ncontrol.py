# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""Decouple deep-model R2-vs-mean decline from training-set SIZE (load-bearing gap before headline).
A. random anchors: random train (NO tau cap) at N=150 and N=400, 8 resamples -> is N the deep 'floor'?
B. N=400 budget ladder (tau where natural>=400): does deep decline with tau hold at larger N? (relatedness robust to N)
C. PCA centroid drift of low-tau training sets (selection-effect sanity).
Reuses run_holstein_benchmark models. Resumable. R2-vs-mean primary."""
import os, sys, csv, json, time
sys.path.insert(0, "scripts")
import numpy as np
import run_holstein_benchmark as H

OUT = os.environ.get("HOUT", "benchmarks/holstein")  # per-trait output dir via env
R = 8; N_TEST = 109
MODELS = ["GBLUP_full", "PCARidge_PCA80", "Ridge_top50k", "FullSNP_MLP_top50k", "SparseGate_MLP_top20k"]
N400_TAUS = [("t200", 0.200), ("t150", 0.150), ("t100", 0.100)]  # natural>=574 -> matched-400 feasible

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

M = np.load(H.GENO).astype(np.float32)
row_for_id, ids_by_row = H.load_alignment(); n = len(ids_by_row)
y = H.load_pheno(row_for_id, n)
log("GRM + PCA ...")
G = H.vanraden_grm(M)
w, V = np.linalg.eigh(G); V = V[:, ::-1][:, :5]  # top-5 PCs (eigenvectors)
pop_centroid = V.mean(axis=0)

met = f"{OUT}/budget_Ncontrol.csv"
done = set()
if os.path.exists(met):
    for ln in open(met, errors="ignore"):
        p = ln.rstrip("\n").split(",")
        if len(p) >= 5 and p[0] in MODELS: done.add((p[0], p[1], p[2], p[4]))
mf = open(met, "a")
if os.path.getsize(met) == 0: mf.write("model,kind,tag,N,r,pearson,r2_vs_mean,var_yhat,centroid_drift\n")
log(f"RESUME {len(done)}")

def run_one(kind, tag, N, r, tr, test):
    mu_y, sd_y = y[tr].mean(), y[tr].std()
    ytr = (y[tr] - mu_y) / sd_y; yte = (y[test] - mu_y) / sd_y; yraw = y[test]
    ssb = float(np.sum((yraw - mu_y) ** 2))
    drift = float(np.linalg.norm(V[tr].mean(axis=0) - pop_centroid))
    for nm in MODELS:
        if (nm, kind, tag, str(r)) in done: continue
        try:
            pred = H.run_model(nm, M, G, tr, test, ytr, yte, "strict", {}); pr = pred * sd_y + mu_y
            mt = H.metrics(yte, pred); r2m = 1 - float(np.sum((yraw - pr) ** 2)) / ssb if ssb > 0 else float("nan")
            mf.write(f"{nm},{kind},{tag},{N},{r},{mt['pearson']:.5f},{r2m:.5f},{float(np.var(pr)):.1f},{drift:.5f}\n"); mf.flush()
            log(f"{nm}/{kind}/{tag}/N{N}/r{r} pear={mt['pearson']:.3f} r2mean={r2m:.3f}")
        except Exception as e:
            mf.write(f"{nm},{kind},{tag},{N},{r},ERR,ERR,ERR,{drift:.5f}\n"); mf.flush(); log(f"ERR {nm} {e}")

# A: random anchors (no tau cap)
for N in [150, 400]:
    for r in range(R):
        rng = np.random.default_rng(900 + N + r)
        test = np.sort(rng.choice(n, N_TEST, replace=False))
        pool = np.array([i for i in range(n) if i not in set(test.tolist())])
        tr = np.sort(rng.choice(pool, N, replace=False))
        run_one("random", f"randN{N}", N, r, tr, test)

# B: N=400 budget ladder
for (name, tau) in N400_TAUS:
    for r in range(R):
        rng = np.random.default_rng(20260529 + hash(name) % 1000 + r) if False else np.random.default_rng(5000 + int(tau * 1000) + r)
        test = np.sort(rng.choice(n, N_TEST, replace=False))
        tset = set(test.tolist()); maxrel = G[:, test].max(axis=1)
        nat = np.array([i for i in range(n) if i not in tset and maxrel[i] <= tau])
        if len(nat) < 400: log(f"skip {name} r{r}: natural {len(nat)}<400"); continue
        rng2 = np.random.default_rng(6000 + int(tau * 1000) + r); tm = nat.copy(); rng2.shuffle(tm)
        run_one("matched400", name, 400, r, np.sort(tm[:400]), test)

mf.close()
open(f"{OUT}/ALL_DONE_ncontrol", "w").write("done\n")
log("N-CONTROL DONE")
