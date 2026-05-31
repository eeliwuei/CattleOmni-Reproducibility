# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""Step 4 + lco-reversal verification: Holstein relatedness-budget continuous sweep.
Absolute tau ladder 0.20 -> 0.025 (fills the crossover region between moderate budgets and lco).
For each tau, R resampled 10% test sets; train = max(train-test relatedness) <= tau.
natural (train shrinks) + matched (FIXED N=50). Saves Pearson, RMSE, R2-vs-train-mean, var(yhat),
and per-sample (y_true_raw, y_pred_raw, max_train_rel). Reuses run_holstein_benchmark models. Resumable.
"""
import os, sys, csv, json, time
sys.path.insert(0, "scripts")
import numpy as np
import run_holstein_benchmark as H

OUT = os.environ.get("HOUT", "benchmarks/holstein")  # per-trait output dir via env
R = 8
N_TEST = 109
N_MATCH = 150  # fixed train size large enough to avoid small-n over-regularization (GBLUP->constant) artifact
MODELS = ["GBLUP_full", "PCARidge_PCA80", "Ridge_top50k", "FullSNP_MLP_top50k", "SparseGate_MLP_top20k"]
TAU_LADDER = [("t200", 0.200), ("t150", 0.150), ("t100", 0.100), ("t070", 0.070), ("t050", 0.050)]  # natural>=155 so matched-150 feasible

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

M = np.load(H.GENO).astype(np.float32)
row_for_id, ids_by_row = H.load_alignment(); n = len(ids_by_row)
y = H.load_pheno(row_for_id, n)
log(f"M{M.shape}; computing GRM ...")
G = H.vanraden_grm(M)

plan = {}
for ti, (name, tau) in enumerate(TAU_LADDER):
    for r in range(R):
        rng = np.random.default_rng(20260529 + ti * 1000 + r)
        test = np.sort(rng.choice(n, size=N_TEST, replace=False))
        tset = set(test.tolist()); maxrel = G[:, test].max(axis=1)
        train_nat = np.array([i for i in range(n) if i not in tset and maxrel[i] <= tau], dtype=int)
        plan[(name, r)] = dict(test=test, train_nat=train_nat)
for (name, tau) in TAU_LADDER:
    szs = [len(plan[(name, r)]["train_nat"]) for r in range(R)]
    log(f"tau {name}={tau}: natural train min/med/max = {min(szs)}/{int(np.median(szs))}/{max(szs)}")

met_path = f"{OUT}/budget_metrics.csv"; pred_path = f"{OUT}/budget_predictions.csv"
done = set()
if os.path.exists(met_path):
    with open(met_path, errors="ignore") as f:
        for ln in f:
            p = ln.replace("\x00", "").rstrip("\n").split(",")
            if len(p) >= 5 and p[0] in MODELS: done.add((p[0], p[1], p[2], p[4]))
mf = open(met_path, "a")
if os.path.getsize(met_path) == 0: mf.write("model,kind,tau_name,tau,r,n_train,n_test,pearson,r2,rmse,mu_y,var_yhat,r2_vs_mean\n")
pf = open(pred_path, "a")
if os.path.getsize(pred_path) == 0: pf.write("model,kind,tau_name,tau,r,sample_id,y_true_raw,y_pred_raw,max_train_rel\n")
log(f"RESUME: {len(done)} done")

t0 = time.time(); cnt = 0
for ti, (name, tau) in enumerate(TAU_LADDER):
    for r in range(R):
        test = plan[(name, r)]["test"]; train_nat = plan[(name, r)]["train_nat"]
        kinds = [("natural", train_nat)]
        if len(train_nat) >= N_MATCH:
            rng = np.random.default_rng(777 + ti * 1000 + r)
            tm = train_nat.copy(); rng.shuffle(tm); kinds.append(("matched", np.sort(tm[:N_MATCH])))
        for kind, tr in kinds:
            if len(tr) < 5: continue
            mu_y, sd_y = y[tr].mean(), y[tr].std()
            ytr = (y[tr] - mu_y) / sd_y; yte = (y[test] - mu_y) / sd_y
            yte_raw = y[test]; te_maxrel = G[np.ix_(test, tr)].max(axis=1)
            ss_base = float(np.sum((yte_raw - mu_y) ** 2))
            for nm in MODELS:
                cnt += 1
                if (nm, kind, name, str(r)) in done: continue
                ts = time.time()
                try:
                    pred = H.run_model(nm, M, G, tr, test, ytr, yte, "strict", {})
                    mt = H.metrics(yte, pred); pr = pred * sd_y + mu_y
                    r2m = 1.0 - float(np.sum((yte_raw - pr) ** 2)) / ss_base if ss_base > 0 else float("nan")
                    mf.write(f"{nm},{kind},{name},{tau:.4f},{r},{len(tr)},{len(test)},{mt['pearson']:.5f},{mt['r2']:.5f},{mt['rmse']:.5f},{mu_y:.4f},{float(np.var(pr)):.4f},{r2m:.5f}\n"); mf.flush()
                    for i in range(len(test)):
                        pf.write(f"{nm},{kind},{name},{tau:.4f},{r},{ids_by_row[test[i]]},{yte_raw[i]:.4f},{pr[i]:.4f},{te_maxrel[i]:.4f}\n")
                    pf.flush()
                    log(f"[{cnt}] {nm}/{kind}/{name}/r{r} n_tr={len(tr)} pear={mt['pearson']:.3f} r2mean={r2m:.3f} var={np.var(pr):.3f} ({time.time()-ts:.1f}s)")
                except Exception as e:
                    mf.write(f"{nm},{kind},{name},{tau:.4f},{r},{len(tr)},{len(test)},ERR,ERR,ERR,ERR,ERR,ERR\n"); mf.flush()
                    log(f"[{cnt}] ERR {nm}/{kind}/{name}/r{r}: {e}")
            json.dump({"cnt": cnt, "elapsed_s": round(time.time() - t0, 1)}, open(f"{OUT}/budget_progress.json", "w"))
mf.close(); pf.close()
open(f"{OUT}/ALL_DONE_budget", "w").write(f"{cnt}\n")
log(f"BUDGET SWEEP DONE cnt={cnt} in {time.time()-t0:.0f}s")
