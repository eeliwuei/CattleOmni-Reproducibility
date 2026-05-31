# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""Step 5: snp3 (Japanese Black, N=9850) relatedness-budget replication on the Level-B A/D relationship
matrices. NOT deep-vs-linear (all kernel methods on the SAME matrix); replicates the phenomenon
'prediction skill declines as train-test relatedness is removed' on a larger, independent breed.
Models: A_KRR (primary), D_KRR, AD_equal_KRR. Per trait. matched-N. Report R2-vs-mean + Pearson + var.
Resumable. NO SNP-level claims."""
import os, csv, json, time
import numpy as np
IN = "./cattleomni_snp3_ad_matrix_h100/01_input_data"
OUT = "benchmarks/snp3"; os.makedirs(OUT, exist_ok=True)
TRAITS = ["CW", "REA", "RT", "SFT", "YI", "BMS"]   # by Y column index (per v2 report order)
R = 6; N_MATCH = 1500; LAM = [0.1, 1.0, 10.0]
QS = [("q99", 0.99), ("q975", 0.975), ("q95", 0.95), ("q90", 0.90), ("q80", 0.80)]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
A = np.load(f"{IN}/A_float32.npy"); D = np.load(f"{IN}/D_float32.npy"); Y = np.load(f"{IN}/Y_float32.npy")
n = A.shape[0]; log(f"A{A.shape} D{D.shape} Y{Y.shape}")
offdiag = A[np.triu_indices(n, 1)]
TAUS = {nm: float(np.quantile(offdiag, q)) for nm, q in QS}
log(f"A-offdiag TAUS={TAUS}")

def krr_fit_predict(K, ytr, K_te_tr, tr, te):
    best = (1.0, -9); idx = np.arange(len(tr)); rng = np.random.default_rng(7); rng.shuffle(idx)
    folds = np.array_split(idx, 3)
    for lam in LAM:
        ps = []
        for j in range(3):
            vi = folds[j]; ti = np.concatenate([folds[m] for m in range(3) if m != j])
            al = np.linalg.solve(K[np.ix_(ti, ti)] + lam * np.eye(len(ti)), ytr[ti])
            pp = K[np.ix_(vi, ti)] @ al
            c = np.corrcoef(ytr[vi], pp)[0, 1] if np.std(pp) > 1e-9 else -9
            ps.append(c)
        if np.mean(ps) > best[1]: best = (lam, np.mean(ps))
    lam = best[0]
    al = np.linalg.solve(K + lam * np.eye(len(tr)), ytr)
    return K_te_tr @ al

met = f"{OUT}/snp3_budget_metrics.csv"
done = set()
if os.path.exists(met):
    for ln in open(met, errors="ignore"):
        p = ln.rstrip("\n").split(",")
        if len(p) >= 4 and p[0] in TRAITS: done.add((p[0], p[1], p[2], p[3]))
mf = open(met, "a")
if os.path.getsize(met) == 0: mf.write("trait,model,tau_name,r,tau,n_train,n_test,pearson,r2_vs_mean,var_yhat\n")

# natural sizes diag
for nm, _ in QS:
    tau = TAUS[nm]; szs = []
    for r in range(R):
        rng = np.random.default_rng(11 + r); test = rng.choice(n, n // 10, replace=False)
        mr = A[:, test].max(axis=1); szs.append(int(np.sum(mr <= tau)) - 0)
    log(f"tau {nm}={tau:.4f}: natural~{int(np.median(szs))}")

t0 = time.time()
for nm, _q in QS:
    tau = TAUS[nm]
    for r in range(R):
        rng = np.random.default_rng(11 + r); test = np.sort(rng.choice(n, n // 10, replace=False))
        tset = set(test.tolist()); mr = A[:, test].max(axis=1)
        nat = np.array([i for i in range(n) if i not in tset and mr[i] <= tau])
        if len(nat) < N_MATCH: log(f"skip {nm} r{r}: nat {len(nat)}<{N_MATCH}"); continue
        rng2 = np.random.default_rng(99 + r); tm = nat.copy(); rng2.shuffle(tm); tr = np.sort(tm[:N_MATCH])
        Ktr_A = A[np.ix_(tr, tr)]; Kte_A = A[np.ix_(test, tr)]
        Ktr_D = D[np.ix_(tr, tr)]; Kte_D = D[np.ix_(test, tr)]
        for ti, tname in enumerate(TRAITS):
            if (tname, "A_KRR", nm, str(r)) in done and (tname, "AD_equal_KRR", nm, str(r)) in done: continue
            yv = Y[:, ti]; mu = yv[tr].mean(); sd = yv[tr].std() + 1e-9
            ytr = (yv[tr] - mu) / sd; yte_raw = yv[test]; ssb = float(np.sum((yte_raw - mu) ** 2))
            for mdl, Ktr, Kte in [("A_KRR", Ktr_A, Kte_A), ("D_KRR", Ktr_D, Kte_D),
                                   ("AD_equal_KRR", (Ktr_A + Ktr_D) / 2, (Kte_A + Kte_D) / 2)]:
                if (tname, mdl, nm, str(r)) in done: continue
                try:
                    pred = krr_fit_predict(Ktr, ytr, Kte, tr, test); pr = pred * sd + mu
                    pe = np.corrcoef(yte_raw, pr)[0, 1] if np.std(pr) > 1e-9 else float("nan")
                    r2m = 1 - float(np.sum((yte_raw - pr) ** 2)) / ssb if ssb > 0 else float("nan")
                    mf.write(f"{tname},{mdl},{nm},{r},{tau:.4f},{len(tr)},{len(test)},{pe:.5f},{r2m:.5f},{float(np.var(pr)):.4f}\n"); mf.flush()
                except Exception as e:
                    mf.write(f"{tname},{mdl},{nm},{r},{tau:.4f},{len(tr)},{len(test)},ERR,ERR,ERR\n"); mf.flush(); log(f"ERR {tname}/{mdl}/{nm}: {e}")
        log(f"done {nm} r{r} ({time.time()-t0:.0f}s)")
        json.dump({"tau": nm, "r": r, "elapsed_s": round(time.time() - t0, 1)}, open(f"{OUT}/snp3_progress.json", "w"))
mf.close(); open(f"{OUT}/ALL_DONE_snp3", "w").write("done\n"); log("SNP3 BUDGET DONE")
