# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""snp3 (Japanese Black N=9850) relatedness-STRINGENCY ladder on Level-B A/D matrices, new-framework metric.
Per-individual tau-budget is infeasible (population too related); the natural relatedness axis here is the
cluster structure: random -> group5 -> group10 -> leave-cluster-out (decreasing train-test relatedness).
A_KRR/D_KRR/AD_equal_KRR, matched-N=500, report R2-vs-mean + Pearson + var, per trait. NO SNP-level claims.
Replicates 'skill declines as relatedness is removed' on a larger, independent breed."""
import os, csv, glob, json, time
import numpy as np
IN = "./cattleomni_snp3_ad_matrix_h100/01_input_data"
SP = "splits/snp3"
OUT = "benchmarks/snp3"; os.makedirs(OUT, exist_ok=True)
TRAITS = ["CW", "REA", "RT", "SFT", "YI", "BMS"]
N_MATCH = 500; LAM = [0.1, 1.0, 10.0]
FAMS = [("random", f"{SP}/random"), ("group5", f"{SP}/relationship_group/group_5fold"),
        ("group10", f"{SP}/relationship_group/group_10fold"), ("lco", f"{SP}/leave_cluster_out")]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
A = np.load(f"{IN}/A_float32.npy"); D = np.load(f"{IN}/D_float32.npy"); Y = np.load(f"{IN}/Y_float32.npy")
n = A.shape[0]; log(f"A{A.shape} Y{Y.shape}")

def read_idx(p):
    out = []
    for ln in open(p).read().splitlines()[1:]:
        if ln.strip(): out.append(int(float(ln.split(",")[0])) - 1)  # snp3 split ids are 1-based -> 0-based A row
    a = np.array(out)
    assert a.min() >= 0 and a.max() < n, f"idx out of range in {p}: [{a.min()},{a.max()}] vs n={n}"
    return a

def krr(K, ytr, Kte):
    idx = np.arange(len(ytr)); rng = np.random.default_rng(7); rng.shuffle(idx); folds = np.array_split(idx, 3)
    best = (1.0, -9)
    for lam in LAM:
        ps = []
        for j in range(3):
            vi = folds[j]; ti = np.concatenate([folds[m] for m in range(3) if m != j])
            al = np.linalg.solve(K[np.ix_(ti, ti)] + lam * np.eye(len(ti)), ytr[ti])
            pp = K[np.ix_(vi, ti)] @ al
            ps.append(np.corrcoef(ytr[vi], pp)[0, 1] if np.std(pp) > 1e-9 else -9)
        if np.mean(ps) > best[1]: best = (lam, np.mean(ps))
    al = np.linalg.solve(K + best[0] * np.eye(len(ytr)), ytr); return Kte @ al

met = f"{OUT}/snp3_ladder_metrics.csv"
done = set()
if os.path.exists(met):
    for ln in open(met, errors="ignore"):
        p = ln.rstrip("\n").split(",")
        if len(p) >= 4 and p[0] in TRAITS: done.add((p[0], p[1], p[2], p[3]))
mf = open(met, "a")
if os.path.getsize(met) == 0: mf.write("trait,model,family,fold,n_train,n_test,pearson,r2_vs_mean,var_yhat\n")

t0 = time.time()
for fam, d in FAMS:
    trains = sorted(glob.glob(f"{d}/*_train.csv"))
    log(f"{fam}: {len(trains)} folds in {d}")
    for tp in trains:
        fold = os.path.basename(tp)[:-10]; te_p = tp.replace("_train.csv", "_test.csv")
        if not os.path.exists(te_p): continue
        tr_all = read_idx(tp); test = read_idx(te_p)
        if len(tr_all) < 50 or len(test) < 10: continue
        rng = np.random.default_rng(42); trm = tr_all.copy(); rng.shuffle(trm)
        tr = np.sort(trm[:min(N_MATCH, len(tr_all))])
        Ktr_A = A[np.ix_(tr, tr)]; Kte_A = A[np.ix_(test, tr)]; Ktr_D = D[np.ix_(tr, tr)]; Kte_D = D[np.ix_(test, tr)]
        for ti, tn in enumerate(TRAITS):
            yv = Y[:, ti]; mu = yv[tr].mean(); sd = yv[tr].std() + 1e-9
            ytr = (yv[tr] - mu) / sd; yte_raw = yv[test]; ssb = float(np.sum((yte_raw - mu) ** 2))
            for mdl, Ktr, Kte in [("A_KRR", Ktr_A, Kte_A), ("D_KRR", Ktr_D, Kte_D), ("AD_equal_KRR", (Ktr_A + Ktr_D) / 2, (Kte_A + Kte_D) / 2)]:
                if (tn, mdl, fam, fold) in done: continue
                try:
                    pr = krr(Ktr, ytr, Kte) * sd + mu
                    pe = np.corrcoef(yte_raw, pr)[0, 1] if np.std(pr) > 1e-9 else float("nan")
                    r2m = 1 - float(np.sum((yte_raw - pr) ** 2)) / ssb if ssb > 0 else float("nan")
                    mf.write(f"{tn},{mdl},{fam},{fold},{len(tr)},{len(test)},{pe:.5f},{r2m:.5f},{float(np.var(pr)):.4f}\n"); mf.flush()
                except Exception as e:
                    mf.write(f"{tn},{mdl},{fam},{fold},{len(tr)},{len(test)},ERR,ERR,ERR\n"); mf.flush(); log(f"ERR {tn}/{mdl}/{fam}/{fold}:{e}")
        log(f"{fam}/{fold} done ({time.time()-t0:.0f}s)")
mf.close(); open(f"{OUT}/ALL_DONE_snp3", "w").write("done\n"); log("SNP3 LADDER DONE")
