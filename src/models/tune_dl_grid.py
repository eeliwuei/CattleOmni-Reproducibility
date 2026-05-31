# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""#2 fair DL tuning grid (reviewer defense vs 'DL undertuned'). PRESET small grid (NOT fishing):
MLP capacity x dropout x wd x lr x epochs. Evaluate at matched-N=400, strict tau=0.10 (t100), 8 resamples,
per trait (same splits/seeds as budget_Ncontrol N400 ladder -> comparable to the linear baseline).
Report per-config outer R2-vs-mean; grid-MAX is selected on OUTER test = OPTIMISTIC upper bound: if even
that <= linear, the 'undertuned' objection is refuted. Reuses run_holstein_benchmark. Traits via argv."""
import os, sys, csv, time
sys.path.insert(0, "scripts")
import numpy as np, run_holstein_benchmark as H
OUT = "benchmarks/tuning"; os.makedirs(OUT, exist_ok=True)
PHE = "data/holstein_phenotype_table.csv"
TRAITS = sys.argv[1:] or ["milk_le_sum_305", "fat_le_sum_305", "scs_le_ave305"]
TAU = 0.10; N_MATCH = 400; R = 8; TOP = 50000
GRID = [  # (hidden, dropout, lr, wd, epochs) -- preset & documented
 (256,0.35,8e-4,1e-3,30),(512,0.35,8e-4,1e-3,30),(128,0.35,8e-4,1e-3,30),
 (256,0.10,8e-4,1e-3,30),(256,0.60,8e-4,1e-3,30),(256,0.35,8e-4,1e-4,30),
 (256,0.35,8e-4,1e-2,30),(256,0.35,3e-4,1e-3,30),(256,0.35,8e-4,1e-3,80),
 (512,0.60,8e-4,1e-2,30),(128,0.10,8e-4,1e-4,80),(512,0.35,3e-4,1e-3,80)]
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
M = np.load(H.GENO).astype(np.float32); row_for_id, ids = H.load_alignment(); n = len(ids)
phe = {int(d["ID"]): d for d in csv.DictReader(open(PHE))}
G = H.vanraden_grm(M)
mf = open(f"{OUT}/dl_tuning_grid.csv", "a")
if mf.tell() == 0 or os.path.getsize(f"{OUT}/dl_tuning_grid.csv") == 0:
    mf.write("trait,cfg,hidden,dropout,lr,wd,epochs,r,n_train,r2_vs_mean,pearson\n")
for TR in TRAITS:
    y = np.array([float(phe[ids[i]][TR]) for i in range(n)])
    log(f"=== {TR} ===")
    for ci, (h, dp, lr, wd, ep) in enumerate(GRID):
        r2s = []
        for r in range(R):
            rng = np.random.default_rng(5000 + int(TAU*1000) + r); test = np.sort(rng.choice(n, 109, replace=False))
            tset = set(test.tolist()); mr = G[:, test].max(axis=1)
            nat = np.array([i for i in range(n) if i not in tset and mr[i] <= TAU])
            if len(nat) < N_MATCH: continue
            rng2 = np.random.default_rng(6000 + int(TAU*1000) + r); tm = nat.copy(); rng2.shuffle(tm); tr = np.sort(tm[:N_MATCH])
            mu, sd = y[tr].mean(), y[tr].std(); ytr = (y[tr]-mu)/sd; yraw = y[test]; ssb = float(np.sum((yraw-mu)**2))
            idx = H.topk_var_idx(M[tr], TOP); Xtr, Xte = H.std_fit_apply(M[np.ix_(tr, idx)], M[np.ix_(test, idx)])
            pred = H.mlp_fit_predict(Xtr, ytr, Xte, hidden=h, dropout=dp, epochs=ep, lr=lr, wd=wd)
            pr = pred*sd+mu; r2m = 1 - float(np.sum((yraw-pr)**2))/ssb if ssb > 0 else float("nan")
            pe = float(np.corrcoef(yraw, pr)[0,1]) if np.std(pr) > 1e-9 else float("nan")
            mf.write(f"{TR},{ci},{h},{dp},{lr},{wd},{ep},{r},{len(tr)},{r2m:.5f},{pe:.5f}\n"); mf.flush(); r2s.append(r2m)
        log(f"  cfg{ci} h{h} d{dp} lr{lr} wd{wd} ep{ep}: R2vsmean mean={np.mean(r2s):+.3f}" if r2s else f"  cfg{ci} skip")
mf.close(); open(f"{OUT}/ALL_DONE_tuning", "w").write("done\n"); log("TUNING GRID DONE")
