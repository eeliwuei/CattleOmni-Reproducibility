# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""A9-T5 L_hard: repeated RELATIONSHIP-CLUSTER 5-fold CV (R=10) -> per-sample relatedness-
constrained residuals (real y_true). Whole GRM-cluster held out together (train excludes
same-cluster); each sample held out 10x. Strict analog of D's repeated-random CV (L_easy).
Reuses harness model funcs. Polite tenant. seed=20260529."""
import sys, csv, time, os
sys.path.insert(0,"scripts")
import numpy as np, run_holstein_benchmark as H
SEED=20260529; RNG=np.random.default_rng(SEED)
OUT="scrl_mini"; os.makedirs(OUT,exist_ok=True)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
row_for_id, ids_by_row = H.load_alignment(); n=len(ids_by_row); y=H.load_pheno(row_for_id,n)
M=np.load(H.GENO).astype(np.float32); assert M.shape[0]==n, M.shape
log("VanRaden GRM..."); G=H.vanraden_grm(M)
cmpath="splits/holstein/relationship_group/group_5fold/cluster_membership.csv"
hdr=next(csv.reader(open(cmpath))); sidc=[c for c in hdr if "sample" in c.lower() or c.lower()=="id"][0]
clc=[c for c in hdr if "cluster" in c.lower()][0]
cm={}
for d in csv.DictReader(open(cmpath)):
    s=int(d[sidc])
    if s in row_for_id: cm[s]=int(d[clc])
clusters=sorted(set(cm.values())); log(f"clusters={len(clusters)} samples_in_cm={len(cm)}")
models=["GBLUP_full","PCARidge_PCA80","FullSNP_MLP_top50k"]
pf=open(f"{OUT}/lhard_predictions.csv","w"); pf.write("repeat,fold,model,sample_id,y_true_std,y_pred,n_train,n_test\n")
R=10; KF=5; t0=time.time()
for r in range(R):
    cl=clusters[:]; RNG.shuffle(cl); fold_of={c:i%KF for i,c in enumerate(cl)}
    for kf in range(KF):
        te_ids=[s for s,c in cm.items() if fold_of[c]==kf]; tr_ids=[s for s,c in cm.items() if fold_of[c]!=kf]
        if len(te_ids)<2 or len(tr_ids)<20: continue
        te=np.array([row_for_id[s] for s in te_ids]); tr=np.array([row_for_id[s] for s in tr_ids])
        mu,sd=y[tr].mean(),y[tr].std(); ytr=(y[tr]-mu)/sd; yte=(y[te]-mu)/sd
        for name in models:
            try:
                pred=H.run_model(name,M,G,tr,te,ytr,yte,"strict",{})
                for s,yt_,yp in zip(te_ids,yte,pred): pf.write(f"{r},{kf},{name},{s},{yt_:.6f},{yp:.6f},{len(tr)},{len(te)}\n")
                pf.flush()
            except Exception as e: log(f"r{r}f{kf} {name} ERR {e}")
    log(f"repeat {r} done ({time.time()-t0:.0f}s)")
pf.close(); open(f"{OUT}/LHARD_DONE","w").write("ok\n"); log(f"LHARD DONE {time.time()-t0:.0f}s -> {OUT}/lhard_predictions.csv")
