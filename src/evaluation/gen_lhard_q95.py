# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""q95 sensitivity: K=40 budget-q95-matched resamples (each test sample max-train-rel <= q95
threshold; matched train=101). per-sample relatedness-budget residuals for fork incremental.
seed=20260530."""
import sys,time; sys.path.insert(0,"scripts")
import numpy as np, run_holstein_benchmark as H
RNG=np.random.default_rng(20260530); OUT="scrl_mini"
rfi,ibr=H.load_alignment(); n=len(ibr); y=H.load_pheno(rfi,n); M=np.load(H.GENO).astype(np.float32)
print("GRM...",flush=True); G=H.vanraden_grm(M)
tau=float(np.quantile(G[np.triu_indices(n,1)],0.95)); print(f"tau(q95)={tau:.4f}",flush=True)
K=40; o=open(f"{OUT}/lhard_q95_predictions.csv","w"); o.write("k,model,sample_id,y_true_std,y_pred,n_train,n_test\n")
made=0; att=0
while made<K and att<K*4:
    att+=1; perm=RNG.permutation(n); T=perm[:110]
    rel=(G[T]>tau).any(0); rel[T]=True; pool=np.where(~rel)[0]
    if len(pool)<101: continue
    tr=RNG.choice(pool,101,replace=False); te=T
    mu,sd=y[tr].mean(),y[tr].std(); ytr=(y[tr]-mu)/sd; yte=(y[te]-mu)/sd
    for name in ["FullSNP_MLP_top50k","GBLUP_full","PCARidge_PCA80"]:
        try:
            p=H.run_model(name,M,G,tr,te,ytr,yte,"strict",{})
            for i in range(len(te)): o.write(f"{made},{name},{ibr[te[i]]},{yte[i]:.6f},{p[i]:.6f},{len(tr)},{len(te)}\n")
            o.flush()
        except Exception as e: print("ERR",name,e,flush=True)
    made+=1
    if made%10==0: print(f"resample {made}/{K} (attempts {att}) pool={len(pool)}",flush=True)
o.close(); open(f"{OUT}/LHARD_Q95_DONE","w").write(str(made)); print(f"LHARD_Q95 DONE made={made}",flush=True)
