# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""N-control: random-split (NO relatedness constraint) performance vs matched train size N.
Decouples 'small-N floor' from 'relatedness effect'. Same held-out test (200), train=random N
from remainder, K=10 repeats. Models GBLUP/PCARidge/FullSNP_MLP. seed=20260529."""
import sys,time; sys.path.insert(0,"scripts")
import numpy as np, run_holstein_benchmark as H
RNG=np.random.default_rng(20260529); OUT="scrl_mini"
rfi,ibr=H.load_alignment(); n=len(ibr); y=H.load_pheno(rfi,n); M=np.load(H.GENO).astype(np.float32)
print("GRM...",flush=True); G=H.vanraden_grm(M)
models=["GBLUP_full","PCARidge_PCA80","FullSNP_MLP_top50k"]; NL=[101,150,300,500,800]; K=10
o=open(f"{OUT}/ncontrol_results.csv","w"); o.write("N,model,repeat,pearson,r2,n_train,n_test\n")
for N in NL:
    for r in range(K):
        perm=RNG.permutation(n); te=perm[:200]; tr=perm[200:200+N]
        mu,sd=y[tr].mean(),y[tr].std(); ytr=(y[tr]-mu)/sd; yte=(y[te]-mu)/sd
        for name in models:
            try:
                p=H.run_model(name,M,G,tr,te,ytr,yte,"strict",{}); m=H.metrics(yte,p)
                o.write(f"{N},{name},{r},{m['pearson']:.5f},{m['r2']:.5f},{len(tr)},{len(te)}\n"); o.flush()
            except Exception as e: print("ERR",N,name,e,flush=True)
    print(f"N={N} done",flush=True)
o.close(); open(f"{OUT}/NCONTROL_DONE","w").write("ok"); print("NCONTROL DONE",flush=True)
