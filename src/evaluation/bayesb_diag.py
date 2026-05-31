# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""Diagnose BayesB full-164k=0.856 (>ceiling, vs GBLUP 0.344). Uses the EXACT gate data
/tmp/bglr_fold_{X,y,ytrue}.bin. (1) ridge on same X.bin -> isolate model vs data.
(2) train/test ID overlap + near-duplicate genotype rows. (3) ridge perm sanity."""
import numpy as np, glob, csv
W="/tmp/bglr_fold"
mt=open(f"{W}_meta.txt").read().split(); n,p,ntr,nte=int(mt[0]),int(mt[1]),int(mt[2]),int(mt[3])
X=np.fromfile(f"{W}_X.bin",dtype=np.float64).reshape(n,p)   # standardized (train stats), rows=[train,test]
y=np.fromfile(f"{W}_y.bin",dtype=np.float64)                # train real, test NaN
ytrue=np.fromfile(f"{W}_ytrue.bin",dtype=np.float64)
Xtr=X[:ntr]; Xte=X[ntr:]; ytr=y[:ntr]; ym=ytr.mean()
print(f"n={n} p={p} ntr={ntr} nte={nte}  ytrue[:3]={ytrue[:3]}")

# (1) kernel ridge (GBLUP-equiv) on the SAME matrix BGLR used
m=p; K=Xtr@Xtr.T/m; Kte=Xte@Xtr.T/m
a=np.linalg.solve(K+1.0*np.eye(ntr), ytr-ym); pred=Kte@a+ym
print(f"[1] kernel-ridge on SAME X.bin: test Pearson={np.corrcoef(ytrue,pred)[0,1]:.4f}  (expect ~0.34)")

# (1b) ridge with permuted train y (leak sanity; expect ~0)
rng=np.random.default_rng(0); yp=rng.permutation(ytr)
a2=np.linalg.solve(K+1.0*np.eye(ntr), yp-yp.mean()); pr2=Kte@a2+yp.mean()
print(f"[1b] ridge PERM train-y: test Pearson={np.corrcoef(ytrue,pr2)[0,1]:+.4f}  (expect ~0)")

# (2) near-duplicate genotype rows test-vs-train (row correlation on standardized X)
Xtr_c=(Xtr-Xtr.mean(1,keepdims=True)); Xtr_c/=(np.linalg.norm(Xtr_c,axis=1,keepdims=True)+1e-9)
Xte_c=(Xte-Xte.mean(1,keepdims=True)); Xte_c/=(np.linalg.norm(Xte_c,axis=1,keepdims=True)+1e-9)
S=Xte_c@Xtr_c.T   # (nte,ntr) cosine/corr of genotype rows
maxc=S.max(1)
print(f"[2] test-row max corr to any train row: min={maxc.min():.3f} median={np.median(maxc):.3f} "
      f"p90={np.percentile(maxc,90):.3f} max={maxc.max():.3f}")
for thr in [0.99,0.97,0.95,0.90]:
    print(f"    test rows with a train twin >{thr}: {(maxc>thr).sum()}/{nte}")
# is high BayesB driven by near-dup test rows? corr(maxc, |ytrue|)? just report top
order=np.argsort(-maxc)[:5]
print(f"    top-5 most-duplicated test rows: maxcorr={maxc[order].round(3)} ytrue={ytrue[order].round(1)}")

# (3) split ID overlap (from original split files, independent of bin)
SP="splits/holstein/random"
trp=sorted(glob.glob(f"{SP}/*_train.csv"))[0]; tep=trp.replace("_train.csv","_test.csv")
rd=lambda q:set(int(float(l.split(",")[0])) for l in open(q).read().splitlines()[1:] if l.strip())
tr_ids=rd(trp); te_ids=rd(tep)
print(f"[3] split overlap train&test: {len(tr_ids&te_ids)} ids  (train {len(tr_ids)} test {len(te_ids)})")
