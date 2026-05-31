# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
"""Diagnostic: GBLUP on random fold0 with IDENTICAL data loading as bayesr_cpu.py.
If GBLUP~0.43 -> my data pipeline is clean, BayesR 0.85 = overfit/bug in sampler.
If GBLUP~0.85 -> data-loading leak (alignment)."""
import numpy as np, csv, glob
GRM="data/processed/holstein/relationship_splits/holstein_grm.npy"
GENO="data/processed/holstein/parse_qc/holstein_genotype_matrix_qc.npy"
PHE="data/processed/holstein/parse_qc/holstein_phenotype_table.csv"
SP="splits/holstein/random"
TRAIT="milk_le_sum_305"
rows=list(csv.DictReader(open(PHE))); rows.sort(key=lambda r: float(r["order"]))
y=np.array([float(r[TRAIT]) for r in rows]);
id2row={int(float(r["sample_id"])):i for i,r in enumerate(rows)}
trp=sorted(glob.glob(f"{SP}/*_train.csv"))[0]; tep=trp.replace("_train.csv","_test.csv")
rd=lambda p:[int(float(l.split(",")[0])) for l in open(p).read().splitlines()[1:] if l.strip()]
tr=np.array(sorted(id2row[i] for i in rd(trp) if i in id2row))
te=np.array(sorted(id2row[i] for i in rd(tep) if i in id2row))
print(f"train {len(tr)} test {len(te)} overlap {len(set(tr)&set(te))}")

# --- GBLUP via GRM (order-sorted alignment, same as h2) ---
A=np.load(GRM).astype(np.float64); A=(A+A.T)/2
ytr=y[tr]; ym=ytr.mean()
lam=1.0
Atr=A[np.ix_(tr,tr)]+lam*np.eye(len(tr))
sol=np.linalg.solve(Atr, ytr-ym)
pred_grm=A[np.ix_(te,tr)]@sol + ym
pe_grm=np.corrcoef(y[te],pred_grm)[0,1]
print(f"GBLUP(GRM) fold0 Pearson = {pe_grm:.4f}")

# --- GBLUP via genotype (ridge), to check genotype-matrix alignment too ---
X=np.load(GENO,mmap_mode="r")
Xtr=np.asarray(X[tr],dtype=np.float64); Xte=np.asarray(X[te],dtype=np.float64)
mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6; Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
# ridge in sample space (kernel = XX'/m)
m=Xtr.shape[1]; K=Xtr@Xtr.T/m; Kte=Xte@Xtr.T/m
a=np.linalg.solve(K+1.0*np.eye(len(tr)), ytr-ym)
pred_x=Kte@a+ym
pe_x=np.corrcoef(y[te],pred_x)[0,1]
print(f"GBLUP(genotype ridge) fold0 Pearson = {pe_x:.4f}")
print("benchmark random GBLUP ~0.435 (relatedness_decay.csv)")
