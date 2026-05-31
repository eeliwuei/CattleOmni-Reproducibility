# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
"""Prep one fold for BGLR BayesB: top-K SNP (train variance), standardize (train stats),
save X (n x p float64 binary, row-major) + y (NaN for test) + meta. Usage: bglr_prep.py REGIME FOLDTAG TOPK"""
import numpy as np, csv, sys, glob, json
GENO="data/processed/holstein/parse_qc/holstein_genotype_matrix_qc.npy"
PHE="data/processed/holstein/parse_qc/holstein_phenotype_table.csv"
SPROOT="splits/holstein"
TRAIT="milk_le_sum_305"
REG=sys.argv[1] if len(sys.argv)>1 else "random"
TAG=sys.argv[2] if len(sys.argv)>2 else None
TOPK=int(sys.argv[3]) if len(sys.argv)>3 else 50000
OUT="/tmp/bglr_fold"
REGDIR={"random":f"{SPROOT}/random","group5":f"{SPROOT}/relationship_group/group_5fold",
        "group10":f"{SPROOT}/relationship_group/group_10fold","lco":f"{SPROOT}/leave_cluster_out"}[REG]

rows=list(csv.DictReader(open(PHE))); rows.sort(key=lambda r: float(r["order"]))
y=np.array([float(r[TRAIT]) for r in rows],dtype=np.float64)
id2row={int(float(r["sample_id"])):i for i,r in enumerate(rows)}
trp=sorted(glob.glob(f"{REGDIR}/*_train.csv"))
if TAG: trp=[p for p in trp if TAG in p]
trp=trp[0]; tep=trp.replace("_train.csv","_test.csv")
rd=lambda p:[int(float(l.split(",")[0])) for l in open(p).read().splitlines()[1:] if l.strip()]
tr=np.array(sorted(id2row[i] for i in rd(trp) if i in id2row))
te=np.array(sorted(id2row[i] for i in rd(tep) if i in id2row))
keep=np.concatenate([tr,te])  # all rows used this fold
X=np.load(GENO,mmap_mode="r")
Xtr=np.asarray(X[tr],dtype=np.float64)
top=np.argsort(-Xtr.var(0))[:TOPK]
mu=Xtr[:,top].mean(0); sd=Xtr[:,top].std(0)+1e-6
Xall=(np.asarray(X[keep],dtype=np.float64)[:,top]-mu)/sd   # (n,K) train then test
y_na=y[keep].copy(); ntr=len(tr); y_na[ntr:]=np.nan        # test -> NA
Xall.astype(np.float64).tofile(f"{OUT}_X.bin")
y_na.astype(np.float64).tofile(f"{OUT}_y.bin")
np.array(y[te],dtype=np.float64).tofile(f"{OUT}_ytrue.bin")
open(f"{OUT}_meta.txt","w").write(f"{Xall.shape[0]} {Xall.shape[1]} {ntr} {len(te)}\n")
open(f"{OUT}_label.txt","w").write(f"{REG} {trp.split('/')[-1]}\n")
print(f"prepped {REG} {trp.split('/')[-1]}: n={Xall.shape[0]} p={Xall.shape[1]} ntr={ntr} nte={len(te)}")
