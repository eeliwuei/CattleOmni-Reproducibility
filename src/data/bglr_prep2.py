# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""Prep ONE fold for BGLR BayesB full-164k (NO marker selection -> no leak vector).
Alignment IDENTICAL to validated bglr_prep.py (parse_qc PHE, order-sort, sample_id->row).
Standardize X by TRAIN stats. Writes X/y/ytrue/meta(+train_mean,sd)/teids/label.
Usage: bglr_prep2.py TRAIN_CSV WORKDIR FAMILY FOLDTAG"""
import numpy as np, csv, sys, os
GENO="data/processed/holstein/parse_qc/holstein_genotype_matrix_qc.npy"
PHE="data/processed/holstein/parse_qc/holstein_phenotype_table.csv"
TRAIT="milk_le_sum_305"
TRAIN_CSV=sys.argv[1]; WORK=sys.argv[2]; FAM=sys.argv[3]; FOLD=sys.argv[4]
os.makedirs(WORK, exist_ok=True)
TEST_CSV=TRAIN_CSV.replace("_train.csv","_test.csv")
rows=list(csv.DictReader(open(PHE))); rows.sort(key=lambda r: float(r["order"]))   # EXACT original alignment
y=np.array([float(r[TRAIT]) for r in rows],dtype=np.float64)
id2row={int(float(r["sample_id"])):i for i,r in enumerate(rows)}
rd=lambda p:[int(float(l.split(",")[0])) for l in open(p).read().splitlines()[1:] if l.strip()]
tr_ids=[i for i in rd(TRAIN_CSV) if i in id2row]; te_ids=[i for i in rd(TEST_CSV) if i in id2row]
tr=np.array([id2row[i] for i in tr_ids]); te=np.array([id2row[i] for i in te_ids])
keep=np.concatenate([tr,te])
X=np.load(GENO,mmap_mode="r")
Xtr=np.asarray(X[tr],dtype=np.float64)
mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6
Xall=(np.asarray(X[keep],dtype=np.float64)-mu)/sd           # ALL 164312 markers, no selection
ytr=y[tr]; tm=float(ytr.mean()); ts=float(ytr.std())
ntr=len(tr); y_na=y[keep].copy(); y_na[ntr:]=np.nan
Xall.astype(np.float64).tofile(f"{WORK}/X.bin")
y_na.astype(np.float64).tofile(f"{WORK}/y.bin")
y[te].astype(np.float64).tofile(f"{WORK}/ytrue.bin")
open(f"{WORK}/meta.txt","w").write(f"{Xall.shape[0]} {Xall.shape[1]} {ntr} {len(te)} {tm} {ts}\n")
open(f"{WORK}/teids.txt","w").write("\n".join(str(i) for i in te_ids)+"\n")
open(f"{WORK}/label.txt","w").write(f"{FAM} {FOLD}\n")
print(f"prepped {FAM}/{FOLD}: n={Xall.shape[0]} p={Xall.shape[1]} ntr={ntr} nte={len(te)} tm={tm:.2f} ts={ts:.2f}")
