# CattleOmni reproducibility repo -- sanitised analysis script (paths repo-relative; no server paths/credentials/raw data).
#!/usr/bin/env python3
"""GRM fold-internal allele-frequency SENSITIVITY.
Main analysis uses a GLOBAL unsupervised VanRaden GRM (allele freqs over all samples).
Here we rebuild the GRM PER FOLD using TRAIN-ONLY allele frequencies, redo the headline
GBLUP random-CV and relationship-aware (group10/group5/lco) contrasts, and compare to the
locked global-GRM GBLUP. If conclusions are unchanged, the transductive global GRM is not
driving them. NOTHING fabricated; GBLUP solved exactly as the benchmark (dual ridge on GRM)."""
import os, sys, glob, csv, time
import numpy as np
sys.path.insert(0,"src")
import importlib.util
spec=importlib.util.spec_from_file_location("H","src/run_holstein_benchmark.py")
H=importlib.util.module_from_spec(spec); _a=sys.argv; sys.argv=["H"]; spec.loader.exec_module(H); sys.argv=_a
SPR="splits/holstein"
OUT="results/grm_foldint"; os.makedirs(OUT,exist_ok=True)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)

M=np.load(H.GENO).astype(np.float64)
row_for_id,ids=H.load_alignment(); n=len(ids)
y=H.load_pheno(row_for_id,n)
log(f"M{M.shape} n={n} trait={H.TRAIT}")

def grm_global(M):
    p=M.mean(0)/2.0; sd=np.sqrt(2*p*(1-p)); Z=M-2*p; denom=2.0*np.sum(p*(1-p))
    return (Z@Z.T)/denom
def grm_trainfreq(M, tr):
    """VanRaden GRM with allele freqs estimated from TRAIN rows only; centring/scaling by train p."""
    p=M[tr].mean(0)/2.0; denom=2.0*np.sum(p*(1-p)); Z=M-2*p
    return (Z@Z.T)/denom

Gg=grm_global(M)  # locked-style global GRM
rd=lambda q:[int(float(l.split(",")[0])) for l in open(q).read().splitlines()[1:] if l.strip()]
def folds_of(reg):
    if reg=="random": return sorted(glob.glob(f"{SPR}/random/*_train.csv"))[:10]
    if reg=="group10": return sorted(glob.glob(f"{SPR}/relationship_group/group_10fold/*_train.csv"))
    if reg=="group5": return sorted(glob.glob(f"{SPR}/relationship_group/group_5fold/*_train.csv"))
    if reg=="lco": return sorted(glob.glob(f"{SPR}/leave_cluster_out/*_train.csv"))

def gblup_pred(G,tr,te,ytr):
    K=G[np.ix_(tr,tr)]; lam=H.pick_lam(K,ytr); return H.dual_solve(K,ytr,G[np.ix_(te,tr)],lam)
def pear(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    return float(np.corrcoef(a,b)[0,1]) if len(a)>2 and a.std()>1e-9 and b.std()>1e-9 else float("nan")

rows=[]
for reg in ["random","group10","group5","lco"]:
    fs=folds_of(reg)
    pg=[]; pf=[]  # global-GRM, foldinternal-GRM per-fold Pearson
    for trp in fs:
        tep=trp.replace("_train.csv","_test.csv")
        tr=np.array(sorted(row_for_id[i] for i in rd(trp) if i in row_for_id))
        te=np.array(sorted(row_for_id[i] for i in rd(tep) if i in row_for_id))
        if len(te)<3: continue
        ym=y[tr].mean(); ys=y[tr].std()+1e-9
        ytr=(y[tr]-ym)/ys; yte_raw=y[te]
        # global GRM GBLUP
        predg=gblup_pred(Gg,tr,te,ytr)*ys+ym
        # fold-internal GRM GBLUP
        Gf=grm_trainfreq(M,tr)
        predf=gblup_pred(Gf,tr,te,ytr)*ys+ym
        pg.append(pear(yte_raw,predg)); pf.append(pear(yte_raw,predf))
    pg=np.array(pg); pf=np.array(pf)
    # lco uses pooled metric in the paper; report both fold-mean and pooled-ish via mean here for the contrast
    rows.append((reg,len(pg),pg.mean(),pf.mean(),(pf-pg).mean()))
    log(f"{reg}: nfold={len(pg)} global={pg.mean():.4f} foldint={pf.mean():.4f} diff={ (pf-pg).mean():+.4f}")

with open(f"{OUT}/grm_foldint_sens.csv","w") as f:
    f.write("regime,nfold,gblup_globalGRM,gblup_foldintGRM,diff\n")
    for r in rows: f.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f}\n")
print("\n=== GRM fold-internal sensitivity (GBLUP fold-mean Pearson) ===")
print(f"{'regime':10s} {'global-GRM':>11s} {'foldint-GRM':>12s} {'diff':>9s}")
for r in rows: print(f"{r[0]:10s} {r[2]:11.4f} {r[3]:12.4f} {r[4]:+9.4f}")
print(f"\nmax |diff| across regimes = {max(abs(r[4]) for r in rows):.4f}")
print(f"saved {OUT}/grm_foldint_sens.csv")
