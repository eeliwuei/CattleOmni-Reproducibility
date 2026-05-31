# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
"""BayesR CPU-numpy single-site Gibbs (avoids GPU per-op dispatch overhead).
Times one random fold. Real Gibbs. Usage: bayesr_cpu.py [TOPK] [BURN] [SAMP]"""
import numpy as np, csv, time, glob, sys
GENO="data/processed/holstein/parse_qc/holstein_genotype_matrix_qc.npy"
PHE="data/processed/holstein/parse_qc/holstein_phenotype_table.csv"
SP="splits/holstein/random"
TRAIT="milk_le_sum_305"
TOPK=int(sys.argv[1]) if len(sys.argv)>1 else 10000
BURN=int(sys.argv[2]) if len(sys.argv)>2 else 500
SAMP=int(sys.argv[3]) if len(sys.argv)>3 else 500
rng=np.random.default_rng(7); t0=time.time()
rows=list(csv.DictReader(open(PHE))); rows.sort(key=lambda r: float(r["order"]))
y_all=np.array([float(r[TRAIT]) for r in rows],dtype=np.float64)
id2row={int(float(r["sample_id"])):i for i,r in enumerate(rows)}
trp=sorted(glob.glob(f"{SP}/*_train.csv"))[0]; tep=trp.replace("_train.csv","_test.csv")
rd=lambda p:[int(float(l.split(",")[0])) for l in open(p).read().splitlines()[1:] if l.strip()]
tr=np.array(sorted(id2row[i] for i in rd(trp) if i in id2row))
te=np.array(sorted(id2row[i] for i in rd(tep) if i in id2row))
X=np.load(GENO,mmap_mode="r"); Xtr=np.asarray(X[tr],dtype=np.float64)
top=np.argsort(-Xtr.var(0))[:TOPK]; Xtr=np.ascontiguousarray(Xtr[:,top])
Xte=np.ascontiguousarray(np.asarray(X[te],dtype=np.float64)[:,top])
mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6; Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
ytr=y_all[tr]; ym=ytr.mean(); ys=ytr.std()+1e-9; yz=(ytr-ym)/ys
nT,m=Xtr.shape; XtX=(Xtr*Xtr).sum(0)
XtrT=np.asfortranarray(Xtr)  # column access fast
beta=np.zeros(m); kcls=np.zeros(m,dtype=np.int64); resid=yz.copy()
gamma=np.array([0.0,1e-4,1e-3,1e-2]); logpi=np.log(np.array([0.95,0.02,0.02,0.01]))
sig2e=1.0; sig2g=0.5; acc=np.zeros(m); nsamp=0
nu_e,S_e,nu_g,S_g=4.0,0.5,4.0,0.5   # scaled-inv-chi2 priors
print(f"setup {time.time()-t0:.0f}s | train {nT} test {len(te)} K={TOPK}",flush=True)
t1=time.time()
for it in range(BURN+SAMP):
    vk=gamma*sig2g                       # (4,) class variances
    u_all=rng.random(m); z_all=rng.standard_normal(m)
    for j in range(m):
        xj=XtrT[:,j]; bj_old=beta[j]
        if bj_old!=0.0: resid+=xj*bj_old
        rhs=xj@resid
        denom=XtX[j]+sig2e/vk[1:]; meank=rhs/denom; varpost=sig2e/denom
        lp=np.empty(4); lp[0]=logpi[0]
        lp[1:]=logpi[1:]+0.5*np.log(varpost/vk[1:])+0.5*meank*meank/varpost
        lp-=lp.max(); p=np.exp(lp); p/=p.sum()
        c=int(np.searchsorted(np.cumsum(p), u_all[j])); kcls[j]=c
        if c==0: beta[j]=0.0
        else:
            k=c-1; bj=meank[k]+np.sqrt(varpost[k])*z_all[j]
            beta[j]=bj; resid-=xj*bj
    # residual variance (scaled-inv-chi2 posterior)
    sig2e=(resid@resid + nu_e*S_e)/(nT+nu_e)
    # genetic variance: sum_{nonnull} beta_j^2 / gamma_{k_j} ~ sig2g * chi2; proper update
    nzmask=kcls>0
    if nzmask.any():
        ssq=np.sum(beta[nzmask]**2 / gamma[kcls[nzmask]])
        m1=int(nzmask.sum())
        sig2g=(ssq + nu_g*S_g)/(m1+nu_g)
    if it>=BURN: acc+=beta; nsamp+=1
    if it%100==0: print(f"  it{it} sig2e={sig2e:.3f} sig2g={sig2g:.3f} nnz={int(nzmask.sum())} ({time.time()-t1:.0f}s)",flush=True)
bhat=acc/max(1,nsamp); pred=Xte@bhat*ys+ym
pe=float(np.corrcoef(y_all[te],pred)[0,1])
print(f"\nFOLD0 BayesR Pearson={pe:.4f} | gibbs {time.time()-t1:.0f}s ({(time.time()-t1)/(BURN+SAMP):.3f}s/iter)")
