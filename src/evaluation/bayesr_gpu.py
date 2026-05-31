# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
"""BayesR (4-class mixture) GPU Gibbs, SYNC-FREE inner loop (no .item()/.cpu()).
top-K SNP by train-fold variance. Times one random fold; if fast, scale to ladder.
Real Gibbs, no fabrication. Usage: python3 bayesr_gpu.py [TOPK] [BURN] [SAMP]"""
import numpy as np, csv, time, glob, sys
import torch
GENO="data/processed/holstein/parse_qc/holstein_genotype_matrix_qc.npy"
PHE="data/processed/holstein/parse_qc/holstein_phenotype_table.csv"
SP="splits/holstein/random"
TRAIT="milk_le_sum_305"
TOPK=int(sys.argv[1]) if len(sys.argv)>1 else 20000
BURN=int(sys.argv[2]) if len(sys.argv)>2 else 500
SAMP=int(sys.argv[3]) if len(sys.argv)>3 else 500
dev="cuda"; torch.manual_seed(7); np.random.seed(7)
t0=time.time()
rows=list(csv.DictReader(open(PHE))); rows.sort(key=lambda r: float(r["order"]))
y_all=np.array([float(r[TRAIT]) for r in rows],dtype=np.float32)
id2row={int(float(r["sample_id"])):i for i,r in enumerate(rows)}
trp=sorted(glob.glob(f"{SP}/*_train.csv"))[0]; tep=trp.replace("_train.csv","_test.csv")
rd=lambda p:[int(float(l.split(",")[0])) for l in open(p).read().splitlines()[1:] if l.strip()]
tr=np.array(sorted(id2row[i] for i in rd(trp) if i in id2row))
te=np.array(sorted(id2row[i] for i in rd(tep) if i in id2row))
X=np.load(GENO,mmap_mode="r")
Xtr=np.asarray(X[tr],dtype=np.float32); top=np.argsort(-Xtr.var(0))[:TOPK]
Xtr=Xtr[:,top]; Xte=np.asarray(X[te],dtype=np.float32)[:,top]
mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6; Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
ytr=y_all[tr]; ym=ytr.mean(); ys=ytr.std()+1e-9; ytrn=(ytr-ym)/ys
print(f"setup {time.time()-t0:.0f}s | train {len(tr)} test {len(te)} K={TOPK}",flush=True)
Xg=torch.tensor(Xtr,device=dev); yg=torch.tensor(ytrn,device=dev); Xteg=torch.tensor(Xte,device=dev)
nT,m=Xg.shape; xtx=(Xg*Xg).sum(0)
beta=torch.zeros(m,device=dev); resid=yg.clone()
gamma=torch.tensor([0.0,1e-4,1e-3,1e-2],device=dev)
logpi=torch.log(torch.tensor([0.95,0.02,0.02,0.01],device=dev))
sig2e=torch.tensor(1.0,device=dev); sig2g=torch.tensor(0.5,device=dev)
acc=torch.zeros(m,device=dev); nsamp=0; t1=time.time()
for it in range(BURN+SAMP):
    vk=gamma*sig2g  # (4,) class variances
    for j in range(m):
        xj=Xg[:,j]; rj=resid+xj*beta[j]; rhs=(xj*rj).sum()
        denom=xtx[j]+sig2e/vk[1:]          # (3,) non-null classes
        meank=rhs/denom; varpost=sig2e/denom
        lp=torch.empty(4,device=dev)
        lp[0]=logpi[0]
        lp[1:]=logpi[1:]+0.5*torch.log(varpost/vk[1:])+0.5*meank*meank/varpost
        p=torch.softmax(lp,0)
        cum=torch.cumsum(p,0); u=torch.rand((),device=dev)
        kk=(u>cum).sum()                    # class index 0..3, GPU, no sync
        nz=(kk>0).float()
        idx=torch.clamp(kk-1,0,2)
        bj=nz*(meank[idx]+torch.sqrt(varpost[idx])*torch.randn((),device=dev))
        beta[j]=bj; resid=rj-xj*bj
    sig2e=(resid*resid).sum()/nT
    nzm=beta[beta!=0]; sig2g=(nzm*nzm).mean() if nzm.numel()>0 else torch.tensor(0.5,device=dev)
    if it>=BURN: acc+=beta; nsamp+=1
    if it%100==0: print(f"  it{it} sig2e={float(sig2e):.3f} nnz={int((beta!=0).sum())} ({time.time()-t1:.0f}s)",flush=True)
bhat=acc/max(1,nsamp); pred=(Xteg@bhat).cpu().numpy()*ys+ym
pe=float(np.corrcoef(y_all[te],pred)[0,1])
print(f"\nFOLD0 BayesR Pearson={pe:.4f} | gibbs {time.time()-t1:.0f}s ({(time.time()-t1)/(BURN+SAMP):.2f}s/iter)")
