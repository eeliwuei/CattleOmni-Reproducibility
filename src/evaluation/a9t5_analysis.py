# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""A9-T5 Block A(target)+B(maxGRM-centered incremental). Target model = FullSNP_MLP.
L_easy = repeated-random-CV per-sample MSE (full2); L_hard = repeated-cluster-CV per-sample MSE
(lhard, drop <3 held-out). inflation=L_hard-L_easy. collapse= easy<median & hard>q75.
A5 cluster ANOVA R2 (ceiling). B: scalar baselines (maxGRM headline) vs scalars+genome-local,
ΔSpearman/ΔAUROC + bootstrap CI; B4 partial-Spearman | maxGRM. seed=20260529."""
import csv, os, numpy as np
RNG=np.random.default_rng(20260529)
B="benchmarks/holstein"; OUT="scrl_mini"
MODEL="FullSNP_MLP_top50k"
def mse_by_sample(path, filt):
    d={}
    for r in csv.DictReader(open(path)):
        if filt(r):
            s=int(r["sample_id"]); e=(float(r["y_pred"])-float(r["y_true_std"]))**2; d.setdefault(s,[]).append(e)
    return d
easy=mse_by_sample(f"{B}/fold_predictions_full2.csv", lambda r: r["mode"]=="strict" and r["family"]=="random" and r["model"]==MODEL)
hard=mse_by_sample(f"{OUT}/lhard_predictions.csv", lambda r: r["model"]==MODEL)
ne=np.array([len(v) for v in easy.values()]); nh=np.array([len(v) for v in hard.values()])
print(f"L_easy: {len(easy)} samples, held-out/sample min={ne.min()} med={int(np.median(ne))}")
print(f"L_hard: {len(hard)} samples, held-out/sample min={nh.min()} med={int(np.median(nh))}")
S=sorted(set(s for s,v in easy.items() if len(v)>=8) & set(s for s,v in hard.items() if len(v)>=3))
le=np.array([np.mean(easy[s]) for s in S]); lh=np.array([np.mean(hard[s]) for s in S])
infl=lh-le; cliff=((le<np.median(le))&(lh>np.quantile(lh,0.75))).astype(int); collapse=(infl>np.quantile(infl,0.75)).astype(int)
print(f"target samples={len(S)} collapse(top-q-infl)={collapse.mean():.3f} CLIFF(good->bad)={cliff.mean():.4f} inflation mean={infl.mean():.3f}")
# features
F={}
hdr=None
for r in csv.DictReader(open(f"{OUT}/prebefore_features.csv")):
    F[int(r["sample_id"])]={k:float(r[k]) for k in r if k!="sample_id"}
fn_scalar=["maxGRM","mean_top5_GRM","mean_top10_GRM","neighbor_entropy","pca_dist"]
fn_local=["local_max","local_mean","frac_high","longest_high_run","spike_count"]
Xs=np.array([[F[s][k] for k in fn_scalar] for s in S]); Xl=np.array([[F[s][k] for k in fn_local] for s in S])
maxg=np.array([F[s]["maxGRM"] for s in S])
# A5 cluster ANOVA R2
cm={}; p="splits/holstein/relationship_group/group_5fold/cluster_membership.csv"
h=next(csv.reader(open(p))); sc=[c for c in h if "sample" in c.lower()][0]; cc=[c for c in h if "cluster" in c.lower()][0]
for r in csv.DictReader(open(p)): cm[int(r[sc])]=int(r[cc])
cl=np.array([cm[s] for s in S]); g=infl.mean(); sstot=((infl-g)**2).sum()
ssb=sum((infl[cl==c].mean()-g)**2*np.sum(cl==c) for c in set(cl)); R2c=ssb/sstot
print(f"A5 cluster-ANOVA R2(inflation~cluster)={R2c:.4f}  (ceiling for any per-sample diagnoser)")
def rank(a): o=np.argsort(a); r=np.empty(len(a)); r[o]=np.arange(len(a)); return r
def spearman(a,b): ra,rb=rank(a),rank(b); return float(np.corrcoef(ra,rb)[0,1])
def auroc(y,s):
    n1=int(y.sum()); n0=int((1-y).sum());
    if n1==0 or n0==0: return float('nan')
    r=rank(s); return float((r[y==1].sum()-n1*(n1-1)/2)/(n1*n0))
def ridge_oof(X,y,lam=1.0,k=5):
    n=len(y); idx=RNG.permutation(n); fo=np.array_split(idx,k); oof=np.zeros(n)
    for f in range(k):
        te=fo[f]; tr=np.concatenate([fo[m] for m in range(k) if m!=f])
        mu=X[tr].mean(0); sd=X[tr].std(0)+1e-8; Xtr=(X[tr]-mu)/sd; Xte=(X[te]-mu)/sd
        ym=y[tr].mean(); w=np.linalg.solve(Xtr.T@Xtr+lam*np.eye(X.shape[1]), Xtr.T@(y[tr]-ym)); oof[te]=Xte@w+ym
    return oof
print("\n=== B2 single prediction-before features ===")
for k,fk in enumerate(fn_scalar+fn_local):
    col=np.concatenate([Xs,Xl],1)[:,k]
    print(f"  {fk:18} Spearman_vs_inflation={spearman(col,infl):+.4f}  AUROC_collapse={auroc(collapse,col):.4f}")
# B3 incremental
o1=ridge_oof(Xs,infl); o2=ridge_oof(np.concatenate([Xs,Xl],1),infl)
sp1,sp2=spearman(o1,infl),spearman(o2,infl); au1,au2=auroc(collapse,o1),auroc(collapse,o2)
# bootstrap CI on deltas
ds=[]; da=[]
for _ in range(1000):
    bi=RNG.integers(0,len(S),len(S))
    ds.append(spearman(o2[bi],infl[bi])-spearman(o1[bi],infl[bi]))
    da.append(auroc(collapse[bi],o2[bi])-auroc(collapse[bi],o1[bi]))
ds=np.array(ds); da=np.array(da)
# B4 partial spearman | maxGRM (rank-residualize)
rg=rank(maxg); 
def resid_on(x): rx=rank(x); b=np.polyfit(rg,rx,1); return rx-(b[0]*rg+b[1])
ps=float(np.corrcoef(resid_on(o2),resid_on(infl))[0,1])
print("\n=== B3 ★INCREMENTAL TEST★ (scalars vs scalars+genome-local) ===")
print(f"  model1 scalars:        Spearman={sp1:.4f}  AUROC_collapse={au1:.4f}")
print(f"  model2 scalars+local:  Spearman={sp2:.4f}  AUROC_collapse={au2:.4f}")
print(f"  ΔSpearman={sp2-sp1:+.4f}  95%CI[{np.percentile(ds,2.5):+.4f},{np.percentile(ds,97.5):+.4f}]")
print(f"  ΔAUROC   ={au2-au1:+.4f}  95%CI[{np.percentile(da,2.5):+.4f},{np.percentile(da,97.5):+.4f}]")
print(f"\n=== B4 partial Spearman(model2, inflation | maxGRM) = {ps:+.4f}  (~0 => just re-deriving maxGRM)")
print(f"=== headline maxGRM alone: Spearman_vs_inflation={spearman(maxg,infl):+.4f}  AUROC_collapse={auroc(collapse,maxg):.4f}")
with open(f"{OUT}/diagnoser_vs_maxgrm.csv","w") as f:
    f.write("metric,value\n")
    f.write(f"n_target_samples,{len(S)}\ncollapse_rate,{collapse.mean():.4f}\ncluster_R2_ceiling,{R2c:.4f}\n")
    f.write(f"maxGRM_spearman,{spearman(maxg,infl):.4f}\nmaxGRM_auroc_collapse,{auroc(collapse,maxg):.4f}\n")
    f.write(f"model1_scalars_spearman,{sp1:.4f}\nmodel2_local_spearman,{sp2:.4f}\ndelta_spearman,{sp2-sp1:.4f}\n")
    f.write(f"delta_spearman_lo,{np.percentile(ds,2.5):.4f}\ndelta_spearman_hi,{np.percentile(ds,97.5):.4f}\n")
    f.write(f"model1_auroc,{au1:.4f}\nmodel2_auroc,{au2:.4f}\ndelta_auroc,{au2-au1:.4f}\n")
    f.write(f"delta_auroc_lo,{np.percentile(da,2.5):.4f}\ndelta_auroc_hi,{np.percentile(da,97.5):.4f}\n")
    f.write(f"partial_spearman_given_maxGRM,{ps:.4f}\n")
print("saved diagnoser_vs_maxgrm.csv")
