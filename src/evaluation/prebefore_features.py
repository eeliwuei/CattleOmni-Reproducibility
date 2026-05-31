# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""A9-T5 Block B inputs: prediction-BEFORE features (no y/y_pred/residual).
Scalars: maxGRM, mean_top5/10_GRM, neighbor_entropy, pca_distance.
Genome-local (fast path): per-sample along-genome nearest-neighbor LOCAL-RELATEDNESS trajectory
= per-block standardized-GENOTYPE correlation to nearest other sample (captures local haplotype
sharing; NOT block-summary which saturates) -> local_max/mean, frac_high, longest_high_run,
spike_count. seed=20260529."""
import sys, csv, time, os
sys.path.insert(0,"scripts")
import numpy as np, run_holstein_benchmark as H
OUT="scrl_mini"; GB="genome_behavior"; os.makedirs(OUT,exist_ok=True)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
row_for_id, ids_by_row = H.load_alignment(); n=len(ids_by_row)
M=np.load(H.GENO).astype(np.float32); log("GRM..."); G=H.vanraden_grm(M)
Gd=G.copy(); np.fill_diagonal(Gd,-np.inf); srt=np.sort(Gd,axis=1)[:,::-1]
maxGRM=srt[:,0]; top5=srt[:,:5].mean(1); top10=srt[:,:10].mean(1)
tk=np.clip(srt[:,:20],1e-6,None); pk=tk/tk.sum(1,keepdims=True); nent=-(pk*np.log(pk)).sum(1)
Xc=M-M.mean(0); Kx=Xc@Xc.T; w,V=np.linalg.eigh(Kx); w=w[::-1]; V=V[:,::-1]
Z=V[:,:10]*np.sqrt(np.clip(w[:10],1e-9,None)); pca_dist=np.sqrt(((Z-Z.mean(0))**2).sum(1)); log("scalars done")
# per-block GENOTYPE local relatedness
chrom=[]
for d in csv.DictReader(open(f"{H.QC}/holstein_snp_metadata.csv")): chrom.append(int(d["chromosome"].replace("chr","")))
chrom=np.array(chrom); order=np.load(f"{GB}/genome_order.npy"); chrom_o=chrom[order]; Mo=M[:,order]
blocks=[]
for ch in range(1,30):
    idx=np.where(chrom_o==ch)[0]
    for s in range(0,len(idx),256): blocks.append(idx[s:s+256])
nB=len(blocks); loc=np.zeros((n,nB),dtype=np.float32); log(f"nB={nB} computing per-block genotype NN-corr...")
for b,cols in enumerate(blocks):
    Xb=Mo[:,cols]; mu=Xb.mean(0); sd=Xb.std(0)+1e-8; Zb=(Xb-mu)/sd
    Sb=(Zb@Zb.T)/Xb.shape[1]; np.fill_diagonal(Sb,-np.inf); loc[:,b]=Sb.max(1)
thr=float(np.quantile(loc,0.90)); above=loc>thr
frac=above.mean(1); lmax=loc.max(1); lmean=loc.mean(1)
def lr(a):
    best=cur=0
    for v in a: cur=cur+1 if v else 0; best=max(best,cur)
    return best
lrun=np.array([lr(above[i]) for i in range(n)])
spike=np.array([int(((loc[i,1:-1]>loc[i,:-2])&(loc[i,1:-1]>loc[i,2:])&(loc[i,1:-1]>thr)).sum()) for i in range(n)])
log(f"local done thr={thr:.4f}  loc range [{loc.min():.3f},{loc.max():.3f}] mean={loc.mean():.3f}")
with open(f"{OUT}/prebefore_features.csv","w") as f:
    f.write("sample_id,maxGRM,mean_top5_GRM,mean_top10_GRM,neighbor_entropy,pca_dist,local_max,local_mean,frac_high,longest_high_run,spike_count\n")
    for i in range(n):
        f.write(f"{ids_by_row[i]},{maxGRM[i]:.6f},{top5[i]:.6f},{top10[i]:.6f},{nent[i]:.6f},{pca_dist[i]:.6f},{lmax[i]:.6f},{lmean[i]:.6f},{frac[i]:.6f},{lrun[i]},{spike[i]}\n")
log(f"FEATURES DONE -> {OUT}/prebefore_features.csv")
