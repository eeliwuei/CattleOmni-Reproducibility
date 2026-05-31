# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""A6.3-A6.5 + A6.8 (frozen protocol). PRIMARY trait = fat_le_ave_305 (fat le-average composition trait,
NOT 'fat %'). Fold-internal ADDITIVE ridge importance per regime (random / group5 / group10), aggregated
to LD-block loci (max|beta|), averaged over folds. Stability = top-K(100) consistency across regimes.
Classes: regime-stable / random-only / strict-only(exploratory). DGAT1 locus pre-annotation check.
Frozen params: top-K primary=100, secondary=50; importance from additive shrinkage only (NO deep attribution);
NO outer-test info used (importance from training fits). lambda fixed=1 on per-SNP-normalised GRM kernel.
Pure numpy. Outputs to gate/."""
import os, csv, glob, time
import numpy as np
QC = "data/processed/holstein/parse_qc"
GENO = f"{QC}/holstein_genotype_matrix_qc.npy"; SNPMETA = f"{QC}/holstein_snp_metadata.csv"; ALIGN = f"{QC}/holstein_sample_alignment.csv"
PHE = "data/holstein_phenotype_table.csv"
SP = "splits/holstein"; OUT = "gate"
TRAIT = "fat_le_ave_305"; TOPK = 100; TOPK2 = 50; LAM = 1.0
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
M = np.load(GENO).astype(np.float64); n, m = M.shape
p = M.mean(0) / 2.0; sd = np.sqrt(2*p*(1-p)); sd[sd < 1e-9] = 1e9; Z = (M - 2*p) / sd
s2l = np.load(f"{OUT}/A6_SNP2LOCUS.npy"); nloc = int(s2l.max()) + 1
chrom = []; pos = []
for r in csv.DictReader(open(SNPMETA)): chrom.append(r["chromosome"]); pos.append(int(r["position"]))
chrom = np.array(chrom); pos = np.array(pos)
rows_al = sorted([(int(d["sample_id"]), int(d["order"])) for d in csv.DictReader(open(ALIGN))], key=lambda x: x[1])
ids = [s for s, _ in rows_al]; row_of = {s: i for i, s in enumerate(ids)}
phe = {int(d["ID"]): d for d in csv.DictReader(open(PHE))}
y = np.array([float(phe[ids[i]][TRAIT]) for i in range(n)])
log(f"M{M.shape} loci={nloc} trait={TRAIT}")
def read_idx(pth): return np.array([row_of[int(float(l.split(',')[0]))] for l in open(pth).read().splitlines()[1:] if l.strip()])
REG = {"random": sorted(glob.glob(f"{SP}/random/*_train.csv")),
       "group5": sorted(glob.glob(f"{SP}/relationship_group/group_5fold/*_train.csv")),
       "group10": sorted(glob.glob(f"{SP}/relationship_group/group_10fold/*_train.csv"))}
def locus_imp_fold(tr):
    Zt = Z[tr]; yt = (y[tr] - y[tr].mean()) / (y[tr].std() + 1e-9)
    K = (Zt @ Zt.T) / m; al = np.linalg.solve(K + LAM*np.eye(len(tr)), yt)
    beta = (Zt.T @ al) / m                      # additive per-SNP coef
    li = np.zeros(nloc); np.maximum.at(li, s2l, np.abs(beta)); return li
imp = {}
for reg, files in REG.items():
    acc = np.zeros(nloc)
    for f in files: acc += locus_imp_fold(read_idx(f))
    imp[reg] = acc / len(files); log(f"  {reg}: {len(files)} folds")
def topk_set(v, k): return set(np.argsort(-v)[:k].tolist())
rand_top, g5_top, g10_top = topk_set(imp["random"], TOPK), topk_set(imp["group5"], TOPK), topk_set(imp["group10"], TOPK)
# A6.3 importance + rank
def ranks(v): r = np.empty(nloc, int); r[np.argsort(-v)] = np.arange(1, nloc+1); return r
rr, r5, r10 = ranks(imp["random"]), ranks(imp["group5"]), ranks(imp["group10"])
with open(f"{OUT}/A6_FOLD_INTERNAL_IMPORTANCE.csv", "w") as f:
    f.write("locus_id,imp_random,imp_group5,imp_group10,rank_random,rank_group5,rank_group10\n")
    for L in range(nloc): f.write(f"{L},{imp['random'][L]:.6g},{imp['group5'][L]:.6g},{imp['group10'][L]:.6g},{rr[L]},{r5[L]},{r10[L]}\n")
# A6.4 stability score
with open(f"{OUT}/A6_REGIME_STABILITY_SCORE.csv", "w") as f:
    f.write("locus_id,n_regimes_top100,min_rank,in_random_top,in_group5_top,in_group10_top\n")
    for L in range(nloc):
        nt = int(L in rand_top) + int(L in g5_top) + int(L in g10_top); mr = min(rr[L], r5[L], r10[L])
        f.write(f"{L},{nt},{mr},{int(L in rand_top)},{int(L in g5_top)},{int(L in g10_top)}\n")
# A6.5 signal classes (relationship-aware = group5 primary; group10 reported)
def cls(L):
    rnd = L in rand_top; strict = L in g5_top
    if rnd and strict: return "regime-stable"
    if rnd and not strict: return "random-only"
    if strict and not rnd: return "strict-only-exploratory"
    return "none"
with open(f"{OUT}/A6_SIGNAL_CLASSES.csv", "w") as f:
    f.write("locus_id,signal_class\n")
    for L in range(nloc):
        c = cls(L)
        if c != "none": f.write(f"{L},{c}\n")
nstab = sum(1 for L in range(nloc) if cls(L) == "regime-stable")
nrand = sum(1 for L in range(nloc) if cls(L) == "random-only")
nstr = sum(1 for L in range(nloc) if cls(L) == "strict-only-exploratory")
log(f"classes (top{TOPK}): regime-stable={nstab} random-only={nrand} strict-only={nstr}")
# A6.8 DGAT1 pre-annotation (peak chr14:1455997 -> its locus)
dg = np.where((chrom == "chr14") & (pos == 1455997))[0]
L = []
if len(dg):
    Ld = int(s2l[dg[0]]); cl = cls(Ld)
    L = [f"# A6.8 DGAT1 positive-control (pre-annotation)\nPeak SNP chr14:1455997 -> locus_id {Ld}\n",
         f"- importance: random={imp['random'][Ld]:.4g}(rank {rr[Ld]}), group5={imp['group5'][Ld]:.4g}(rank {r5[Ld]}), group10={imp['group10'][Ld]:.4g}(rank {r10[Ld]})",
         f"- in_random_top100={Ld in rand_top}, in_group5_top100={Ld in g5_top}, in_group10_top100={Ld in g10_top}",
         f"- **signal class = {cl}**",
         f"- interpretation: regime-stable=ideal positive control; random-only=relatedness-shortcut warning; not-top-K=check trait/build/importance."]
    open(f"{OUT}/A6_DGAT1_POSITIVE_CONTROL_PREANNOTATION.md", "w").write("\n".join(L) + "\n")
    log(f"DGAT1 locus {Ld} class={cl} ranks r{rr[Ld]}/g5 {r5[Ld]}/g10 {r10[Ld]}")
else:
    log("DGAT1 peak SNP not found")
open(f"{OUT}/ALL_DONE_stability", "w").write("done\n"); log("A6.3-A6.5 + A6.8 DONE")
