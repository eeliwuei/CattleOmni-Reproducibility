# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""Relatedness-aware LMM-GWAS, FaST-LMM-style with LOCO (leave-one-chromosome-out).
GRM (VanRaden) as random effect; per-chromosome GRM excludes the tested chromosome so the SNP
is not in its own background. NOT naive SNP+PC (which inflates on inbred Holstein). Pure numpy
(scipy/sklearn broken on H100). Outputs per-SNP -log10(p), genomic inflation lambda, top hits.
Deterministic (no RNG). Usage: gwas_lmm_loco.py <trait1> [trait2 ...]"""
import os, sys, csv, math, time
import numpy as np
QC = "data/processed/holstein/parse_qc"
GENO = f"{QC}/holstein_genotype_matrix_qc.npy"; SNPMETA = f"{QC}/holstein_snp_metadata.csv"
ALIGN = f"{QC}/holstein_sample_alignment.csv"; PHE = "data/holstein_phenotype_table.csv"
OUT = "gate"; os.makedirs(OUT, exist_ok=True)
TRAITS = sys.argv[1:] or ["fat_le_ave_305"]
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# ---- genotype + standardize (VanRaden) ----
M = np.load(GENO).astype(np.float64); n, m = M.shape
p = M.mean(0) / 2.0; sd = np.sqrt(2 * p * (1 - p)); sd[sd < 1e-9] = 1e9
Z = (M - 2 * p) / sd                          # n x m standardized
chrom = []; pos = []
for r in csv.DictReader(open(SNPMETA)): chrom.append(r["chromosome"]); pos.append(int(r["position"]))
chrom = np.array(chrom); pos = np.array(pos, float)
chrs = sorted(set(chrom.tolist()), key=lambda c: int(c.replace("chr", "")))
log(f"M{M.shape}; {len(chrs)} chromosomes")
ZZt = Z @ Z.T                                  # n x n unscaled (sum over ALL snps)
maskc = {c: (chrom == c) for c in chrs}
ZcZct = {c: (Z[:, maskc[c]] @ Z[:, maskc[c]].T) for c in chrs}
ncs = {c: int(maskc[c].sum()) for c in chrs}

# ---- id<->row ----
rows_al = sorted([(int(d["sample_id"]), int(d["order"])) for d in csv.DictReader(open(ALIGN))], key=lambda x: x[1])
ids_by_row = [s for s, _ in rows_al]; row_of_id = {s: i for i, s in enumerate(ids_by_row)}
phe = {int(d["ID"]): d for d in csv.DictReader(open(PHE))}

def neglog10p_chi2_1(x):                        # stable -log10 p for 1-df chi2 (x = t^2)
    if x <= 0: return 0.0
    z = math.sqrt(x) / math.sqrt(2.0)
    if z < 25:
        pp = math.erfc(z); return -math.log10(pp) if pp > 0 else (z*z + math.log(z*math.sqrt(math.pi)))/math.log(10)
    return (z*z + math.log(z*math.sqrt(math.pi))) / math.log(10)

def reml_delta(L, yt, x1t):
    n1 = len(yt) - 1
    def ll(d):
        w = 1.0/(L+d); a = np.sum(w*x1t*x1t); bh = np.sum(w*x1t*yt)/a; r = yt-bh*x1t
        s2 = np.sum(w*r*r)/n1
        return -0.5*(n1*math.log(2*math.pi*s2)+np.sum(np.log(L+d))+math.log(a)+n1)
    best=(1.0,-1e18)
    for ld in np.linspace(-6,6,49):
        d=math.exp(ld); v=ll(d)
        if v>best[1]: best=(d,v)
    for ld in np.linspace(math.log(best[0])-0.4,math.log(best[0])+0.4,41):
        d=math.exp(ld); v=ll(d)
        if v>best[1]: best=(d,v)
    return best[0]

for TR in TRAITS:
    y = np.array([float(phe[ids_by_row[i]][TR]) for i in range(n)])
    log(f"=== {TR}: n={n} mean={y.mean():.3f} std={y.std():.3f} ===")
    res = []  # (snp_idx, chi2)
    for c in chrs:
        Kloco = (ZZt - ZcZct[c]) / max(1, (m - ncs[c]))
        Lval, U = np.linalg.eigh(Kloco); Lval = np.clip(Lval, 1e-9, None)
        yt = U.T @ y; x1t = U.T @ np.ones(n)
        d = reml_delta(Lval, yt, x1t); w = 1.0/(Lval+d)
        a = float(np.sum(w*x1t*x1t)); Sx1y = float(np.sum(w*x1t*yt)); Syy = float(np.sum(w*yt*yt))
        Zc = Z[:, maskc[c]]; Zct = U.T @ Zc                       # n x ncs[c]
        b = (w*x1t) @ Zct; Sxx = (w) @ (Zct*Zct); Sxy = (w*yt) @ Zct
        denom = Sxx - b*b/a; denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)
        slope = (Sxy - b*Sx1y/a)/denom
        Syy_adj = Syy - Sx1y*Sx1y/a
        rss = Syy_adj - slope*slope*denom; rss = np.clip(rss, 1e-12, None)
        s2 = rss/(n-2); se = np.sqrt(s2/denom); t = slope/se; chi2 = t*t
        idxs = np.where(maskc[c])[0]
        for j, ii in enumerate(idxs):
            if np.isfinite(chi2[j]): res.append((int(ii), float(chi2[j]), float(d)))
        log(f"  {c}: nsnp={ncs[c]} delta={d:.3f} h2~{1/(1+d):.2f} maxchi2={np.nanmax(chi2):.1f}")
    chi2all = np.array([r[1] for r in res]); lam = float(np.median(chi2all)/0.4549)
    order = sorted(res, key=lambda r: -r[1])
    # write per-snp
    fn = f"{OUT}/gwas_{TR}.csv"
    with open(fn, "w") as f:
        f.write("snp_id,chromosome,position,maf,chi2,neglog10p\n")
        for ii, ch2, _ in res:
            f.write(f"{chrom[ii]}:{int(pos[ii])},{chrom[ii]},{int(pos[ii])},{p[ii]*2 if False else ''},{ch2:.4f},{neglog10p_chi2_1(ch2):.4f}\n")
    log(f"  -> {fn}; lambda_GC={lam:.3f}; top5:")
    for ii, ch2, _ in order[:5]:
        log(f"     {chrom[ii]}:{int(pos[ii])}  chi2={ch2:.1f}  -log10p={neglog10p_chi2_1(ch2):.2f}")
    open(f"{OUT}/gwas_{TR}_lambda.txt","w").write(f"lambda_GC={lam:.4f}\nn_snp={len(res)}\n")
log("GWAS LMM-LOCO done")
