# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""A6.2 LD-block / clumped-locus construction (frozen params: r^2<0.2, window 1Mb).
Trait-independent loci (greedy distance+LD clumping by position order) -> unbiased units for stability
+ matched permutation. Output A6_LD_BLOCK_INDEX.csv. Pure numpy."""
import os, csv, time
import numpy as np
QC = "data/processed/holstein/parse_qc"
GENO = f"{QC}/holstein_genotype_matrix_qc.npy"; SNPMETA = f"{QC}/holstein_snp_metadata.csv"
OUT = "gate"; os.makedirs(OUT, exist_ok=True)
R2 = 0.2; WIN = 1_000_000
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
M = np.load(GENO).astype(np.float64); n, m = M.shape
mu = M.mean(0); sdv = M.std(0); sdv[sdv < 1e-9] = 1e9
Zc = (M - mu) / sdv                                  # standardized cols (corr = (1/n) Zc_i . Zc_j)
chrom = []; pos = []; maf = []
for r in csv.DictReader(open(SNPMETA)):
    chrom.append(r["chromosome"]); pos.append(int(r["position"])); maf.append(float(r["maf"]))
chrom = np.array(chrom); pos = np.array(pos); maf = np.array(maf)
chrs = sorted(set(chrom.tolist()), key=lambda c: int(c.replace("chr", "")))
log(f"M{M.shape}; {len(chrs)} chr; r2<{R2} window {WIN}")
rows = []; lid = 0; snp2locus = np.full(m, -1, dtype=int)
for c in chrs:
    idx = np.where(chrom == c)[0]; idx = idx[np.argsort(pos[idx])]
    used = np.zeros(len(idx), bool)
    P = pos[idx]
    for a in range(len(idx)):
        if used[a]: continue
        ix = idx[a]; zi = Zc[:, ix]
        # candidates within +/- WIN of index, not used
        lo = np.searchsorted(P, P[a] - WIN); hi = np.searchsorted(P, P[a] + WIN)
        members = [ix]; mn = mx = P[a]
        cand = [b for b in range(lo, hi) if b != a and not used[b]]
        if cand:
            cols = idx[np.array(cand)]
            r2v = (Zc[:, cols].T @ zi / n) ** 2          # vectorized r^2 of index vs all window candidates
            for b, ok in zip(cand, r2v >= R2):
                if ok:
                    used[b] = True; members.append(idx[b]); mn = min(mn, P[b]); mx = max(mx, P[b])
        used[a] = True
        for mm in members: snp2locus[mm] = lid
        span = int(mx - mn) + 1
        rows.append((lid, f"{c}:{int(pos[ix])}", c, int(mn), int(mx), span, len(members), round(float(maf[ix]), 4), round(len(members) / (span / 1e6 + 1e-9), 2)))
        lid += 1
    log(f"  {c}: {len(idx)} snp -> {sum(1 for r in rows if r[2]==c)} loci")
with open(f"{OUT}/A6_LD_BLOCK_INDEX.csv", "w") as f:
    f.write("locus_id,index_snp,chromosome,start_pos,end_pos,span_bp,n_snps,index_maf,snp_density_per_Mb\n")
    for r in rows: f.write(",".join(map(str, r)) + "\n")
ns = np.array([r[6] for r in rows])
log(f"DONE: {len(rows)} LD-blocks; median n_snp/block={int(np.median(ns))}; singletons={int((ns==1).sum())}")
np.save(f"{OUT}/A6_SNP2LOCUS.npy", snp2locus)
open(f"{OUT}/ALL_DONE_ldblocks", "w").write(f"{len(rows)}\n")
