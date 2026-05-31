# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""Is the chr14:1455997 (bosTau7) fat peak tagging DGAT1 via long-range LD, or an independent proximal-BTA14
signal? DGAT1 in ARS-UCD1.2 = chr14:0.604Mb -> bosTau7 ~0.40Mb (from liftover monotone segment).
Compute r^2 of the peak vs proximal-BTA14 SNP bins. High r^2 to DGAT1-bin => tags DGAT1; low => independent."""
import csv
import numpy as np
QC = "data/processed/holstein/parse_qc"
M = np.load(f"{QC}/holstein_genotype_matrix_qc.npy").astype(np.float64)
chrom = []; pos = []
for r in csv.DictReader(open(f"{QC}/holstein_snp_metadata.csv")):
    chrom.append(r["chromosome"]); pos.append(int(r["position"]))
chrom = np.array(chrom); pos = np.array(pos)
peak = np.where((chrom == "chr14") & (pos == 1455997))[0][0]
g0 = M[:, peak] - M[:, peak].mean()
def r2col(j):
    b = M[:, j] - M[:, j].mean()
    d = np.sqrt((g0 * g0).sum() * (b * b).sum())
    return 0.0 if d == 0 else float(((g0 * b).sum() / d) ** 2)
idx = np.where(chrom == "chr14")[0]
print(f"peak col={peak} chr14:1455997 (bosTau7) ; n_chr14={len(idx)}")
print(f"{'bin (bosTau7)':>22} {'n':>5} {'max_r2':>7} {'mean_r2':>8} {'n_r2>0.2':>8}")
for label, lo, hi in [("DGAT1reg 0.30-0.55Mb", 300000, 550000), ("0.55-0.90Mb", 550000, 900000),
                      ("0.90-1.30Mb", 900000, 1300000), ("PEAK 1.30-1.60Mb", 1300000, 1600000),
                      ("1.60-2.00Mb", 1600000, 2000000)]:
    sel = idx[(pos[idx] >= lo) & (pos[idx] < hi)]
    if len(sel) == 0:
        print(f"{label:>22} {0:>5}"); continue
    rs = np.array([r2col(j) for j in sel])
    print(f"{label:>22} {len(sel):>5} {rs.max():>7.3f} {rs.mean():>8.3f} {int((rs>0.2).sum()):>8}")
# nearest DGAT1-region SNP r2 to peak (bosTau7 ~0.40Mb)
dgat = idx[(pos[idx] >= 350000) & (pos[idx] <= 500000)]
if len(dgat):
    rs = [(r2col(j), int(pos[j])) for j in dgat]
    rs.sort(reverse=True)
    print(f"top r2 of peak vs DGAT1-region SNPs (bosTau7 0.35-0.50Mb): {rs[:3]}")
