# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""Robustness of build inference: is the chr-length overflow a BULK distribution (=> genuinely longer-chr
old build) or 1-2 outliers (=> mismapped SNPs over an UMD3.1 bulk)? Authoritative ARS-UCD1.2 lengths from GFF."""
import csv
from collections import defaultdict
SNPMETA = "data/processed/holstein/parse_qc/holstein_snp_metadata.csv"
# authoritative ARS-UCD1.2 (from on-disk GFF header NC_0373xx.1) + UMD3.1
ARS = {1:158534110, 14:85007780, 18:65820629, 19:63449741, 20:71974595, 25:42350435, 29:51098607}
UMD = {1:158337067, 14:84648390, 18:66004023, 19:65367414, 20:72042655, 25:42904170, 29:51505224}
pos = defaultdict(list)
for r in csv.DictReader(open(SNPMETA)):
    pos[int(r["chromosome"].replace("chr", ""))].append(int(r["position"]))
def pct(a, q):
    b = sorted(a); return b[min(len(b)-1, int(len(b)*q/100))]
print(f"{'chr':>3} {'n_snp':>7} {'max':>10} {'p99.9':>10} {'n>ARS':>7} {'n>UMD':>7} {'ARSlen':>10} {'UMDlen':>10}")
for c in sorted(pos):
    if c not in ARS: continue
    a = pos[c]; nA = sum(1 for x in a if x > ARS[c]); nU = sum(1 for x in a if x > UMD[c])
    print(f"{c:>3} {len(a):>7} {max(a):>10} {pct(a,99.9):>10} {nA:>7} {nU:>7} {ARS[c]:>10} {UMD[c]:>10}")
