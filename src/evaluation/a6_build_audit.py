# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""A6.6 gate -- SNP map genome-build audit. Build UNKNOWN -> STOP (no annotation overlap). No silent liftover."""
import csv, glob, os
QC = "data/processed/holstein/parse_qc"
SNPMETA = f"{QC}/holstein_snp_metadata.csv"; OUT = "gate"
# autosome lengths (bp): ARS-UCD1.2 (bosTau9) vs UMD3.1 -- key chrs
ARS = {1:158534110, 2:136231102, 5:120829699, 6:117806340, 12:85358539, 14:85007780, 18:65938284, 19:64057457, 20:71974595, 25:42350435, 29:51098607}
UMD = {1:158337067, 2:137060424, 5:121191424, 6:119458736, 12:85358539, 14:84648390, 18:66004023, 19:65367414, 20:72042655, 25:42904170, 29:51505224}
mx = {}
for r in csv.DictReader(open(SNPMETA)):
    c = int(r["chromosome"].replace("chr", "")); p = int(r["position"])
    if p > mx.get(c, 0): mx[c] = p
L = ["# A6 SNP-map genome-build audit", "", "Positional SNP ids (chr:pos:ref:alt), NO rsID. Build inferred from chr-max vs assembly lengths + DGAT1 + source README.", "",
     "| chr | chip_max | ARS-UCD1.2 | UMD3.1 | max<=ARS | max<=UMD |", "|---|---|---|---|---|---|"]
ars_ok = umd_ok = True
for c in sorted(mx):
    if c not in ARS: continue
    a, u = ARS[c], UMD[c]; ao, uo = mx[c] <= a, mx[c] <= u
    ars_ok &= ao; umd_ok &= uo
    L.append(f"| {c} | {mx[c]} | {a} | {u} | {ao} | {uo} |")
L += ["", "## DGAT1 positional evidence",
      "- peak chr14:1,455,997. ARS-UCD1.2 DGAT1 ~0.61 Mb; UMD3.1 DGAT1 ~1.80 Mb. Peak (chip SNP in LD) at 1.46 Mb is closer to UMD3.1 but not definitive.",
      "", "## Source README / build-keyword search"]
hits = []
for d in [QC, QC + "/..", QC + "/../..", "data"]:
    for f in glob.glob(d + "/*.md") + glob.glob(d + "/*.txt") + glob.glob(d + "/*README*") + glob.glob(d + "/*.csv"):
        if "snp_metadata" in f or "genotype" in f: continue
        try:
            t = open(f, errors="ignore").read(200000)
            for kw in ["ARS-UCD1.2", "ARS-UCD2.0", "ARS-UCD1.3", "ARS_UCD", "UMD3.1", "UMD_3", "bosTau9", "bosTau8", "btau"]:
                if kw.lower() in t.lower(): hits.append(f"{os.path.basename(f)}: {kw}")
        except Exception: pass
L += [f"- {h}" for h in sorted(set(hits))] or ["- NO build keyword found in any README/txt/md/csv near the SNP map."]
if hits: v = "see README hits (confirm matches SNP coordinates)"
elif ars_ok and not umd_ok: v = "chr-lengths consistent with ARS-UCD1.2 ONLY"
elif umd_ok and not ars_ok: v = "chr-lengths consistent with UMD3.1 ONLY"
else: v = "UNKNOWN -- chr-lengths fit BOTH builds; DGAT1 leans UMD3.1 but not definitive"
stop = not hits and not (ars_ok ^ umd_ok)
L += ["", f"## VERDICT: {v}",
      f"## A6.6 GATE: {'STOP -- build not confirmable from on-disk evidence; annotation overlap forbidden until the original-dataset build is confirmed (no silent liftover, no forced overlap).' if stop else 'PROCEED only if the above source confirms a single build matching the SNP coordinates.'}"]
open(f"{OUT}/A6_SNP_BUILD_AUDIT.md", "w").write("\n".join(L) + "\n")
print("\n".join(L[-14:]))
