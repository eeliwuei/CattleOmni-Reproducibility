# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
import csv, numpy as np
B = "benchmarks"
TR = {"milk_le_sum_305": "holstein", "fat_le_sum_305": "holstein_fat", "scs_le_ave305": "holstein_scs"}
g = {}
for r in csv.DictReader(open(B + "/tuning/dl_tuning_grid.csv")):
    if r["r2_vs_mean"] in ("ERR", ""): continue
    g.setdefault((r["trait"], r["cfg"]), []).append(float(r["r2_vs_mean"]))
def linbest(d):
    vals = {}
    for r in csv.DictReader(open(B + "/" + d + "/budget_Ncontrol.csv")):
        if r["tag"] == "t100" and r["r2_vs_mean"] not in ("ERR", ""):
            vals.setdefault(r["model"], []).append(float(r["r2_vs_mean"]))
    return {m: float(np.mean(v)) for m, v in vals.items()}
print("=== #2 fair-tuning readout (matched-N=400, tau=t100=0.10); DL grid-MAX = optimistic upper bound (selected on outer test) ===")
for tr, d in TR.items():
    cfgm = {c: float(np.mean(v)) for (t, c), v in g.items() if t == tr}
    dlmax = max(cfgm.values()); dlcfg = max(cfgm, key=cfgm.get)
    lin = linbest(d); linmax = max([lin.get(m, -9) for m in ["GBLUP_full", "Ridge_top50k", "PCARidge_PCA80"]])
    if dlmax < linmax - 0.005: v = "DL < linear  => undertuning REFUTED (best of 12 cfg, test-optimal, still loses)"
    elif dlmax > linmax + 0.005: v = "DL >= linear => tuning matters; soften model-class claim"
    else: v = "~tie"
    print("%-18s DLmax=%+.3f(cfg%s)  bestLinear=%+.3f   %s" % (tr, dlmax, dlcfg, linmax, v))
    print("   12-cfg DL means: " + ", ".join("c%s:%+.2f" % (c, m) for c, m in sorted(cfgm.items(), key=lambda x: int(x[0]))))
