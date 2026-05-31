# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/bin/bash
# First-wave multi-trait: run budget sweep + N-control for fat (high h2) and SCS (low h2),
# SEQUENTIALLY (no GPU contention; polite tenant). Reuses same splits/genotype; only trait+outdir change.
cd .
BASE=benchmarks
rm -f $BASE/MULTITRAIT_DONE
run_trait () {
  T="$1"; TAG="$2"; D="$BASE/holstein_$TAG"; mkdir -p "$D"
  echo "[multitrait] $TAG ($T) budget $(date +%H:%M:%S)" >> "$D/run.log"
  HTRAIT="$T" HOUT="$D" nice -n 10 python3 scripts/run_budget_sweep.py >> "$D/run.log" 2>&1
  echo "[multitrait] $TAG ($T) ncontrol $(date +%H:%M:%S)" >> "$D/run.log"
  HTRAIT="$T" HOUT="$D" nice -n 10 python3 scripts/budget_Ncontrol.py >> "$D/run.log" 2>&1
  echo "[multitrait] $TAG done $(date +%H:%M:%S)" >> "$D/run.log"
}
run_trait fat_le_sum_305 fat
run_trait scs_le_ave305 scs
echo done > "$BASE/MULTITRAIT_DONE"
echo "[multitrait] ALL DONE $(date +%H:%M:%S)" >> "$BASE/holstein_fat/run.log"
