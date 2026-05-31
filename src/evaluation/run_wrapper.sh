# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/bin/bash
# Auto-restart wrapper: relaunch the resumable Holstein benchmark until ALL_DONE sentinel appears.
# Robust to silent SIGKILLs (shared-host OOM/contention) — each attempt resumes from done folds.
cd .
B=benchmarks/holstein
rm -f "$B/ALL_DONE_full"
for i in $(seq 1 40); do
  echo "[wrapper] attempt $i start $(date +%H:%M:%S)" >> "$B/run_full.log"
  nice -n 10 python3 scripts/run_holstein_benchmark.py --mode both \
    --families random,group,lco,budget \
    --models GBLUP_full,Ridge_top50k,Ridge_top20k,PCARidge_PCA80,FullSNP_MLP_top50k,SparseGate_MLP_top20k \
    --tag full >> "$B/run_full.log" 2>&1
  if [ -f "$B/ALL_DONE_full" ]; then
    echo "[wrapper] ALL_DONE after attempt $i at $(date +%H:%M:%S)" >> "$B/run_full.log"; break
  fi
  echo "[wrapper] attempt $i ended WITHOUT sentinel at $(date +%H:%M:%S); resuming in 5s..." >> "$B/run_full.log"
  sleep 5
done
echo "[wrapper] wrapper exiting at $(date +%H:%M:%S)" >> "$B/run_full.log"
