# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/bin/bash
# Auto-restart wrapper for the resumable budget sweep (robust to shared-host SIGKILL).
cd .
B=benchmarks/holstein
rm -f "$B/ALL_DONE_budget"
for i in $(seq 1 40); do
  echo "[budget-wrapper] attempt $i start $(date +%H:%M:%S)" >> "$B/budget_run.log"
  nice -n 10 python3 scripts/run_budget_sweep.py >> "$B/budget_run.log" 2>&1
  if [ -f "$B/ALL_DONE_budget" ]; then echo "[budget-wrapper] ALL_DONE after attempt $i at $(date +%H:%M:%S)" >> "$B/budget_run.log"; break; fi
  echo "[budget-wrapper] attempt $i ended WITHOUT sentinel at $(date +%H:%M:%S); resuming in 5s" >> "$B/budget_run.log"
  sleep 5
done
echo "[budget-wrapper] exiting $(date +%H:%M:%S)" >> "$B/budget_run.log"
