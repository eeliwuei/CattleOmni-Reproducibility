# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/bin/bash
cd .
B=benchmarks/holstein
rm -f "$B/ALL_DONE_ncontrol"
for i in $(seq 1 30); do
  echo "[ncontrol-wrapper] attempt $i $(date +%H:%M:%S)" >> "$B/ncontrol_run.log"
  nice -n 10 python3 scripts/budget_Ncontrol.py >> "$B/ncontrol_run.log" 2>&1
  [ -f "$B/ALL_DONE_ncontrol" ] && { echo "[ncontrol-wrapper] DONE attempt $i" >> "$B/ncontrol_run.log"; break; }
  echo "[ncontrol-wrapper] attempt $i no sentinel; resume 5s" >> "$B/ncontrol_run.log"; sleep 5
done
