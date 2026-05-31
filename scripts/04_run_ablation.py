#!/usr/bin/env python3
"""Step 04: tuning grid + matched-N budget sweep (under-tuning and sample-size controls)."""
import runpy
runpy.run_path("src/models/tune_dl_grid.py", run_name="__main__")
runpy.run_path("src/models/run_budget_sweep_N400.py", run_name="__main__")
