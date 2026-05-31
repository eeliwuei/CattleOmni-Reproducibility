#!/usr/bin/env python3
"""Step 06: regenerate tables (postprocess + within-cluster verification)."""
import runpy
runpy.run_path("src/evaluation/holstein_postprocess.py", run_name="__main__")
runpy.run_path("src/evaluation/wc_verify_all5.py", run_name="__main__")
