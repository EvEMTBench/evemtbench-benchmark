#!/usr/bin/env python3
"""Standalone builder → paper/manuscript/generated/supp_tab_baseline_cfg.tex
(S6: reference-baseline configurations, introspected from the shipped baseline classes).

    PYTHONPATH=src python hpc/paper_artifacts/build_supp_tab_baseline_cfg.py            # write
    PYTHONPATH=src python hpc/paper_artifacts/build_supp_tab_baseline_cfg.py --check    # verify, write nothing
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_lib import run  # noqa: E402
from supplement_lib import baseline_cfg_table  # noqa: E402

if __name__ == "__main__":
    run("supp_tab_baseline_cfg.tex", baseline_cfg_table())
