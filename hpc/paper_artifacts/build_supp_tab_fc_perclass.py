#!/usr/bin/env python3
"""Standalone builder → paper/manuscript/generated/supp_tab_fc_perclass.tex (F03: per-class F1 for
fault-type classification, validation-selected reference)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_lib import fc_per_class_f1_table, run

if __name__ == "__main__":
    run("supp_tab_fc_perclass.tex", fc_per_class_f1_table())
