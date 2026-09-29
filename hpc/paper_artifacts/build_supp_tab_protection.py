#!/usr/bin/env python3
"""Standalone builder → paper/manuscript/generated/supp_tab_protection.tex (F03: dependability /
security error rates for the binary functions, validation-selected reference)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_lib import protection_rate_table, run

if __name__ == "__main__":
    run("supp_tab_protection.tex", protection_rate_table())
