#!/usr/bin/env python3
"""Standalone builder → paper/manuscript/generated/tab_task_spec.tex (static specification table)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_lib import run, spec_table  # noqa: E402

if __name__ == "__main__":
    run("tab_task_spec.tex", spec_table("task_spec"))
