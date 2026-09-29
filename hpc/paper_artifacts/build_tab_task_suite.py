#!/usr/bin/env python3
"""Standalone builder → paper/manuscript/generated/tab_task_suite.tex (static spec table).

The full 24-task, 3-tier scored suite (12 functions × observability views). Static/authored
content (not data-derived), so it is exempt from the pre-freeze ``\res`` policy like the other
spec tables; it is regenerated + byte-checked with them by ``check_results_fresh.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_lib import run, spec_table

if __name__ == "__main__":
    run("tab_task_suite.tex", spec_table("task_suite"))
