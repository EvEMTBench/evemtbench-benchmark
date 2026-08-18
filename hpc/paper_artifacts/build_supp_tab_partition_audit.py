#!/usr/bin/env python3
"""Standalone builder → paper/manuscript/generated/supp_tab_partition_audit.tex
(S5: partition + leakage audit of the committed splits (target grids and transfer source corpora)).

    PYTHONPATH=src python hpc/paper_artifacts/build_supp_tab_partition_audit.py            # write
    PYTHONPATH=src python hpc/paper_artifacts/build_supp_tab_partition_audit.py --check    # verify, write nothing
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_lib import run  # noqa: E402
from supplement_lib import partition_audit_table  # noqa: E402

if __name__ == "__main__":
    run("supp_tab_partition_audit.tex", partition_audit_table())
