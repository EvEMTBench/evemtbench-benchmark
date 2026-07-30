"""Regenerate the manuscript's five held_out result tables from outputs/<version>/.

Thin shim over the per-artifact builders in ``hpc/paper_artifacts/`` (one standalone script → one
``paper/manuscript/generated/tab_res_<key>.tex``). The table logic lives in
``hpc/paper_artifacts/paper_lib.py``; this module stays the historical entry point so
``make_results_bundle.py``, ``check_results_fresh.py``, ``make_audit.py``, and the tests keep
importing ``build_paper_tables`` + the shared config/selection names (``GRID_ROWS``, ``OBS``,
``PAPER_FUNCTIONS``, ``TEST_SETS``, ``best_cell_choice``) from here unchanged.

    PYTHONPATH=src python hpc/make_paper_tables.py            # write the 5 fragments

Wire numbers ONLY from a frozen benchmark version tag (see ``paper/AGENTS.md``); rehearsal runs
stay ``\\res``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# hpc/ is not a package; the per-artifact builders live in the paper_artifacts/ sibling dir.
sys.path.insert(0, str(Path(__file__).resolve().parent / "paper_artifacts"))
from paper_lib import (  # noqa: E402,F401  (re-exported for downstream importers)
    GEN_DIR,
    GRID_ROWS,
    OBS,
    PAPER_FUNCTIONS,
    TEST_SETS,
    best_cell_choice,
    build_paper_tables,
    result_table,
)


def main() -> None:
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    tables = build_paper_tables()
    for fname, content in tables.items():
        (GEN_DIR / fname).write_text(content)
    n_res = sum(c.count(r"\res") for c in tables.values())
    print(f"wrote {len(tables)} paper tables to {GEN_DIR} ({n_res} empty cells -> \\res)")


if __name__ == "__main__":
    main()
