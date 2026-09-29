"""Import-only compatibility shim — **no longer writes any artifact**.

Historically this module regenerated the five held_out result fragments at once. Under the artifact
rule the manuscript and Supplementary Information are now built with —

    one scientific table  ->  exactly one Python generator  ->  exactly one .tex artifact

— each fragment is owned by exactly one standalone ``hpc/paper_artifacts/build_<name>.py``
(see ``paper/manuscript/generated/README.md``). Build them individually, or all at once through the
Makefile, which invokes the individual generators rather than a multi-output script::

    make paper-artifacts         # regenerate every manuscript + SI fragment
    make check-supp-artifacts    # verify the committed fragments are fresh (writes nothing)

    PYTHONPATH=src python hpc/paper_artifacts/build_tab_res_fd.py [--check]

The table LOGIC lives in ``hpc/paper_artifacts/paper_lib.py`` (shared, artifact-free); the pure
aggregator ``build_paper_tables`` and the shared config/selection names (``GRID_ROWS``, ``OBS``,
``PAPER_FUNCTIONS``, ``TEST_SETS``, ``best_cell_choice``) are re-exported here so the freshness
gate, the audit dashboard and the tests keep importing them from this path unchanged. Aggregators
return ``{filename: latex}``; they do not write.
"""

from __future__ import annotations

import sys
from pathlib import Path

# hpc/ is not a package; the per-artifact builders live in the paper_artifacts/ sibling dir.
sys.path.insert(0, str(Path(__file__).resolve().parent / "paper_artifacts"))
from paper_lib import (  # noqa: F401  (re-exported for downstream importers)
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
    sys.exit(
        "hpc/make_paper_tables.py no longer writes artifacts (one artifact = one generator).\n"
        "Use `make paper-artifacts`, or a single owner such as\n"
        "  PYTHONPATH=src python hpc/paper_artifacts/build_tab_res_fd.py\n"
        "See paper/manuscript/generated/README.md for the full artifact -> generator map."
    )


if __name__ == "__main__":
    main()
