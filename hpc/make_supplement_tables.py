"""Import-only compatibility shim — **no longer writes any artifact**.

Historically this module regenerated ten supplement fragments at once. That violates the artifact
rule the Supplementary Information is now built under:

    one scientific table  ->  exactly one Python generator  ->  exactly one .tex artifact

Every ``paper/manuscript/generated/*.tex`` is therefore owned by exactly one standalone
``hpc/paper_artifacts/build_<name>.py`` (see ``paper/manuscript/generated/README.md`` for the
artifact -> generator -> input map). Build them individually, or all at once through the Makefile,
which invokes the individual generators rather than a multi-output script::

    make supp-artifacts          # regenerate every SI fragment
    make check-supp-artifacts    # verify the committed fragments are fresh (writes nothing)

    PYTHONPATH=src python hpc/paper_artifacts/build_supp_tab_fc_support.py [--check]

The table LOGIC still lives in ``hpc/paper_artifacts/paper_lib.py`` (shared, artifact-free), and the
pure aggregator ``build_supplement_tables`` is re-exported here so the freshness gate and the tests
keep importing it from this path unchanged. Aggregators return ``{filename: latex}``; they do not
write.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "paper_artifacts"))
from paper_lib import (  # noqa: F401  (re-exported for downstream importers)
    GEN_DIR,
    build_supplement_tables,
)


def main() -> None:
    sys.exit(
        "hpc/make_supplement_tables.py no longer writes artifacts (one artifact = one generator).\n"
        "Use `make supp-artifacts`, or a single owner such as\n"
        "  PYTHONPATH=src python hpc/paper_artifacts/build_supp_tab_fc_support.py\n"
        "See paper/manuscript/generated/README.md for the full artifact -> generator map."
    )


if __name__ == "__main__":
    main()
