"""Regenerate the Supplementary's ten full per-baseline tables from outputs/<version>/.

Thin shim over the per-artifact builders in ``hpc/paper_artifacts/`` (one standalone script → one
``paper/manuscript/generated/supp_tab_<key>_<test_set>.tex``). The table logic lives in
``hpc/paper_artifacts/paper_lib.py``; this module stays the historical entry point so
``make_results_bundle.py``, ``check_results_fresh.py``, and the tests keep importing
``build_supplement_tables`` from here unchanged.

    PYTHONPATH=src python hpc/make_supplement_tables.py       # write the 10 fragments
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "paper_artifacts"))
from paper_lib import (  # noqa: E402,F401  (re-exported for downstream importers)
    GEN_DIR,
    build_supplement_tables,
    supp_table,
)


def main() -> None:
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    tables = build_supplement_tables()
    for fname, content in tables.items():
        (GEN_DIR / fname).write_text(content)
    n_res = sum(c.count(r"\res") for c in tables.values())
    print(f"wrote {len(tables)} supplement tables to {GEN_DIR} ({n_res} empty cells -> \\res)")


if __name__ == "__main__":
    main()
