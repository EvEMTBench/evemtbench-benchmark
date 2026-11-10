"""The COMMITTED splits regenerate bit-for-bit from the real dataset (the determinism claim in
docs/determinism.md). Marked data_integrity -> excluded from CI; run where the dataset is mounted:
    pytest -m data_integrity tests/cluster/test_committed_splits_regenerate.py
"""

import pytest

from evemtbench.data import roots
from evemtbench.splits.leakage import verify_protocol_regenerates

pytestmark = pytest.mark.data_integrity

CASES = [("held_out", g) for g in ("double_line", "cigre_mv", "testgrid_110kv", "ieee39")] + [
    ("multi_grid_pretrain", g) for g in ("template_20kv", "template_110kv", "template_345kv")
]


def _have(grid: str, family: str) -> bool:
    try:
        return roots.windows_dir(grid, family).is_dir()
    except Exception:  # no data root resolvable on this machine
        return False


@pytest.mark.parametrize(("protocol", "grid"), CASES)
def test_committed_split_regenerates_from_the_dataset(protocol, grid):
    family = "multi_grid" if protocol == "multi_grid_pretrain" else "adapt_grid"
    if not _have(grid, family):
        pytest.skip(f"{grid}/{family} windows not mounted")
    verify_protocol_regenerates(protocol, grid)  # raises LeakageError on any mismatch
