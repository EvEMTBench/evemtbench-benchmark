"""Sanity checks against the REAL mounted dataset. Marked data_integrity -> excluded from CI.

Run on LME/NHR where the dataset is mounted:
    pytest -m data_integrity
"""

import numpy as np
import pytest

from evemtbench.data import roots
from evemtbench.data.windows import load_windows

pytestmark = pytest.mark.data_integrity


def _have_real_data() -> bool:
    try:
        roots.windows_dir("double_line", "adapt_grid")  # resolves only if a root is found
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_real_data(), reason="real dataset not mounted")
def test_double_line_finetune_windows_load():
    ws = load_windows("double_line", "adapt_grid")
    assert ws.X.ndim == 3 and ws.X.shape[2] == 66
    assert ws.n_cubicles == 11
    # v0.1 reconfigured dataset: ~10k adapt_grid simulation episodes.
    n_eps = len(np.unique(ws.sample_ids))
    assert 9000 <= n_eps <= 10_000
    assert ws.contract.window_length == 0.05
