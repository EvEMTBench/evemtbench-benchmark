"""Fault grounded-vs-ungrounded label — binary, on fault-active windows.

A protection-relevant distinction: ground faults (single/two-phase-to-ground, incipient, HIF) engage
ground/residual-current elements, whereas phase faults (2-phase, 3-phase, ungrounded) do not.
Positive (1) = grounded fault. Source column ``y_is_grounded`` (int8 0/1), episode-constant. The
valid-sample mask restricts scoring to fault-active windows (where the value is defined), so all
three observability settings apply. A change here is a **major** version bump.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_COL = "y_is_grounded"


def derive_fault_grounded_series(y: pd.DataFrame) -> np.ndarray:
    """Binary grounded-fault labels -> int array (n,) in {0,1}. Non-fault rows (masked out) -> 0."""
    if _COL not in y.columns:
        raise KeyError(
            f"windows y is missing column {_COL!r}; cannot derive grounded-fault labels. "
            "It is a contract-required fault descriptor (see contract/contract.json)."
        )
    return y[_COL].fillna(0).astype(np.int64).to_numpy()
