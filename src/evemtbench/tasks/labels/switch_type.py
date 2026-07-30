"""Switch-type label — multiclass switching-event category, on switching-active windows.

Classifies *what kind* of benign switching a window contains (the "what benign disturbance is this?"
companion to switch detection). Source: ``y_switch_category`` (contract-optional string, no frozen
contract enum — the vocabulary is frozen HERE). Global-only (switching windows have no faulted
line).

Grid-restricted categories (``cable``/``motor_start`` MV-only, ``generator`` EHV-only, ``ibr`` on
MV+TestGrid) have zero support on grids that lack them (NaN per-class F1). Universal categories
first so indices stay stable. Valid samples = switching-active windows with an in-vocab category;
out-of-vocab/null -> -1 (dropped). A change to this vocabulary is a **major** version bump.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Frozen class order (index = label id). Universal first, then grid-restricted (appended last).
SWITCH_TYPE_CLASSES: tuple[str, ...] = (
    "load",
    "capacitor",
    "ohl",
    "inrush",
    "ibr",
    "cable",
    "motor_start",
    "generator",
)
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(SWITCH_TYPE_CLASSES)}

_COL = "y_switch_category"


def derive_switch_type_series(y: pd.DataFrame) -> np.ndarray:
    """Switch-type class index per window; -1 for non-switching / null / out-of-vocab rows."""
    if _COL not in y.columns:
        raise KeyError(
            f"windows y is missing column {_COL!r}; cannot derive switch-type labels. "
            "It is a contract-optional switching descriptor (see contract/contract.json)."
        )
    return y[_COL].map(lambda v: CLASS_TO_IDX.get(str(v), -1)).to_numpy().astype(np.int64)
