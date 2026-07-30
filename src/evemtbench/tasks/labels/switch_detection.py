"""Switch-detection label (SD) — active switching event vs. everything else (binary).

Window is positive iff it overlaps an active event (``overlap_ratio > 0``) of ``event_family ==
"switching"``. Faults (active or pre-fault) and quiescent windows are negatives. The benign-event
counterpart of fault detection; together ED = FD-positives ∪ SD-positives.

Re-derived from ``overlap_ratio`` × ``event_family`` (≡ the producer's ``y_switch_present``).
Global-only. A change here is a **major** version bump.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_OVERLAP_COL = "overlap_ratio"
_FAMILY_COL = "event_family"


def derive_switch_present_series(y: pd.DataFrame) -> np.ndarray:
    """Binary switch-detection labels for a windows ``y`` frame -> int array (n,) in {0,1}."""
    if _OVERLAP_COL not in y.columns:
        raise KeyError(
            f"windows y is missing required column {_OVERLAP_COL!r}; cannot derive "
            "switch-detection labels (see data.schema.WINDOWS_Y_REQUIRED)."
        )
    active = y[_OVERLAP_COL].fillna(0.0).to_numpy() > 0.0
    family = y[_FAMILY_COL].astype(str).to_numpy()
    return (active & (family == "switching")).astype(np.int64)
