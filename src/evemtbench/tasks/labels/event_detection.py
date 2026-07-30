"""Event-detection label (ED) — any active grid event vs. quiescent (binary).

Window is positive iff it overlaps a real event (``overlap_ratio > 0``) whose ``event_family`` is a
disturbance (fault OR switching). Every other window — the pre-/post-event quiescent windows of any
episode, and any ``no_event`` window — is negative. This is the broad "is *anything* happening?"
detector; it sees benign switching as a positive, unlike fault detection (FD).

Equivalent to the producer's ``y_event_present`` column, but re-derived from
``overlap_ratio`` × ``event_family`` (the benchmark re-derives labels rather than echoing a
precomputed column). Global-only: quiescent/switching windows have no faulted line for the
local/line oracle. A change here is a **major** version bump.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_OVERLAP_COL = "overlap_ratio"
_FAMILY_COL = "event_family"


def derive_event_present_series(y: pd.DataFrame) -> np.ndarray:
    """Binary event-detection labels for a windows ``y`` frame -> int array (n,) in {0,1}."""
    if _OVERLAP_COL not in y.columns:
        raise KeyError(
            f"windows y is missing required column {_OVERLAP_COL!r}; cannot derive "
            "event-detection labels (see data.schema.WINDOWS_Y_REQUIRED)."
        )
    active = y[_OVERLAP_COL].fillna(0.0).to_numpy() > 0.0
    family = y[_FAMILY_COL].astype(str).to_numpy()
    return (active & ((family == "fault") | (family == "switching"))).astype(np.int64)
