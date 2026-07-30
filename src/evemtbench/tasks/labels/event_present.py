"""Fault-detection label derivation (window-level fault activity).

Pure, deterministic mapping from a window's metadata to the binary fault-detection label: a
window is positive iff its ``event_type`` is a fault (``flt_*``) AND the window overlaps the
event (not a pre-fault "clean" window). Re-derives the label rather than echoing a precomputed
column; golden tests pin it. Any change here is a **major** version bump.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...data.schema import is_fault

# Activity signal in the windows y schema. ``overlap_ratio`` is a required column of the
# windows contract (data.schema.WINDOWS_Y_REQUIRED), so it is the single source of "activity".
_OVERLAP_COL = "overlap_ratio"
_ETYPE_COL = "event_type"


def derive_fault_active_series(y: pd.DataFrame) -> np.ndarray:
    """Vectorized fault-detection labels for a windows ``y`` frame -> int array (n,).

    A window is positive iff its ``event_type`` is a fault (``flt_*``) AND it overlaps the event
    (``overlap_ratio > 0``). We fail loud if ``overlap_ratio`` is missing rather than silently
    degrade: a previous ``status``-based fallback was **unreachable and incorrect** — the
    producer's ``WindowStatus`` vocabulary never contains ``"event"``/``"active"``, so that branch
    would have labelled every window negative. Removing it changes no scores on any real dataset
    (all carry ``overlap_ratio``); it only converts a latent silent-mislabel into a loud failure.
    """
    if _OVERLAP_COL not in y.columns:
        raise KeyError(
            f"windows y is missing required column {_OVERLAP_COL!r}; cannot derive "
            "fault-detection labels (see data.schema.WINDOWS_Y_REQUIRED)."
        )
    is_flt = y[_ETYPE_COL].map(is_fault).to_numpy()
    active = y[_OVERLAP_COL].fillna(0.0).to_numpy() > 0.0
    return (is_flt & active).astype(np.int64)
