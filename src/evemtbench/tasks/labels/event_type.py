"""Event-classification label (EC) — full event-type taxonomy over every window (multiclass).

The hard-frontier classifier: *what kind* of event a window contains across the whole event space
(faults + switching + quiescent). Label = the window's ``event_type`` when it is event-active
(``overlap_ratio > 0``), else ``no_event`` — so pre-/post-event windows fold into the quiescent
class. This makes EC the fine-grained refinement of the 3-class event-state task (ES families =
{fault, switching, normal} are the coarse grouping of these 23 types), and a superset of FC
(fault-active windows only, 9 fault classes).

Frozen 23-class vocabulary: the 9 ``flt_*`` types (indices 0..8, shared with FC), the 13
``switch_*`` types (9..21), and ``no_event`` (22). Grid-restricted types (HIF → MV, gen_trip → EHV,
cable/motor → MV, ibr → MV+TestGrid) simply have zero support elsewhere (NaN per-class F1).
Global-only. Active windows whose type is out-of-vocab -> -1 (dropped). Change = **major** bump.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fault_class import FAULT_CLASS_CLASSES

# Frozen class order (index = label id): 9 faults (aligned with FC), 13 switching, then no_event.
_SWITCHING_CLASSES: tuple[str, ...] = (
    "switch_load_on",
    "switch_load_off",
    "switch_cap_on",
    "switch_cap_off",
    "switch_ohl_on",
    "switch_ohl_off",
    "switch_inrush_lv",
    "switch_inrush_hv",
    "switch_ibr_trip",
    "switch_cable_on",
    "switch_cable_off",
    "switch_motor_start",
    "switch_gen_trip",
)
EVENT_TYPE_CLASSES: tuple[str, ...] = FAULT_CLASS_CLASSES + _SWITCHING_CLASSES + ("no_event",)
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(EVENT_TYPE_CLASSES)}

_OVERLAP_COL = "overlap_ratio"
_ETYPE_COL = "event_type"
_NO_EVENT = "no_event"


def derive_event_type_series(y: pd.DataFrame) -> np.ndarray:
    """Event-type class index per window; non-active -> no_event; out-of-vocab active -> -1."""
    if _OVERLAP_COL not in y.columns:
        raise KeyError(
            f"windows y is missing required column {_OVERLAP_COL!r}; cannot derive "
            "event-classification labels (see data.schema.WINDOWS_Y_REQUIRED)."
        )
    active = y[_OVERLAP_COL].fillna(0.0).to_numpy() > 0.0
    etype = y[_ETYPE_COL].astype(str).to_numpy()
    # Non-active windows are quiescent regardless of their episode's event_type.
    labels_str = np.where(active, etype, _NO_EVENT)
    idx = np.fromiter((CLASS_TO_IDX.get(s, -1) for s in labels_str), dtype=np.int64, count=len(y))
    return idx
