"""Per-task valid-sample rules (Benchmark Plan §4: targets and valid samples).

Each function is scored on a subset of windows. The rule is applied *after* the episode-level
split, so it only narrows which windows of already-assigned episodes are scored — it never
moves a simulation across splits.

v0.1 (DoubleLine, fault-only). All three functions operate **within fault episodes**, where the
episode-constant ``y_fault_line`` is defined — this keeps the LOCAL/LINE oracle views well
defined for every scored window:

* FAULT_DETECTION — all windows of fault episodes (positives = fault-active windows, negatives =
  the pre-fault windows of the same episode). Gate: ``y_fault_line`` not null.
* FAULT_CLASS — fault-active windows only (a non-active window has no fault type to classify).
* FAULT_LOCATION — fault-active windows of **line-fault** episodes only (location is NaN on bus
  faults). Gate: fault-active AND ``y_fault_location`` not null.

EVENT_STATE breaks that within-fault-episode pattern: it scores **all** windows (every window has
a defined normal/switching/fault state), which is why it is global-only — the local/line oracle
would need a faulted line that switching/normal windows do not have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .labels.event_present import derive_fault_active_series
from .labels.event_type import derive_event_type_series
from .labels.fault_category import derive_fault_category_series
from .labels.fault_class import derive_fault_class_series
from .labels.fault_phase import derive_fault_phase_series
from .labels.switch_detection import derive_switch_present_series
from .labels.switch_type import derive_switch_type_series
from .taxonomy import Function, TaskDefinition


def valid_sample_mask(y: pd.DataFrame, td: TaskDefinition) -> np.ndarray:
    """Boolean mask (n,) of windows admissible for the task."""
    if td.function is Function.FAULT_DETECTION:
        # windows of fault episodes (y_fault_line is episode-constant, NaN only on non-faults)
        return y["y_fault_line"].notna().to_numpy()
    if td.function is Function.FAULT_CLASS:
        # fault-active windows whose fault type is in the class vocabulary (a type outside it would
        # map to -1 and crash a CrossEntropy head; this guard keeps FC robust to taxonomy gaps).
        return derive_fault_active_series(y).astype(bool) & (derive_fault_class_series(y) != -1)
    if td.function is Function.FAULT_LOCATION:
        return derive_fault_active_series(y).astype(bool) & y["y_fault_location"].notna().to_numpy()
    if td.function is Function.EVENT_STATE:
        # Every window has a defined grid state (normal/switching/fault); all are scored. This is a
        # global-only task, so it needs no y_fault_line oracle (which is null on non-fault windows).
        return np.ones(len(y), dtype=bool)
    if td.function in (Function.EVENT_DETECTION, Function.SWITCH_DETECTION):
        # Binary event-scope detection over every window (event-active vs. rest). Global-only.
        return np.ones(len(y), dtype=bool)
    if td.function is Function.EVENT_CLASSIFICATION:
        # All windows (non-active fold into no_event); drop only out-of-vocab active types (-1).
        return derive_event_type_series(y) != -1
    if td.function is Function.SWITCH_TYPE:
        # Switching-active windows whose category is in the frozen vocabulary. Global-only.
        return derive_switch_present_series(y).astype(bool) & (derive_switch_type_series(y) != -1)
    if td.function is Function.FAULT_GROUNDED:
        # Grounded-vs-ungrounded is defined on fault-active windows (y_is_grounded episode-const).
        return derive_fault_active_series(y).astype(bool)
    if td.function is Function.FAULT_PHASE:
        # Fault-active windows with ≥1 phase bit set (7-class combination in vocabulary).
        return derive_fault_active_series(y).astype(bool) & (derive_fault_phase_series(y) != -1)
    if td.function is Function.FAULT_CATEGORY:
        # Fault-active windows whose category is in {shc, incipient, hif}.
        return derive_fault_active_series(y).astype(bool) & (derive_fault_category_series(y) != -1)
    if td.function is Function.FAULT_ORIGIN:
        # Line-vs-bus origin on fault-active windows (global-only; y_fault_line defined here).
        return derive_fault_active_series(y).astype(bool)
    raise NotImplementedError(f"valid-sample rule for {td.function} not defined")
