"""Event-state labels (ES) — high-level grid-state classification (3-class multiclass).

Classifies *what state the grid is in* for every window: ``normal`` / ``switching`` / ``fault``.
This is the high-level "middle-ground" task between fault detection (FD, within-fault-episode
onset) and fault classification (FC, 9-class fault subtype): broader than FD (it sees switching
and quiescent windows, not just fault episodes) yet coarser than FC.

The label keys off **window activity × event family** — a window is:

* ``fault`` (2) iff it overlaps the event (``overlap_ratio > 0``) AND ``event_family == "fault"``;
* ``switching`` (1) iff it overlaps the event AND ``event_family == "switching"``;
* ``normal`` (0) otherwise — i.e. every pre-/post-event (non-overlapping) window of *any* episode,
  plus the fully quiescent ``no_event`` (family ``normal``) episodes.

Because every fault and switching episode contributes its own quiescent pre-event windows, the
``normal`` class has support on **all four grids** — it does not depend on the HV-only ``no_event``
episodes. Every window has a defined state, so the deriver never returns -1 (unlike FC).

Presentation note: the ``switching`` class is what the user-facing "Normal / **Event** / Fault"
framing calls "Event". The frozen vocabulary uses the raw ``event_family`` values for a robust,
column-faithful deriver. The class order is frozen; a change is a **major** version bump.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Frozen class order (index = label id). Matches the ``event_family`` vocabulary; "switching" is
# surfaced as "Event" in the Normal/Event/Fault framing. Changing this set is a major version bump.
EVENT_STATE_CLASSES: tuple[str, ...] = ("normal", "switching", "fault")
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(EVENT_STATE_CLASSES)}

# Activity signal (shared convention with the fault-detection deriver in ``event_present``).
_OVERLAP_COL = "overlap_ratio"
_FAMILY_COL = "event_family"


def derive_event_state_series(y: pd.DataFrame) -> np.ndarray:
    """Vectorized event-state labels for a windows ``y`` frame -> int array (n,) in {0,1,2}.

    A window is ``fault`` (2) / ``switching`` (1) iff it overlaps the event
    (``overlap_ratio > 0``) and its ``event_family`` is ``fault`` / ``switching`` respectively;
    every other window (pre-/post-event and all ``no_event`` windows) is ``normal`` (0). We fail
    loud if ``overlap_ratio`` is missing rather than silently mislabel — it is a required column of
    the windows contract (``data.schema.WINDOWS_Y_REQUIRED``).
    """
    if _OVERLAP_COL not in y.columns:
        raise KeyError(
            f"windows y is missing required column {_OVERLAP_COL!r}; cannot derive "
            "event-state labels (see data.schema.WINDOWS_Y_REQUIRED)."
        )
    active = y[_OVERLAP_COL].fillna(0.0).to_numpy() > 0.0
    family = y[_FAMILY_COL].astype(str).to_numpy()
    labels = np.zeros(len(y), dtype=np.int64)  # default: normal (0)
    labels[active & (family == "fault")] = CLASS_TO_IDX["fault"]
    labels[active & (family == "switching")] = CLASS_TO_IDX["switching"]
    return labels
