"""Faulted-phase identification label — 7-class single-label, on fault-active windows.

The relay phase-selection task: which phase(s) are involved in the fault. The three per-phase bits
``y_phase_{A,B,C}`` are packed into one of 7 frozen combinations (single, double, or three-phase).
We use a single-label 7-class encoding (not multi-label) so the task reuses the standard multiclass
pipeline; a true independent-per-phase (multi-label) variant is a separate, mechanism-heavier task
(see docs/task_suite_roadmap.md). Valid samples are fault-active windows with ≥1 phase bit set;
all three observability settings apply. A change to this vocabulary is a **major** version bump.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Frozen class order (index = label id).
PHASE_CLASSES: tuple[str, ...] = ("A", "B", "C", "AB", "AC", "BC", "ABC")
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(PHASE_CLASSES)}

# code = A*4 + B*2 + C  ->  class index; 0 (no phase set) -> -1 sentinel (dropped by valid mask).
_CODE_TO_IDX: dict[int, int] = {
    4: CLASS_TO_IDX["A"],
    2: CLASS_TO_IDX["B"],
    1: CLASS_TO_IDX["C"],
    6: CLASS_TO_IDX["AB"],
    5: CLASS_TO_IDX["AC"],
    3: CLASS_TO_IDX["BC"],
    7: CLASS_TO_IDX["ABC"],
    0: -1,
}
_PHASE_COLS = ("y_phase_A", "y_phase_B", "y_phase_C")


def derive_fault_phase_series(y: pd.DataFrame) -> np.ndarray:
    """Faulted-phase class index per window; -1 for no-phase rows (dropped by the valid mask)."""
    missing = [c for c in _PHASE_COLS if c not in y.columns]
    if missing:
        raise KeyError(
            f"windows y is missing phase columns {missing}; cannot derive faulted-phase labels. "
            "They are contract-required fault descriptors (see contract/contract.json)."
        )
    a, b, c = (y[col].fillna(0).astype(np.int64).to_numpy() for col in _PHASE_COLS)
    code = a * 4 + b * 2 + c
    out = np.full(len(y), -1, dtype=np.int64)
    for k, v in _CODE_TO_IDX.items():
        out[code == k] = v
    return out
