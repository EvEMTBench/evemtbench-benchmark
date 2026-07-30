"""Fault-classification labels (FC) — full ``flt_*`` event-type taxonomy (multiclass).

Unlike the retired collapsed fault-type task, FC classifies the **full** fault event_type
string at the granularity the simulation produces (short-circuit subtypes + incipient, with
their arc variants). The class order is frozen so multiclass scores are comparable across
splits. Valid samples are fault-active windows only (see ``tasks/valid_samples.py``).

The frozen vocabulary is the **union** of ``flt_*`` event types across all reference grids: the
seven short-circuit / incipient subtypes present on every grid, plus the two high-impedance
(HIF) subtypes that occur only on the MV grid (CigreMV). HV/EHV grids simply have zero support
for the HIF classes. The HIF classes are appended last so the original indices 0..6 are stable.
A change to this set is a **major** version bump.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...data.schema import is_fault

# Frozen class order (index = label id). 0..6 = the universal SHC/incipient subtypes; 7..8 = the
# MV-only HIF subtypes (no support on HV/EHV grids). Appended last to keep 0..6 stable.
FAULT_CLASS_CLASSES: tuple[str, ...] = (
    "flt_1phg_shc",
    "flt_1phg_shc_w_arc",
    "flt_2ph_shc",
    "flt_2phg_shc",
    "flt_3ph_shc",
    "flt_1phg_incipient",
    "flt_1phg_incipient_w_arc",
    "flt_1phg_hif",
    "flt_1phg_hif_w_arc",
)
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(FAULT_CLASS_CLASSES)}


def fault_class_idx(event_type: str) -> int:
    """Class index for a fault ``event_type``; -1 if not a known fault class."""
    if not is_fault(event_type):
        return -1
    return CLASS_TO_IDX.get(str(event_type), -1)


def derive_fault_class_series(y: pd.DataFrame) -> np.ndarray:
    """Class index per window; -1 for non-fault / unknown rows (filtered by valid_samples)."""
    return y["event_type"].map(fault_class_idx).to_numpy().astype(np.int64)
