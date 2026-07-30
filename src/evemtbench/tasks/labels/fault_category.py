"""Fault-category label — 3-class {shc, incipient, hif}, on fault-active windows.

Coarser than the 9-class fault subtype (FC): the physical fault mechanism. ``shc`` = solid/low-
impedance short circuit; ``incipient`` = self-extinguishing high-impedance fault; ``hif`` = high-
impedance ground fault. HIF is the notoriously hard, safety-critical class and occurs **only on the
MV grid** (CigreMV) — it has zero support on HV/EHV grids (per-class F1 = NaN there, like FC's HIF).

Source: ``y_fault_category`` (contract-optional enum) — the deriver fails loud if the column is
absent. Universal classes first (indices 0,1), MV-only ``hif`` last (index 2) so indices stay
stable. Out-of-vocab / null -> -1 (dropped by the valid mask). A change is a **major** version bump.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Frozen class order (index = label id). shc/incipient universal; hif MV-only, appended last.
FAULT_CATEGORY_CLASSES: tuple[str, ...] = ("shc", "incipient", "hif")
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(FAULT_CATEGORY_CLASSES)}

_COL = "y_fault_category"


def derive_fault_category_series(y: pd.DataFrame) -> np.ndarray:
    """Fault-category class index per window; -1 for non-fault / null / out-of-vocab rows."""
    if _COL not in y.columns:
        raise KeyError(
            f"windows y is missing column {_COL!r}; cannot derive fault-category labels. "
            "It is a contract-optional fault descriptor (see contract/contract.json)."
        )
    return y[_COL].map(lambda v: CLASS_TO_IDX.get(str(v), -1)).to_numpy().astype(np.int64)
