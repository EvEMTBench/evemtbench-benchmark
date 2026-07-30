"""Fault-origin label — line fault vs bus fault (binary), on fault-active windows.

Where the fault sits: on a line/feeder (``y_fault_line`` prefix ``MainLn*``) vs at a busbar
(``MainBus*``) — a protection-scheme distinction (line protection vs busbar protection). Positive
(1) = bus fault, 0 = line fault.

**Global-only** (same rule as faulted-line/zone ID): ``y_fault_line`` *is* the observability oracle,
so at local/line the input window is selected to sit on the faulted element — the origin would be
leaked by the view construction. At global the whole main-line universe is shown regardless, so the
origin must be inferred from the waveforms. A change here is a **major** version bump.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_COL = "y_fault_line"


def derive_fault_origin_series(y: pd.DataFrame) -> np.ndarray:
    """Binary fault-origin labels -> int array (n,): 1 = bus fault, 0 = line/non-fault (masked)."""
    if _COL not in y.columns:
        raise KeyError(
            f"windows y is missing required column {_COL!r}; cannot derive fault-origin labels "
            "(it is the observability oracle; see data.schema)."
        )
    return y[_COL].astype(str).str.contains("Bus", na=False).astype(np.int64).to_numpy()
