"""Fault-location labels (FL) — distance along the faulted line, % (regression target).

Reads the pipeline's ``y_fault_location`` column (% along the faulted line, ~1..99). It is
**episode-constant** and present only on **line faults** — NaN on bus faults and non-fault
episodes. The valid-sample rule (``tasks/valid_samples.py``) restricts FL to fault-active
windows of line-fault episodes, so the returned targets are finite there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_LOCATION_COL = "y_fault_location"


def derive_fault_location_series(y: pd.DataFrame) -> np.ndarray:
    """Continuous fault-location % per window as float32 (NaN where undefined)."""
    return y[_LOCATION_COL].to_numpy().astype(np.float32)
