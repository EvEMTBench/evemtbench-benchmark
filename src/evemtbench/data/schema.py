"""Column contracts for the windowed ``y.parquet`` + the fault event-type predicate.

The v0.1 benchmark consumes pre-built windows, so only the windows-y contract is enforced here
(fail loud, never warn-and-continue). ``is_fault`` is the shared fault predicate used by the
label derivers.
"""

from __future__ import annotations

import pandas as pd

# --- windows y.parquet columns this benchmark READS (a subset of the contract's required set;
#     a conformance test asserts WINDOWS_Y_REQUIRED ⊆ contract required columns) ---
# sample_id is the simulation-episode id used for episode-level splits (Benchmark Plan §7.6).
# overlap_ratio (fault-detection activity), y_fault_line (observability oracle) and
# y_fault_location (fault-location target) were previously used but not enforced — a producer
# rename would have surfaced as a deep KeyError instead of at this schema gate.
WINDOWS_Y_REQUIRED = [
    "sample_id",
    "window_idx",
    "event_type",
    "event_family",
    "overlap_ratio",
    "y_fault_line",
    "y_fault_location",
]

FAULT_PREFIX = "flt_"  # event_type values for faults all start with this


class SchemaError(ValueError):
    """Raised when a windows table violates the required column contract."""


def is_fault(event_type: str) -> bool:
    """True iff ``event_type`` denotes a fault (``flt_*``)."""
    return str(event_type).startswith(FAULT_PREFIX)


def assert_windows_y_schema(df: pd.DataFrame) -> None:
    """Validate a windows ``y.parquet``. Raise :class:`SchemaError` on any missing core column."""
    missing = [c for c in WINDOWS_Y_REQUIRED if c not in df.columns]
    if missing:
        raise SchemaError(
            f"windows y is missing required columns {missing}. "
            f"Present columns ({len(df.columns)}): {sorted(df.columns)}."
        )
