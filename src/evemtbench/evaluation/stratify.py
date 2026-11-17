"""Stratified metric breakdowns (Benchmark Plan §7.5, §10.5).

Aggregate scores hide failure modes, so ``evaluate`` reports metrics broken down by physically
meaningful strata. Generic over the metric function so it serves binary and multiclass tasks.
Step 3/4 stratify by ``grid`` / ``event_family`` / ``event_type`` (always in the windows schema)
plus ``origin`` (line vs bus, derived from ``y_fault_line``) and ``resistance_bin`` (present only
when the aux settings ``resistance`` column has been joined into the metadata via
``load_task(..., attach_aux=("resistance",))`` — otherwise skipped).

For binary tasks a breakdown group can hold a single true class (fault origin by origin, the
``normal`` event family of a detection task). Binary ``macro_f1`` is then not interpretable (see
``SingleClassBinaryBreakdownWarning``); ``stratified_metrics(..., binary=True)`` says so, without
touching any returned value.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd

DEFAULT_STRATA = ("grid", "event_family", "event_type", "origin", "resistance_bin")

MetricFn = Callable[[np.ndarray, np.ndarray], dict[str, float]]


class SingleClassBinaryBreakdownWarning(UserWarning):
    """A binary breakdown group holds only one true class, so its ``macro_f1`` depends on whether
    the absent class is ever predicted (1.0 if never; one wrong prediction adds the absent class
    with F1 = 0 and roughly halves it). See docs/known_issues.md section 3. Interpretation only: the
    frozen value is still reported unchanged. A dedicated class so callers that run with warnings
    as errors can filter exactly this one."""


def _with_origin(metadata: pd.DataFrame) -> pd.DataFrame:
    """Add a derived ``origin`` column (line / bus / none) from ``y_fault_line`` so every fault task
    is reportable by fault origin. This stratifies *metrics* by ``y_fault_line`` (metadata, not a
    model input), so it introduces no leakage (unlike using it as a prediction label)."""
    if "origin" in metadata.columns or "y_fault_line" not in metadata.columns:
        return metadata
    fl = metadata["y_fault_line"].astype(str)
    origin = np.where(
        fl.str.contains("Bus", na=False),
        "bus",
        np.where(fl.str.contains("Ln", na=False), "line", "none"),
    )
    return metadata.assign(origin=origin)


def _with_resistance_bin(metadata: pd.DataFrame) -> pd.DataFrame:
    """Add a ``resistance_bin`` stratum from a joined ``resistance`` column (aux settings), so fault
    tasks can be read per fault-resistance band. A discrete sweep (few values) is grouped by value;
    a continuous column is quartile-binned. No-op if ``resistance`` was not attached."""
    if "resistance_bin" in metadata.columns or "resistance" not in metadata.columns:
        return metadata
    r = pd.to_numeric(metadata["resistance"], errors="coerce")
    if r.dropna().empty:
        return metadata
    if r.dropna().nunique() <= 6:  # discrete sweep -> one bin per value
        binned = r.map(lambda v: "na" if pd.isna(v) else f"{v:g}Ω")
    else:  # continuous -> quartiles
        binned = pd.qcut(r, q=4, duplicates="drop").astype(str).where(~r.isna(), "na")
    return metadata.assign(resistance_bin=binned)


def stratified_metrics(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: MetricFn,
    strata: tuple[str, ...] = DEFAULT_STRATA,
    *,
    binary: bool = False,
) -> dict[str, dict[str, dict[str, float]]]:
    """Return ``{stratum_col: {group_value: {metric: value, n: count}}}`` for available strata.

    ``binary`` (the caller states the task's label kind; it is never inferred from label values)
    enables one ``SingleClassBinaryBreakdownWarning`` per call that lists every group whose
    ``y_true`` holds a single class. The predicate reads ``y_true`` only, so it fires whether the
    predictions in that group are all correct or not. The returned values are identical either way.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metadata = _with_resistance_bin(_with_origin(metadata))
    out: dict[str, dict[str, dict[str, float]]] = {}
    single_class: list[str] = []
    for col in strata:
        if col not in metadata.columns:
            continue
        groups: dict[str, dict[str, float]] = {}
        values = metadata[col].to_numpy()
        for g in pd.unique(values):
            mask = values == g
            groups[str(g)] = {**metric_fn(y_true[mask], y_pred[mask]), "n": int(mask.sum())}
            if binary:
                present = np.unique(y_true[mask])
                if present.size < 2:  # same predicate as compute_binary_metrics' single_class
                    only = f"only class {present[0]}" if present.size else "no samples"
                    single_class.append(f"{col}={g} (n={int(mask.sum())}, {only})")
        out[col] = groups
    if single_class:
        warnings.warn(
            f"binary breakdown: {len(single_class)} group(s) hold a single true class: "
            f"{'; '.join(single_class)}. Their macro_f1 depends on whether the absent class is "
            "ever predicted (1.0 if never; one wrong prediction roughly halves it), so it is not "
            "comparable across groups; read accuracy and macro_f1_supported there. Values are "
            "reported unchanged (see docs/known_issues.md section 3)",
            SingleClassBinaryBreakdownWarning,
            stacklevel=3,
        )
    return out
