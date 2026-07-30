"""Stratified metric breakdowns (Benchmark Plan §7.5, §10.5).

Aggregate scores hide failure modes, so ``evaluate`` reports metrics broken down by physically
meaningful strata. Generic over the metric function so it serves binary and multiclass tasks.
Step 3/4 stratify by ``grid`` / ``event_family`` / ``event_type`` (always in the windows schema)
plus ``origin`` (line vs bus, derived from ``y_fault_line``) and ``resistance_bin`` (present only
when the aux settings ``resistance`` column has been joined into the metadata via
``load_task(..., attach_aux=("resistance",))`` — otherwise skipped).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

DEFAULT_STRATA = ("grid", "event_family", "event_type", "origin", "resistance_bin")

MetricFn = Callable[[np.ndarray, np.ndarray], dict[str, float]]


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
) -> dict[str, dict[str, dict[str, float]]]:
    """Return ``{stratum_col: {group_value: {metric: value, n: count}}}`` for available strata."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metadata = _with_resistance_bin(_with_origin(metadata))
    out: dict[str, dict[str, dict[str, float]]] = {}
    for col in strata:
        if col not in metadata.columns:
            continue
        groups: dict[str, dict[str, float]] = {}
        values = metadata[col].to_numpy()
        for g in pd.unique(values):
            mask = values == g
            groups[str(g)] = {**metric_fn(y_true[mask], y_pred[mask]), "n": int(mask.sum())}
        out[col] = groups
    return out
