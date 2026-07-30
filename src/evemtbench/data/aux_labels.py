"""Per-episode simulator settings (auxiliary overlay).

Reads the ``settings.csv`` packaged alongside the windowed data (extracted into the aux overlay by
``evemtbench.data.extract_aux``) and joins selected columns onto a windows ``metadata`` frame by
**episode**: windows ``sample_id`` == settings ``general/sim_idx``. This surfaces per-episode
physical parameters (fault resistance, arc model, grounding, operating point) that are *not* in the
windowed ``y`` — enabling stratified difficulty curves (e.g. resistance-binned FL/HIF) without
touching the frozen windows contract.

**Not a model input.** Joining these onto *metadata* stratifies metrics only; using a settings value
as a feature would be label leakage.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import pandas as pd

from .roots import GridSpec, aux_dir

_SETTINGS_FILE = "settings.csv"
_SIM_IDX = "general/sim_idx"

# Column-group predicates (prefixes vary a little across grid schemas; keep them permissive).
_GROUPS: dict[str, Callable[[str], bool]] = {
    "resistance": lambda c: "resistance" in c.lower(),
    "arc": lambda c: c.startswith("arc/"),
    "grounding": lambda c: c.startswith("grounding/"),
    "operating_point": lambda c: c.startswith("loads/") or c.startswith("ExtGrid/"),
    "events": lambda c: c.startswith("events/"),
}

# Preference order for "the" fault-resistance column (the schema varies across grids).
_RESISTANCE_PREF = (
    "events/event_flt_shc_resistance",
    "events/event_flt_hif_resistance",
    "grounding/solid_ground_resistance",
    "grounding/resistive_ground_resistance",
)


def load_settings(
    grid: str | GridSpec, family: str = "benchmark", root: str | os.PathLike | None = None
) -> pd.DataFrame:
    """Load the per-episode ``settings.csv`` for ``grid``/``family`` from the aux overlay.

    Fails loud if absent (run ``python -m evemtbench.data.extract_aux`` first)."""
    path = aux_dir(grid, family, root) / _SETTINGS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"no settings.csv at {path}. Extract it first: "
            f"python -m evemtbench.data.extract_aux --grid <key> --family {family}"
        )
    return pd.read_csv(path)


def group_columns(settings: pd.DataFrame, group: str) -> list[str]:
    """Settings columns belonging to a semantic group (resistance / arc / grounding /
    operating_point / events)."""
    if group not in _GROUPS:
        raise KeyError(f"unknown settings group {group!r}; known: {sorted(_GROUPS)}")
    pred = _GROUPS[group]
    return [c for c in settings.columns if pred(c)]


def fault_resistance_column(settings: pd.DataFrame) -> str | None:
    """The best per-episode fault-resistance column present, or None."""
    for cand in _RESISTANCE_PREF:
        if cand in settings.columns:
            return cand
    hits = group_columns(settings, "resistance")
    return hits[0] if hits else None


def join_to_metadata(
    metadata: pd.DataFrame,
    grid: str | GridSpec,
    family: str,
    cols: list[str],
    *,
    rename: dict[str, str] | None = None,
    root: str | os.PathLike | None = None,
) -> pd.DataFrame:
    """Left-join settings ``cols`` onto ``metadata`` by ``sample_id`` == ``general/sim_idx``.

    Returns a new frame; unmatched episodes get NaN. ``rename`` maps settings column names to the
    output names attached to metadata (e.g. ``{"events/event_flt_shc_resistance": "resistance"}``).
    """
    settings = load_settings(grid, family, root)
    if _SIM_IDX not in settings.columns:
        raise KeyError(f"settings.csv missing the join key {_SIM_IDX!r} (grid={grid}, {family}).")
    missing = [c for c in cols if c not in settings.columns]
    if missing:
        raise KeyError(f"settings.csv missing requested columns {missing} (grid={grid}, {family}).")
    sub = settings[[_SIM_IDX, *cols]].rename(columns={_SIM_IDX: "sample_id"})
    if rename:
        sub = sub.rename(columns=rename)
    return metadata.merge(sub, on="sample_id", how="left")
