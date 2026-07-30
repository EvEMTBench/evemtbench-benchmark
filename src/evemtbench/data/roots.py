"""Data-root resolution and the grid registry.

The data root is resolved from the environment (never hardcoded into logic). Precedence:

1. explicit ``root=`` argument,
2. ``$EVEMTBENCH_DATA_ROOT``,
3. ``$EVEMT_ROOT`` (the convention shared with the sibling production repo).

Each *family* (``benchmark`` / ``adapt_grid`` / ``multi_grid``) is a subdirectory of the root, and
each *grid* maps to a ``raw_foldername`` (windows/preprocessed directory) and a ``topology``
id (used in window file names). These mappings mirror the sibling repo's dataset configs so a
single grid resolves identically in both.

The large-scale multi-topology **pre-training** family was named ``train`` through v0.2-dev; the
producer now writes it on disk as ``multi_grid/`` (a contract relabel, mirroring the earlier
``fine_tune`` -> ``adapt_grid`` rename). ``train`` is accepted as a read alias — see
:func:`resolve_data_root`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

FAMILIES = ("benchmark", "adapt_grid", "multi_grid")


@dataclass(frozen=True)
class GridSpec:
    """Identity of one grid topology across the dataset.

    ``key`` is the short canonical name used in the benchmark API; ``topology`` is the
    sibling/file-naming id (appears in ``X_<topology>_W..._S....raw``); ``raw_foldername`` is
    the windows/preprocessed subdirectory.
    """

    key: str
    topology: str
    raw_foldername: str
    voltage_kv: float
    frequency_hz: int
    level: str  # "MV" | "HV" | "EHV"


# Canonical grid registry — the four reference grids plus the multi_grid pre-training corpus.
# ieee39 is registered and benchmark-windowed (its adapt_grid windowing is in progress).
# ``template_20kv`` is the first windowed grid of the multi_grid (cross-topology) pre-training
# family; more topologies join as the corpus grows. Its windows use the portable 12-channel
# "double_line" view (feature axis carries no bus/line identity — that lives in ``y``), so it is
# only observable at LOCAL/LINE; see ``tasks/line_view.py``.
GRIDS: dict[str, GridSpec] = {
    "double_line": GridSpec("double_line", "hv_double_line_110kv", "DoubleLine", 110.0, 50, "HV"),
    "cigre_mv": GridSpec("cigre_mv", "mv_cigre_distribution_20kv", "CigreMVGrid", 20.0, 50, "MV"),
    "testgrid_110kv": GridSpec(
        "testgrid_110kv", "hv_testgrid_110kv", "TestGrid110kV", 110.0, 50, "HV"
    ),
    "ieee39": GridSpec(
        "ieee39", "ehv_ieee39_transmission_345kv", "IEEE39BusSystem", 345.0, 60, "EHV"
    ),
    "template_20kv": GridSpec("template_20kv", "mv_template_20kv", "Template20kV", 20.0, 50, "MV"),
    "template_110kv": GridSpec(
        "template_110kv", "hv_template_110kv", "Template110kV", 110.0, 50, "HV"
    ),
    "template_345kv": GridSpec(
        "template_345kv", "ehv_template_345kv", "Template345kV", 345.0, 50, "EHV"
    ),
}

# topology id -> short key (for resolving manifests/configs back to a GridSpec).
_TOPOLOGY_TO_KEY = {g.topology: k for k, g in GRIDS.items()}


def get_grid(grid: str | GridSpec) -> GridSpec:
    """Resolve a grid key, topology id, or pass-through :class:`GridSpec`."""
    if isinstance(grid, GridSpec):
        return grid
    if grid in GRIDS:
        return GRIDS[grid]
    if grid in _TOPOLOGY_TO_KEY:
        return GRIDS[_TOPOLOGY_TO_KEY[grid]]
    raise KeyError(
        f"Unknown grid {grid!r}. Known keys: {sorted(GRIDS)}; "
        f"known topologies: {sorted(_TOPOLOGY_TO_KEY)}."
    )


def resolve_data_root(family: str, root: str | os.PathLike | None = None) -> Path:
    """Return ``{root}/{family}``, resolving the root from arg/env/fallback.

    Raises if ``family`` is unknown or if no usable root can be found.
    """
    # Backward-compat: the in-distribution family was named ``fine_tune`` in the v0.1 contract
    # and is ``adapt_grid`` from v0.2 on (matching the producer's on-disk directory); the
    # multi-topology pre-training family was ``train`` and is ``multi_grid`` on disk. Accept both
    # legacy names as read aliases so old callers / external data still resolve.
    if family == "fine_tune":
        family = "adapt_grid"
    elif family == "train":
        family = "multi_grid"
    if family not in FAMILIES:
        raise ValueError(f"Unknown family {family!r}; expected one of {FAMILIES}.")

    base = (
        str(root)
        if root is not None
        else os.environ.get("EVEMTBENCH_DATA_ROOT") or os.environ.get("EVEMT_ROOT")
    )
    if base is None:
        raise RuntimeError(
            "No data root found. Set $EVEMTBENCH_DATA_ROOT (or $EVEMT_ROOT) to the EvEMTBench "
            "dataset root, or pass root=."
        )
    return Path(base) / family


def windows_dir(grid: str | GridSpec, family: str, root: str | os.PathLike | None = None) -> Path:
    """Directory holding the windowed memmap + parquet for ``grid`` in ``family``."""
    g = get_grid(grid)
    return resolve_data_root(family, root) / "windows" / g.raw_foldername


def raw_partition_dir(
    grid: str | GridSpec, family: str, root: str | os.PathLike | None = None
) -> Path:
    """Directory holding raw ``data/``, ``labels/``, ``graphs/`` for ``grid`` in ``family``."""
    g = get_grid(grid)
    return resolve_data_root(family, root) / "raw" / g.raw_foldername


def resolve_aux_root(root: str | os.PathLike | None = None) -> Path:
    """Root of the **auxiliary overlay** (topology graphs + per-episode ``settings.csv``), kept
    separate from the producer-owned windowed data root so nothing writes into it.

    Precedence: explicit ``root=`` → ``$EVEMTBENCH_AUX_ROOT`` → a sibling ``evemtbench-aux`` next to
    the data root (``$EVEMTBENCH_DATA_ROOT`` / ``$EVEMT_ROOT``). Raises if none can be resolved.
    Populated by ``python -m evemtbench.data.extract_aux`` (which pulls the small artifacts out of
    the per-grid ``*.tar.gz``); ``multi_grid`` slices already carry them extracted.
    """
    base = str(root) if root is not None else os.environ.get("EVEMTBENCH_AUX_ROOT")
    if base:
        return Path(base)
    data_root = os.environ.get("EVEMTBENCH_DATA_ROOT") or os.environ.get("EVEMT_ROOT")
    if data_root:
        return Path(data_root).parent / "evemtbench-aux"
    raise RuntimeError(
        "No aux root found. Set $EVEMTBENCH_AUX_ROOT (or $EVEMTBENCH_DATA_ROOT / $EVEMT_ROOT so a "
        "sibling 'evemtbench-aux' can be derived), or pass root=."
    )


def aux_dir(grid: str | GridSpec, family: str, root: str | os.PathLike | None = None) -> Path:
    """Directory holding the aux artifacts (``graph.pickle`` + ``settings.csv``) for ``grid`` in
    ``family``: ``<aux-root>/<family>/<raw_foldername>/``. Non-destructive overlay; see
    :func:`resolve_aux_root`."""
    g = get_grid(grid)
    if family == "fine_tune":
        family = "adapt_grid"
    elif family == "train":
        family = "multi_grid"
    return resolve_aux_root(root) / family / g.raw_foldername
