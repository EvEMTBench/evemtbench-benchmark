"""Topology-graph access (auxiliary overlay).

Loads the per-grid networkx topology graph packaged alongside the windowed data (extracted into the
aux overlay by ``evemtbench.data.extract_aux``). Nodes are buses/elements and edges are
transformers/lines/switching branches; each carries nested ``<name>_param_dict`` attributes
(electrical nameplate + impedances). This module hides the pickle/networkx handling behind a small
interface and provides a best-effort flattener from the nested param dicts to per-node/edge scalar
feature vectors (the raw material for a future GNN baseline / cross-topology protocol).

``networkx`` is an optional dependency (``pip install 'evemtbench[graphs]'``); it is imported lazily
so the core product layer stays torch/networkx-free. A change to what a graph *contains* is a data
(producer) change, not a benchmark-contract change — this module only reads.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .roots import GridSpec, aux_dir

if TYPE_CHECKING:  # pragma: no cover - typing only
    import networkx as nx

_GRAPH_FILE = "graph.pickle"


def _require_networkx():
    try:
        import networkx as nx
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "topology graphs need networkx; install the extra: pip install 'evemtbench[graphs]'"
        ) from e
    return nx


def _graph_path(grid: str | GridSpec, family: str, root: str | os.PathLike | None = None) -> Path:
    return aux_dir(grid, family, root) / _GRAPH_FILE


def load_grid_graph(
    grid: str | GridSpec, family: str = "benchmark", root: str | os.PathLike | None = None
) -> nx.Graph:
    """Load the topology graph for ``grid``/``family`` from the aux overlay. Fails loud if absent
    (run ``python -m evemtbench.data.extract_aux`` first)."""
    _require_networkx()  # needed to unpickle the nx graph object
    path = _graph_path(grid, family, root)
    if not path.exists():
        raise FileNotFoundError(
            f"no topology graph at {path}. Extract it first: "
            f"python -m evemtbench.data.extract_aux --grid <key> --family {family}"
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


def graph_summary(g: nx.Graph) -> dict[str, Any]:
    """Cheap structural summary (nodes/edges/attribute-key sets) — no feature flattening."""
    node_keys: set[str] = set()
    for _, data in g.nodes(data=True):
        node_keys.update(data.keys())
    edge_keys: set[str] = set()
    for *_, data in g.edges(data=True):
        edge_keys.update(data.keys())
    return {
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "directed": g.is_directed(),
        "multigraph": g.is_multigraph(),
        "node_attr_keys": sorted(node_keys),
        "edge_attr_keys": sorted(edge_keys),
    }


def _flatten_scalars(attrs: dict[str, Any]) -> dict[str, float]:
    """Flatten an element's nested ``*_param_dict`` attributes into scalar features.

    Each node/edge carries one or more ``<name>_param_dict`` values (dicts of nameplate params).
    We keep only int/float leaf values (bools excluded), keyed ``<param>`` (later params win on a
    name clash). Strings/lists (types, curves) are dropped here — a GNN encoder can one-hot them
    separately from :func:`graph_summary`. Best-effort: robust to schema variation across grids.
    """
    out: dict[str, float] = {}
    for val in attrs.values():
        if isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)):
                    out[k] = float(v)
    return out


def graph_node_features(g: nx.Graph) -> dict[Any, dict[str, float]]:
    """Per-node scalar feature dict (flattened from nested param dicts)."""
    return {n: _flatten_scalars(data) for n, data in g.nodes(data=True)}


def graph_edge_features(g: nx.Graph) -> dict[Any, dict[str, float]]:
    """Per-edge scalar feature dict, keyed by the edge endpoints tuple."""
    return {(u, v): _flatten_scalars(data) for u, v, data in g.edges(data=True)}


def list_topologies(
    family: str = "multi_grid",
    grid: str | GridSpec | None = None,
    root: str | os.PathLike | None = None,
) -> list[Path]:
    """List all graph pickles in a family's aux dir (e.g. the varied ``multi_grid`` corpus).

    For the reference grids this is just the single ``graph.pickle``; for ``multi_grid`` it is the
    per-topology corpus used by the (planned) cross-topology protocol.
    """
    if grid is None:
        # multi_grid graphs live outside the per-grid layout; glob the family root recursively.
        from .roots import resolve_aux_root

        base = resolve_aux_root(root) / family
    else:
        base = aux_dir(grid, family, root)
    return sorted(base.rglob("*.pickle")) if base.exists() else []
