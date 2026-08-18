"""Leakage validators (Benchmark Plan §7.6) — run as CI gates.

* :func:`assert_disjoint_triples` — no ``(family, grid, sim_idx)`` appears in more than one
  partition of a held_out split. (train/val/test come from the adapt_grid family; the benchmark
  partition is a different family, so it is inherently disjoint from training.)
* :func:`assert_group_disjoint` — no *group* (simulation ``result_id`` / topology ``graph_id``)
  appears in more than one of train/val/test — the grouped-split guarantee for the multi_grid
  source corpora, where one episode is one line of one simulation and episode-disjointness alone
  does not prevent sibling-line or shared-topology leakage.
* :func:`verify_held_out_regenerates` — committed held_out files reproduce from the generator at
  the recorded seed (guards against silent generator drift).
* :func:`verify_protocol_regenerates` — same guarantee for any generated protocol
  (``multi_grid_pretrain``), honoring the recorded ``grouping``.
"""

from __future__ import annotations

import os

from ..data.roots import get_grid
from .generator import _GENERATED_PROTOCOLS, generate_held_out
from .loader import read_held_out_splits, read_meta, read_splits


class LeakageError(AssertionError):
    """Raised when a split set violates a leakage guarantee."""


def assert_disjoint_triples(splits: dict[str, list[tuple[str, str, int]]]) -> None:
    """No ``(family, grid, sim_idx)`` triple may appear in more than one partition."""
    seen: dict[tuple, str] = {}
    for name, triples in splits.items():
        for t in triples:
            if t in seen:
                raise LeakageError(f"{t} appears in both {seen[t]!r} and {name!r} partitions")
            seen[t] = name


def assert_group_disjoint(
    splits: dict[str, list[tuple[str, str, int]]],
    groups: dict[int, str],
    *,
    level: str = "group",
) -> None:
    """No group key may span more than one of the train/val/test partitions, and every episode
    must carry a group key (missing group metadata is itself a leakage-check failure)."""
    seen: dict[str, str] = {}
    for name in ("train", "val", "test"):
        for _family, _grid, sim_idx in splits.get(name, []):
            if sim_idx not in groups:
                raise LeakageError(f"episode {sim_idx} has no {level} group key (metadata missing)")
            key = str(groups[sim_idx])
            prev = seen.get(key)
            if prev is not None and prev != name:
                raise LeakageError(
                    f"{level} {key!r} appears in both {prev!r} and {name!r} partitions"
                )
            seen[key] = name


def verify_protocol_regenerates(
    protocol: str,
    grid: str,
    *,
    root: str | os.PathLike | None = None,
    version: str | None = None,
    splits_root: str | os.PathLike | None = None,
) -> None:
    """Assert a committed generated-protocol split (``held_out`` / ``multi_grid_pretrain``) equals
    a fresh generation at the recorded seed, honoring the recorded ``stratify_by`` AND ``grouping``
    (both absent/None on legacy metas)."""
    key = get_grid(grid).key
    committed = read_splits(protocol, key, version=version, splits_root=splits_root)
    meta = read_meta(protocol, key, version=version, splits_root=splits_root)
    train_family, benchmark_family, _grids = _GENERATED_PROTOCOLS[protocol]
    art = generate_held_out(
        grid,
        protocol=protocol,
        train_family=train_family,
        benchmark_family=benchmark_family,
        root=root,
        seed=meta["seed"],
        stratify_by=meta.get("stratify_by"),
        grouping=meta.get("grouping"),
    )
    for name, triples in committed.items():
        regen = [tuple(t.split(":")) for t in art.splits[name]]
        regen_typed = [(f, g, int(i)) for f, g, i in regen]
        if triples != regen_typed:
            raise LeakageError(f"committed {protocol}/{key}/{name} does not match regeneration")


def verify_held_out_regenerates(
    grid: str,
    *,
    root: str | os.PathLike | None = None,
    version: str | None = None,
    splits_root: str | os.PathLike | None = None,
) -> None:
    """Assert committed held_out splits equal a fresh generation at the recorded seed —
    honoring the recorded ``stratify_by`` (absent/None on pre-stratification metas)."""
    key = get_grid(grid).key
    committed = read_held_out_splits(key, version=version, splits_root=splits_root)
    meta = read_meta("held_out", key, version=version, splits_root=splits_root)
    art = generate_held_out(grid, root=root, seed=meta["seed"], stratify_by=meta.get("stratify_by"))
    for name, triples in committed.items():
        regen = [tuple(t.split(":")) for t in art.splits[name]]
        regen_typed = [(f, g, int(i)) for f, g, i in regen]
        if triples != regen_typed:
            raise LeakageError(f"committed held_out/{key}/{name} does not match regeneration")
