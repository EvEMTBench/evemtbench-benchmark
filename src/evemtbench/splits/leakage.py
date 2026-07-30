"""Leakage validators (Benchmark Plan §7.6) — run as CI gates.

* :func:`assert_disjoint_triples` — no ``(family, grid, sim_idx)`` appears in more than one
  partition of a held_out split. (train/val/test come from the adapt_grid family; the benchmark
  partition is a different family, so it is inherently disjoint from training.)
* :func:`verify_held_out_regenerates` — committed held_out files reproduce from the generator at
  the recorded seed (guards against silent generator drift).
"""

from __future__ import annotations

import os

from ..data.roots import get_grid
from ..version import SPLITS_VERSION
from .generator import generate_held_out
from .loader import read_held_out_splits, read_meta


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


def verify_held_out_regenerates(
    grid: str,
    *,
    root: str | os.PathLike | None = None,
    version: str = SPLITS_VERSION,
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
