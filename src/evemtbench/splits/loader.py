"""Read committed split files — the source of truth consumed by ``load_task``.

A held_out split for a grid lives at ``v<version>/held_out/<grid>/`` as
``train/val/test/benchmark.txt`` (each line a ``family:grid:sim_idx`` triple) plus
``meta.json``. Reading never recomputes anything; the files are authoritative so a generator
bug cannot silently change anyone's splits.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..version import splits_version
from .generator import PACKAGED_SPLITS_DIR
from .protocols import HELD_OUT_SPLIT_NAMES


def split_dir(
    protocol: str,
    grid: str,
    *,
    version: str | None = None,
    splits_root: str | os.PathLike | None = None,
) -> Path:
    """Committed split directory for ``protocol``/``grid``. ``version=None`` (the default)
    resolves per protocol (:func:`evemtbench.version.splits_version`): held_out and the transfer
    protocols read the frozen v1.0 tree, ``multi_grid_pretrain`` the corrected v1.1 grouped tree.
    An explicit ``version`` always wins (historical inspection, fixtures)."""
    root = Path(splits_root) if splits_root is not None else PACKAGED_SPLITS_DIR
    version = splits_version(protocol) if version is None else version
    return root / f"v{version}" / protocol / grid


def read_meta(
    protocol: str,
    grid: str,
    *,
    version: str | None = None,
    splits_root: str | os.PathLike | None = None,
) -> dict:
    path = split_dir(protocol, grid, version=version, splits_root=splits_root) / "meta.json"
    return json.loads(path.read_text())


def read_triple_split(
    protocol: str,
    grid: str,
    split: str,
    *,
    version: str | None = None,
    splits_root: str | os.PathLike | None = None,
) -> list[tuple[str, str, int]]:
    """Read one split file of ``family:grid:sim_idx`` triples."""
    path = split_dir(protocol, grid, version=version, splits_root=splits_root) / f"{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(
            f"split file not found: {path}. Generate it with `evemtbench-splits`."
        )
    triples = []
    for line in path.read_text().split():
        family, g, sim_idx = line.split(":")
        triples.append((family, g, int(sim_idx)))
    return triples


def read_splits(
    protocol: str,
    grid: str,
    *,
    version: str | None = None,
    splits_root: str | os.PathLike | None = None,
) -> dict[str, list[tuple[str, str, int]]]:
    """Read all 4 partitions (train/val/test/benchmark) of ``protocol``/``grid`` as
    family:grid:sim_idx triples. Used by the committed ``held_out`` and ``multi_grid_pretrain``
    splits (the transfer protocols reuse the target grid's ``held_out`` split at load time)."""
    return {
        name: read_triple_split(protocol, grid, name, version=version, splits_root=splits_root)
        for name in HELD_OUT_SPLIT_NAMES
    }


def read_held_out_splits(
    grid: str,
    *,
    version: str | None = None,
    splits_root: str | os.PathLike | None = None,
) -> dict[str, list[tuple[str, str, int]]]:
    """Read all 4 held_out partitions (train/val/test/benchmark) as family:grid:sim_idx triples."""
    return read_splits("held_out", grid, version=version, splits_root=splits_root)
