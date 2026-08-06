"""Deterministic split generation for the ``held_out`` protocol.

The committed ``.txt`` files are the source of truth; this regenerates them byte-identically
given the same (episodes, fractions, seed). A held_out split for a grid is a directory of
``train.txt`` / ``val.txt`` / ``test.txt`` / ``benchmark.txt`` (each line a
``family:grid:sim_idx`` triple) plus ``meta.json`` recording seed, fractions, counts, version.

train/val/test partition the **adapt_grid** family episodes (70/15/15); benchmark is the entire
**benchmark** family of the same grid (a second, never-trained-on held-out test).

Run ``evemtbench-splits`` (this module's ``main``) on a machine with the dataset mounted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data.roots import GridSpec, get_grid
from ..data.windows import read_episode_ids, read_episode_labels
from ..version import SPLITS_VERSION
from .protocols import HELD_OUT_FRACTIONS, HELD_OUT_SPLIT_NAMES

# Packaged splits live next to this module: src/evemtbench/splits/v<version>/...
PACKAGED_SPLITS_DIR = Path(__file__).resolve().parent

# Per-grid seed offset keeps grids independent yet reproducible.
BASE_SEED = 20260618


def split_episodes(
    episode_ids: list[int],
    fractions: tuple[float, float, float],
    seed: int,
    strata: dict[int, str] | None = None,
) -> dict[str, list[int]]:
    """Deterministically partition sorted episode ids into train/val/test by fraction.

    ``strata=None`` → a single seeded permutation (uniform). ``strata`` (episode id → class label,
    e.g. ``event_type``) → a **stratified group split**: each class is partitioned 70/15/15
    independently, so rare, grid-restricted classes (HIF, gen_trip, …) are spread across the
    partitions instead of landing wholesale in one. Both paths are byte-reproducible for a given
    seed (sorted ids; classes consumed in sorted order).

    **Limit of that guarantee, stated precisely:** the per-class sizes are rounded independently
    (``round(0.70*m)``, ``round(0.15*m)``, remainder), so a class needs **m >= 6 episodes** to be
    present in all three partitions. Below that some partition is empty for that class: m=1 →
    (1,0,0); m=2 → (1,0,1); m=3 → (2,0,1) — no val; m=4 → (3,1,0) and m=5 → (4,1,0) — no test. This
    is not hypothetical arithmetic to be relied on: it is what
    ``test_stratified_split_small_classes`` pins. Every committed v1.0 split has >= 31 episodes per
    class per partition, so no published split is affected — but a future grid with a genuinely
    rare class would silently lose it from a partition, and per-class metrics there would read n/a
    rather than wrong. The rounding is NOT changed here: doing so would alter the committed splits.
    """
    ids = np.array(sorted(int(i) for i in episode_ids), dtype=np.int64)
    rng = np.random.default_rng(seed)
    if strata is None:
        n = len(ids)
        n_train = int(round(fractions[0] * n))
        n_val = int(round(fractions[1] * n))
        perm = rng.permutation(n)
        chunks = {
            "train": perm[:n_train],
            "val": perm[n_train : n_train + n_val],
            "test": perm[n_train + n_val :],
        }
        return {name: sorted(int(i) for i in ids[idx]) for name, idx in chunks.items()}

    by_class: dict[str, list[int]] = {}
    for i in ids:
        by_class.setdefault(str(strata.get(int(i), "__unlabeled__")), []).append(int(i))
    out: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for cls in sorted(by_class):
        cids = np.array(by_class[cls], dtype=np.int64)
        cids = cids[rng.permutation(len(cids))]  # rng consumed in sorted-class order → reproducible
        m = len(cids)
        m_train = int(round(fractions[0] * m))
        m_val = int(round(fractions[1] * m))
        out["train"] += cids[:m_train].tolist()
        out["val"] += cids[m_train : m_train + m_val].tolist()
        out["test"] += cids[m_train + m_val :].tolist()
    return {name: sorted(int(i) for i in v) for name, v in out.items()}


def _grid_seed(grid_key: str) -> int:
    # Stable across processes/machines (builtin hash() is salted by PYTHONHASHSEED).
    digest = int(hashlib.sha256(grid_key.encode()).hexdigest()[:8], 16)
    return BASE_SEED + (digest % 100_000)


def _triple(family: str, grid: str, sim_idx: int) -> str:
    return f"{family}:{grid}:{sim_idx}"


@dataclass
class HeldOutArtifact:
    protocol: str
    grid: str
    seed: int
    splits: dict[str, list[str]]  # split name -> list of "family:grid:sim_idx" triples
    stratify_by: str | None = None  # strata key used (e.g. "event_type"), or None for uniform
    strata: dict[int, str] | None = None  # episode -> class (for meta.json histograms), optional


def generate_held_out(
    grid: str | GridSpec,
    *,
    protocol: str = "held_out",
    train_family: str = "adapt_grid",
    benchmark_family: str | None = "benchmark",
    root: str | os.PathLike | None = None,
    seed: int | None = None,
    stratify_by: str | None = None,
) -> HeldOutArtifact:
    """Build a 4-partition episode split for one grid under ``protocol``.

    train/val/test = the ``train_family`` episodes split 70/15/15; benchmark = ALL episodes of
    ``benchmark_family`` (held out, never trained on), or empty when ``benchmark_family`` is None.
    Defaults reproduce the ``held_out`` protocol (adapt_grid + benchmark). The
    ``multi_grid_pretrain`` protocol passes ``train_family="multi_grid", benchmark_family=None``
    (in-corpus split of the
    pre-training family, no separate held-out family). With ``stratify_by`` (e.g. ``"event_type"``)
    the train-family split is class-stratified so rare classes are spread across the partitions
    rather than landing wholesale in one (presence in *all three* requires >= 6 episodes of the
    class — see :func:`split_episodes`); default None reproduces the uniform split byte-for-byte.
    The per-grid seed is namespaced by protocol so different protocols' splits of the same grid are
    independent.
    """
    g = get_grid(grid)
    seed = _grid_seed(protocol + "_" + g.key) if seed is None else seed

    ft_eps = read_episode_ids(g, train_family, root=root)
    strata = (
        read_episode_labels(g, train_family, column=stratify_by, root=root) if stratify_by else None
    )
    parts = split_episodes(ft_eps, HELD_OUT_FRACTIONS, seed, strata=strata)

    # The benchmark family may be absent (a not-yet-windowed grid, or a pre-training family that has
    # no separate held-out family); in that case the split carries an empty benchmark partition.
    bm_eps: list[int] = []
    if benchmark_family is not None:
        try:
            bm_eps = read_episode_ids(g, benchmark_family, root=root)
        except FileNotFoundError:
            bm_eps = []

    splits = {
        "train": [_triple(train_family, g.key, i) for i in parts["train"]],
        "val": [_triple(train_family, g.key, i) for i in parts["val"]],
        "test": [_triple(train_family, g.key, i) for i in parts["test"]],
        "benchmark": (
            [_triple(benchmark_family, g.key, i) for i in bm_eps] if benchmark_family else []
        ),
    }
    return HeldOutArtifact(protocol, g.key, seed, splits, stratify_by=stratify_by, strata=strata)


def write_held_out_artifact(
    artifact: HeldOutArtifact, splits_root: Path, version: str = SPLITS_VERSION
) -> Path:
    """Write the 4 partition .txt files + meta.json. Returns the directory."""
    out = Path(splits_root) / f"v{version}" / artifact.protocol / artifact.grid
    out.mkdir(parents=True, exist_ok=True)
    for name in HELD_OUT_SPLIT_NAMES:
        lines = "\n".join(artifact.splits[name])
        (out / f"{name}.txt").write_text(lines + ("\n" if lines else ""))
    meta = {
        "protocol": artifact.protocol,
        "grid": artifact.grid,
        "seed": artifact.seed,
        "splits_version": version,
        "fractions": list(HELD_OUT_FRACTIONS),
        "counts": {n: len(artifact.splits[n]) for n in HELD_OUT_SPLIT_NAMES},
        "stratify_by": artifact.stratify_by,
    }
    if artifact.strata is not None:  # per-split class histograms (self-documenting stratification)
        hist: dict[str, dict[str, int]] = {}
        for name in ("train", "val", "test"):
            counts = Counter(
                artifact.strata.get(int(tr.split(":")[2]), "__unlabeled__")
                for tr in artifact.splits[name]
            )
            hist[name] = dict(sorted(counts.items()))
        meta["class_histograms"] = hist
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return out


# protocol -> (train_family, benchmark_family, default grids). held_out splits the adapt_grid
# family of each reference grid; multi_grid_pretrain splits the multi_grid pre-training corpus
# in-place (no separate benchmark family). Transfer protocols reuse the target grid's held_out
# split at load time and need no committed files of their own.
_GENERATED_PROTOCOLS: dict[str, tuple[str, str | None, list[str]]] = {
    "held_out": (
        "adapt_grid",
        "benchmark",
        ["double_line", "cigre_mv", "testgrid_110kv", "ieee39"],
    ),
    "multi_grid_pretrain": (
        "multi_grid",
        None,
        ["template_20kv", "template_110kv", "template_345kv"],
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate committed split files.")
    parser.add_argument("--protocol", default="held_out", choices=sorted(_GENERATED_PROTOCOLS))
    parser.add_argument("--root", default=None, help="data root override (else env/default)")
    parser.add_argument(
        "--out", default=str(PACKAGED_SPLITS_DIR), help="splits root (default: packaged dir)"
    )
    parser.add_argument(
        "--grids", nargs="*", default=None, help="grid keys (else protocol default)"
    )
    parser.add_argument(
        "--stratify-by",
        default=None,
        help="episode-label column for a class-stratified train split (e.g. event_type); "
        "recorded in meta.json so leakage.verify_held_out_regenerates reproduces it",
    )
    args = parser.parse_args()

    train_family, benchmark_family, default_grids = _GENERATED_PROTOCOLS[args.protocol]
    for grid_key in args.grids or default_grids:
        art = generate_held_out(
            grid_key,
            protocol=args.protocol,
            train_family=train_family,
            benchmark_family=benchmark_family,
            root=args.root,
            stratify_by=args.stratify_by,
        )
        out = write_held_out_artifact(art, Path(args.out))
        counts = {n: len(art.splits[n]) for n in HELD_OUT_SPLIT_NAMES}
        print(
            f"[{args.protocol}/{grid_key}] seed={art.seed} stratify_by={art.stratify_by} "
            f"counts={counts} -> {out}"
        )


if __name__ == "__main__":
    main()
