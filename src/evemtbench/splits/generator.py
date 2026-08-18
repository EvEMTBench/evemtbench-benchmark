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


def split_grouped_episodes(
    episode_ids: list[int],
    groups: dict[int, str],
    fractions: tuple[float, float, float],
    seed: int,
    strata: dict[int, str] | None = None,
) -> dict[str, list[int]]:
    """Deterministically partition episodes into train/val/test with **whole groups** kept together.

    ``groups`` maps every episode id to a group key (e.g. its simulation ``result_id`` or its
    generated-topology ``graph_id``); no group ever straddles a partition boundary, so sibling
    episodes of one simulation — or all simulations of one topology — always land in the same
    partition. Fractions are targeted on **episode counts** (groups vary in size).

    Assignment is a proportional greedy against ONE global per-partition episode budget
    (largest relative remaining capacity wins), with stratification controlling the ORDER: each
    group is labeled with the **rarest event class it contains** (by corpus-wide totals), strata
    are processed rarest-first, groups within a stratum largest-first. Rare-class groups are
    therefore placed while the budgets are still balanced, which spreads them ~proportionally
    (for simulation grouping, where a fault simulation's stratum is its fault class, this is
    near-exact per-class stratification — the one fault episode plus its no_event sibling lines
    move as one unit); for topology grouping, where every event class spans most topologies, it
    degrades gracefully into proportional size-balancing, and class proportions follow because
    classes are near-uniform across topologies (the split audit verifies coverage, never assumes
    it). Two rejected designs, for the record: per-class-demand assignment (with near-universal
    classes it hands out almost every group during the rarest class's phase — observed 59/20/21
    instead of 70/15/15 on the real corpora) and per-stratum budgets (degenerate on small corpora:
    singleton strata all land in train).

    Deterministic and byte-reproducible for a given seed: every tie falls back to a seeded
    permutation of the sorted group keys, then to the fixed partition order.

    **Coverage limit, stated precisely:** a class can appear in all three partitions only if it
    occurs in >= 3 distinct groups; a class confined to a single group lands wholesale in one
    partition. That is a property of the data (class-topology concentration), not of the assignment
    — the split audit reports per-class group counts so such classes are visible, never silent.
    """
    ids = sorted(int(i) for i in episode_ids)
    missing = [i for i in ids if i not in groups]
    if missing:
        raise ValueError(
            f"{len(missing)} episode(s) have no group key (first: {missing[:5]}); "
            "grouped splitting requires complete group metadata"
        )
    names = ("train", "val", "test")

    # group -> episodes, group -> class histogram (single pseudo-class when unstratified)
    by_group: dict[str, list[int]] = {}
    for i in ids:
        by_group.setdefault(str(groups[i]), []).append(i)
    cls_of = (lambda i: str(strata.get(i, "__unlabeled__"))) if strata else (lambda i: "__all__")
    group_hist: dict[str, Counter] = {
        g: Counter(cls_of(i) for i in eps) for g, eps in by_group.items()
    }
    class_total: Counter = Counter()
    for h in group_hist.values():
        class_total.update(h)

    # seeded permutation of the sorted group keys -> the deterministic tie-break order
    keys = sorted(by_group)
    rng = np.random.default_rng(seed)
    order = {k: r for r, k in enumerate(np.array(keys)[rng.permutation(len(keys))])}

    # stratum of a group = the rarest class it contains (corpus-wide totals; ties lexicographic)
    def rarest(hist: Counter) -> str:
        return min(hist, key=lambda c: (class_total[c], c))

    strata_groups: dict[str, list[str]] = {}
    for g, h in group_hist.items():
        strata_groups.setdefault(rarest(h), []).append(g)

    n_total = len(ids)
    target = {p: f * n_total for p, f in zip(names, fractions, strict=True)}
    filled = dict.fromkeys(names, 0)
    assignment: dict[str, str] = {}
    for label in sorted(strata_groups, key=lambda c: (class_total[c], c)):  # rarest stratum first
        # largest group first tightens the endgame fit; ties resolve by the seeded permutation
        for g in sorted(strata_groups[label], key=lambda k: (-len(by_group[k]), order[k])):
            # partition with the largest relative remaining global capacity; ties -> fixed order
            p = max(
                names,
                key=lambda q: (
                    (target[q] - filled[q]) / target[q] if target[q] > 0 else 0.0,
                    -names.index(q),
                ),
            )
            assignment[g] = p
            filled[p] += len(by_group[g])

    out: dict[str, list[int]] = {p: [] for p in names}
    for g, p in assignment.items():
        out[p] += by_group[g]
    return {p: sorted(v) for p, v in out.items()}


def _grid_seed(grid_key: str) -> int:
    # Stable across processes/machines (builtin hash() is salted by PYTHONHASHSEED).
    digest = int(hashlib.sha256(grid_key.encode()).hexdigest()[:8], 16)
    return BASE_SEED + (digest % 100_000)


def _triple(family: str, grid: str, sim_idx: int) -> str:
    return f"{family}:{grid}:{sim_idx}"


# Grouping level -> the episode-constant y-parquet column carrying the group key. The multi_grid
# corpora encode one *line* of one simulation per episode (sample_id = result_id*10^4 + line_idx),
# so an ungrouped episode split scatters sibling lines of a simulation — and whole generated
# topologies — across partitions. "simulation" keeps each simulation (result_id) together;
# "topology" keeps each generated topology (graph_id), and with it all its simulations and lines.
GROUPING_COLUMNS = {"simulation": "result_id", "topology": "graph_id"}


@dataclass
class HeldOutArtifact:
    protocol: str
    grid: str
    seed: int
    splits: dict[str, list[str]]  # split name -> list of "family:grid:sim_idx" triples
    stratify_by: str | None = None  # strata key used (e.g. "event_type"), or None for uniform
    strata: dict[int, str] | None = None  # episode -> class (for meta.json histograms), optional
    grouping: str | None = None  # "simulation" | "topology" | None (legacy episode-level)
    group_column: str | None = None  # y-parquet column the group keys came from
    groups: dict[int, str] | None = None  # episode -> group key (for meta.json group counts)


def generate_held_out(
    grid: str | GridSpec,
    *,
    protocol: str = "held_out",
    train_family: str = "adapt_grid",
    benchmark_family: str | None = "benchmark",
    root: str | os.PathLike | None = None,
    seed: int | None = None,
    stratify_by: str | None = None,
    grouping: str | None = None,
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

    ``grouping`` (``"simulation"`` / ``"topology"``) switches to the **group-aware** split
    (:func:`split_grouped_episodes`): every episode of one group (its ``result_id`` simulation, or
    its ``graph_id`` generated topology) stays in one partition — the leakage-free source-corpus
    splits for ``multi_grid_pretrain``. The seed is then additionally namespaced by the grouping so
    the simulation- and topology-grouped splits of one grid are independent draws. Default ``None``
    reproduces the legacy episode-level split byte-for-byte.
    """
    g = get_grid(grid)
    seed_key = f"{protocol}_{grouping}_{g.key}" if grouping else f"{protocol}_{g.key}"
    seed = _grid_seed(seed_key) if seed is None else seed

    ft_eps = read_episode_ids(g, train_family, root=root)
    strata = (
        read_episode_labels(g, train_family, column=stratify_by, root=root) if stratify_by else None
    )
    groups: dict[int, str] | None = None
    group_column: str | None = None
    if grouping is not None:
        if grouping not in GROUPING_COLUMNS:
            raise ValueError(f"grouping {grouping!r} not in {sorted(GROUPING_COLUMNS)}")
        group_column = GROUPING_COLUMNS[grouping]
        groups = read_episode_labels(g, train_family, column=group_column, root=root)
        parts = split_grouped_episodes(ft_eps, groups, HELD_OUT_FRACTIONS, seed, strata=strata)
    else:
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
    return HeldOutArtifact(
        protocol,
        g.key,
        seed,
        splits,
        stratify_by=stratify_by,
        strata=strata,
        grouping=grouping,
        group_column=group_column,
        groups=groups,
    )


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
    if artifact.grouping is not None:  # group-aware split: record how to regenerate + verify it
        meta["grouping"] = artifact.grouping
        meta["group_column"] = artifact.group_column
        assert artifact.groups is not None
        meta["group_counts"] = {
            name: len({artifact.groups[int(tr.split(":")[2])] for tr in artifact.splits[name]})
            for name in ("train", "val", "test")
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
    parser.add_argument(
        "--grouping",
        default=None,
        choices=sorted(GROUPING_COLUMNS),
        help="group-aware split level (multi_grid source corpora): 'simulation' keeps each "
        "result_id together, 'topology' keeps each graph_id (and thus all its simulations and "
        "sibling lines) together; recorded in meta.json for regeneration verification",
    )
    parser.add_argument(
        "--splits-version",
        default=SPLITS_VERSION,
        help="version directory to write under the splits root (e.g. 1.1)",
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
            grouping=args.grouping,
        )
        out = write_held_out_artifact(art, Path(args.out), version=args.splits_version)
        counts = {n: len(art.splits[n]) for n in HELD_OUT_SPLIT_NAMES}
        print(
            f"[{args.protocol}/{grid_key}] seed={art.seed} stratify_by={art.stratify_by} "
            f"grouping={art.grouping} counts={counts} -> {out}"
        )


if __name__ == "__main__":
    main()
