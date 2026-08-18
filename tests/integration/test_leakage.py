"""held_out + multi_grid split structure and leakage gates (fixture + committed packaged splits)."""

import hashlib
import json

import pytest

from evemtbench.data.roots import GRIDS
from evemtbench.splits import leakage
from evemtbench.splits.generator import generate_held_out, write_held_out_artifact
from evemtbench.splits.loader import read_held_out_splits, read_meta, read_splits, split_dir
from evemtbench.splits.protocols import HELD_OUT_SPLIT_NAMES

PRETRAIN_GRIDS = ("template_20kv", "template_110kv", "template_345kv")


def test_generate_held_out_partition_families_and_grids(fixture_root):
    art = generate_held_out("double_line", root=fixture_root)
    assert set(art.splits) == set(HELD_OUT_SPLIT_NAMES)
    for name in ("train", "val", "test"):
        fams = {t.split(":")[0] for t in art.splits[name]}
        assert fams == {"adapt_grid"}
    assert {t.split(":")[0] for t in art.splits["benchmark"]} == {"benchmark"}
    grids = {t.split(":")[1] for part in art.splits.values() for t in part}
    assert grids == {"double_line"}


def test_train_val_test_are_episode_disjoint(fixture_root):
    art = generate_held_out("double_line", root=fixture_root)
    triples = {n: [tuple(t.split(":")) for t in art.splits[n]] for n in HELD_OUT_SPLIT_NAMES}
    leakage.assert_disjoint_triples(triples)  # must not raise


def test_disjoint_validator_catches_overlap():
    with pytest.raises(leakage.LeakageError, match="appears in both"):
        leakage.assert_disjoint_triples(
            {"train": [("adapt_grid", "double_line", 1)], "val": [("adapt_grid", "double_line", 1)]}
        )


def test_held_out_regenerates_identically(fixture_root, held_out_splits):
    leakage.verify_held_out_regenerates(
        "double_line", root=fixture_root, splits_root=held_out_splits
    )


def test_stratified_split_regenerates_from_meta(fixture_root, tmp_path):
    """A committed STRATIFIED split must round-trip: verify_held_out_regenerates reads the
    recorded ``stratify_by`` from meta.json and reproduces byte-identical partitions (the M7
    flip makes this the production path)."""
    from evemtbench.splits.generator import generate_held_out, write_held_out_artifact
    from evemtbench.version import SPLITS_VERSION

    art = generate_held_out("double_line", root=fixture_root, stratify_by="event_type")
    write_held_out_artifact(art, tmp_path, version=SPLITS_VERSION)
    leakage.verify_held_out_regenerates(
        "double_line", root=fixture_root, splits_root=tmp_path
    )  # must not raise


@pytest.mark.parametrize("grid", sorted(GRIDS))
def test_committed_splits_are_disjoint_and_family_pure(grid):
    """The COMMITTED packaged splits (what users consume) are episode-disjoint with the right
    family per partition — a data-free gate on the split artifacts themselves (CI-safe: reads
    only files shipped inside the package, no dataset mount)."""
    try:
        triples = read_held_out_splits(grid)
    except FileNotFoundError:
        pytest.skip(f"no committed held_out split for {grid!r}")
    leakage.assert_disjoint_triples(triples)  # must not raise
    for name in ("train", "val", "test"):
        assert triples[name], f"{grid}: empty {name} partition"
        assert all(fam == "adapt_grid" for fam, _g, _i in triples[name])
    # benchmark partition (empty until that grid's benchmark family is windowed) is family-pure
    assert all(fam == "benchmark" for fam, _g, _i in triples["benchmark"])


@pytest.mark.parametrize("grid", PRETRAIN_GRIDS)
def test_committed_multi_grid_splits_are_grouped_and_audited(grid):
    """The COMMITTED multi_grid source-corpus splits are family-pure, episode-disjoint,
    topology-grouped by declaration, and their committed audit.json (produced on-host against the
    dataset metadata) reports zero overlap at every level and hashes that still match the split
    files — a data-free CI gate against artifact drift. (The on-host counterparts that recompute
    everything from the dataset are ``evemtbench.splits.audit --protocol multi_grid_pretrain``
    and ``leakage.verify_protocol_regenerates``.)"""
    triples = read_splits("multi_grid_pretrain", grid)
    leakage.assert_disjoint_triples(triples)
    for name in ("train", "val", "test"):
        assert triples[name], f"{grid}: empty {name} partition"
        assert all(fam == "multi_grid" for fam, _g, _i in triples[name])
    assert triples["benchmark"] == []  # in-corpus split: no separate benchmark family

    meta = read_meta("multi_grid_pretrain", grid)
    assert meta.get("grouping") == "topology"
    assert meta.get("group_column") == "graph_id"
    assert isinstance(meta.get("seed"), int)
    assert set(meta.get("group_counts", {})) == {"train", "val", "test"}
    assert all(v > 0 for v in meta["group_counts"].values())

    d = split_dir("multi_grid_pretrain", grid)
    audit = json.loads((d / "audit.json").read_text())
    assert audit["ok"], audit["issues"]
    assert audit["grouping"] == "topology"
    for level in ("episode", "simulation", "topology"):
        assert all(v == 0 for v in audit["overlaps"][level].values()), (level, audit["overlaps"])
    # the audited files are the committed files (hash pin, incl. meta.json)
    for fname, digest in audit["partition_hashes"].items():
        actual = hashlib.sha256((d / fname).read_bytes()).hexdigest()
        assert actual == digest, f"{grid}/{fname} drifted since its committed audit"
    # audit covers exactly the committed episode counts
    assert audit["n_episodes"] == sum(len(triples[n]) for n in ("train", "val", "test"))
    for name in ("train", "val", "test"):
        assert audit["counts"][name]["episodes"] == len(triples[name])
        assert audit["counts"][name]["topologies"] == meta["group_counts"][name]


def test_grouped_split_regenerates_from_meta(fixture_root, tmp_path):
    """A committed GROUPED split round-trips: verify_protocol_regenerates reads the recorded
    ``grouping`` + ``stratify_by`` + seed from meta.json and reproduces byte-identical
    partitions — the determinism gate for the corrected multi_grid splits."""
    art = generate_held_out(
        "template_20kv",
        protocol="multi_grid_pretrain",
        train_family="multi_grid",
        benchmark_family=None,
        root=fixture_root,
        stratify_by="event_type",
        grouping="topology",
    )
    write_held_out_artifact(art, tmp_path, version="1.1")
    leakage.verify_protocol_regenerates(
        "multi_grid_pretrain", "template_20kv", root=fixture_root, splits_root=tmp_path
    )  # must not raise


def test_grouped_fixture_split_is_group_disjoint(fixture_root):
    """End-to-end on the synthetic corpus: the generated topology-grouped split keeps every
    simulation AND topology in one partition (checked against the fixture's own metadata)."""
    from evemtbench.data.windows import read_episode_labels

    art = generate_held_out(
        "template_20kv",
        protocol="multi_grid_pretrain",
        train_family="multi_grid",
        benchmark_family=None,
        root=fixture_root,
        stratify_by="event_type",
        grouping="topology",
    )
    triples = {n: [tuple(t.split(":")) for t in art.splits[n]] for n in ("train", "val", "test")}
    typed = {n: [(f, g, int(i)) for f, g, i in v] for n, v in triples.items()}
    topos = read_episode_labels("template_20kv", "multi_grid", column="graph_id", root=fixture_root)
    sims = read_episode_labels("template_20kv", "multi_grid", column="result_id", root=fixture_root)
    leakage.assert_group_disjoint(typed, topos, level="topology")
    leakage.assert_group_disjoint(typed, sims, level="simulation")
    for name in ("train", "val", "test"):
        assert typed[name], f"empty {name} partition on the fixture corpus"


def test_loader_round_trips(fixture_root, held_out_splits):
    loaded = read_held_out_splits("double_line", splits_root=held_out_splits)
    assert set(loaded) == set(HELD_OUT_SPLIT_NAMES)
    # benchmark partition equals the fixture's full benchmark family
    assert all(fam == "benchmark" for fam, _g, _i in loaded["benchmark"])
