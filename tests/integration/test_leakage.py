"""held_out split structure + leakage gates (fixture + the committed packaged splits)."""

import pytest

from evemtbench.data.roots import GRIDS
from evemtbench.splits import leakage
from evemtbench.splits.generator import generate_held_out
from evemtbench.splits.loader import read_held_out_splits
from evemtbench.splits.protocols import HELD_OUT_SPLIT_NAMES


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


def test_loader_round_trips(fixture_root, held_out_splits):
    loaded = read_held_out_splits("double_line", splits_root=held_out_splits)
    assert set(loaded) == set(HELD_OUT_SPLIT_NAMES)
    # benchmark partition equals the fixture's full benchmark family
    assert all(fam == "benchmark" for fam, _g, _i in loaded["benchmark"])
