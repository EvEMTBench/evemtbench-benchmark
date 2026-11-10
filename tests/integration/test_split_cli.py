"""The split generator's CLI defaults reproduce the committed splits, and it refuses to overwrite
them silently; the committed metas record the documented seed derivation and recipe."""

import shutil
import warnings

import pytest

from evemtbench.splits.generator import _COMMITTED_RECIPES, _grid_seed, generate_held_out, main
from evemtbench.splits.loader import read_meta
from evemtbench.version import splits_version

HELD_OUT_GRIDS = ("double_line", "cigre_mv", "testgrid_110kv", "ieee39")
PRETRAIN_GRIDS = ("template_20kv", "template_110kv", "template_345kv")


COMMITTED = [("held_out", g) for g in HELD_OUT_GRIDS] + [
    ("multi_grid_pretrain", g) for g in PRETRAIN_GRIDS
]


@pytest.mark.parametrize(("protocol", "grid"), COMMITTED)
def test_committed_meta_matches_seed_derivation_and_recipe(protocol, grid):
    # Data-free: every committed split's recorded seed is BASE_SEED + the per-grid offset derived
    # from (protocol[, grouping], grid) -- the documented derivation, not a hand-picked seed -- and
    # its recorded recipe is what the CLI uses by default.
    meta = read_meta(protocol, grid)
    recipe = _COMMITTED_RECIPES[protocol]
    grouping = recipe["grouping"]
    key = f"{protocol}_{grouping}_{grid}" if grouping else f"{protocol}_{grid}"
    assert meta["seed"] == _grid_seed(key)
    assert meta["stratify_by"] == recipe["stratify_by"]
    assert meta.get("grouping") == grouping
    assert meta["splits_version"] == splits_version(protocol)


def test_cli_defaults_follow_the_committed_recipe(fixture_root, tmp_path):
    main(["--root", str(fixture_root), "--out", str(tmp_path), "--grids", "double_line"])
    meta = read_meta("held_out", "double_line", splits_root=tmp_path)
    assert meta["splits_version"] == "1.0"
    assert meta["stratify_by"] == "event_type"
    assert "grouping" not in meta

    main(
        [
            "--protocol",
            "multi_grid_pretrain",
            "--root",
            str(fixture_root),
            "--out",
            str(tmp_path),
            "--grids",
            "template_20kv",
        ]
    )
    meta = read_meta("multi_grid_pretrain", "template_20kv", splits_root=tmp_path)
    assert meta["splits_version"] == "1.1"  # not the frozen v1.0 tree
    assert (meta["stratify_by"], meta["grouping"]) == ("event_type", "topology")


def test_cli_refuses_to_change_an_existing_split_without_force(fixture_root, tmp_path, capsys):
    base = ["--root", str(fixture_root), "--out", str(tmp_path), "--grids", "double_line"]
    main(base)
    main(base)  # identical regeneration is fine
    assert "identical to the committed split" in capsys.readouterr().out
    split = tmp_path / "v1.0" / "held_out" / "double_line" / "train.txt"
    before = split.read_text()

    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main([*base, "--stratify-by", "none"])
    assert split.read_text() == before  # nothing written

    main([*base, "--stratify-by", "none", "--force"])
    assert "CHANGED" in capsys.readouterr().out
    assert read_meta("held_out", "double_line", splits_root=tmp_path)["stratify_by"] is None


def _copy_fixture(fixture_root, tmp_path):
    root = tmp_path / "data"
    shutil.copytree(fixture_root, root)
    return root, root / "benchmark" / "windows" / "DoubleLine"


def test_missing_benchmark_windows_warn_and_yield_empty_partition(fixture_root, tmp_path):
    root, bm_dir = _copy_fixture(fixture_root, tmp_path)
    shutil.rmtree(bm_dir)
    with pytest.warns(UserWarning, match="EMPTY benchmark partition"):
        art = generate_held_out("double_line", root=root)
    assert art.splits["benchmark"] == []


def test_unreadable_benchmark_windows_raise_instead_of_an_empty_partition(fixture_root, tmp_path):
    # A windowed benchmark directory that cannot be resolved (here: a second X file matching the
    # contract tag) must fail loudly -- an empty benchmark.txt would drop the benchmark test set.
    root, bm_dir = _copy_fixture(fixture_root, tmp_path)
    (x,) = bm_dir.glob("X_*.raw")
    shutil.copy(x, bm_dir / x.name.replace(".raw", "_dup.raw"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(FileNotFoundError, match="expected exactly one"):
            generate_held_out("double_line", root=root)


def test_audit_cli_honours_splits_version(fixture_root, tmp_path, capsys):
    from evemtbench.splits import audit

    main(["--root", str(fixture_root), "--out", str(tmp_path), "--grids", "double_line"])
    capsys.readouterr()
    args = ["--grids", "double_line", "--root", str(fixture_root), "--splits-root", str(tmp_path)]
    with pytest.raises(SystemExit):
        audit.main([*args, "--splits-version", "1.0"])
    assert "AUDIT ERROR" not in capsys.readouterr().out  # the v1.0 split was found and audited
    # a version with no split must error, i.e. --splits-version is not silently ignored
    with pytest.raises(SystemExit) as missing:
        audit.main([*args, "--splits-version", "9.9"])
    assert missing.value.code == 1
    assert "AUDIT ERROR" in capsys.readouterr().out
