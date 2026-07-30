"""Windowed memmap loading, contract gating, episode keys, local view (DoubleLine fixture)."""

import numpy as np
import pytest

from evemtbench.data.windows import WindowContract, find_window_files, load_windows

GRID = "double_line"


def test_loads_fixture_windows(fixture_root):
    ws = load_windows(GRID, "adapt_grid", root=fixture_root)
    # 21 episodes x 4 windows = 84; 11 cubicles -> 66 features; 8 samples/window.
    assert ws.X.shape == (84, 8, 66)
    assert ws.n_cubicles == 11
    assert ws.contract == WindowContract(0.05, 0.005)


def test_episode_key_groups_windows(fixture_root):
    ws = load_windows(GRID, "adapt_grid", root=fixture_root)
    ids, counts = np.unique(ws.sample_ids, return_counts=True)
    assert len(ids) == 21
    assert set(counts.tolist()) == {4}  # 4 windows per episode


def test_local_view_selects_one_cubicle(fixture_root):
    ws = load_windows(GRID, "adapt_grid", root=fixture_root)
    lv = ws.local_view(1)
    assert lv.shape == (84, 8, 6)
    assert np.allclose(lv, ws.X[:, :, 6:12])


def test_contract_mismatch_fails_loud(fixture_root):
    with pytest.raises(ValueError, match="contract mismatch"):
        load_windows(
            GRID,
            "adapt_grid",
            root=fixture_root,
            expected_contract=WindowContract(window_length=0.1, step_length_seconds=0.01),
        )


def _touch_config(d, topology, tag):
    """One window-file quartet for a geometry tag (contents irrelevant to file *finding*)."""
    stem = f"{topology}_{tag}"
    (d / f"X_{stem}.raw").write_bytes(b"")
    (d / f"X_{stem}.raw.manifest.json").write_text("{}")
    (d / f"X_{stem}.raw.meta.json").write_text("{}")
    (d / f"y_{stem}.parquet").write_bytes(b"")


def test_multi_config_dir_narrows_by_contract(tmp_path):
    # The production mount holds several geometries side by side (e.g. W0p020 + W0p050); the
    # pinned contract must select ITS file instead of failing "expected exactly one".
    topo = "hv_double_line_110kv"
    _touch_config(tmp_path, topo, "W0p020_S0p005")
    _touch_config(tmp_path, topo, "W0p050_S0p005")
    files = find_window_files(tmp_path, topology=topo, contract=WindowContract(0.05, 0.005))
    assert files["x_path"].name == f"X_{topo}_W0p050_S0p005.raw"
    files = find_window_files(tmp_path, topology=topo, contract=WindowContract(0.02, 0.005))
    assert files["x_path"].name == f"X_{topo}_W0p020_S0p005.raw"
    # without a contract the ambiguity still fails loud (inspection mode never guesses)
    with pytest.raises(FileNotFoundError, match="expected exactly one"):
        find_window_files(tmp_path, topology=topo)
    # a contract matching NEITHER file also fails loud (never a silent wrong pick)
    with pytest.raises(FileNotFoundError, match="W0p100_S0p010"):
        find_window_files(tmp_path, topology=topo, contract=WindowContract(0.1, 0.01))


def test_pre_event_windows_are_negative(fixture_root):
    ws = load_windows(GRID, "adapt_grid", root=fixture_root)
    # Within a fault episode, early ("clean") windows are y_event_present=0.
    ep0 = ws.y[ws.y["sample_id"] == 0].sort_values("window_idx")
    assert ep0["y_event_present"].tolist() == [0, 0, 1, 1]
