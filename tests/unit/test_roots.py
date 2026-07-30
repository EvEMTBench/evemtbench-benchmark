"""Grid registry + env-based root resolution (no hardcoded paths in logic)."""

import pytest

from evemtbench.data import roots


def test_grid_lookup_by_key_and_topology():
    g = roots.get_grid("double_line")
    assert g.topology == "hv_double_line_110kv"
    assert roots.get_grid("hv_double_line_110kv") is g  # topology id resolves to same spec
    assert g.frequency_hz == 50


def test_registry_has_windowed_reference_grids():
    # The four reference grids (transfer targets) + the multi_grid pre-training grid.
    assert {"double_line", "cigre_mv", "testgrid_110kv", "ieee39"} <= set(roots.GRIDS)
    tmpl = roots.GRIDS["template_20kv"]  # pre-training corpus (multi_grid family)
    assert tmpl.level == "MV" and tmpl.voltage_kv == 20.0
    assert roots.GRIDS["double_line"].frequency_hz == 50
    assert roots.GRIDS["cigre_mv"].level == "MV"
    assert roots.GRIDS["testgrid_110kv"].topology == "hv_testgrid_110kv"
    # IEEE-39 is the only 60 Hz / EHV grid.
    assert roots.GRIDS["ieee39"].frequency_hz == 60
    assert roots.GRIDS["ieee39"].level == "EHV"


def test_unknown_grid_raises():
    with pytest.raises(KeyError):
        roots.get_grid("nope")


def test_unknown_family_raises():
    with pytest.raises(ValueError, match="Unknown family"):
        roots.resolve_data_root("not_a_family", root="/tmp")


def test_root_precedence_env(monkeypatch, tmp_path):
    monkeypatch.setenv("EVEMTBENCH_DATA_ROOT", str(tmp_path))
    assert roots.resolve_data_root("adapt_grid") == tmp_path / "adapt_grid"
    # explicit arg overrides env
    assert roots.resolve_data_root("benchmark", root="/other") == roots.Path("/other") / "benchmark"


def test_windows_and_raw_dirs(tmp_path):
    wd = roots.windows_dir("double_line", "adapt_grid", root=tmp_path)
    assert wd == tmp_path / "adapt_grid" / "windows" / "DoubleLine"
    rd = roots.raw_partition_dir("double_line", "benchmark", root=tmp_path)
    assert rd == tmp_path / "benchmark" / "raw" / "DoubleLine"
