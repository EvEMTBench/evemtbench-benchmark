"""Auxiliary data layer: settings join + column groups, graph load/features, data manifest.

Uses a temp aux root (no real dataset) so it runs in CI; the graph tests importorskip networkx.
"""

from __future__ import annotations

import pickle

import pandas as pd
import pytest

from evemtbench.data import aux_labels, manifest, roots


def _aux_grid_dir(tmp_path, grid="double_line", family="benchmark"):
    d = roots.aux_dir(grid, family, root=tmp_path)
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_resolve_aux_root_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("EVEMTBENCH_AUX_ROOT", str(tmp_path / "auxenv"))
    assert roots.resolve_aux_root() == tmp_path / "auxenv"
    assert roots.resolve_aux_root(root=tmp_path / "explicit") == tmp_path / "explicit"
    monkeypatch.delenv("EVEMTBENCH_AUX_ROOT", raising=False)
    monkeypatch.setenv("EVEMTBENCH_DATA_ROOT", str(tmp_path / "data" / "evemtbench"))
    # sibling of the data root, so the producer root is untouched
    assert roots.resolve_aux_root() == tmp_path / "data" / "evemtbench-aux"


def test_settings_join_groups_and_resistance(tmp_path):
    d = _aux_grid_dir(tmp_path)
    pd.DataFrame(
        {
            "general/sim_idx": [0, 1, 2],
            "events/event_flt_shc_resistance": [0.1, 5.0, 20.0],
            "arc/arc_tau": [1e-3, 2e-3, 3e-3],
            "grounding/grnd_type": ["solid", "resonant", "resistive"],
        }
    ).to_csv(d / "settings.csv", index=False)

    md = pd.DataFrame({"sample_id": [0, 1, 2, 2], "window_idx": [0, 1, 0, 1]})
    out = aux_labels.join_to_metadata(
        md,
        "double_line",
        "benchmark",
        ["events/event_flt_shc_resistance"],
        rename={"events/event_flt_shc_resistance": "resistance"},
        root=tmp_path,
    )
    assert list(out["resistance"]) == [0.1, 5.0, 20.0, 20.0]  # joined per episode

    s = aux_labels.load_settings("double_line", "benchmark", root=tmp_path)
    assert aux_labels.fault_resistance_column(s) == "events/event_flt_shc_resistance"
    assert aux_labels.group_columns(s, "resistance") == ["events/event_flt_shc_resistance"]
    assert aux_labels.group_columns(s, "arc") == ["arc/arc_tau"]


def test_settings_missing_fails_loud(tmp_path):
    with pytest.raises(FileNotFoundError):
        aux_labels.load_settings("double_line", "benchmark", root=tmp_path)
    with pytest.raises(KeyError):  # requested column not in settings
        d = _aux_grid_dir(tmp_path)
        pd.DataFrame({"general/sim_idx": [0]}).to_csv(d / "settings.csv", index=False)
        aux_labels.join_to_metadata(
            pd.DataFrame({"sample_id": [0]}), "double_line", "benchmark", ["nope"], root=tmp_path
        )


def test_graph_roundtrip_and_scalar_features(tmp_path):
    nx = pytest.importorskip("networkx")
    from evemtbench.data import graphs

    d = _aux_grid_dir(tmp_path)
    g = nx.MultiGraph()
    g.add_node("Bus1", Trf_param_dict={"strn": 80.0, "trafo_type": "X", "cgnd_h": 0})
    g.add_edge("Bus1", "Bus2", Ln_param_dict={"length": 4.2, "r0pu": 0.01, "name": "L", "on": True})
    with open(d / "graph.pickle", "wb") as fh:
        pickle.dump(g, fh)

    gg = graphs.load_grid_graph("double_line", "benchmark", root=tmp_path)
    summ = graphs.graph_summary(gg)
    assert summ["n_nodes"] == 2 and summ["n_edges"] == 1 and summ["multigraph"]

    nf = graphs.graph_node_features(gg)
    assert nf["Bus1"]["strn"] == 80.0  # float kept
    assert nf["Bus1"]["cgnd_h"] == 0.0  # int kept (as float)
    assert "trafo_type" not in nf["Bus1"]  # string dropped

    ef = graphs.graph_edge_features(gg)
    (edge_feats,) = ef.values()
    assert edge_feats["length"] == 4.2 and edge_feats["r0pu"] == 0.01
    assert "name" not in edge_feats and "on" not in edge_feats  # string + bool dropped


def test_graph_missing_fails_loud(tmp_path):
    pytest.importorskip("networkx")
    from evemtbench.data import graphs

    with pytest.raises(FileNotFoundError):
        graphs.load_grid_graph("double_line", "benchmark", root=tmp_path)


def test_manifest_scan_detects_fixture_windows(fixture_root):
    m = manifest.scan(root=fixture_root, aux_root=fixture_root)
    assert set(m) >= {"adapt_grid", "benchmark"}
    assert "W0p050" in m["adapt_grid"]["double_line"]["windows"]  # the fixture is W0p050
    assert m["benchmark"]["double_line"]["graph"] is False  # no aux graph for the fixture
