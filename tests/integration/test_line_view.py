"""Main-line projection + LOCAL/LINE/GLOBAL faulted-element oracle, derived from feature_names."""

import pytest

from evemtbench.tasks import line_view as lv

# DoubleLine cubicle layout (on-disk order, 6 channels each) — mirrors the real grid + fixture.
_CUBICLES = [
    "Bus_1_Line_1-2A",
    "Bus_1_Line_1-2B",
    "Bus_1_Transformer_1-ExtGrid1",
    "Bus_2_Load_2",
    "Bus_2_Line_1-2A",
    "Bus_2_Line_1-2B",
    "Bus_2_Line_2-3A",
    "Bus_2_Line_2-3B",
    "Bus_3_Line_2-3A",
    "Bus_3_Line_2-3B",
    "Bus_3_Transformer_3-ExtGrid3",
]
_FEATURES = [
    f"{k}_{q}_{ph}_{u}"
    for k in _CUBICLES
    for q, u in (("Isec", "A"), ("Usec", "V"))
    for ph in ("L1", "L2", "L3")
]


def _oracle():
    return lv.build_oracle("double_line", _FEATURES)


def test_line_channels_are_48_main_line_columns():
    o = _oracle()
    # main lines: cubicles 0,1 (Bus1) + 4..9 (Bus2/Bus3 bays); transformers/load dropped
    assert list(o.line_channels) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] + list(range(24, 60))
    assert o.n_line_cubicles == 8


def test_observation_columns_widths():
    o = _oracle()
    assert len(o.observation_columns("MainLn1-2A", "local")) == 6
    assert len(o.observation_columns("MainLn1-2A", "line")) == 12
    assert len(o.observation_columns("MainLn1-2A", "global")) == 48


def test_line_view_picks_both_terminals():
    # MainLn1-2A -> projected cubicles 0 (Bus_1) and 2 (Bus_2) -> cols 0..5 + 12..17
    cols = _oracle().observation_columns("MainLn1-2A", "line")
    assert cols == [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17]


def test_local_is_near_terminal_subset_of_line():
    o = _oracle()
    local = o.observation_columns("MainLn2-3B", "local")
    line = o.observation_columns("MainLn2-3B", "line")
    assert local == line[:6]  # nested hierarchy


def test_bus_fault_maps_to_incident_line_bays():
    # MainBus2 -> two Bus_2 line bays (projected cubicles 2 and 3)
    cols = _oracle().observation_columns("MainBus2", "line")
    assert cols == [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]


def test_unknown_fault_line_raises():
    with pytest.raises(KeyError, match="unknown y_fault_line"):
        _oracle().observation_columns("MainLnX", "line")


def test_global_ignores_fault_line():
    o = _oracle()
    a = o.observation_columns("MainBus1", "global")
    b = o.observation_columns("MainLn2-3A", "global")
    assert a == b == list(range(48))


def test_oracle_generalizes_to_a_different_topology():
    # A 3-cubicle toy grid with one line at two terminals + a transformer -> 1 line, 12 ch.
    feats = [
        f"{k}_{q}_{ph}_{u}"
        for k in ("Bus_1_Line_1-2", "Bus_2_Line_1-2", "Bus_2_Transformer_x")
        for q, u in (("Isec", "A"), ("Usec", "V"))
        for ph in ("L1", "L2", "L3")
    ]
    o = lv.build_oracle("toy", feats)
    assert o.n_line_cubicles == 2
    assert list(o.line_channels) == list(range(12))  # first two cubicles, transformer dropped
    assert o.observation_columns("MainLn1-2", "line") == list(range(12))
    assert o.observation_columns("MainLn1-2", "local") == list(range(6))
