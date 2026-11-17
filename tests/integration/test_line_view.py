"""Main-line projection + LOCAL/LINE/GLOBAL faulted-element oracle, derived from feature_names."""

import uuid
import warnings

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


def test_two_ended_grid_has_no_one_ended_lines_and_no_warning():
    # A grid name of its own keeps the build uncached, so a spurious warning would be raised here.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        o = lv.build_oracle(f"two_ended_{uuid.uuid4().hex}", _FEATURES)
    assert o.one_ended_lines == ()


def test_one_ended_line_is_flagged_and_its_frozen_padding_pinned():
    # Drop the Bus_2 end of 2-3B: like CIGRE MV line 8-14, only one terminal is exported.
    cubicles = [c for c in _CUBICLES if c != "Bus_2_Line_2-3B"]
    features = [
        f"{k}_{q}_{ph}_{u}"
        for k in cubicles
        for q, u in (("Isec", "A"), ("Usec", "V"))
        for ph in ("L1", "L2", "L3")
    ]
    # A grid name of its own keeps the build uncached: the warning fires on build only.
    with pytest.warns(lv.OneEndedLineWarning, match="MainLn2-3B has a single measured terminal"):
        o = lv.build_oracle(f"one_ended_{uuid.uuid4().hex}", features)
    assert o.one_ended_lines == ("MainLn2-3B",)
    cols = o.observation_columns("MainLn2-3B", "line")
    # Frozen v1 behaviour: the far slot repeats the only terminal (see docs/known_issues.md).
    assert cols[:6] == cols[6:]
    assert o.observation_columns("MainLn2-3B", "local") == cols[:6]


def test_one_ended_warning_can_be_filtered_under_warnings_as_errors():
    # Downstream CI that runs with warnings as errors must be able to silence exactly this class.
    features = [
        f"{k}_{q}_{ph}_{u}"
        for k in _CUBICLES
        if k != "Bus_2_Line_2-3B"
        for q, u in (("Isec", "A"), ("Usec", "V"))
        for ph in ("L1", "L2", "L3")
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=lv.OneEndedLineWarning)
        o = lv.build_oracle(f"filtered_{uuid.uuid4().hex}", features)
    assert o.one_ended_lines == ("MainLn2-3B",)
    assert issubclass(lv.OneEndedLineWarning, UserWarning)


def _features_of(cubicles: list[str]) -> list[str]:
    return [
        f"{k}_{q}_{ph}_{u}"
        for k in cubicles
        for q, u in (("Isec", "A"), ("Usec", "V"))
        for ph in ("L1", "L2", "L3")
    ]


def _channels_of(cubicles: list[str]) -> list[dict]:
    """The structured channel descriptor (contract C-META-2) for a cubicle layout: one entry per
    channel, six per cubicle, carrying the fields the oracle reads."""
    out: list[dict] = []
    for key in cubicles:
        _, bus, element_type, element_id = key.split("_", 3)
        out += [{"bus": bus, "element_type": element_type, "element_id": element_id}] * 6
    return out


def test_one_ended_line_is_identified_by_both_metadata_routes():
    """The feature-name route and the channel-descriptor route (the one production uses) flag the
    same one-ended line and build the same oracle. The descriptor build gets opaque feature names,
    so it cannot be falling back to name parsing."""
    cubicles = [c for c in _CUBICLES if c != "Bus_2_Line_2-3B"]
    features = _features_of(cubicles)
    opaque = [f"ch{i}" for i in range(len(features))]
    with pytest.warns(lv.OneEndedLineWarning, match="MainLn2-3B has a single measured terminal"):
        by_name = lv.build_oracle(f"names_{uuid.uuid4().hex}", features)
    with pytest.warns(lv.OneEndedLineWarning, match="MainLn2-3B has a single measured terminal"):
        by_descriptor = lv.build_oracle(
            f"descriptor_{uuid.uuid4().hex}", opaque, _channels_of(cubicles)
        )
    assert by_name.one_ended_lines == by_descriptor.one_ended_lines == ("MainLn2-3B",)
    assert by_name._pairs == by_descriptor._pairs
    assert by_name.line_channels == by_descriptor.line_channels
    with pytest.raises(ValueError, match="no line-bay cubicles"):
        lv.build_oracle(f"opaque_{uuid.uuid4().hex}", opaque)  # names alone carry no line bays


def test_two_ended_grid_is_not_flagged_by_the_descriptor_route():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        o = lv.build_oracle(
            f"descriptor_two_ended_{uuid.uuid4().hex}", _FEATURES, _channels_of(_CUBICLES)
        )
    assert o.one_ended_lines == ()


def test_single_bay_bus_is_padded_but_is_not_a_one_ended_line():
    """A bus fault at a bus with one line bay repeats that bay in the LINE view. That is the
    bus-adjacent-bay labelling convention, not a missing terminal: no line is flagged and no
    warning is raised, while the line between the two buses keeps two distinct terminals."""
    cubicles = ["Bus_1_Line_1-2", "Bus_2_Line_1-2", "Bus_2_Transformer_x"]
    for name, feats, channels in (
        ("names", _features_of(cubicles), None),
        ("descriptor", _features_of(cubicles), _channels_of(cubicles)),
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            o = lv.build_oracle(f"single_bay_{name}_{uuid.uuid4().hex}", feats, channels)
        assert o.one_ended_lines == ()
        for bus in ("MainBus1", "MainBus2"):
            cols = o.observation_columns(bus, "line")
            assert cols[:6] == cols[6:] == o.observation_columns(bus, "local")
        line = o.observation_columns("MainLn1-2", "line")
        assert line[:6] != line[6:]


def test_one_ended_line_and_single_bay_bus_stay_distinct_in_one_grid():
    """With the Bus_2 end of 2-3B missing, the line is flagged; the buses are not, whatever
    their number of bays."""
    cubicles = [c for c in _CUBICLES if c != "Bus_2_Line_2-3B"]
    with pytest.warns(lv.OneEndedLineWarning):
        o = lv.build_oracle(f"mixed_{uuid.uuid4().hex}", _features_of(cubicles))
    assert o.one_ended_lines == ("MainLn2-3B",)
    assert not any(label.startswith("MainBus") for label in o.one_ended_lines)
    assert {"MainBus1", "MainBus2", "MainBus3"} <= set(o._pairs)
