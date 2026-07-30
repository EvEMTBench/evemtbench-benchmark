"""FD/FC/FL/ES label derivers and per-task valid-sample masks (on the fixture)."""

import numpy as np

from evemtbench.data.windows import load_windows
from evemtbench.tasks.labels.event_detection import derive_event_present_series
from evemtbench.tasks.labels.event_state import (
    EVENT_STATE_CLASSES,
    derive_event_state_series,
)
from evemtbench.tasks.labels.event_type import (
    EVENT_TYPE_CLASSES,
    derive_event_type_series,
)
from evemtbench.tasks.labels.fault_category import (
    FAULT_CATEGORY_CLASSES,
    derive_fault_category_series,
)
from evemtbench.tasks.labels.fault_class import (
    FAULT_CLASS_CLASSES,
    derive_fault_class_series,
)
from evemtbench.tasks.labels.fault_grounded import derive_fault_grounded_series
from evemtbench.tasks.labels.fault_location import derive_fault_location_series
from evemtbench.tasks.labels.fault_origin import derive_fault_origin_series
from evemtbench.tasks.labels.fault_phase import PHASE_CLASSES, derive_fault_phase_series
from evemtbench.tasks.labels.switch_detection import derive_switch_present_series
from evemtbench.tasks.labels.switch_type import (
    SWITCH_TYPE_CLASSES,
    derive_switch_type_series,
)
from evemtbench.tasks.registry import get_task
from evemtbench.tasks.valid_samples import valid_sample_mask


def _ws(fixture_root):
    return load_windows("double_line", "adapt_grid", root=fixture_root)


def test_hif_fault_types_are_mapped_not_minus_one():
    # MV grids carry HIF fault types; they must map into the vocabulary (indices 7,8), never -1,
    # or a CrossEntropy head crashes on them (the CigreMV FC bug this fixes).
    import pandas as pd

    y = pd.DataFrame(
        {"event_type": ["flt_1phg_hif", "flt_1phg_hif_w_arc", "flt_3ph_shc", "no_event"]}
    )
    idx = derive_fault_class_series(y)
    assert FAULT_CLASS_CLASSES[idx[0]] == "flt_1phg_hif"
    assert FAULT_CLASS_CLASSES[idx[1]] == "flt_1phg_hif_w_arc"
    assert idx[0] >= 0 and idx[1] >= 0  # HIF is in-vocab now
    assert idx[3] == -1  # non-fault still sentinel (filtered by the valid-sample mask)


def test_fault_class_indices_match_taxonomy(fixture_root):
    ws = _ws(fixture_root)
    idx = derive_fault_class_series(ws.y)
    et = ws.y["event_type"].to_numpy()
    for i, e in zip(idx, et, strict=True):
        if str(e).startswith("flt_"):
            assert FAULT_CLASS_CLASSES[i] == e
        else:
            assert i == -1  # non-fault -> sentinel (filtered by valid mask)


def test_fault_location_is_float_with_nan_on_nonline(fixture_root):
    ws = _ws(fixture_root)
    loc = derive_fault_location_series(ws.y)
    assert loc.dtype == np.float32
    # bus-fault / non-fault windows carry NaN
    is_bus_or_nonfault = (
        ws.y["y_fault_line"].isin(["MainBus1", "MainBus2", "MainBus3"])
        | ws.y["y_fault_line"].isna()
    )
    assert np.isnan(loc[is_bus_or_nonfault.to_numpy()]).all()


def test_fd_mask_is_fault_episode_windows(fixture_root):
    ws = _ws(fixture_root)
    mask = valid_sample_mask(ws.y, get_task("fault_detection_local"))
    # exactly the windows whose episode has a defined faulted element
    assert np.array_equal(mask, ws.y["y_fault_line"].notna().to_numpy())
    # includes pre-fault (negative) and active (positive) windows
    assert set(ws.y["y_fault_present"].to_numpy()[mask].tolist()) == {0, 1}


def test_fc_mask_is_fault_active_only(fixture_root):
    ws = _ws(fixture_root)
    mask = valid_sample_mask(ws.y, get_task("fault_classification_local"))
    # all kept windows are fault-active
    assert (ws.y["y_fault_present"].to_numpy()[mask] == 1).all()


def test_fl_mask_excludes_bus_faults(fixture_root):
    ws = _ws(fixture_root)
    mask = valid_sample_mask(ws.y, get_task("fault_location_local"))
    kept_lines = set(ws.y["y_fault_line"].to_numpy()[mask].tolist())
    assert kept_lines and all(line.startswith("MainLn") for line in kept_lines)
    # and every kept location is finite
    loc = derive_fault_location_series(ws.y)[mask]
    assert np.isfinite(loc).all()


def test_event_state_label_mapping():
    # Golden mapping: activity x event_family -> normal(0) / switching(1) / fault(2). Pre-event
    # (non-overlapping) windows of ANY family, and all no_event windows, are normal.
    import pandas as pd

    y = pd.DataFrame(
        {
            "event_family": [
                "fault",  # active fault      -> fault
                "fault",  # pre-fault window  -> normal
                "switching",  # active switching  -> switching
                "switching",  # pre-switch window -> normal
                "normal",  # no_event          -> normal
                "normal",  # no_event (active) -> normal (family wins)
            ],
            "overlap_ratio": [0.7, 0.0, 0.5, 0.0, 0.0, 1.0],
        }
    )
    idx = derive_event_state_series(y)
    got = [EVENT_STATE_CLASSES[i] for i in idx]
    assert got == ["fault", "normal", "switching", "normal", "normal", "normal"]


def test_event_state_mask_is_all_windows_and_labels_in_range(fixture_root):
    ws = _ws(fixture_root)
    td = get_task("event_state_global")
    mask = valid_sample_mask(ws.y, td)
    assert mask.all() and len(mask) == len(ws.y)  # every window is scored
    labels = derive_event_state_series(ws.y)
    assert set(np.unique(labels)).issubset({0, 1, 2})
    # normal support comes from pre-event windows, so it must exist even on the fixture
    assert (labels == 0).any()


# --- event-scope detection (binary) ---------------------------------------------------------------


def test_event_and_switch_detection_mapping():
    import pandas as pd

    y = pd.DataFrame(
        {
            "event_family": ["fault", "fault", "switching", "switching", "normal", "normal"],
            "overlap_ratio": [0.7, 0.0, 0.5, 0.0, 0.0, 1.0],
        }
    )
    # ED: any active disturbance (fault OR switching) is positive; normal never is.
    assert list(derive_event_present_series(y)) == [1, 0, 1, 0, 0, 0]
    # SD: only active switching is positive.
    assert list(derive_switch_present_series(y)) == [0, 0, 1, 0, 0, 0]


# --- fault attributes ---


def test_fault_phase_bit_packing():
    import pandas as pd

    y = pd.DataFrame(
        {
            "y_phase_A": [1, 0, 0, 1, 1, 0, 1, 0],
            "y_phase_B": [0, 1, 0, 1, 0, 1, 1, 0],
            "y_phase_C": [0, 0, 1, 0, 1, 1, 1, 0],
        }
    )
    got = [PHASE_CLASSES[i] if i >= 0 else None for i in derive_fault_phase_series(y)]
    assert got == ["A", "B", "C", "AB", "AC", "BC", "ABC", None]  # (0,0,0) -> -1 sentinel


def test_fault_category_mapping():
    import pandas as pd

    y = pd.DataFrame({"y_fault_category": ["shc", "incipient", "hif", "nope", None]})
    idx = derive_fault_category_series(y)
    assert [FAULT_CATEGORY_CLASSES[i] if i >= 0 else -1 for i in idx] == [
        "shc",
        "incipient",
        "hif",
        -1,
        -1,
    ]


def test_fault_grounded_and_phase_masks_on_fixture(fixture_root):
    ws = _ws(fixture_root)
    # grounded: scored on fault-active windows; both classes present (fixture has grounded 1phg/2phg
    # + ungrounded 2ph/3ph faults).
    gmask = valid_sample_mask(ws.y, get_task("fault_grounded_local"))
    assert (ws.y["y_fault_present"].to_numpy()[gmask] == 1).all()
    assert set(np.unique(derive_fault_grounded_series(ws.y)[gmask])) == {0, 1}
    # phase: fault-active with a phase set; labels within 0..6.
    pmask = valid_sample_mask(ws.y, get_task("fault_phase_line"))
    assert (ws.y["y_fault_present"].to_numpy()[pmask] == 1).all()
    assert set(np.unique(derive_fault_phase_series(ws.y)[pmask])).issubset(set(range(7)))


# --- switch-type & full event-type (EC) ---


def test_switch_type_mapping():
    import pandas as pd

    y = pd.DataFrame({"y_switch_category": ["load", "generator", "cable", "bogus", None]})
    idx = derive_switch_type_series(y)
    assert [SWITCH_TYPE_CLASSES[i] if i >= 0 else -1 for i in idx] == [
        "load",
        "generator",
        "cable",
        -1,
        -1,
    ]


def test_event_type_active_folds_nonactive_to_no_event():
    import pandas as pd

    y = pd.DataFrame(
        {
            "event_type": [
                "flt_1phg_shc",
                "flt_1phg_shc",
                "switch_load_on",
                "switch_load_on",
                "flt_x",
            ],
            "overlap_ratio": [1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )
    got = [EVENT_TYPE_CLASSES[i] if i >= 0 else -1 for i in derive_event_type_series(y)]
    # active -> its type; non-active -> no_event; unknown active type -> -1
    assert got == ["flt_1phg_shc", "no_event", "switch_load_on", "no_event", -1]


def test_event_classification_on_fixture(fixture_root):
    ws = _ws(fixture_root)
    labels = derive_event_type_series(ws.y)
    mask = valid_sample_mask(ws.y, get_task("event_classification_global"))
    assert mask.all()  # fixture has no out-of-vocab active types
    no_event_idx = EVENT_TYPE_CLASSES.index("no_event")
    assert (labels == no_event_idx).any()  # pre-event windows fold into no_event
    # at least one fault class (indices 0..8) is present among the derived labels
    assert (labels[:, None] == np.arange(len(FAULT_CLASS_CLASSES))).any()


# --- fault origin (line vs bus) + origin stratification ---


def test_fault_origin_mapping():
    import pandas as pd

    y = pd.DataFrame({"y_fault_line": ["MainLn1-2A", "MainBus1", "MainLn2-3", "MainBus10", None]})
    assert list(derive_fault_origin_series(y)) == [0, 1, 0, 1, 0]  # bus=1; line / non-fault=0


def test_fault_origin_mask_and_both_classes_on_fixture(fixture_root):
    ws = _ws(fixture_root)
    mask = valid_sample_mask(ws.y, get_task("fault_origin_global"))
    assert (ws.y["y_fault_present"].to_numpy()[mask] == 1).all()  # fault-active windows only
    labels = derive_fault_origin_series(ws.y)[mask]
    assert set(np.unique(labels)) == {0, 1}  # fixture has both line and bus faults


def test_origin_stratification_breakdown():
    import pandas as pd

    from evemtbench.evaluation.stratify import stratified_metrics

    md = pd.DataFrame({"y_fault_line": ["MainLn1-2A", "MainBus1", "MainLn2-3", None]})
    yt = np.array([0, 1, 0, 0])
    yp = np.array([0, 1, 1, 0])
    out = stratified_metrics(md, yt, yp, lambda a, b: {"acc": float((a == b).mean())})
    assert "origin" in out
    assert set(out["origin"]) == {"line", "bus", "none"}
    assert out["origin"]["bus"]["n"] == 1 and out["origin"]["line"]["n"] == 2


def test_resistance_bin_stratification_discrete():
    import pandas as pd

    from evemtbench.evaluation.stratify import stratified_metrics

    md = pd.DataFrame({"resistance": [0.1, 0.1, 5.0, 20.0]})  # 3 unique -> per-value bins
    yt = np.array([0, 1, 0, 1])
    yp = np.array([0, 1, 1, 1])
    out = stratified_metrics(md, yt, yp, lambda a, b: {"acc": float((a == b).mean())})
    assert "resistance_bin" in out
    assert set(out["resistance_bin"]) == {"0.1Ω", "5Ω", "20Ω"}
    assert out["resistance_bin"]["0.1Ω"]["n"] == 2


def test_resistance_bin_stratification_quartiles():
    import pandas as pd

    from evemtbench.evaluation.stratify import stratified_metrics

    md = pd.DataFrame({"resistance": list(range(20))})  # continuous -> quartile bins
    out = stratified_metrics(md, np.zeros(20, int), np.zeros(20, int), lambda a, b: {"acc": 1.0})
    assert "resistance_bin" in out and len(out["resistance_bin"]) == 4
