"""The fault-location valid-sample gate (benchmark v1.2.0): fault-active AND located AND a line
target. Synthetic label frames, one row per case, so each condition is pinned in isolation."""

import numpy as np
import pandas as pd
import pytest

from evemtbench.tasks.registry import get_task
from evemtbench.tasks.taxonomy import LINE_LABEL_PREFIX
from evemtbench.tasks.valid_samples import line_target_fault_location_mask, valid_sample_mask

FAULT, SWITCH = "flt_abc", "sw_line_open"

# (case, event_type, overlap_ratio, y_fault_location, y_fault_line, admitted)
CASES = [
    ("active line fault, located", FAULT, 0.6, 37.0, "MainLn1-2", True),
    ("active line fault, other line id", FAULT, 1.0, 99.0, "MainLn8-14", True),
    ("inactive (pre-fault) window of a line fault", FAULT, 0.0, 37.0, "MainLn1-2", False),
    ("missing overlap of a line fault", FAULT, np.nan, 37.0, "MainLn1-2", False),
    ("active line fault, location missing", FAULT, 0.6, np.nan, "MainLn1-2", False),
    ("active bus fault, location missing (benchmark)", FAULT, 0.6, np.nan, "MainBus3", False),
    ("active bus fault, located (adapt_grid)", FAULT, 0.6, 50.0, "MainBus3", False),
    ("inactive bus fault, located", FAULT, 0.0, 50.0, "MainBus3", False),
    ("active fault, unrecognised target", FAULT, 0.6, 50.0, "Trafo_T1", False),
    ("active fault, lower-case line prefix", FAULT, 0.6, 50.0, "mainln1-2", False),
    ("active fault, target missing (NaN)", FAULT, 0.6, 50.0, np.nan, False),
    ("active fault, target missing (None)", FAULT, 0.6, 50.0, None, False),
    ("active fault, empty target", FAULT, 0.6, 50.0, "", False),
    ("active non-fault event on a line", SWITCH, 0.6, 50.0, "MainLn1-2", False),
]


def _frame(cases=CASES) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_type": [c[1] for c in cases],
            "overlap_ratio": [c[2] for c in cases],
            "y_fault_location": [c[3] for c in cases],
            "y_fault_line": pd.Series([c[4] for c in cases], dtype=object),
        }
    )


def test_line_prefix_is_the_documented_grammar():
    assert LINE_LABEL_PREFIX == "MainLn"


@pytest.mark.parametrize(
    "task", ["fault_location_local", "fault_location_line", "fault_location_global"]
)
def test_gate_admits_exactly_the_active_located_line_targets(task):
    mask = valid_sample_mask(_frame(), get_task(task))
    assert mask.dtype == bool and mask.shape == (len(CASES),)
    got = {c[0]: bool(m) for c, m in zip(CASES, mask, strict=True)}
    assert got == {c[0]: c[5] for c in CASES}


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_each_case_alone(case):
    """Row-wise: the verdict on a window does not depend on the other rows of the frame."""
    mask = valid_sample_mask(_frame([case]), get_task("fault_location_line"))
    assert mask.tolist() == [case[5]]


def test_line_target_mask_is_an_alias_of_the_gate():
    y = _frame()
    np.testing.assert_array_equal(
        line_target_fault_location_mask(y), valid_sample_mask(y, get_task("fault_location_line"))
    )


def test_gate_needs_the_fault_line_column():
    """No silent fallback to the pre-1.2.0 rule when the target column is absent."""
    with pytest.raises(KeyError):
        valid_sample_mask(_frame().drop(columns="y_fault_line"), get_task("fault_location_line"))


def test_other_fault_tasks_still_admit_bus_targets():
    """The correction is scoped to fault location: FD/FC keep scoring bus-target fault windows."""
    y = _frame([c for c in CASES if c[0] == "active bus fault, located (adapt_grid)"])
    y["y_fault_present"] = 1
    assert valid_sample_mask(y, get_task("fault_detection_local")).tolist() == [True]
