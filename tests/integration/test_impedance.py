"""Impedance fault locator against the committed fixture, the registry and the runner seams.

The synthetic fixture is fabricated noise with 8 samples per window and ships no topology graph,
so it cannot carry a physical fault-location test (those are in
``tests/unit/test_impedance_locator.py``, on a synthetic line with a known fault position). What
it can test is that the baseline refuses, loudly and for the right reason, instead of guessing.
"""

import pytest

from evemtbench import coverage as cov
from evemtbench.baselines.impedance import (
    ImpedanceBaseline,
    LineParameters,
    lines_measured_at_far_end,
)
from evemtbench.tasks.registry import get_task, list_tasks
from evemtbench.tasks.task import load_task

_LINES = ("MainLn1-2A", "MainLn1-2B", "MainLn2-3A", "MainLn2-3B")
_PARAMS = {n: LineParameters(n, 20.0, 0.0334 + 0.26j, 0.118 + 0.979j) for n in _LINES}


def _load(task_id, fixture_root, held_out_splits):
    return load_task(
        task_id, "held_out", grid="double_line", root=fixture_root, splits_root=held_out_splits
    )


def test_registered_and_constructible():
    from evemtbench.baselines.runner import BASELINE_FACTORIES, _make_baseline

    assert "impedance" in BASELINE_FACTORIES and "impedance" in cov.ALL_BASELINES
    model = _make_baseline("impedance", seed=3, device="cpu", epochs=1, batch_size=8, lr=1e-3)
    assert isinstance(model, ImpedanceBaseline) and model.name == "impedance"


def test_applies_only_to_fault_location_local_and_line():
    applies = {t for t in list_tasks() if cov.baseline_applies("impedance", get_task(t))}
    assert applies == {"fault_location_local", "fault_location_line"}
    # nothing is trained, so the pre-training / transfer protocols never apply
    td = get_task("fault_location_line")
    for protocol in ("multi_grid_pretrain", "transfer_zeroshot", "transfer_finetune"):
        assert not cov.baseline_applies("impedance", td, protocol)


def test_a_reference_outside_the_learned_selection_pool():
    assert "impedance" in cov.CONVENTIONAL_BASELINES
    assert "impedance" not in cov.LEARNED_BASELINES
    assert "impedance" not in cov.DEEP_BASELINES


def test_matrix_cells_are_held_out_on_the_reference_grids():
    cells = cov.iter_cells(
        protocols=("held_out", "multi_grid_pretrain", "transfer_zeroshot", "transfer_finetune"),
        baselines=("impedance",),
    )
    assert {c.protocol for c in cells} == {"held_out"}
    assert {c.task for c in cells} == {"fault_location_local", "fault_location_line"}
    assert {c.grid for c in cells} == set(cov.REFERENCE_GRIDS)


def test_fixture_view_exposes_its_feature_axes(fixture_root, held_out_splits):
    task = _load("fault_location_line", fixture_root, held_out_splits)
    ((grid, feature_names, _channels),) = task.train.feature_axes()
    assert grid == "double_line" and len(feature_names) == 66
    assert lines_measured_at_far_end(task.train) == ((), ())  # every fixture line has two ends


def test_missing_topology_graph_fails_loud(fixture_root, held_out_splits, tmp_path):
    task = _load("fault_location_local", fixture_root, held_out_splits)
    with pytest.raises(FileNotFoundError, match="no topology graph"):
        ImpedanceBaseline(aux_root=str(tmp_path)).fit(task)


def test_non_integer_samples_per_cycle_fails_loud(fixture_root, held_out_splits):
    """The fixture's 8-sample windows hold no integer number of samples per 50 Hz cycle."""
    task = _load("fault_location_local", fixture_root, held_out_splits)
    with pytest.raises(ValueError, match="integer multiple"):
        ImpedanceBaseline(line_parameters=_PARAMS).fit(task)


@pytest.mark.parametrize("task_id", ["fault_detection_local", "fault_location_global"])
def test_rejects_tasks_outside_its_scope(task_id, fixture_root, held_out_splits):
    task = _load(task_id, fixture_root, held_out_splits)
    with pytest.raises(ValueError, match="fault locator|local or line"):
        ImpedanceBaseline(line_parameters=_PARAMS).fit(task)
