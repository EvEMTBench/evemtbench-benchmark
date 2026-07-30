"""End-to-end load + evaluate smoke for the expanded task suite (on the synthetic fixture).

Exercises the full path for each new task — split -> windows -> valid-sample mask -> observability
projection -> label derivation -> evaluate (with the frozen class vocab / per-class F1) — without a
trained model (constant predictions). Correctness of the label mappings themselves is pinned by the
golden tests in ``test_fault_labels.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evemtbench.evaluation.evaluate import evaluate
from evemtbench.tasks.task import load_task

# (task_id, num_classes) for all 14 new task ids (13 from the first batch + fault_origin_global).
_NEW_TASKS = [
    ("event_detection_global", 2),
    ("switch_detection_global", 2),
    ("switch_type_global", 8),
    ("event_classification_global", 23),
    ("fault_origin_global", 2),
    ("fault_grounded_local", 2),
    ("fault_grounded_line", 2),
    ("fault_grounded_global", 2),
    ("fault_phase_local", 7),
    ("fault_phase_line", 7),
    ("fault_phase_global", 7),
    ("fault_category_local", 3),
    ("fault_category_line", 3),
    ("fault_category_global", 3),
]


def _load(task_id, fixture_root, held_out_splits):
    return load_task(
        task_id, "held_out", grid="double_line", root=fixture_root, splits_root=held_out_splits
    )


def _first_nonempty(task):
    for name in ("test", "benchmark", "train", "val"):
        v = task.view(name)
        if len(v) > 0:
            return name, v
    return None, None


@pytest.mark.parametrize("task_id,ncls", _NEW_TASKS)
def test_new_task_loads_masks_and_evaluates(task_id, ncls, fixture_root, held_out_splits):
    task = _load(task_id, fixture_root, held_out_splits)
    name, view = _first_nonempty(task)
    assert view is not None, f"{task_id}: all views empty on the fixture"
    labels = np.asarray(view.labels)
    assert labels.min() >= 0 and labels.max() < ncls  # in the frozen class range
    # feature projection is well-defined for every scored window
    assert view.feature_shape[0] == len(view)
    # evaluate runs with a constant prediction and emits the headline metric
    md = view.metadata
    preds = pd.DataFrame(
        {"sample_id": md["sample_id"], "window_idx": md["window_idx"], "prediction": 0}
    )
    res = evaluate(task, preds, test_set=name)
    assert "balanced_accuracy" in res.metrics


def test_attach_aux_is_graceful_without_overlay(
    fixture_root, held_out_splits, tmp_path, monkeypatch
):
    # attach_aux must no-op (not crash) when the aux overlay is absent — the default path.
    monkeypatch.setenv("EVEMTBENCH_AUX_ROOT", str(tmp_path / "no_aux"))
    task = load_task(
        "fault_detection_local",
        "held_out",
        grid="double_line",
        root=fixture_root,
        splits_root=held_out_splits,
        attach_aux=("resistance",),
    )
    assert "resistance" not in task.test.metadata.columns
