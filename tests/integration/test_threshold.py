"""Overcurrent threshold baseline — valid only for fault_detection_local."""

import numpy as np
import pytest

from evemtbench.baselines.threshold import ThresholdBaseline, _current_statistic
from evemtbench.evaluation.evaluate import evaluate
from evemtbench.tasks.task import load_task


def _load(task_id, fixture_root, held_out_splits):
    return load_task(
        task_id, "held_out", grid="double_line", root=fixture_root, splits_root=held_out_splits
    )


def test_current_statistic_shape():
    x = np.random.randn(4, 8, 6).astype("float32")
    assert _current_statistic(x).shape == (4,)


def test_threshold_fits_and_predicts_local(fixture_root, held_out_splits):
    task = _load("fault_detection_local", fixture_root, held_out_splits)
    model = ThresholdBaseline(seed=0)
    model.fit(task)
    assert model.threshold_ is not None
    preds = model.predict(task.test)
    result = evaluate(task, preds, test_set="test")
    assert result.n_samples == len(task.test)
    assert set(preds["prediction"].unique()) <= {0, 1}


def test_threshold_rejects_non_local(fixture_root, held_out_splits):
    task = _load("fault_detection_global", fixture_root, held_out_splits)
    with pytest.raises(ValueError, match="local-relay"):
        ThresholdBaseline(seed=0).fit(task)


def test_threshold_rejects_non_binary(fixture_root, held_out_splits):
    task = _load("fault_classification_local", fixture_root, held_out_splits)
    with pytest.raises(ValueError, match="binary-only"):
        ThresholdBaseline(seed=0).fit(task)
