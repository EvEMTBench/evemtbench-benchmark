"""1D-CNN baseline smoke — classification (FD) and regression (FL) paths. Needs torch."""

import numpy as np
import pytest

pytest.importorskip("torch")

from evemtbench.baselines.cnn import CNNBaseline  # noqa: E402
from evemtbench.evaluation.evaluate import evaluate  # noqa: E402
from evemtbench.tasks.task import load_task  # noqa: E402


def _load(task_id, fixture_root, held_out_splits):
    return load_task(
        task_id, "held_out", grid="double_line", root=fixture_root, splits_root=held_out_splits
    )


def test_cnn_classification_path(fixture_root, held_out_splits):
    task = _load("fault_detection_global", fixture_root, held_out_splits)
    model = CNNBaseline(epochs=2, batch_size=8, width=8, device="cpu", seed=0)
    model.fit(task)
    assert model._mean is not None and not model._regression
    preds = model.predict(task.test)
    result = evaluate(task, preds, test_set="test")
    assert result.n_samples == len(task.test)
    assert set(preds["prediction"].unique()) <= {0, 1}


def test_cnn_regression_path(fixture_root, held_out_splits):
    task = _load("fault_location_line", fixture_root, held_out_splits)
    model = CNNBaseline(epochs=2, batch_size=8, width=8, device="cpu", seed=0)
    model.fit(task)
    assert model._regression  # MSE head + target standardization
    preds = model.predict(task.benchmark)
    result = evaluate(task, preds, test_set="benchmark")
    assert "mae" in result.metrics
    assert np.issubdtype(preds["prediction"].to_numpy().dtype, np.floating)
