"""Regression metrics (fault location) + the metric-direction table."""

import numpy as np

from evemtbench.evaluation.metrics import METRIC_DIRECTION, compute_regression_metrics


def test_perfect_prediction():
    y = np.array([10.0, 20.0, 50.0, 80.0])
    m = compute_regression_metrics(y, y.copy())
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["median_abs_error"] == 0.0
    assert m["r2"] == 1.0


def test_known_errors():
    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([1.0, -1.0, 2.0, -2.0])  # abs errors 1,1,2,2
    m = compute_regression_metrics(y_true, y_pred)
    assert m["mae"] == 1.5
    assert np.isclose(m["rmse"], np.sqrt((1 + 1 + 4 + 4) / 4))
    assert m["median_abs_error"] == 1.5


def test_r2_can_be_negative():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([4.0, 3.0, 2.0, 1.0])  # anti-correlated -> R2 < 0
    m = compute_regression_metrics(y_true, y_pred)
    assert m["r2"] < 0


def test_metric_direction_signs():
    assert METRIC_DIRECTION["mae"] == -1
    assert METRIC_DIRECTION["rmse"] == -1
    assert METRIC_DIRECTION["r2"] == +1
    assert METRIC_DIRECTION["balanced_accuracy"] == +1
