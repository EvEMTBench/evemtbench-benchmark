"""RandomForest baseline across the three label kinds (classification + regression)."""

import numpy as np
import pytest

from evemtbench.baselines.random_forest import RandomForestBaseline, _features
from evemtbench.evaluation.evaluate import evaluate
from evemtbench.tasks.task import load_task


def test_feature_extraction_shape():
    x = np.random.randn(5, 8, 6).astype("float32")
    assert _features(x).shape == (5, 6 * 7)  # 7 stats per channel


@pytest.mark.parametrize(
    "task_id,kind",
    [
        ("fault_detection_local", "binary"),
        ("fault_classification_line", "multiclass"),
        ("fault_location_global", "regression"),
    ],
)
def test_rf_fits_and_predicts(task_id, kind, fixture_root, held_out_splits):
    task = load_task(
        task_id, "held_out", grid="double_line", root=fixture_root, splits_root=held_out_splits
    )
    model = RandomForestBaseline(seed=0, n_estimators=10)
    model.fit(task)
    for ts in ("test", "benchmark"):
        preds = model.predict(task.view(ts))
        assert len(preds) == len(task.view(ts))
        result = evaluate(task, preds, test_set=ts)
        assert result.test_set == ts
        if kind == "regression":
            assert {"mae", "rmse", "r2", "median_abs_error"} <= set(result.metrics)
            assert np.issubdtype(preds["prediction"].to_numpy().dtype, np.floating)
        else:
            assert "balanced_accuracy" in result.metrics
            assert np.issubdtype(preds["prediction"].to_numpy().dtype, np.integer)
