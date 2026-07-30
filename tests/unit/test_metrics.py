"""compute_multiclass_metrics: a class with no support in this eval set has an UNDEFINED (NaN) F1 —
not a misleading 0.0. This is the source of the leaderboard's "n/a vs 0.000" distinction; a
regression here reintroduces the original bug (HIF F1 reading 0.000 on grids that have no HIF)."""

import math

import numpy as np

from evemtbench.evaluation.metrics import compute_multiclass_metrics

CLASSES = ("a", "b", "c")  # 'c' (index 2) is the class we leave absent from y_true


def test_absent_class_f1_is_nan_but_present_zero_stays_real():
    # a(0) and b(1) occur; c(2) never does. Model predicts everything as a, so b scores a genuine 0.
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    m = compute_multiclass_metrics(y_true, y_pred, classes=CLASSES)

    assert math.isnan(m["f1_c"])  # absent -> undefined (NaN), NOT 0.0
    assert m["f1_b"] == 0.0 and not math.isnan(m["f1_b"])  # present but never right -> genuine zero
    assert m["f1_a"] > 0.0 and math.isfinite(m["f1_a"])
    for k in ("accuracy", "macro_f1", "balanced_accuracy"):
        assert math.isfinite(m[k])  # aggregates defined (>=2 classes present)


def test_all_classes_present_no_nan():
    m = compute_multiclass_metrics(np.array([0, 1, 2]), np.array([0, 1, 2]), classes=CLASSES)
    assert all(not math.isnan(m[f"f1_{c}"]) for c in CLASSES)
    assert m["f1_a"] == 1.0 and m["f1_c"] == 1.0  # perfect prediction
