"""compute_multiclass_metrics: a class with no support in this eval set has an UNDEFINED (NaN) F1 —
not a misleading 0.0. This is the source of the leaderboard's "n/a vs 0.000" distinction; a
regression here reintroduces the original bug (HIF F1 reading 0.000 on grids that have no HIF)."""

import math

import numpy as np
import pytest
from sklearn.metrics import f1_score

from evemtbench.evaluation.metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    macro_f1_supported,
)

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


def test_frozen_macro_f1_still_zero_fills_absent_classes():
    """REGRESSION PIN — do not "fix" this.

    ``macro_f1`` is frozen benchmark contract and the manuscript specifies it exactly this way:
    per-class F1 with zero-division mapped to 0, averaged over the FULL frozen vocabulary. An
    external review read the resulting 0.667 below as a bug; changing it would break the published
    contract and invalidate every committed macro_f1. The complement is reported separately as
    ``macro_f1_supported`` (next test).
    """
    m = compute_multiclass_metrics(np.array([0, 1]), np.array([0, 1]), classes=CLASSES)
    assert m["macro_f1"] == pytest.approx(2 / 3)  # perfect on a,b; absent c contributes 0
    assert m["accuracy"] == 1.0 and m["balanced_accuracy"] == 1.0
    assert math.isnan(m["f1_c"])


def test_macro_f1_supported_excludes_unsupported_classes():
    m = compute_multiclass_metrics(np.array([0, 1]), np.array([0, 1]), classes=CLASSES)
    assert m["macro_f1_supported"] == 1.0  # perfect over the classes that are actually present
    # equals the nanmean of the emitted per-class F1s (sklearn computes each label independently)
    per_class = [m[f"f1_{c}"] for c in CLASSES]
    assert m["macro_f1_supported"] == pytest.approx(np.nanmean(per_class))


def test_macro_f1_supported_equals_macro_f1_under_full_support():
    """With every vocabulary class supported the two definitions must coincide — otherwise the new
    metric would be silently rescaling results that were never affected by the zero-fill."""
    y_true, y_pred = np.array([0, 1, 2, 0]), np.array([0, 1, 0, 0])
    m = compute_multiclass_metrics(y_true, y_pred, classes=CLASSES)
    assert m["macro_f1_supported"] == pytest.approx(m["macro_f1"])


def test_macro_f1_supported_on_empty_input_is_nan_not_zero():
    """Asserted on the helper directly: sklearn refuses zero-sample input, so the full
    ``compute_multiclass_metrics`` path raises before reaching this metric. An empty stratum must
    read undefined, never 0.0 — that is what keeps an empty group out of a mean."""
    assert math.isnan(macro_f1_supported(np.array([], dtype=int), np.array([], dtype=int)))


def test_macro_f1_supported_for_binary_single_class_truth():
    """Binary macro_f1 passes no `labels`, so sklearn averages over y_true ∪ y_pred: a class present
    only in the PREDICTIONS still dilutes it. The supported-only variant keys off y_true alone."""
    y_true, y_pred = np.array([0, 0, 0, 0]), np.array([0, 0, 0, 1])
    b = compute_binary_metrics(y_true, y_pred)
    assert b["macro_f1"] < b["macro_f1_supported"]
    assert b["macro_f1_supported"] == pytest.approx(
        f1_score(y_true, y_pred, average="macro", labels=[0], zero_division=0)
    )
