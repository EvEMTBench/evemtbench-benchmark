"""Predictions alignment is by key (sample_id, window_idx), never by order; fails on mismatch."""

import numpy as np
import pandas as pd
import pytest

from evemtbench.evaluation.predictions_io import (
    align_to_view,
    check_prediction_values,
    predictions_frame,
)


def _meta(pairs):
    return pd.DataFrame(pairs, columns=["sample_id", "window_idx"])


def test_aligns_by_key_not_order():
    meta = _meta([(1, 0), (1, 1), (2, 0)])
    # predictions deliberately shuffled relative to meta order
    preds = predictions_frame([2, 1, 1], [0, 1, 0], [9, 8, 7])
    aligned = align_to_view(meta, preds)
    assert list(aligned) == [7, 8, 9]  # matched by key, not row position


def test_missing_prediction_raises():
    meta = _meta([(1, 0), (1, 1)])
    preds = predictions_frame([1], [0], [1])
    with pytest.raises(ValueError, match="no prediction"):
        align_to_view(meta, preds)


def test_duplicate_key_raises():
    meta = _meta([(1, 0)])
    preds = predictions_frame([1, 1], [0, 0], [1, 0])
    with pytest.raises(ValueError, match="duplicate"):
        align_to_view(meta, preds)


def test_extra_predictions_raise_by_default():
    """Extra keys are an error, as the docstring and docs/submission.md say.

    This previously asserted the opposite (extras silently dropped): the left merge ignored any key
    outside the view, so a file built against a different view scored as if it matched.
    """
    meta = _meta([(1, 0)])
    preds = predictions_frame([1, 2], [0, 0], [5, 6])  # extra (2,0) not in test
    with pytest.raises(ValueError, match="not in this test set"):
        align_to_view(meta, preds)


def test_extra_predictions_allowed_only_with_opt_in():
    meta = _meta([(1, 0)])
    preds = predictions_frame([1, 2], [0, 0], [5, 6])
    assert list(align_to_view(meta, preds, allow_extra=True)) == [5]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_predictions_raise(bad):
    """NaN/Inf must be refused for every task kind.

    Unchecked, a NaN class id casts to INT64_MIN and lands outside labels=[0,1], so the confusion
    matrix drops the row and missed_fault_rate silently improves (or goes NaN) instead of failing.
    """
    with pytest.raises(ValueError, match="not finite"):
        check_prediction_values(np.array([0.0, bad]), regression=False, n_classes=2)
    with pytest.raises(ValueError, match="not finite"):
        check_prediction_values(np.array([1.0, bad]), regression=True)


def test_non_integral_class_ids_raise_but_near_integral_pass():
    # 2.4 is almost certainly a score, not a class id — rounding it would invent class 2
    with pytest.raises(ValueError, match="not integral class ids"):
        check_prediction_values(np.array([0.0, 2.4]), regression=False, n_classes=5)
    # documented behaviour preserved: a float that IS integral within tolerance is accepted
    check_prediction_values(np.array([0.0, 2.9999999]), regression=False, n_classes=5)


def test_out_of_vocabulary_class_ids_raise():
    with pytest.raises(ValueError, match="outside the task vocabulary"):
        check_prediction_values(np.array([0, 2]), regression=False, n_classes=2)  # binary got a 2
    with pytest.raises(ValueError, match="outside the task vocabulary"):
        check_prediction_values(np.array([-1, 0]), regression=False, n_classes=2)
    check_prediction_values(np.array([0, 1]), regression=False, n_classes=2)


def test_regression_allows_out_of_range_but_not_non_finite():
    """A fault location outside [0, 100] % is a BAD prediction, not an invalid submission — MAE
    should punish it. Only NaN/Inf, which poison the aggregate, are refused."""
    check_prediction_values(np.array([-50.0, 1e6]), regression=True)
    with pytest.raises(ValueError, match="not finite"):
        check_prediction_values(np.array([-50.0, float("nan")]), regression=True)


def test_non_numeric_predictions_raise():
    with pytest.raises(ValueError, match="must be numeric"):
        check_prediction_values(np.array(["a", "b"], dtype=object), regression=False, n_classes=2)
