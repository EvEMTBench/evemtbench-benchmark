"""Predictions alignment is by key (sample_id, window_idx), never by order; fails on mismatch."""

import pandas as pd
import pytest

from evemtbench.evaluation.predictions_io import align_to_view, predictions_frame


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


def test_extra_predictions_are_ignored_if_all_test_covered():
    meta = _meta([(1, 0)])
    preds = predictions_frame([1, 2], [0, 0], [5, 6])  # extra (2,0) not in test
    aligned = align_to_view(meta, preds)
    assert list(aligned) == [5]
