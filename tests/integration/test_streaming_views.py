"""Streaming TaskView: iter_batches / channel_stats / RF-streamed must match the dense path."""

import numpy as np

from evemtbench.baselines.random_forest import _features, _features_streamed
from evemtbench.tasks.task import load_task


def _task(fixture_root, held_out_splits, task_id="fault_detection_global"):
    return load_task(
        task_id, "held_out", grid="double_line", root=fixture_root, splits_root=held_out_splits
    )


def test_iter_batches_concatenates_to_features(fixture_root, held_out_splits):
    view = _task(fixture_root, held_out_splits).train
    dense = view.features
    streamed = np.concatenate([x for x, _ in view.iter_batches(7)], axis=0)  # odd batch size
    assert streamed.shape == dense.shape
    assert np.array_equal(streamed, dense)  # streaming is order-preserving + bit-identical
    # feature_shape agrees without materializing
    assert view.feature_shape == dense.shape


def test_channel_stats_match_dense(fixture_root, held_out_splits):
    view = _task(fixture_root, held_out_splits).train
    dense = view.features
    mean, std = view.channel_stats(batch_size=5)
    assert np.allclose(mean, dense.mean(axis=(0, 1), keepdims=True), atol=1e-4)
    assert np.allclose(std, dense.std(axis=(0, 1), keepdims=True) + 1e-6, atol=1e-4)


def test_rf_streamed_summary_matches_dense(fixture_root, held_out_splits):
    view = _task(fixture_root, held_out_splits).train
    assert np.allclose(_features_streamed(view), _features(view.features), atol=1e-5)


def test_labels_align_with_batches(fixture_root, held_out_splits):
    view = _task(fixture_root, held_out_splits).train
    ys = np.concatenate([y for _, y in view.iter_batches(7)])
    assert np.array_equal(ys, view.labels)


def test_projection_cache_matches_live(fixture_root, held_out_splits, tmp_path):
    # The projection cache must be a transparent speed-up: reads from it are bit-identical to the
    # live full-channel projection, in any order, and iter_batches output is unchanged.
    view = _task(fixture_root, held_out_splits).train
    n = len(view)
    live = view._project_positions(np.arange(n))  # live read-all-channels + project

    view.materialize_cache(None)  # no dir -> no-op
    assert view._cache is None

    view.materialize_cache(tmp_path)  # build the (view-channels-only) scratch memmap
    assert view._cache is not None
    assert np.array_equal(view._read_positions(np.arange(n)), live)  # bit-identical
    order = np.random.default_rng(0).permutation(n)
    assert np.array_equal(view._read_positions(order), live[order])  # order-preserving
    streamed = np.concatenate([x for x, _ in view.iter_batches(7)], axis=0)
    assert np.array_equal(streamed, live)  # iter_batches (via the cache) unchanged
