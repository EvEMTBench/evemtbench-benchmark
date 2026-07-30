"""End-to-end smoke: fixture -> held_out split -> load_task -> majority -> evaluate -> file audit.

Proves the full user path works without the real dataset and without importing baselines into
the evaluation path.
"""

import json

import pytest

from evemtbench import BENCHMARK_VERSION
from evemtbench.baselines.majority import MajorityBaseline
from evemtbench.evaluation.evaluate import evaluate
from evemtbench.evaluation.predictions_io import load_predictions, save_predictions
from evemtbench.tasks.task import load_task


@pytest.fixture
def fd_task(fixture_root, held_out_splits):
    return load_task(
        "fault_detection_local",
        "held_out",
        grid="double_line",
        root=fixture_root,
        splits_root=held_out_splits,
    )


def test_load_task_shapes_and_views(fd_task):
    t = fd_task
    assert t.train.features.shape[1:] == (8, 6)  # local observability -> 6 channels
    assert len(t.train) > 0 and len(t.test) > 0 and len(t.benchmark) > 0
    # episode-level splits: train/test/benchmark disjoint
    assert set(t.train.sample_ids).isdisjoint(set(t.test.sample_ids))


def test_majority_evaluate_both_test_sets(fd_task):
    model = MajorityBaseline()
    model.fit(fd_task)
    for ts in ("test", "benchmark"):
        result = evaluate(fd_task, model.predict(fd_task.view(ts)), test_set=ts)
        assert result.benchmark_version == BENCHMARK_VERSION
        assert result.test_set == ts
        for m in ("balanced_accuracy", "missed_fault_rate", "false_alarm_rate", "accuracy"):
            assert m in result.metrics


def test_predictions_file_roundtrip_matches(fd_task, tmp_path):
    model = MajorityBaseline()
    model.fit(fd_task)
    preds = model.predict(fd_task.test)
    direct = evaluate(fd_task, preds).metrics
    path = tmp_path / "preds.parquet"
    save_predictions(preds, path)
    from_file = evaluate(fd_task, load_predictions(path)).metrics
    assert direct == from_file


def test_result_json_serializable(fd_task):
    model = MajorityBaseline()
    model.fit(fd_task)
    json.loads(evaluate(fd_task, model.predict(fd_task.test)).to_json())
