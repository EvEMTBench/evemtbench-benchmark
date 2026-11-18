"""Negative checks for the guards added after the external code review of v1.3.0 (git5 #44-#49).

Each test feeds an invalid input that used to produce a normal score or a silent success and
asserts that it now fails with a message naming the problem."""

import math
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from evemtbench import evaluate, load_task
from evemtbench.data import input_digests
from evemtbench.evaluation.predictions_io import align_to_view

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "hpc"))
TASK = "fault_detection_local"


def _load(fixture_root, splits, **kw):
    return load_task(
        TASK, "held_out", grid="double_line", root=fixture_root, splits_root=splits, **kw
    )


def test_single_directory_cannot_serve_two_families(fixture_root, held_out_splits):
    """#44: one --directory would alias the benchmark family onto the training waveforms."""
    directory = fixture_root / "adapt_grid" / "windows" / "DoubleLine"
    with pytest.raises(ValueError, match="cannot also serve"):
        _load(fixture_root, held_out_splits, directory=directory)
    _load(fixture_root, held_out_splits)  # the data root still works


def test_split_naming_an_absent_episode_is_rejected(fixture_root, held_out_splits, tmp_path):
    """#45: a listed episode that is not in the partition must not shrink the test set silently."""
    splits = tmp_path / "splits"
    shutil.copytree(held_out_splits, splits)
    test_txt = next(splits.rglob("double_line/test.txt"))
    test_txt.write_text(test_txt.read_text().rstrip("\n") + "\nadapt_grid:double_line:999999\n")
    with pytest.raises(ValueError, match="999999"):
        _load(fixture_root, splits)


def test_duplicate_or_null_reference_keys_are_rejected():
    """#45: the reference side of the join must be a key, like the prediction side."""
    md = pd.DataFrame({"sample_id": [1, 1, 2], "window_idx": [0, 0, 0]})
    preds = pd.DataFrame({"sample_id": [1, 2], "window_idx": [0, 0], "prediction": [0, 1]})
    with pytest.raises(ValueError, match="duplicate"):
        align_to_view(md, preds)
    md_null = pd.DataFrame({"sample_id": [1, None], "window_idx": [0, 0]})
    with pytest.raises(ValueError, match="null"):
        align_to_view(md_null, preds)


def test_payload_bytes_are_verified(fixture_root, tmp_path, monkeypatch):
    """#46: one changed waveform value, label file or channel descriptor each fails verification."""
    root = tmp_path / "root"
    shutil.copytree(fixture_root, root)
    table = input_digests.build_table(root, payload=("x", "y", "meta"))
    monkeypatch.setattr(input_digests, "load_table", lambda: table)
    assert input_digests.verify(root, deep=True) == []
    part = root / "adapt_grid" / "windows" / "DoubleLine"
    for pattern, name, deep in (
        ("X_*.raw", "x", True),
        ("y_*.parquet", "y", False),
        ("*.meta.json", "meta", False),
    ):
        target = next(part.glob(pattern))
        original = target.read_bytes()
        target.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        problems = input_digests.verify(root, deep=deep)
        assert any(f"{name} file sha256 mismatch" in p for p in problems), (name, problems)
        target.write_bytes(original)
    assert input_digests.verify(root, deep=True) == []


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, "0.9", True])
def test_non_finite_validation_mean_is_unusable(bad):
    """#48: a NaN/Inf validation mean must not be able to win model selection."""
    import make_results

    assert make_results.mean_val({"aggregate": {"m": {"mean": bad}}}, "m") is None
    assert make_results.mean_val({"aggregate": {"m": {"mean": 0.5}}}, "m") == 0.5


def test_evaluator_result_names_its_transfer_source(fixture_root, held_out_splits):
    """#51: a held_out result keeps its shape (no ``source`` key)."""
    task = _load(fixture_root, held_out_splits)
    md = task.test.metadata
    preds = pd.DataFrame(
        {"sample_id": md["sample_id"], "window_idx": md["window_idx"], "prediction": 0}
    )
    assert "source" not in evaluate(task, preds).to_dict()


def test_missing_validation_support_is_unknown_not_zero(tmp_path):
    """#51: a result without ``n_test`` (legacy validation re-scores) exports as unknown."""
    import json

    from evemtbench.leaderboard import scan_results

    d = tmp_path / "outputs" / "1.0.0" / TASK / "held_out" / "double_line" / "cnn" / "val"
    d.mkdir(parents=True)
    doc = {
        "benchmark_version": "1.0.0", "task": TASK, "protocol": "held_out", "grid": "double_line",
        "baseline": "cnn", "test_set": "val", "seeds": [0],
        "aggregate": {"balanced_accuracy": {"mean": 0.9, "std": 0.0, "ci95": None}},
        "per_seed": [{"seed": 0, "metrics": {"balanced_accuracy": 0.9}}],
    }  # fmt: skip
    (d / "result.json").write_text(json.dumps(doc))
    (row,) = scan_results(tmp_path / "outputs")
    assert row.n_test is None
    (d / "result.json").write_text(json.dumps({**doc, "n_test": 123}))
    (row,) = scan_results(tmp_path / "outputs")
    assert row.n_test == 123
