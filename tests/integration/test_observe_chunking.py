"""Chunked feature projection must be bit-identical to a single-shot read (memory-only change)."""

import numpy as np

from evemtbench.tasks import task as task_mod
from evemtbench.tasks.task import load_task


def _build(fixture_root, held_out_splits, view="benchmark"):
    t = load_task(
        "fault_detection_global",
        "held_out",
        grid="double_line",
        root=fixture_root,
        splits_root=held_out_splits,
    )
    return getattr(t, view).features.copy()


def test_chunk_size_invariant(fixture_root, held_out_splits, monkeypatch):
    monkeypatch.setattr(task_mod, "_OBSERVE_CHUNK", 100000)  # single pass
    big = _build(fixture_root, held_out_splits)
    monkeypatch.setattr(task_mod, "_OBSERVE_CHUNK", 3)  # many tiny chunks
    small = _build(fixture_root, held_out_splits)
    assert big.shape == small.shape
    assert np.array_equal(big, small)
