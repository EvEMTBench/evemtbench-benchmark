"""Persisted channel-stats (2026-07-18 GPU-audit follow-up): the per-cell full read+project pass
(up to 635 s on wide views) is computed once and reused across processes/links via a content-
addressed npz next to the projection cache. Invariants: loaded stats are the EXACT arrays a fresh
compute returns; the key covers view content + transform + accumulation batch size; no env vars ⇒
no disk I/O (hermetic memo-only behavior, exactly as before)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evemtbench.tasks.task import TaskView, add_zero_sequence, build_input_transform, load_task
from evemtbench.version import BENCHMARK_VERSION


def _task(fixture_root, held_out_splits, task_id="fault_detection_local"):
    return load_task(
        task_id, "held_out", grid="double_line", root=fixture_root, splits_root=held_out_splits
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Persistence is env-gated; start every test from the hermetic (no-persistence) state."""
    monkeypatch.delenv("EVEMTBENCH_PROJECTION_CACHE", raising=False)
    monkeypatch.delenv("EVEMTBENCH_PROJECTION_CACHE_ROOT", raising=False)
    monkeypatch.delenv("EVEMTBENCH_CACHE_READONLY", raising=False)


def test_roundtrip_and_no_recompute(fixture_root, held_out_splits, tmp_path, monkeypatch):
    monkeypatch.setenv("EVEMTBENCH_PROJECTION_CACHE_ROOT", str(tmp_path))
    v1 = _task(fixture_root, held_out_splits).view("train")
    mean1, std1 = v1.channel_stats(transform=None)
    files = list((tmp_path / BENCHMARK_VERSION).glob("stats_*.npz"))
    assert len(files) == 1 and files[0].name == v1.stats_filename("none")

    v2 = _task(fixture_root, held_out_splits).view("train")  # fresh Task, empty memo
    monkeypatch.setattr(
        v2, "iter_batches", lambda *a, **k: pytest.fail("persisted stats must skip the full pass")
    )
    mean2, std2 = v2.channel_stats(transform=None)
    assert mean2.dtype == np.float32 and std2.dtype == np.float32
    assert np.array_equal(mean1, mean2) and np.array_equal(std1, std2)


def test_key_includes_transform_and_batch_size(fixture_root, held_out_splits):
    v = _task(fixture_root, held_out_splits).view("train")
    composed = build_input_transform(True, "per_window")
    names = {
        v.stats_filename("none"),
        v.stats_filename(add_zero_sequence.__name__),
        v.stats_filename(composed.__name__),
        v.stats_filename("none", batch_size=4096),
    }
    assert len(names) == 4  # transform and accumulation chunking never collide


def test_content_addressed_across_tasks(fixture_root, held_out_splits):
    names = {
        _task(fixture_root, held_out_splits, t).view("train").stats_filename("none")
        for t in ("fault_detection_local", "fault_classification_local", "fault_location_local")
    }
    assert len(names) == 3  # different window subsets -> different content keys


def test_readonly_does_not_block_stats_write(fixture_root, held_out_splits, tmp_path, monkeypatch):
    """CACHE_READONLY guards projection-cache BUILDS; the ~10 KB stats write-back stays allowed
    (self-healing across links). channel_stats must not raise and must persist."""
    monkeypatch.setenv("EVEMTBENCH_PROJECTION_CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("EVEMTBENCH_CACHE_READONLY", "1")
    v = _task(fixture_root, held_out_splits).view("train")
    v.channel_stats(transform=None)
    assert list((tmp_path / BENCHMARK_VERSION).glob("stats_*.npz"))


def test_two_level_probe_order(fixture_root, held_out_splits, tmp_path, monkeypatch):
    local, root = tmp_path / "local", tmp_path / "root"
    monkeypatch.setenv("EVEMTBENCH_PROJECTION_CACHE", str(local))
    monkeypatch.setenv("EVEMTBENCH_PROJECTION_CACHE_ROOT", str(root))
    v1 = _task(fixture_root, held_out_splits).view("train")
    mean1, _ = v1.channel_stats(transform=None)
    fname = v1.stats_filename("none")
    assert (local / fname).exists() and (root / BENCHMARK_VERSION / fname).exists()

    # corrupt the node-local copy -> loader falls through to the persistent root, then heals local
    (local / fname).write_bytes(b"garbage")
    v2 = _task(fixture_root, held_out_splits).view("train")
    mean2, _ = v2.channel_stats(transform=None)
    assert np.array_equal(mean1, mean2)


def test_corrupt_everywhere_recomputes_and_rewrites(
    fixture_root, held_out_splits, tmp_path, monkeypatch
):
    monkeypatch.setenv("EVEMTBENCH_PROJECTION_CACHE_ROOT", str(tmp_path))
    v1 = _task(fixture_root, held_out_splits).view("train")
    mean1, _ = v1.channel_stats(transform=None)
    path = tmp_path / BENCHMARK_VERSION / v1.stats_filename("none")
    path.write_bytes(b"\x00" * 16)

    v2 = _task(fixture_root, held_out_splits).view("train")
    mean2, _ = v2.channel_stats(transform=None)  # falls back to compute, rewrites atomically
    assert np.array_equal(mean1, mean2)
    with np.load(path) as z:  # the corrupt file was replaced by a valid one
        assert np.array_equal(z["mean"], mean1)


def test_no_env_means_no_disk_io(fixture_root, held_out_splits, tmp_path):
    v = _task(fixture_root, held_out_splits).view("train")
    v.channel_stats(transform=None)
    assert not list(tmp_path.rglob("stats_*.npz"))  # nothing written anywhere we can see
    from evemtbench.tasks.task import _stats_dirs

    assert _stats_dirs() == []


def test_empty_dense_view_skips_persistence(tmp_path, monkeypatch):
    monkeypatch.setenv("EVEMTBENCH_PROJECTION_CACHE_ROOT", str(tmp_path))
    empty = TaskView(
        "benchmark", [], np.array([]), pd.DataFrame({"sample_id": []}), dense=np.empty((0, 0, 0))
    )
    assert empty.stats_filename("none") is None
