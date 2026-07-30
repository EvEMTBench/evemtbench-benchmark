"""Persistent projection-cache policy (evemtbench.data.staging) + the read-only guard in
TaskView.materialize_cache. These encode the I/O-redesign invariants: caches are content-addressed
and shared across tasks of the same observability; GPU training never *builds* a cache."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from evemtbench.data import staging
from evemtbench.tasks.task import TaskView, load_task
from evemtbench.version import BENCHMARK_VERSION


def _task(fixture_root, held_out_splits, task_id):
    return load_task(
        task_id, "held_out", grid="double_line", root=fixture_root, splits_root=held_out_splits
    )


def test_cache_filename_stable_and_content_addressed(fixture_root, held_out_splits):
    t = _task(fixture_root, held_out_splits, "fault_detection_local")
    v = t.view("train")
    fn1, fn2 = v.cache_filename(), v.cache_filename()
    assert fn1 == fn2 and fn1.startswith("proj_train_double_line_local_") and fn1.endswith(".f32")
    # a different view (different name/shape) gets a different filename
    assert t.view("test").cache_filename() != fn1


def test_different_window_sets_get_different_caches(fixture_root, held_out_splits):
    """Detection/classification/location keep different window subsets (fault+normal vs faulted),
    so their content keys differ — a cache is never mistakenly shared across task families. (Sharing
    is across seeds/models of the *same* task, which is the per-job-rebuild win.)"""
    names = {
        _task(fixture_root, held_out_splits, f).view("train").cache_filename()
        for f in ("fault_detection_local", "fault_classification_local", "fault_location_local")
    }
    assert len(names) == 3


def test_persistent_cache_root_env_and_version_scoped(tmp_path, monkeypatch):
    monkeypatch.setenv("EVEMTBENCH_PROJECTION_CACHE_ROOT", str(tmp_path))
    assert staging.persistent_cache_root() == tmp_path / BENCHMARK_VERSION


def test_view_cache_files_dedup_and_absent(fixture_root, held_out_splits, tmp_path, monkeypatch):
    monkeypatch.setenv("EVEMTBENCH_PROJECTION_CACHE_ROOT", str(tmp_path))  # empty → all absent
    # duplicate task entries must dedup to that task's cacheable views (≤ 4 views)
    files = staging.view_cache_files(
        "double_line",
        ["fault_detection_local", "fault_detection_local"],
        root=fixture_root,
        splits_root=held_out_splits,
    )
    assert files, "expected some cacheable views"
    assert all(not exists for _, exists in files)  # nothing built yet
    assert 1 <= len(files) <= 4  # deduped across the duplicate entry


def test_readonly_guard_raises_when_cache_missing(
    fixture_root, held_out_splits, tmp_path, monkeypatch
):
    v = _task(fixture_root, held_out_splits, "fault_detection_local").view("train")
    monkeypatch.setenv("EVEMTBENCH_CACHE_READONLY", "1")
    with pytest.raises(RuntimeError, match="EVEMTBENCH_CACHE_READONLY"):
        v.materialize_cache(tmp_path)


def test_prep_builds_then_readonly_reads_bit_identical(
    fixture_root, held_out_splits, tmp_path, monkeypatch
):
    """Prep (no readonly) builds the cache; a fresh view with readonly=1 then *reads* it
    (cache hit, no build) and yields the same features as the live projection."""
    live = _task(fixture_root, held_out_splits, "fault_detection_local")
    live_feats = live.view("train").features.copy()

    builder = _task(fixture_root, held_out_splits, "fault_detection_local").view("train")
    builder.materialize_cache(tmp_path)  # builds (no readonly)

    monkeypatch.setenv("EVEMTBENCH_CACHE_READONLY", "1")
    reader = _task(fixture_root, held_out_splits, "fault_detection_local").view("train")
    reader.materialize_cache(tmp_path)  # must NOT raise — cache present
    assert np.array_equal(reader.features, live_feats)


def test_readonly_guard_noop_on_empty_dense_view(tmp_path, monkeypatch):
    """The runner materializes ALL views incl. an empty benchmark; the read-only guard must NOT
    raise for an empty/dense view (cache_filename is None there). Regression for the exact
    runner-on-uncached-empty-benchmark path."""
    monkeypatch.setenv("EVEMTBENCH_CACHE_READONLY", "1")
    empty = TaskView(
        "benchmark", [], np.array([]), pd.DataFrame({"sample_id": []}), dense=np.empty((0, 0, 0))
    )
    assert empty.cache_filename() is None
    empty.materialize_cache(tmp_path)  # must NOT raise, must stay uncached
    assert empty._cache is None


def test_readonly_guard_noop_when_already_cached(
    fixture_root, held_out_splits, tmp_path, monkeypatch
):
    """A view already holding its cache must not re-trigger the guard."""
    v = _task(fixture_root, held_out_splits, "fault_detection_local").view("train")
    v.materialize_cache(tmp_path)  # builds + sets _cache
    monkeypatch.setenv("EVEMTBENCH_CACHE_READONLY", "1")
    v.materialize_cache(tmp_path)  # _cache already set -> early return, no raise
    assert v._cache is not None


def test_cache_status_counts(fixture_root, held_out_splits, tmp_path, monkeypatch):
    monkeypatch.setenv("EVEMTBENCH_PROJECTION_CACHE_ROOT", str(tmp_path))
    croot = staging.persistent_cache_root()
    # Build only fault_detection_local's caches into the persistent root.
    t = _task(fixture_root, held_out_splits, "fault_detection_local")
    for vn in ("train", "val", "test", "benchmark"):
        t.view(vn).materialize_cache(croot)
    fd = staging.cache_status(
        "double_line", ["fault_detection_local"], root=fixture_root, splits_root=held_out_splits
    )
    assert fd["missing"] == [] and fd["present"] == fd["total"] > 0
    # A task whose caches were not built shows up as missing.
    fc = staging.cache_status(
        "double_line",
        ["fault_classification_local"],
        root=fixture_root,
        splits_root=held_out_splits,
    )
    assert len(fc["missing"]) == fc["total"] and fc["present"] == 0


def test_prune_stale_versions_only_touches_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("EVEMTBENCH_PROJECTION_CACHE_ROOT", str(tmp_path))
    cur = staging.persistent_cache_root()  # tmp_path/<BENCHMARK_VERSION>
    cur.mkdir(parents=True)
    (cur / "proj_x.f32").write_bytes(b"0")
    old = tmp_path / "0.0.1-old"
    old.mkdir()
    (old / "proj_y.f32").write_bytes(b"0")
    junk = tmp_path / "not-a-cache"  # no proj_*.f32 -> must be untouched
    junk.mkdir()
    (junk / "keepme.txt").write_bytes(b"0")
    listed = staging.prune_stale_versions(dry_run=True)
    assert [p.name for p in listed] == ["0.0.1-old"]
    staging.prune_stale_versions(dry_run=False)
    assert not old.exists() and cur.exists() and junk.exists()  # only the stale cache removed


def test_cli_files_and_status(fixture_root, held_out_splits, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EVEMTBENCH_PROJECTION_CACHE_ROOT", str(tmp_path))  # empty -> all missing
    assert staging._main(["cache-root"]) == 0
    assert str(BENCHMARK_VERSION) in capsys.readouterr().out
    args = [
        "--grid",
        "double_line",
        "--tasks",
        "fault_detection_local",
        "--root",
        str(fixture_root),
        "--splits-root",
        str(held_out_splits),
    ]
    assert staging._main(["files", *args]) == 0
    assert all(ln.startswith("proj_") for ln in capsys.readouterr().out.split() if ln)
    assert staging._main(["status", *args]) == 1  # nonzero exit when caches missing (shell gate)


def test_runner_end_to_end_persistent_cache(fixture_root, held_out_splits, tmp_path, monkeypatch):
    """End-to-end of the production path: a CPU-prep-style build populates the cache; a read-only
    run then trains from it (what GPU jobs do) and writes results; an empty cache under read-only
    fails fast (never builds/reads raw under GPU)."""
    pytest.importorskip("torch")  # --baseline mlp trains a deep model; skip on the torch-free CI
    from evemtbench.baselines import runner

    cdir = tmp_path / "cache"
    common = [
        "--task",
        "fault_detection_local",
        "--protocol",
        "held_out",
        "--grid",
        "double_line",
        "--baseline",
        "mlp",
        "--seeds",
        "0",
        "--epochs",
        "1",
        "--root",
        str(fixture_root),
        "--splits-root",
        str(held_out_splits),
    ]
    monkeypatch.setenv("EVEMTBENCH_PROJECTION_CACHE", str(cdir))

    # 1) prep: build the cache (no readonly) and write results
    runner.main([*common, "--out", str(tmp_path / "out1")])
    assert any(cdir.glob("proj_*.f32")), "prep run should have built caches"

    # 2) production: read-only run trains from the prebuilt cache and writes results
    monkeypatch.setenv("EVEMTBENCH_CACHE_READONLY", "1")
    out2 = tmp_path / "out2"
    runner.main([*common, "--out", str(out2)])
    base = out2 / BENCHMARK_VERSION / "fault_detection_local" / "held_out" / "double_line" / "mlp"
    assert json.loads((base / "test" / "result.json").read_text())["baseline"] == "mlp"

    # 3) fail-fast: read-only with an empty cache dir must raise (no GPU-held build, no raw read)
    monkeypatch.setenv("EVEMTBENCH_PROJECTION_CACHE", str(tmp_path / "empty"))
    with pytest.raises(RuntimeError, match="EVEMTBENCH_CACHE_READONLY"):
        runner.main([*common, "--out", str(tmp_path / "out3")])
