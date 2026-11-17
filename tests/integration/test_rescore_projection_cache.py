"""hpc/rescore_validation.py reads the projection cache when configured, as the runner does."""

import pytest


class _View:
    def __init__(self, n: int):
        self.n, self.calls = n, []

    def __len__(self) -> int:
        return self.n

    def materialize_cache(self, cdir: str) -> None:
        self.calls.append(cdir)


class _Task:
    def __init__(self, n_test: int = 3):
        self.train, self.val, self.test = _View(5), _View(2), _View(n_test)


@pytest.fixture
def rv():
    pytest.importorskip("torch")  # rescore_validation loads checkpoints; torch-free CI skips
    import rescore_validation

    return rescore_validation


def test_no_cache_configured_reads_nothing_new(rv, monkeypatch):
    monkeypatch.delenv("EVEMTBENCH_PROJECTION_CACHE", raising=False)
    task = _Task()
    rv.attach_projection_cache(task)
    assert task.train.calls == task.val.calls == task.test.calls == []


def test_configured_cache_is_attached_to_every_non_empty_view(rv, monkeypatch, tmp_path):
    monkeypatch.setenv("EVEMTBENCH_PROJECTION_CACHE", str(tmp_path))
    task = _Task(n_test=0)
    rv.attach_projection_cache(task)
    assert task.train.calls == [str(tmp_path)]
    assert task.val.calls == [str(tmp_path)]
    assert task.test.calls == []  # an empty view has no cache file to read


def test_rescore_cell_attaches_the_cache_before_scoring(rv, monkeypatch):
    seen = []
    monkeypatch.setattr(rv, "attach_projection_cache", lambda t: seen.append(t))
    monkeypatch.setattr(rv, "_committed_test_headline", lambda cell, headline: {})

    class _Def:
        metrics = ("mae",)

    monkeypatch.setattr(rv, "load_task", lambda *a, **k: type("T", (), {"definition": _Def()})())
    status = rv.rescore_cell(None, "fault_location_local", "double_line", "mlp", tol=1e-3)
    assert status.startswith("skip (no committed test")
    assert len(seen) == 1
