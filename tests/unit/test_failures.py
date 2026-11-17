"""Per-cell failure records (evemtbench.failures)."""

import os
import signal

import pytest

from evemtbench import failures


def test_lifecycle_open_close_clear(tmp_path):
    cell = tmp_path / "cell"
    n = failures.open_attempt(cell)
    assert n == 1
    doc = failures.load(cell)
    assert doc["attempts"][0]["status"] == "running"
    try:
        raise RuntimeError("CUDA out of memory")
    except RuntimeError as e:
        failures.close_attempt(cell, n, stage="fit (seed 3)", exc=e)
    a = failures.load(cell)["attempts"][0]
    assert a["status"] == "failed" and a["stage"] == "fit (seed 3)"
    assert a["error_type"] == "RuntimeError" and "out of memory" in a["message"]
    assert a["traceback_tail"]
    assert failures.reason(a) == "fit: RuntimeError"  # per-seed detail dropped for grouping
    assert failures.open_attempt(cell) == 2
    assert failures.n_attempts(failures.load(cell)) == 2
    failures.clear(cell)
    assert failures.load(cell) is None


def test_unclosed_attempt_reads_as_no_clean_exit(tmp_path):
    failures.open_attempt(tmp_path)
    assert failures.last_reason(failures.load(tmp_path)) == failures.NO_CLEAN_EXIT


def test_corrupt_record_is_ignored(tmp_path):
    failures.failure_path(tmp_path).write_text("{not json")
    assert failures.load(tmp_path) is None


def test_sigterm_becomes_an_exception_and_handler_is_restored():
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(failures.TerminatedBySignal):
        with failures.sigterm_as_exception():
            os.kill(os.getpid(), signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM) == before
