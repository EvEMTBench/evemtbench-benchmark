"""Per-cell failure records — why a cell is not ``done``, and how often it has been tried.

A cell that crashes leaves no ``result.json``, so the coverage manifest alone cannot tell
"never started" from "died three times in the same place"; the reason used to live only in a
SLURM ``.out`` file. The runner now keeps ``failure.json`` in the cell directory:

* each run **opens an attempt** (``status: "running"``) before it loads anything;
* an exception (including SIGTERM, which SLURM sends at the time limit) **closes** it as
  ``failed`` with the stage it happened in, the exception type, a short message and the
  traceback tail;
* a run that completes **removes** the file. Success leaves no record, and failures that were
  later fixed do not linger.

An attempt still marked ``running`` when nothing runs (SIGKILL, OOM kill, node failure) is
reported as ``no clean exit``. :mod:`evemtbench.coverage` uses the attempt count as a **retry
cap**: ``--emit-pending`` skips a cell after :data:`MAX_ATTEMPTS` failed attempts, so the sweep
stops resubmitting a deterministic failure; ``--failures`` lists the reasons. The Agent4CT
retry feeder does the same with ``MAX_RETRY``. Unlike Agent4CT, a failure is never stored as a
score: these records are operational state, read by no result builder, and not part of the
committed coverage snapshot.

This module must not import :mod:`evemtbench.baselines` (the evaluation layer stays importable
without it).
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import signal
import traceback
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from ._io import atomic_write_text

FAILURE_FILE = "failure.json"
SCHEMA_VERSION = 1
#: default retry cap for ``coverage --emit-pending`` (override with ``--max-attempts``).
MAX_ATTEMPTS = 3
#: the reason reported for an attempt that never closed (killed without running Python code).
NO_CLEAN_EXIT = "no clean exit (killed: SIGKILL / OOM / node failure?)"
_MESSAGE_CHARS = 500
_TRACEBACK_LINES = 25


class TerminatedBySignal(BaseException):
    """Raised in the main thread when the process receives SIGTERM (e.g. the SLURM time limit),
    so the failure record is closed with the stage the run was in. A ``BaseException``, like
    ``KeyboardInterrupt``, so ordinary ``except Exception`` handlers do not swallow it."""


def failure_path(cell_dir: str | Path) -> Path:
    return Path(cell_dir) / FAILURE_FILE


def load(cell_dir: str | Path) -> dict | None:
    """The cell's failure record, or ``None`` (none / unreadable)."""
    path = failure_path(cell_dir)
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(doc, dict) or not isinstance(doc.get("attempts"), list):
        return None
    return doc


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(cell_dir: Path, doc: dict) -> None:
    cell_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(failure_path(cell_dir), json.dumps(doc, indent=2, sort_keys=True) + "\n")


def open_attempt(cell_dir: str | Path) -> int:
    """Append a ``running`` attempt and return its number (1-based)."""
    cell_dir = Path(cell_dir)
    doc = load(cell_dir) or {"schema_version": SCHEMA_VERSION, "attempts": []}
    n = len(doc["attempts"]) + 1
    doc["attempts"].append(
        {
            "attempt": n,
            "status": "running",
            "started": _now(),
            "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
    )
    _write(cell_dir, doc)
    return n


def close_attempt(cell_dir: str | Path, attempt: int, *, stage: str, exc: BaseException) -> None:
    """Mark ``attempt`` as failed in ``stage`` because of ``exc``. Never raises: recording a
    failure must not mask the failure itself."""
    try:
        cell_dir = Path(cell_dir)
        doc = load(cell_dir) or {"schema_version": SCHEMA_VERSION, "attempts": []}
        entry = next((a for a in doc["attempts"] if a.get("attempt") == attempt), None)
        if entry is None:
            entry = {"attempt": attempt}
            doc["attempts"].append(entry)
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        entry.update(
            {
                "status": "failed",
                "ended": _now(),
                "stage": stage,
                "error_type": type(exc).__name__,
                "message": str(exc)[:_MESSAGE_CHARS],
                "traceback_tail": "".join(tb).splitlines()[-_TRACEBACK_LINES:],
            }
        )
        _write(cell_dir, doc)
    except Exception:  # best effort by design
        pass


def clear(cell_dir: str | Path) -> None:
    """Remove the record after a successful run."""
    failure_path(cell_dir).unlink(missing_ok=True)


def n_attempts(doc: dict | None) -> int:
    return len(doc["attempts"]) if doc else 0


def reason(attempt: dict) -> str:
    """A grouping key for an attempt: ``"<stage>: <ErrorType>"``, or :data:`NO_CLEAN_EXIT` for
    an attempt that never closed. Built from fixed fields only, never from the free-text message."""
    if attempt.get("status") != "failed":
        return NO_CLEAN_EXIT
    stage = str(attempt.get("stage") or "?")
    # per-seed / per-test-set detail is useful in the record, but groups better without it
    stage = stage.split(" (")[0]
    return f"{stage}: {attempt.get('error_type') or '?'}"


def last_reason(doc: dict | None) -> str | None:
    if not doc or not doc["attempts"]:
        return None
    return reason(doc["attempts"][-1])


@contextlib.contextmanager
def sigterm_as_exception() -> Iterator[None]:
    """Turn SIGTERM into :class:`TerminatedBySignal` for the duration of the block (main thread
    only; elsewhere this is a no-op). The previous handler is restored afterwards."""
    try:
        previous = signal.getsignal(signal.SIGTERM)

        def _handler(signum, frame):
            raise TerminatedBySignal(f"received signal {signum} (SIGTERM)")

        signal.signal(signal.SIGTERM, _handler)
    except ValueError:  # not in the main thread
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
