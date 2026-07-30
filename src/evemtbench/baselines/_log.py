"""Lightweight training/data-loading logging for the deep baselines.

A long IEEE-39 `global` cell used to print nothing between the cell marker and the final metrics —
"flying blind" for hours. These helpers emit timestamped, flushed lines (captured in the SLURM
`.out`) showing per-epoch loss, wall-time, throughput, and crucially the **data-loading fraction**
of each epoch, so a GPU starved by the data pipeline is visible at a glance (and the improvement
from the parallel loader is measurable).

Two optional, tracker-agnostic hooks (both no-ops unless the runner wires them up) give the v1.0
logging redundancy without the torch/eval layers knowing about any tracker:

* **File mirror** (``set_log_file``): the runner points this at ``outputs/<cell>/train.log`` so the
  per-epoch stream is durable and CELL-ADDRESSABLE — the SLURM ``.out`` is job-named and interleaves
  many cells, so it is not a per-result artifact.
* **Numeric sink** (``set_epoch_sink`` / ``emit_epoch``): an experiment tracker (W&B) registers a
  callable; ``_train`` feeds it the raw per-epoch floats alongside the formatted stdout line.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

# Per-cell durable log mirror (set by the runner around a cell; None clears/closes).
_mirror = None  # type: ignore[assignment]  # open text file handle or None


def set_log_file(path: str | Path | None) -> None:
    """Mirror subsequent ``log()`` lines to ``path`` (append, line-buffered). ``None`` clears."""
    global _mirror
    if _mirror is not None:
        try:
            _mirror.close()
        except Exception:
            pass
        _mirror = None
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        _mirror = open(p, "a", buffering=1, encoding="utf-8")


def log(msg: str) -> None:
    """Timestamped, flushed stdout line (so it lands in the SLURM .out promptly); also mirrored to
    the per-cell ``train.log`` when a file is registered via :func:`set_log_file`."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _mirror is not None:
        try:
            _mirror.write(line + "\n")
        except Exception:
            pass


# Optional per-epoch numeric sink (an experiment tracker registers here; no-op when unset).
_epoch_sink: Callable[[dict], None] | None = None


def set_epoch_sink(sink: Callable[[dict], None] | None) -> None:
    """Register (or clear with ``None``) a callable that receives raw per-epoch metric dicts."""
    global _epoch_sink
    _epoch_sink = sink


def emit_epoch(metrics: dict) -> None:
    """Feed raw per-epoch numeric metrics to the active sink; silent no-op if unset or on error
    (logging must never break training)."""
    if _epoch_sink is not None:
        try:
            _epoch_sink(metrics)
        except Exception:
            pass


def epoch_line(
    epoch: int, epochs: int, *, loss: float, wall_s: float, data_s: float, n: int, extra: str = ""
) -> str:
    """Format one per-epoch progress line: loss, wall-time, data-loading %, throughput."""
    wall = max(wall_s, 1e-9)
    return (
        f"  epoch {epoch}/{epochs} loss={loss:.4f} {wall:.1f}s "
        f"data={100 * data_s / wall:.0f}% {n / wall:.0f} samp/s{extra}"
    )
