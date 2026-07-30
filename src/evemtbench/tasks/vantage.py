"""Vantage-conditioned views — the relay-selectivity seam (designed, NOT built).

A deployed relay protects ONE zone from ONE vantage (cubicle); selectivity asks it to trip for
in-zone faults only — an out-of-zone fault or benign event must not trip. The current tasks
never pose this question (the observability oracle hands the model the faulted line). Design:
``docs/design/selectivity.md`` + ``docs/task_cards/relay_selectivity.md``.

This module freezes the *interface* so the seam is importable and visible today; every function
raises :class:`NotImplementedError` with a pointer to the design doc. Filling it is post-v0.2.0
work (a paper Open Challenge). Product layer: no baselines/torch imports here, ever.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .task import Task, TaskView

_DESIGN = "designed but not built - see docs/design/selectivity.md"


def iter_vantages(grid: str) -> list[dict]:
    """Enumerate the frozen vantage set for ``grid``: one entry per line cubicle with its
    protected line (zone), derived from the same channel map the observability oracle uses
    (``tasks/line_view.py``). Bus faults map to the incident-line bays, mirroring
    ``y_fault_line``."""
    raise NotImplementedError(f"iter_vantages({grid!r}): {_DESIGN}")


def vantage_view(task: Task, cubicle_idx: int) -> TaskView:
    """A view of the vantage's own channels over ALL valid windows (not just oracle-selected
    episodes), with labels rewritten to ``in_zone`` ∈ {0,1} by a pure, golden-tested deriver
    (fault present AND on the vantage's protected line)."""
    raise NotImplementedError(f"vantage_view(cubicle_idx={cubicle_idx}): {_DESIGN}")
