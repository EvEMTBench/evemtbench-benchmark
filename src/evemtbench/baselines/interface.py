"""Baseline interface. A baseline fits on a task's train view and predicts a test view.

Baselines live ONLY here (the demo layer). Evaluation never imports baselines; a user supplies
their own predictions file. Implement this interface + register a config to add a baseline.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from ..tasks.task import Task, TaskView


@runtime_checkable
class Baseline(Protocol):
    name: str

    def fit(self, task: Task) -> None:
        """Fit on ``task.train`` (and optionally ``task.val``). Must not touch ``task.test``."""

    def predict(self, view: TaskView) -> pd.DataFrame:
        """Return a canonical predictions frame for ``view`` (keyed by sample_id, window_idx)."""
