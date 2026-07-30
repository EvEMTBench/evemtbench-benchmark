"""Overcurrent threshold baseline — transparent classical-protection reference (§8.2).

A single current-magnitude threshold separates fault from non-fault windows. The decision
statistic is the peak per-phase RMS current in the window; the threshold is selected on the
**training** split only (maximizing balanced accuracy), never on test (§9.1). This is a
domain reference, not a manufacturer-grade relay.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from ..evaluation.predictions_io import predictions_frame
from ..tasks.task import Task, TaskView

# Local view channel order: [Ia, Ib, Ic, Va, Vb, Vc]; currents are the first three.
_CURRENT_CHANNELS = (0, 1, 2)
_STAT_BATCH = 8192


def _current_statistic(features: np.ndarray) -> np.ndarray:
    """Per-window decision statistic: max over phases of RMS current. (n, l, c) -> (n,)."""
    cur = features[:, :, _CURRENT_CHANNELS]  # (n, l, 3)
    rms = np.sqrt((cur**2).mean(axis=1))  # (n, 3)
    return rms.max(axis=1)  # (n,)


def _current_statistic_streamed(view: TaskView) -> np.ndarray:
    """Stream the view from the memmap and build the (n,) decision statistic per batch, without ever
    materializing the full (n, l, c) window array — mirrors random_forest's streaming so the large
    grids (e.g. cigre_mv) don't thrash host RAM."""
    parts = [_current_statistic(x) for x, _ in view.iter_batches(_STAT_BATCH)]
    return np.concatenate(parts, axis=0) if parts else np.empty((0,), np.float32)


class ThresholdBaseline:
    name = "threshold"

    def __init__(self, *, n_candidates: int = 200, seed: int = 0):
        self.n_candidates = n_candidates
        self.threshold_: float | None = None

    def fit(self, task: Task) -> None:
        from ..tasks.taxonomy import LabelKind, Observability

        if task.definition.label_kind is not LabelKind.BINARY:
            raise ValueError(
                "threshold baseline is binary-only (e.g. fault_detection); "
                f"task {task.definition.task_id!r} is {task.definition.label_kind.value}"
            )
        if task.definition.observability is not Observability.LOCAL:
            raise ValueError(
                "threshold (overcurrent) baseline is a local-relay reference; "
                f"task {task.definition.task_id!r} uses {task.definition.observability.value} input"
            )
        stat = _current_statistic_streamed(task.train)
        y = np.asarray(task.train.labels).astype(int)
        # Candidate thresholds across the observed statistic range (quantiles for robustness).
        qs = np.linspace(0.0, 1.0, self.n_candidates)
        candidates = np.unique(np.quantile(stat, qs))
        best_thr, best_score = candidates[0], -1.0
        for thr in candidates:
            score = balanced_accuracy_score(y, (stat > thr).astype(int))
            if score > best_score:
                best_thr, best_score = float(thr), score
        self.threshold_ = best_thr

    def predict(self, view: TaskView) -> pd.DataFrame:
        if self.threshold_ is None:
            raise RuntimeError("call fit() before predict()")
        stat = _current_statistic_streamed(view)
        preds = (stat > self.threshold_).astype(int)
        return predictions_frame(view.metadata["sample_id"], view.metadata["window_idx"], preds)
