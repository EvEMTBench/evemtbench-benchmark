"""Majority/mean baseline — the sanity lower bound (Benchmark Plan §5.4).

Classification: predicts the most frequent train class. Regression (fault location): predicts
the train **mean** target — the constant-prediction baseline any model must beat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluation.predictions_io import predictions_frame
from ..tasks.task import Task, TaskView
from ..tasks.taxonomy import LabelKind


class MajorityBaseline:
    """Predicts the most frequent train class (or the train mean, for regression)."""

    name = "majority"

    def __init__(self) -> None:
        self.constant: float | int | None = None

    def fit(self, task: Task) -> None:
        labels = np.asarray(task.train.labels)
        if task.definition.label_kind is LabelKind.REGRESSION:
            self.constant = float(np.mean(labels.astype(float)))
        else:
            # Most frequent class on train (ties -> lower class id).
            values, counts = np.unique(labels.astype(int), return_counts=True)
            self.constant = int(values[np.argmax(counts)])

    def predict(self, view: TaskView) -> pd.DataFrame:
        if self.constant is None:
            raise RuntimeError("call fit() before predict()")
        preds = np.full(len(view), self.constant)
        return predictions_frame(view.metadata["sample_id"], view.metadata["window_idx"], preds)
