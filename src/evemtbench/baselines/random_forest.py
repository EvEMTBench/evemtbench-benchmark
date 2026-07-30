"""Feature-based Random Forest baseline (classical ML reference, Benchmark Plan §5.4/§8.3).

Extracts per-channel summary features from each window (RMS, peak, min, max, std, mean,
peak-to-peak) and fits a class-balanced Random Forest. Uses only scikit-learn (no extra
dependency), so it runs anywhere — CPU, CI-with-data, or HPC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ..evaluation.predictions_io import predictions_frame
from ..tasks.task import Task, TaskView
from ..tasks.taxonomy import LabelKind

_RF_BATCH = 8192


def _features_streamed(view: TaskView) -> np.ndarray:
    """Stream the view from the memmap and build the small (n, c*7) summary matrix without ever
    materializing the full (n, l, c) window array (so the wide global view fits in RAM)."""
    parts = [_features(x) for x, _ in view.iter_batches(_RF_BATCH)]
    return np.concatenate(parts, axis=0) if parts else np.empty((0, 0), np.float32)


def _features(windows: np.ndarray) -> np.ndarray:
    """(n, l, c) -> (n, c*7) per-channel summary statistics over the time axis."""
    x = np.asarray(windows, dtype=np.float32)
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    xmin = x.min(axis=1)
    xmax = x.max(axis=1)
    ptp = xmax - xmin
    rms = np.sqrt((x**2).mean(axis=1))
    peak = np.abs(x).max(axis=1)
    return np.concatenate([mean, std, xmin, xmax, ptp, rms, peak], axis=1)


class RandomForestBaseline:
    name = "random_forest"

    def __init__(self, *, n_estimators: int = 200, max_depth: int | None = None, seed: int = 0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self.model = None
        self._regression = False

    def fit(self, task: Task) -> None:
        self._regression = task.definition.label_kind is LabelKind.REGRESSION
        if self._regression:
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.seed,
                n_jobs=-1,
            )
            y = np.asarray(task.train.labels).astype(float)
        else:
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                class_weight="balanced",
                random_state=self.seed,
                n_jobs=-1,
            )
            y = np.asarray(task.train.labels).astype(int)
        assert self.model is not None  # set in both branches above
        self.model.fit(_features_streamed(task.train), y)

    def predict(self, view: TaskView) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("call fit() before predict()")
        raw = self.model.predict(_features_streamed(view))
        preds = raw.astype(float) if self._regression else raw.astype(int)
        return predictions_frame(view.metadata["sample_id"], view.metadata["window_idx"], preds)
