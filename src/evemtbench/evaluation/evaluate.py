"""The official ``evaluate`` entry point — frozen metrics + stratified breakdowns.

Aligns a predictions file/frame to a task's test view by ``(sample_id, window_idx)``, computes
the task's headline metrics and stratified breakdowns, and stamps the benchmark version so
every result is attributable to a frozen contract.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..tasks.labels.event_state import EVENT_STATE_CLASSES
from ..tasks.labels.event_type import EVENT_TYPE_CLASSES
from ..tasks.labels.fault_category import FAULT_CATEGORY_CLASSES
from ..tasks.labels.fault_class import FAULT_CLASS_CLASSES
from ..tasks.labels.fault_phase import PHASE_CLASSES
from ..tasks.labels.switch_type import SWITCH_TYPE_CLASSES
from ..tasks.task import Task
from ..tasks.taxonomy import Function, LabelKind
from ..version import BENCHMARK_VERSION
from .metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    compute_regression_metrics,
)
from .predictions_io import align_to_view
from .stratify import stratified_metrics

# Multiclass tasks → their frozen class vocabulary (so absent classes still score and stay
# comparable). Binary/regression tasks are absent from this map.
_CLASS_VOCAB = {
    Function.FAULT_CLASS: FAULT_CLASS_CLASSES,
    Function.EVENT_STATE: EVENT_STATE_CLASSES,
    Function.EVENT_CLASSIFICATION: EVENT_TYPE_CLASSES,
    Function.SWITCH_TYPE: SWITCH_TYPE_CLASSES,
    Function.FAULT_PHASE: PHASE_CLASSES,
    Function.FAULT_CATEGORY: FAULT_CATEGORY_CLASSES,
}


def _nan_to_none(obj):
    """Recursively replace NaN/inf floats with None so the JSON is strictly valid."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


@dataclass
class EvalResult:
    benchmark_version: str
    task: str
    protocol: str
    grid: str
    n_samples: int
    metrics: dict[str, float]
    test_set: str = "test"  # which held-out set: "test" (adapt_grid) or "benchmark"
    breakdowns: dict[str, dict] = field(default_factory=dict)
    seed_summary: dict | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        sanitized = _nan_to_none(json.loads(json.dumps(self.to_dict(), default=float)))
        return json.dumps(sanitized, indent=indent, sort_keys=True, allow_nan=False)


def evaluate(task: Task, predictions: pd.DataFrame, *, test_set: str = "test") -> EvalResult:
    """Evaluate ``predictions`` against one held-out view using the task's official metrics.

    ``test_set`` selects the view: ``"test"`` (adapt_grid in-distribution) or ``"benchmark"``
    (the held-out benchmark family). Recorded on the result for attribution.
    """
    td = task.definition
    test = task.view(test_set)
    is_regression = td.label_kind is LabelKind.REGRESSION

    def _cast(arr):
        # Regression keeps floats; classification rounds to the nearest class id rather than
        # truncating toward zero, so a third-party float prediction like 2.9999999 scores as 3.
        arr = np.asarray(arr)
        return arr.astype(float) if is_regression else np.round(arr).astype(int)

    y_true = _cast(test.labels)
    y_pred = _cast(align_to_view(test.metadata, predictions))

    if td.label_kind is LabelKind.BINARY:

        def metric_fn(yt, yp):
            return compute_binary_metrics(yt, yp, td.metrics)
    elif td.label_kind is LabelKind.MULTICLASS:
        classes = _CLASS_VOCAB.get(td.function)

        def metric_fn(yt, yp):
            return compute_multiclass_metrics(yt, yp, td.metrics, classes=classes)
    elif is_regression:

        def metric_fn(yt, yp):
            return compute_regression_metrics(yt, yp, td.metrics)
    else:
        raise NotImplementedError(f"evaluate does not support label kind {td.label_kind}")

    metrics = metric_fn(y_true, y_pred)
    breakdowns = stratified_metrics(test.metadata, y_true, y_pred, metric_fn)

    return EvalResult(
        benchmark_version=BENCHMARK_VERSION,
        task=td.task_id,
        protocol=task.protocol,
        grid=task.grid,
        n_samples=len(y_true),
        metrics=metrics,
        test_set=test_set,
        breakdowns=breakdowns,
    )
