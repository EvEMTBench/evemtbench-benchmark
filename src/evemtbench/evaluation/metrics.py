"""Frozen metric implementations (binary protection tasks).

These definitions are part of the benchmark contract; changing them is a **major** version
bump. Protection-oriented rates use the convention positive = fault/trip:

* ``missed_fault_rate`` = FN / (TP + FN)  — fraction of true faults the model missed.
* ``false_alarm_rate``  = FP / (FP + TN)  — fraction of non-faults flagged as faults.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

MULTICLASS_METRICS = ("balanced_accuracy", "macro_f1", "accuracy")

BINARY_METRICS = (
    "balanced_accuracy",
    "macro_f1",
    "missed_fault_rate",
    "false_alarm_rate",
    "accuracy",
)

REGRESSION_METRICS = ("mae", "rmse", "r2", "median_abs_error")

# Optimization direction per metric: +1 = higher is better, -1 = lower is better.
# Consumed by the leaderboard so it does not assume [0,1]-higher-better (e.g. MAE, R²).
METRIC_DIRECTION: dict[str, int] = {
    "balanced_accuracy": +1,
    "macro_f1": +1,
    "macro_f1_supported": +1,
    "accuracy": +1,
    "missed_fault_rate": -1,
    "missed_event_rate": -1,
    "false_alarm_rate": -1,
    "false_event_trip_rate": -1,
    "mae": -1,
    "rmse": -1,
    "median_abs_error": -1,
    "r2": +1,
}


def macro_f1_supported(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 averaged over ONLY the classes with support in this evaluation set.

    A companion to ``macro_f1``, not a replacement for it. ``macro_f1`` is frozen benchmark contract
    and stays exactly as specified in the paper: per-class F1 with zero-division mapped to 0,
    averaged over the full frozen vocabulary. That definition is deliberate — it keeps the number
    comparable across grids — but it also means a grid where only 3 of 5 classes can occur is capped
    at 0.6, so the value is not an achievable-skill measure on grids with partial class support (FC
    with MV-only HIF classes, EC with grid-specific event types).

    This variant answers the other question — how well the model did on the classes that were
    actually there — by averaging over ``support > 0`` in ``y_true``. Reported alongside, so neither
    reading is lost and the contract is untouched.

    For multiclass it equals the ``nanmean`` of the emitted per-class ``f1_<class>`` values: sklearn
    computes each label's F1 over all samples independently, so restricting ``labels`` changes only
    which are averaged. NaN (never 0.0) on an empty set — undefined, not zero skill.
    """
    y_true = np.asarray(y_true)
    labels = np.unique(y_true)
    if labels.size == 0:
        return float("nan")
    return float(f1_score(y_true, y_pred, average="macro", labels=labels.tolist(), zero_division=0))


def _rates(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """(missed_fault_rate, false_alarm_rate) with positive class = 1."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    missed = float(fn / (tp + fn)) if (tp + fn) > 0 else float("nan")
    false_alarm = float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")
    return missed, false_alarm


def compute_binary_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, metric_names: tuple[str, ...] = BINARY_METRICS
) -> dict[str, float]:
    """Compute the requested binary metrics. NaN where a metric is undefined for the subset."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out: dict[str, float] = {}
    missed, false_alarm = _rates(y_true, y_pred)
    single_class = len(np.unique(y_true)) < 2
    for name in metric_names:
        if name == "accuracy":
            out[name] = float((y_true == y_pred).mean()) if len(y_true) else float("nan")
        elif name == "balanced_accuracy":
            out[name] = (
                float("nan") if single_class else float(balanced_accuracy_score(y_true, y_pred))
            )
        elif name == "macro_f1":
            out[name] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        elif name in ("missed_fault_rate", "missed_event_rate"):
            # FN rate among positives — "missed fault" or "missed event" by task framing
            out[name] = missed
        elif name in ("false_alarm_rate", "false_event_trip_rate"):
            # FP rate among negatives — false alarm, or (fault_vs_event) tripping on a benign event
            out[name] = false_alarm
        else:
            raise KeyError(f"unknown metric {name!r}")
    # Emitted additively (like the per-class f1_<class> keys), NOT via metric_names: the frozen
    # td.metrics tuples, BINARY/MULTICLASS_METRICS and HEADLINE_PRIORITY stay untouched, so
    # headline selection and the metric contract are unaffected.
    out["macro_f1_supported"] = macro_f1_supported(y_true, y_pred)
    return out


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_names: tuple[str, ...] = REGRESSION_METRICS,
) -> dict[str, float]:
    """Compute regression metrics for the fault-location task (target = distance %)."""
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    err = y_pred - y_true
    out: dict[str, float] = {}
    n = len(y_true)
    for name in metric_names:
        if name == "mae":
            out[name] = float(np.mean(np.abs(err))) if n else float("nan")
        elif name == "rmse":
            out[name] = float(np.sqrt(np.mean(err**2))) if n else float("nan")
        elif name == "median_abs_error":
            out[name] = float(np.median(np.abs(err))) if n else float("nan")
        elif name == "r2":
            if n < 2:
                out[name] = float("nan")
            else:
                ss_res = float(np.sum(err**2))
                ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
                out[name] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        else:
            raise KeyError(f"unknown metric {name!r}")
    return out


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_names: tuple[str, ...] = MULTICLASS_METRICS,
    *,
    classes: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Compute multiclass metrics + per-class F1. ``classes`` fixes the label set so every class
    appears as a column (keeping scores comparable across grids), but a class with **no support**
    in this eval set (e.g. HIF on a non-MV grid) has an *undefined* F1 — reported as NaN, not a
    misleading 0.0, so it aggregates to ``n_defined=0`` and renders as n/a (not zero skill)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels = list(range(len(classes))) if classes else None
    out: dict[str, float] = {}
    single_class = len(np.unique(y_true)) < 2
    for name in metric_names:
        if name == "accuracy":
            out[name] = float((y_true == y_pred).mean()) if len(y_true) else float("nan")
        elif name == "balanced_accuracy":
            out[name] = (
                float("nan") if single_class else float(balanced_accuracy_score(y_true, y_pred))
            )
        elif name == "macro_f1":
            out[name] = float(
                f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
            )
        else:
            raise KeyError(f"unknown metric {name!r}")
    if classes:
        per = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
        # Support per class in this eval set; a class with zero true samples has an undefined F1
        # (no instances to detect) — emit NaN so it is not mistaken for genuine zero performance.
        support = (
            np.bincount(y_true, minlength=len(classes))
            if len(y_true)
            else np.zeros(len(classes), dtype=int)
        )
        for i, (c, score) in enumerate(zip(classes, per, strict=True)):
            out[f"f1_{c}"] = float("nan") if support[i] == 0 else float(score)
    # See compute_binary_metrics: additive, so the frozen macro_f1 contract is untouched. Equals the
    # nanmean of the f1_<class> values just emitted.
    out["macro_f1_supported"] = macro_f1_supported(y_true, y_pred)
    return out
