"""Evaluator self-test — prove the scoring harness itself is sound, with zero data.

Adapted from UnderstandingVCS's ``benchmark/run.py --self-test`` idiom: before trusting any
score, verify that the *harness* scores known inputs correctly. Every check routes through the
REAL :func:`evemtbench.evaluation.evaluate` — the same choke point the sweep runner uses — via a
synthetic in-memory :class:`Task`, so a third party can audit the scoring path with a pip
install only (no dataset, no cluster):

* a **perfect-predictions control** must hit the metric ceiling (balanced accuracy / F1 = 1.0,
  MAE = 0.0) — if the oracle fed to the harness does not score perfectly, the harness is broken,
  not the model;
* a **majority/constant control** must land exactly on the analytic ``1/k`` chance floor — the
  same closed-form anchor the coverage floor gate checks over the real matrix;
* **malformed predictions must raise** (duplicate keys, missing rows, missing columns) — the
  alignment contract that keeps third-party submissions honest;
* every declared metric must have a **direction** (higher/lower-better) in the frozen contract.

    evemtbench-evaluate --self-test          # PASS/FAIL per check; exit 0/1
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..tasks.registry import get_task, list_tasks
from ..tasks.task import Task, TaskView
from .evaluate import evaluate
from .metrics import METRIC_DIRECTION
from .predictions_io import predictions_frame

_CEILING_TOL = 1e-9


def _synthetic_task(task_id: str, labels: np.ndarray) -> Task:
    """A minimal in-memory Task whose test view carries ``labels`` — enough for evaluate()."""
    n = len(labels)
    meta = pd.DataFrame(
        {"sample_id": [f"ep{i // 4}" for i in range(n)], "window_idx": [i % 4 for i in range(n)]}
    )
    view = TaskView("test", [], labels, meta)
    empty = TaskView("empty", [], np.empty(0), pd.DataFrame({"sample_id": [], "window_idx": []}))
    return Task(
        definition=get_task(task_id),
        protocol="held_out",
        grid="selftest",
        version="selftest",
        train=empty,
        val=empty,
        test=view,
        benchmark=empty,
    )


def _preds(task: Task, values: np.ndarray) -> pd.DataFrame:
    meta = task.test.metadata
    return predictions_frame(meta["sample_id"], meta["window_idx"], values)


def _score(task: Task, values: np.ndarray) -> dict[str, float]:
    return evaluate(task, _preds(task, values)).metrics


Check = tuple[str, bool, str]


def _check_ceiling(task_id: str, labels: np.ndarray, headline: str, ceiling: float) -> Check:
    task = _synthetic_task(task_id, labels)
    val = _score(task, labels)[headline]  # the oracle fed to the harness itself
    ok = abs(val - ceiling) <= _CEILING_TOL
    return (
        f"{task_id}: perfect predictions -> {headline} ceiling",
        ok,
        f"{headline}={val!r}, expected {ceiling}",
    )


def _check_floor(task_id: str, labels: np.ndarray, constant, floor: float) -> Check:
    task = _synthetic_task(task_id, labels)
    val = _score(task, np.full(len(labels), constant))["balanced_accuracy"]
    ok = abs(val - floor) <= _CEILING_TOL
    return (
        f"{task_id}: constant prediction -> analytic 1/k floor",
        ok,
        f"balanced_accuracy={val!r}, expected {floor}",
    )


def _check_raises(name: str, task: Task, frame: pd.DataFrame) -> Check:
    try:
        evaluate(task, frame)
    except (ValueError, KeyError) as e:
        return (name, True, f"raised as required ({type(e).__name__})")
    return (name, False, "malformed predictions were scored silently")


def run_selftest() -> tuple[bool, list[Check]]:
    """All controls; (overall_ok, [(check id, ok, detail)])."""
    checks: list[Check] = []

    # binary: fault_detection_local, balanced two-class labels
    y_bin = np.array([0] * 8 + [1] * 8)
    checks.append(_check_ceiling("fault_detection_local", y_bin, "balanced_accuracy", 1.0))
    checks.append(_check_floor("fault_detection_local", y_bin, 0, 1.0 / 2))

    # multiclass: fault_classification_local over the full frozen 9-class vocabulary
    y_mc = np.tile(np.arange(9), 4)
    checks.append(_check_ceiling("fault_classification_local", y_mc, "balanced_accuracy", 1.0))
    checks.append(_check_ceiling("fault_classification_local", y_mc, "macro_f1", 1.0))
    checks.append(_check_floor("fault_classification_local", y_mc, 0, 1.0 / 9))

    # regression: fault_location_local — perfect MAE/RMSE = 0; a constant control must NOT be 0
    y_reg = np.linspace(0.5, 24.5, 16)
    checks.append(_check_ceiling("fault_location_local", y_reg, "mae", 0.0))
    checks.append(_check_ceiling("fault_location_local", y_reg, "rmse", 0.0))
    task = _synthetic_task("fault_location_local", y_reg)
    mae = _score(task, np.full(len(y_reg), float(y_reg.mean())))["mae"]
    checks.append(
        (
            "fault_location_local: constant prediction -> nonzero MAE",
            mae > 0,
            f"mae={mae!r}, must be > 0",
        )
    )

    # alignment contract: malformed frames must raise, never score
    task = _synthetic_task("fault_detection_local", y_bin)
    good = _preds(task, y_bin)
    checks.append(
        _check_raises(
            "alignment: duplicate (sample_id, window_idx) raises",
            task,
            pd.concat([good, good.iloc[[0]]], ignore_index=True),
        )
    )
    checks.append(_check_raises("alignment: missing prediction rows raise", task, good.iloc[:-1]))
    checks.append(
        _check_raises(
            "alignment: missing 'prediction' column raises",
            task,
            good.drop(columns=["prediction"]),
        )
    )

    # frozen contract completeness: every declared metric has a direction
    missing = sorted(
        {
            m
            for t in list_tasks()
            for m in get_task(t).metrics
            if not m.startswith("f1_") and m not in METRIC_DIRECTION
        }
    )
    checks.append(
        (
            "contract: every declared metric has a direction",
            not missing,
            f"missing from METRIC_DIRECTION: {missing}" if missing else "all present",
        )
    )

    return all(ok for _, ok, _ in checks), checks


def main() -> int:
    ok, checks = run_selftest()
    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {name}" + ("" if passed else f"  [{detail}]"))
    n_pass = sum(1 for _, p, _ in checks if p)
    print(
        f"evaluator self-test: {'OK' if ok else 'FAILED'} — {n_pass}/{len(checks)} checks "
        "(scored through the real evaluate() path; no data required)"
    )
    return 0 if ok else 1
