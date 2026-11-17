"""CODE-02: single-class binary breakdowns warn, and no returned value changes.

Binary ``macro_f1`` averages over the classes in ``y_true`` ∪ ``y_pred`` (no pinned label set), so
on a breakdown group with one true class it depends on whether the absent class is ever predicted
(docs/known_issues.md §3). ``stratified_metrics(..., binary=True)`` — passed by ``evaluate`` for
binary tasks — emits one ``SingleClassBinaryBreakdownWarning`` listing those groups. The frozen
values are deliberately left as they are; these tests pin both the warning and the invariance.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from evemtbench.evaluation.evaluate import _CLASS_VOCAB, evaluate
from evemtbench.evaluation.metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    compute_regression_metrics,
)
from evemtbench.evaluation.stratify import (
    SingleClassBinaryBreakdownWarning,
    stratified_metrics,
)
from evemtbench.tasks.task import load_task
from evemtbench.tasks.taxonomy import LabelKind

# --- the minimal example (pins existing behaviour; values are frozen, NOT corrected here) -------


def test_minimal_single_class_example_values_are_unchanged():
    y_true = np.array([1, 1, 1, 1])
    right = compute_binary_metrics(y_true, np.array([1, 1, 1, 1]))
    wrong = compute_binary_metrics(y_true, np.array([1, 1, 1, 0]))
    assert right["macro_f1"] == 1.0
    # one wrong prediction adds the absent class 0 with F1 = 0: (6/7 + 0) / 2
    assert wrong["macro_f1"] == pytest.approx(3 / 7)
    assert wrong["macro_f1_supported"] == pytest.approx(6 / 7)
    assert wrong["accuracy"] == 0.75


# --- the warning on stratified_metrics directly --------------------------------------------------


def _md(groups):
    return pd.DataFrame({"event_family": groups})


def _binary_fn(yt, yp):
    return compute_binary_metrics(yt, yp)


@pytest.mark.parametrize(
    "y_pred", [[1, 1, 1, 1, 0, 1], [1, 1, 1, 0, 0, 1]], ids=["all_correct", "one_wrong"]
)
def test_single_class_group_warns_whether_or_not_predictions_are_right(y_pred):
    md = _md(["fault"] * 4 + ["mixed"] * 2)
    y_true = np.array([1, 1, 1, 1, 0, 1])
    with pytest.warns(SingleClassBinaryBreakdownWarning) as rec:
        stratified_metrics(md, y_true, np.array(y_pred), _binary_fn, binary=True)
    assert len(rec) == 1  # one aggregated warning per call
    msg = str(rec[0].message)
    assert "event_family=fault (n=4, only class 1)" in msg  # identifies the subgroup
    assert "mixed" not in msg  # the two-class group is not listed
    assert "macro_f1" in msg and "docs/known_issues.md section 3" in msg


def test_all_single_class_groups_are_listed_in_one_warning():
    md = pd.DataFrame({"event_family": ["normal", "normal", "fault", "fault"]})
    y_true = np.array([0, 0, 1, 1])
    with pytest.warns(SingleClassBinaryBreakdownWarning) as rec:
        stratified_metrics(md, y_true, np.array([0, 1, 1, 1]), _binary_fn, binary=True)
    assert len(rec) == 1
    msg = str(rec[0].message)
    assert "2 group(s)" in msg
    assert "event_family=normal (n=2, only class 0)" in msg
    assert "event_family=fault (n=2, only class 1)" in msg


def test_two_class_groups_do_not_warn():
    md = _md(["a", "a", "b", "b"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stratified_metrics(
            md, np.array([0, 1, 0, 1]), np.array([1, 1, 1, 1]), _binary_fn, binary=True
        )


def test_default_is_off_and_never_infers_binary_from_labels():
    """Generic callers (any metric_fn) see no warning unless they declare a binary task."""
    md = _md(["a", "a"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stratified_metrics(md, np.array([1, 1]), np.array([1, 0]), _binary_fn)


def test_warning_can_be_filtered_on_its_own():
    md = _md(["a", "a"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=SingleClassBinaryBreakdownWarning)
        stratified_metrics(md, np.array([1, 1]), np.array([1, 0]), _binary_fn, binary=True)


def test_returned_values_identical_with_and_without_the_flag():
    md = pd.DataFrame(
        {"event_family": ["normal"] * 3 + ["fault"] * 3, "y_fault_line": ["MainLn1-2"] * 6}
    )
    y_true = np.array([0, 0, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 1, 1])
    off = stratified_metrics(md, y_true, y_pred, _binary_fn)
    with pytest.warns(SingleClassBinaryBreakdownWarning):
        on = stratified_metrics(md, y_true, y_pred, _binary_fn, binary=True)
    assert json.dumps(on, sort_keys=True) == json.dumps(off, sort_keys=True)


# --- through evaluate() on the committed synthetic fixture ---------------------------------------

# every binary task, plus multiclass and regression controls
_BINARY = [
    "fault_detection_local",
    "event_detection_global",
    "switch_detection_global",
    "fault_grounded_local",
    "fault_grounded_line",
    "fault_grounded_global",
    "fault_origin_global",
]
_NON_BINARY = [
    "fault_classification_local",
    "fault_phase_local",
    "fault_category_global",
    "event_state_global",
    "event_classification_global",
    "fault_location_local",
]


def _metric_fn(task):
    td = task.definition
    if td.label_kind is LabelKind.BINARY:
        return lambda yt, yp: compute_binary_metrics(yt, yp, td.metrics)
    if td.label_kind is LabelKind.MULTICLASS:
        classes = _CLASS_VOCAB.get(td.function)
        return lambda yt, yp: compute_multiclass_metrics(yt, yp, td.metrics, classes=classes)
    return lambda yt, yp: compute_regression_metrics(yt, yp, td.metrics)


def _predictions(view, kind, flip_every):
    """True labels with every ``flip_every``-th one perturbed (``0``: all correct)."""
    y = np.asarray(view.labels)
    if kind is LabelKind.REGRESSION:
        y = y.astype(float).copy()
        if flip_every:
            y[::flip_every] += 1.0
    else:
        y = np.round(y).astype(int).copy()
        if flip_every:
            y[::flip_every] = 1 - y[::flip_every] if kind is LabelKind.BINARY else 0
    md = view.metadata
    return pd.DataFrame(
        {"sample_id": md["sample_id"], "window_idx": md["window_idx"], "prediction": y}
    ), y


def _eval_both_sets(task_id, fixture_root, held_out_splits, flip_every):
    task = load_task(
        task_id, "held_out", grid="double_line", root=fixture_root, splits_root=held_out_splits
    )
    kind = task.definition.label_kind
    for ts in ("test", "benchmark"):
        view = task.view(ts)
        if len(view) == 0:
            continue
        preds, y_pred = _predictions(view, kind, flip_every)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            res = evaluate(task, preds, test_set=ts)
        # "without the warning machinery": the pre-CODE-02 call, i.e. no binary flag
        y_true = np.asarray(view.labels)
        y_true = y_true.astype(float) if kind is LabelKind.REGRESSION else y_true.astype(int)
        baseline = stratified_metrics(view.metadata, y_true, y_pred, _metric_fn(task))
        yield task, ts, res, baseline, rec


@pytest.mark.parametrize("flip_every", [0, 3], ids=["all_correct", "partly_wrong"])
@pytest.mark.parametrize("task_id", _BINARY + _NON_BINARY)
def test_fixture_breakdowns_identical_with_and_without_warning(
    task_id, flip_every, fixture_root, held_out_splits
):
    ran = False
    for task, ts, res, baseline, rec in _eval_both_sets(
        task_id, fixture_root, held_out_splits, flip_every
    ):
        ran = True
        assert json.dumps(res.breakdowns, sort_keys=True) == json.dumps(baseline, sort_keys=True), (
            f"{task_id}/{ts}: breakdowns moved"
        )
        ours = [w for w in rec if issubclass(w.category, SingleClassBinaryBreakdownWarning)]
        if task.definition.label_kind is not LabelKind.BINARY:
            assert not ours, f"{task_id}/{ts}: non-binary task warned"
        else:
            assert len(ours) <= 1  # at most one aggregated warning per evaluation
            single = [
                f"{col}={g}"
                for col, groups in res.breakdowns.items()
                for g, row in groups.items()
                if np.isnan(row["balanced_accuracy"])  # NaN exactly when y_true is single-class
            ]
            if single:
                assert len(ours) == 1, f"{task_id}/{ts}: single-class groups {single} not warned"
                for name in single:
                    assert f"{name} (n=" in str(ours[0].message)
            else:
                assert not ours
    assert ran, f"{task_id}: no non-empty test view on the fixture"


def test_fixture_has_a_single_class_binary_breakdown(fixture_root, held_out_splits):
    """Guard that the invariance test above exercises the warning path at all."""
    seen = False
    for task_id in _BINARY:
        for *_, rec in _eval_both_sets(task_id, fixture_root, held_out_splits, 3):
            seen |= any(issubclass(w.category, SingleClassBinaryBreakdownWarning) for w in rec)
    assert seen
