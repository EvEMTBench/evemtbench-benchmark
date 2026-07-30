"""Coherence gates: tasks/protocols/baselines are mutually consistent (v0.1)."""

import pytest

from evemtbench.data.roots import GRIDS
from evemtbench.splits.protocols import PROTOCOLS
from evemtbench.tasks.registry import get_task, list_tasks
from evemtbench.tasks.taxonomy import Function, LabelKind, Observability

# Objectives that score switching/normal windows have no faulted-line oracle -> global-only.
_GLOBAL_ONLY = {
    Function.EVENT_STATE,
    Function.EVENT_DETECTION,
    Function.SWITCH_DETECTION,
    Function.SWITCH_TYPE,
    Function.EVENT_CLASSIFICATION,
    Function.FAULT_ORIGIN,  # fault-active but global-only (y_fault_line oracle would leak origin)
}
# Fault-active-only objectives -> all three observability variants (y_fault_line defined).
_THREE_OBS = {
    Function.FAULT_DETECTION,
    Function.FAULT_CLASS,
    Function.FAULT_LOCATION,
    Function.FAULT_GROUNDED,
    Function.FAULT_PHASE,
    Function.FAULT_CATEGORY,
}


def test_task_suite_shape():
    tids = list_tasks()
    # 6 three-obs objectives x {local,line,global} = 18, plus 6 global-only objectives = 24.
    assert len(tids) == 24
    functions = {get_task(t).function for t in tids}
    assert functions == _THREE_OBS | _GLOBAL_ONLY
    # global-only objectives appear exactly once, as *_global; three-obs ones appear 3x.
    for fn in _GLOBAL_ONLY:
        ids = [t for t in tids if get_task(t).function is fn]
        assert len(ids) == 1 and ids[0].endswith("_global")
        assert get_task(ids[0]).observability is Observability.GLOBAL
    for fn in _THREE_OBS:
        obs = {get_task(t).observability for t in tids if get_task(t).function is fn}
        assert obs == {Observability.LOCAL, Observability.LINE, Observability.GLOBAL}


def test_label_kinds_match_function():
    expected = {  # function -> (label_kind, num_classes)
        Function.FAULT_DETECTION: (LabelKind.BINARY, 2),
        Function.FAULT_CLASS: (LabelKind.MULTICLASS, 9),  # 7 SHC/incipient + 2 MV-only HIF
        Function.FAULT_LOCATION: (LabelKind.REGRESSION, None),
        Function.EVENT_STATE: (LabelKind.MULTICLASS, 3),
        Function.EVENT_DETECTION: (LabelKind.BINARY, 2),
        Function.SWITCH_DETECTION: (LabelKind.BINARY, 2),
        Function.FAULT_GROUNDED: (LabelKind.BINARY, 2),
        Function.FAULT_PHASE: (LabelKind.MULTICLASS, 7),
        Function.FAULT_CATEGORY: (LabelKind.MULTICLASS, 3),
        Function.SWITCH_TYPE: (LabelKind.MULTICLASS, 8),
        Function.EVENT_CLASSIFICATION: (LabelKind.MULTICLASS, 23),  # 9 fault + 13 switch + no_event
        Function.FAULT_ORIGIN: (LabelKind.BINARY, 2),  # line vs bus
    }
    for tid in list_tasks():
        td = get_task(tid)
        assert td.function in expected, f"{tid}: unexpected function {td.function}"
        kind, ncls = expected[td.function]
        assert td.label_kind is kind and td.num_classes == ncls, tid


def test_every_task_has_a_valid_tier():
    from evemtbench.tasks.registry import tasks_by_tier

    for tid in list_tasks():
        assert get_task(tid).tier in (1, 2, 3), f"{tid}: tier must be 1/2/3"
    grouped = tasks_by_tier()
    assert set(grouped) == {1, 2, 3}  # every task is tiered (no None bucket)
    assert sum(len(v) for v in grouped.values()) == len(list_tasks())
    # tiers partition the suite (disjoint, complete)
    flat = [t for ids in grouped.values() for t in ids]
    assert sorted(flat) == list_tasks()


def test_multiclass_tasks_have_frozen_class_vocab():
    # Every MULTICLASS task must have a _CLASS_VOCAB entry of matching length, else per-class F1 is
    # silently dropped and cross-grid comparability breaks (with an otherwise-green suite).
    from evemtbench.evaluation.evaluate import _CLASS_VOCAB

    for tid in list_tasks():
        td = get_task(tid)
        if td.label_kind is LabelKind.MULTICLASS:
            assert td.function in _CLASS_VOCAB, f"{tid}: MULTICLASS but no _CLASS_VOCAB entry"
            assert len(_CLASS_VOCAB[td.function]) == td.num_classes, tid


def test_every_task_uses_known_protocol_and_grid():
    for tid in list_tasks():
        td = get_task(tid)
        assert "held_out" in td.protocols  # every task is scored under held_out
        for proto in td.protocols:
            assert proto in PROTOCOLS
        # the pre-training/transfer protocols are portable-view only (fault-family local/line)
        if any(p in td.protocols for p in ("transfer_zeroshot", "transfer_finetune")):
            assert td.observability is not Observability.GLOBAL
            assert td.function in _THREE_OBS
        assert td.default_grid in GRIDS


@pytest.mark.parametrize("name", ["majority", "threshold", "random_forest"])
def test_baselines_constructible(name):
    from evemtbench.baselines.runner import _make_baseline

    model = _make_baseline(name, seed=0, device="cpu", epochs=1, batch_size=8, lr=1e-3)
    assert hasattr(model, "fit") and hasattr(model, "predict")


def test_unknown_baseline_rejected():
    from evemtbench.baselines.runner import _make_baseline

    with pytest.raises(ValueError, match="unknown baseline"):
        _make_baseline("nope", seed=0, device="cpu", epochs=1, batch_size=8, lr=1e-3)
