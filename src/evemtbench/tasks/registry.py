"""Registry of official benchmark tasks (24 total; DoubleLine default grid).

The suite spans a difficulty ladder (``tier`` on each :class:`TaskDefinition`; see
``docs/task_suite_roadmap.md``), all under the single ``held_out`` protocol:

* **Fault triad** — **FD** detection (binary), **FC** classification (``flt_*`` multiclass),
  **FL** location (regression) — each × {**local**, **line**, **global**} observability
  (``tasks/line_view.py``).
* **Fault-attribute** (fault-active windows → all three observability settings): **grounded**
  (binary), **phase** (7-class A/B/C combo), **category** (shc/incipient/hif).
* **Event-scope** (scored on switching/normal windows too → ``global``-only, since those windows
  have no faulted line for the local/line oracle keyed on ``y_fault_line``): **ES** event-state
  (3-class), **ED** event-detection, **SD** switch-detection (binary), **ST** switch-type,
  **EC** event-classification (23-class), and **fault-origin** (line-vs-bus, ``global``-only to
  avoid oracle leakage).

Adding a task is a *minor* bump; changing a frozen label vocabulary or split is *major*.
"""

from __future__ import annotations

from ..splits.protocols import PRETRAIN_TRANSFER_PROTOCOLS
from .labels.event_state import EVENT_STATE_CLASSES
from .labels.event_type import EVENT_TYPE_CLASSES
from .labels.fault_category import FAULT_CATEGORY_CLASSES
from .labels.fault_class import FAULT_CLASS_CLASSES
from .labels.fault_phase import PHASE_CLASSES
from .labels.switch_type import SWITCH_TYPE_CLASSES
from .taxonomy import Function, LabelKind, Observability, TaskDefinition

_PROTOCOLS = ("held_out",)
# The portable LOCAL/LINE views (fixed 6/12 ch) additionally support the pre-training/transfer
# protocols; GLOBAL is grid-specific width and has no multi_grid (double_line) counterpart.
_PORTABLE_PROTOCOLS = (*_PROTOCOLS, *PRETRAIN_TRANSFER_PROTOCOLS)
_GRID = "double_line"

# Metric sets per objective.
_FD_METRICS = ("balanced_accuracy", "macro_f1", "missed_fault_rate", "false_alarm_rate", "accuracy")
_FC_METRICS = ("balanced_accuracy", "macro_f1", "accuracy")
_ES_METRICS = ("balanced_accuracy", "macro_f1", "accuracy")  # multiclass, like FC
_FL_METRICS = ("mae", "rmse", "r2", "median_abs_error")
# Event-scope detection: protection-framed binary rates (dependability / security), event-named.
_ED_METRICS = (
    "balanced_accuracy",
    "macro_f1",
    "missed_event_rate",
    "false_event_trip_rate",
    "accuracy",
)
_BIN_METRICS = ("balanced_accuracy", "macro_f1", "accuracy")  # plain binary attribute (grounded)

# Difficulty tier per objective (1=easy anchor near-solved on ideal data .. 3=hard frontier).
# All observability variants of an objective share its tier. See docs/task_suite_roadmap.md.
_TIER: dict[Function, int] = {
    Function.FAULT_DETECTION: 1,
    Function.EVENT_DETECTION: 1,
    Function.SWITCH_DETECTION: 1,
    Function.FAULT_GROUNDED: 1,
    Function.EVENT_STATE: 2,
    Function.FAULT_CLASS: 2,
    Function.FAULT_PHASE: 2,
    Function.FAULT_CATEGORY: 2,
    Function.SWITCH_TYPE: 2,
    Function.FAULT_ORIGIN: 2,
    Function.EVENT_CLASSIFICATION: 3,
    Function.FAULT_LOCATION: 3,
}

_OBS = {
    Observability.LOCAL: ("local", "near terminal of the faulted line (1 cubicle, 6 ch)"),
    Observability.LINE: ("line", "both terminals of the faulted line (2 cubicles, 12 ch)"),
    Observability.GLOBAL: ("global", "all 4 main lines (8 cubicles, 48 ch)"),
}


def _make(function: Function, label_kind: LabelKind, metrics, num_classes, prefix, blurb):
    """Build the three observability variants of one objective.

    LOCAL/LINE are the portable fixed-width views, so they also carry the pre-training/transfer
    protocols; GLOBAL keeps ``held_out`` only (grid-specific width, no double_line counterpart)."""
    out = {}
    for obs, (suffix, obs_desc) in _OBS.items():
        tid = f"{prefix}_{suffix}"
        protocols = (
            _PORTABLE_PROTOCOLS if obs in (Observability.LOCAL, Observability.LINE) else _PROTOCOLS
        )
        out[tid] = TaskDefinition(
            task_id=tid,
            function=function,
            observability=obs,
            label_kind=label_kind,
            description=f"{blurb} Observability: {obs_desc}.",
            metrics=metrics,
            protocols=protocols,
            default_grid=_GRID,
            num_classes=num_classes,
            tier=_TIER.get(function),
        )
    return out


def _make_global(function: Function, label_kind: LabelKind, metrics, num_classes, task_id, blurb):
    """Build a single global-only task (for objectives scored on switching/normal windows, which
    have no faulted line for the local/line oracle)."""
    return {
        task_id: TaskDefinition(
            task_id=task_id,
            function=function,
            observability=Observability.GLOBAL,
            label_kind=label_kind,
            description=f"{blurb} Observability: {_OBS[Observability.GLOBAL][1]}.",
            metrics=metrics,
            protocols=_PROTOCOLS,
            default_grid=_GRID,
            num_classes=num_classes,
            tier=_TIER.get(function),
        )
    }


BENCHMARK_TASKS: dict[str, TaskDefinition] = {
    **_make(
        Function.FAULT_DETECTION,
        LabelKind.BINARY,
        _FD_METRICS,
        2,
        "fault_detection",
        "Window-level fault detection: fault-active window vs. a pre-fault window of the same "
        "fault episode.",
    ),
    **_make(
        Function.FAULT_CLASS,
        LabelKind.MULTICLASS,
        _FC_METRICS,
        len(FAULT_CLASS_CLASSES),
        "fault_classification",
        "Fault classification over the full flt_* event-type taxonomy "
        f"({len(FAULT_CLASS_CLASSES)} classes), on fault-active windows.",
    ),
    **_make(
        Function.FAULT_LOCATION,
        LabelKind.REGRESSION,
        _FL_METRICS,
        None,
        "fault_location",
        "Fault-location regression: distance along the faulted line (%), on fault-active windows "
        "of line-fault episodes.",
    ),
    # Event-state (ES): global-only (no faulted-line oracle for switching/normal windows).
    "event_state_global": TaskDefinition(
        task_id="event_state_global",
        function=Function.EVENT_STATE,
        observability=Observability.GLOBAL,
        label_kind=LabelKind.MULTICLASS,
        description=(
            f"High-level event-state classification over every window ({len(EVENT_STATE_CLASSES)} "
            "classes: normal / switching / fault). "
            f"Observability: {_OBS[Observability.GLOBAL][1]}."
        ),
        metrics=_ES_METRICS,
        protocols=_PROTOCOLS,
        default_grid=_GRID,
        num_classes=len(EVENT_STATE_CLASSES),
        tier=_TIER[Function.EVENT_STATE],
    ),
    # --- Event-scope detection/classification: global-only (switching/normal lack a line) ---
    **_make_global(
        Function.EVENT_DETECTION,
        LabelKind.BINARY,
        _ED_METRICS,
        2,
        "event_detection_global",
        "Event detection: any active grid event (fault or switching) vs. quiescent, over every "
        "window.",
    ),
    **_make_global(
        Function.SWITCH_DETECTION,
        LabelKind.BINARY,
        _ED_METRICS,
        2,
        "switch_detection_global",
        "Switching-event detection: active switching event vs. everything else, over every window.",
    ),
    **_make_global(
        Function.SWITCH_TYPE,
        LabelKind.MULTICLASS,
        _FC_METRICS,
        len(SWITCH_TYPE_CLASSES),
        "switch_type_global",
        f"Switching-event type classification ({len(SWITCH_TYPE_CLASSES)} categories), on "
        "switching-active windows.",
    ),
    **_make_global(
        Function.EVENT_CLASSIFICATION,
        LabelKind.MULTICLASS,
        _FC_METRICS,
        len(EVENT_TYPE_CLASSES),
        "event_classification_global",
        f"Full event-type classification ({len(EVENT_TYPE_CLASSES)} classes: faults + switching + "
        "no_event) over every window.",
    ),
    # --- Fault-attribute tasks: fault-active windows -> all three observability settings apply ---
    **_make(
        Function.FAULT_GROUNDED,
        LabelKind.BINARY,
        _BIN_METRICS,
        2,
        "fault_grounded",
        "Grounded-vs-ungrounded fault classification, on fault-active windows.",
    ),
    **_make(
        Function.FAULT_PHASE,
        LabelKind.MULTICLASS,
        _FC_METRICS,
        len(PHASE_CLASSES),
        "fault_phase",
        f"Faulted-phase identification ({len(PHASE_CLASSES)}-class A/B/C combination), on "
        "fault-active windows.",
    ),
    **_make(
        Function.FAULT_CATEGORY,
        LabelKind.MULTICLASS,
        _FC_METRICS,
        len(FAULT_CATEGORY_CLASSES),
        "fault_category",
        f"Fault-category classification ({len(FAULT_CATEGORY_CLASSES)} classes: shc / incipient / "
        "hif), on fault-active windows.",
    ),
    # Fault origin (line vs bus): global-only — y_fault_line is the oracle (local/line would leak).
    **_make_global(
        Function.FAULT_ORIGIN,
        LabelKind.BINARY,
        _BIN_METRICS,
        2,
        "fault_origin_global",
        "Fault-origin classification: line fault vs bus fault, on fault-active windows.",
    ),
}


def get_task(task_id: str) -> TaskDefinition:
    if task_id not in BENCHMARK_TASKS:
        raise KeyError(f"Unknown task {task_id!r}; known: {sorted(BENCHMARK_TASKS)}")
    return BENCHMARK_TASKS[task_id]


def list_tasks() -> list[str]:
    return sorted(BENCHMARK_TASKS)


def tasks_by_tier() -> dict[int | None, list[str]]:
    """Task ids grouped by difficulty tier (sorted within each tier); untiered under ``None``.

    Drives tier-grouped rendering in RESULTS.md / the leaderboard and the roadmap narrative.
    """
    out: dict[int | None, list[str]] = {}
    for tid in list_tasks():
        out.setdefault(BENCHMARK_TASKS[tid].tier, []).append(tid)
    return {
        tier: sorted(ids)
        for tier, ids in sorted(out.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))
    }
