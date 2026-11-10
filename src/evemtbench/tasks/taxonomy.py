"""Task taxonomy — protection objective × input setting, plus label/metric metadata.

A *task* is a protection objective (Benchmark Plan §4) bound to an input setting (§2.4) and a
deterministic label rule. The protocol (``held_out`` — v0.1's single protocol) is orthogonal
and supplied at ``load_task`` time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# Faulted-element label grammar of ``y_fault_line``: ``MainLn<lineid>`` / ``MainBus<bus>``.
LINE_LABEL_PREFIX = "MainLn"


class Function(str, Enum):
    """Protection / analysis objective. Fault triad FD/FC/FL + the event-scope & attribute suite.

    Difficulty spans easy anchors (detection) to a hard frontier (fine-grained EC, precise FL);
    see ``docs/task_suite_roadmap.md``.
    """

    FAULT_DETECTION = "fault_detection"  # FD: fault active vs. pre-fault (binary)
    FAULT_CLASS = "fault_class"  # FC: full flt_* event-type taxonomy (multiclass)
    FAULT_LOCATION = "fault_location"  # FL: distance-along-line % (regression, line faults)
    EVENT_STATE = "event_state"  # ES: high-level grid state normal/switching/fault (multiclass)
    # Event-scope detection/classification (global-only: switching/normal windows have no line)
    EVENT_DETECTION = "event_detection"  # ED: any active event vs. quiescent (binary)
    SWITCH_DETECTION = "switch_detection"  # SD: active switching event vs. rest (binary)
    EVENT_CLASSIFICATION = "event_classification"  # EC: full event_type incl. no_event (multiclass)
    SWITCH_TYPE = "switch_type"  # ST: switching-event category (multiclass, switching windows)
    FAULT_ORIGIN = "fault_origin"  # line-fault vs bus-fault (binary); global-only (oracle leakage)
    # Fault-attribute tasks (fault-active windows -> all three observability settings apply)
    FAULT_GROUNDED = "fault_grounded"  # grounded vs. ungrounded fault (binary)
    FAULT_PHASE = "fault_phase"  # faulted-phase combination A/B/C -> 7-class (multiclass)
    FAULT_CATEGORY = "fault_category"  # shc / incipient / hif (multiclass)


class Observability(str, Enum):
    """Measurement input setting, centred on the faulted element (oracle from y_fault_line).

    A nested hierarchy within the 48-channel main-line universe (see ``tasks/line_view.py``):
    LOCAL ⊂ LINE ⊂ GLOBAL. Fairness-critical — never compare across settings blindly.
    """

    LOCAL = "local"  # near terminal of the faulted line (1 cubicle, 6 ch)
    LINE = "line"  # both terminals of the faulted line (2 cubicles, 12 ch)
    GLOBAL = "global"  # all 4 main lines (8 cubicles, 48 ch)


class LabelKind(str, Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


@dataclass(frozen=True)
class TaskDefinition:
    """Complete specification of one benchmark task."""

    task_id: str
    function: Function
    observability: Observability
    label_kind: LabelKind
    description: str
    metrics: tuple[str, ...]
    protocols: tuple[str, ...]
    default_grid: str
    num_classes: int | None = None  # None for REGRESSION
    # Difficulty tier for reporting/roadmap grouping (1=easy anchor .. 3=hard frontier); None =
    # untiered. Machine-readable so RESULTS/leaderboard group by it. See docs/task_suite_roadmap.md.
    tier: int | None = None
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"Task({self.task_id}: {self.function.value}/{self.observability.value})"
