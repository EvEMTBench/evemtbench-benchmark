"""Frozen class-index order (index = label id) of every multiclass vocabulary, pinned as LITERALS.

docs/versioning.md lists the class order as frozen contract: reordering, inserting or renaming a
class changes what every stored prediction index means, so it requires a major version bump. The
other label tests decode through the very tuple they test and cannot catch a permutation; this one
compares against an independent copy. If you are deliberately changing a vocabulary, bump the
major version and update this file in the same change.
"""

import pytest

from evemtbench.tasks.labels.event_state import EVENT_STATE_CLASSES
from evemtbench.tasks.labels.event_type import EVENT_TYPE_CLASSES
from evemtbench.tasks.labels.fault_category import FAULT_CATEGORY_CLASSES
from evemtbench.tasks.labels.fault_class import FAULT_CLASS_CLASSES
from evemtbench.tasks.labels.fault_phase import PHASE_CLASSES
from evemtbench.tasks.labels.switch_type import SWITCH_TYPE_CLASSES

_FAULT_CLASS_V1 = (
    "flt_1phg_shc",
    "flt_1phg_shc_w_arc",
    "flt_2ph_shc",
    "flt_2phg_shc",
    "flt_3ph_shc",
    "flt_1phg_incipient",
    "flt_1phg_incipient_w_arc",
    "flt_1phg_hif",
    "flt_1phg_hif_w_arc",
)

FROZEN_V1 = {
    "fault_class": (FAULT_CLASS_CLASSES, _FAULT_CLASS_V1),
    "fault_phase": (PHASE_CLASSES, ("A", "B", "C", "AB", "AC", "BC", "ABC")),
    "fault_category": (FAULT_CATEGORY_CLASSES, ("shc", "incipient", "hif")),
    "event_state": (EVENT_STATE_CLASSES, ("normal", "switching", "fault")),
    "switch_type": (
        SWITCH_TYPE_CLASSES,
        ("load", "capacitor", "ohl", "inrush", "ibr", "cable", "motor_start", "generator"),
    ),
    "event_type": (
        EVENT_TYPE_CLASSES,
        _FAULT_CLASS_V1
        + (
            "switch_load_on",
            "switch_load_off",
            "switch_cap_on",
            "switch_cap_off",
            "switch_ohl_on",
            "switch_ohl_off",
            "switch_inrush_lv",
            "switch_inrush_hv",
            "switch_ibr_trip",
            "switch_cable_on",
            "switch_cable_off",
            "switch_motor_start",
            "switch_gen_trip",
            "no_event",
        ),
    ),
}


@pytest.mark.parametrize("name", sorted(FROZEN_V1))
def test_class_order_is_frozen(name):
    actual, frozen = FROZEN_V1[name]
    assert tuple(actual) == frozen, f"{name} class order changed: that is a major-version change"
