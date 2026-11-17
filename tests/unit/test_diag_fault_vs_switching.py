"""EXP-04 fault-versus-switching diagnostic: the frozen class mapping, the three conditional rates,
zero support -> undefined, the seed aggregate and the per-seed headline control, on tiny frames."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import diag_fault_vs_switching as D

N, S, F = D.NORMAL, D.SWITCHING, D.FAULT


def test_frozen_class_ids_and_mapping():
    assert (N, S, F) == (0, 1, 2)
    # only a predicted fault is a fault prediction; normal and switching are both non-fault
    assert D.is_fault_prediction(np.array([0, 1, 2])).tolist() == [False, False, True]
    assert D.MAPPING["non_fault_prediction"] == ["normal", "switching"]
    assert D.MAPPING["event_detection_used"] is False


def test_three_rates_counted_separately():
    true = np.array([S, S, S, S, F, F, F, N, N, N, N, N])
    pred = np.array([F, S, N, F, F, S, N, N, F, S, N, N])
    eps = np.array([1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 6])
    r = D.conditional_rates(true, pred, eps)
    # (a) switching windows predicted fault: 2 of 4
    assert r["switch_to_fault"] == {"support": 4, "support_episodes": 2, "count": 2, "rate": 0.5}
    # (b) fault windows predicted switching OR normal: 2 of 3 (switching counts as a miss)
    assert r["missed_fault"]["count"] == 2 and r["missed_fault"]["support"] == 3
    assert r["missed_fault"]["rate"] == pytest.approx(2 / 3)
    # (c) inactive windows predicted fault: 1 of 5; a predicted switching is not counted here
    assert r["inactive_to_fault"]["rate"] == pytest.approx(1 / 5)
    assert r["inactive_to_fault"]["support_episodes"] == 2


def test_zero_support_is_undefined_not_zero():
    r = D.conditional_rates(np.array([N, N]), np.array([N, N]))
    assert r["switch_to_fault"]["support"] == 0 and r["switch_to_fault"]["rate"] is None
    assert r["missed_fault"]["rate"] is None
    assert r["inactive_to_fault"]["rate"] == 0.0  # supported and genuinely zero


def _meta():
    return pd.DataFrame(
        {
            "sample_id": [1, 1, 2, 2, 3, 3],
            "window_idx": [0, 1, 0, 1, 0, 1],
            "event_type": ["sw_load", "sw_load", "sw_cap", "sw_cap", "flt_ag", "flt_ag"],
            "event_family": ["switching"] * 4 + ["fault"] * 2,
            "y_switch_category": ["load", "load", "capacitor", "capacitor", None, None],
        }
    )


def test_breakdowns_keep_vocabulary_groups_and_undefined_rates():
    meta = _meta()
    true = np.array([N, S, N, S, N, F])
    pred = np.array([N, F, F, S, N, S])
    rows = pd.DataFrame(D.rate_rows(meta, true, pred))
    by = rows.set_index(["breakdown", "group", "rate_name"])
    assert by.loc[("all", "all", "switch_to_fault"), "rate"] == 0.5
    assert by.loc[("y_switch_category", "load", "switch_to_fault"), "rate"] == 1.0
    assert by.loc[("y_switch_category", "capacitor", "switch_to_fault"), "rate"] == 0.0
    assert by.loc[("y_switch_category", "capacitor", "inactive_to_fault"), "rate"] == 1.0
    # a vocabulary category absent from the frame is listed with zero support and no rate
    cable = by.loc[("y_switch_category", "cable", "switch_to_fault")]
    assert cable["support"] == 0 and pd.isna(cable["rate"])
    # a switching event type has no fault windows: missed-fault rate undefined, not 0
    assert pd.isna(by.loc[("event_type", "sw_load", "missed_fault"), "rate"])
    assert by.loc[("event_type", "flt_ag", "missed_fault"), "rate"] == 1.0
    assert by.loc[("y_switch_category", D.NONE_GROUP, "missed_fault"), "support"] == 1
    # the fixed event_family vocabulary includes 'normal' even without windows
    assert by.loc[("event_family", "normal", "inactive_to_fault"), "support"] == 0


def test_confusion_counts_sum_to_windows():
    c = pd.DataFrame(D.confusion(np.array([N, S, F, F]), np.array([N, F, F, S])))
    assert c["count"].sum() == 4 and len(c) == 9
    assert c.set_index(["true_state", "pred_state"]).loc[("switching", "fault"), "count"] == 1


def test_seed_summary_spread_and_undefined():
    per_seed = pd.DataFrame(
        [
            {"g": "x", "seed": s, "support": 4, "support_episodes": 2, "count": k, "rate": k / 4}
            for s, k in enumerate([1, 2, 3])
        ]
        + [
            {"g": "y", "seed": s, "support": 0, "support_episodes": 0, "count": 0, "rate": None}
            for s in range(3)
        ]
    )
    out = D.summarise_seeds(per_seed, ["g"]).set_index("g")
    assert out.loc["x", "mean"] == pytest.approx(0.5)
    assert out.loc["x", "std"] == pytest.approx(0.25)  # ddof=1 over seeds, not windows
    assert (out.loc["x", "min"], out.loc["x", "max"]) == (0.25, 0.75)
    assert out.loc["y", "n_defined"] == 0 and math.isnan(out.loc["y", "mean"])


def test_seed_summary_rejects_support_drift():
    per_seed = pd.DataFrame(
        [
            {"g": "x", "support": 4, "support_episodes": 2, "count": 1, "rate": 0.25},
            {"g": "x", "support": 5, "support_episodes": 2, "count": 1, "rate": 0.2},
        ]
    )
    with pytest.raises(ValueError, match="support differs"):
        D.summarise_seeds(per_seed, ["g"])


def test_headline_control():
    assert D.headline_control({"balanced_accuracy": 0.7}, {"balanced_accuracy": 0.7}, 0.0) is None
    assert D.headline_control({"balanced_accuracy": 0.7}, {"balanced_accuracy": 0.71}, 0.0)
    assert D.headline_control({"balanced_accuracy": 0.7}, {}, 0.0)
    assert D.headline_control({"balanced_accuracy": float("nan")}, {"balanced_accuracy": 0.7}, 1)


def test_confusion_summary_row_shares_over_seeds():
    rows = []
    for seed, pred in enumerate([np.array([N, S, F, F]), np.array([N, S, F, N])]):
        for r in D.confusion(np.array([N, S, F, F]), pred):
            rows.append({"grid": "g", "baseline": "b", "test_set": "test", "seed": seed, **r})
    out = D.summarise_confusion(pd.DataFrame(rows)).set_index(["true_state", "pred_state"])
    assert out.loc[("fault", "fault"), "share_mean"] == pytest.approx(0.75)  # 1.0 and 0.5
    assert out.loc[("fault", "normal"), "share_mean"] == pytest.approx(0.25)
    assert out.loc[("fault", "fault"), "support"] == 2


def test_si_table_skips_a_set_without_scored_cells():
    # benchmark cells all failed their controls: only test rows reach the summary
    rows = [
        {
            "grid": "double_line",
            "test_set": "test",
            "baseline": b,
            "breakdown": "all",
            "group": "all",
            "rate_name": name,
            "support": 10,
            "n_defined": 5,
            "mean": 0.1,
            "std": 0.0,
        }
        for b in D.LEARNED_BASELINES
        for name in D.RATES
    ]
    tex = D.si_table(pd.DataFrame(rows), {("double_line", "test"): "tst"})
    assert D._SET_LABEL["test"] in tex and D._SET_LABEL["benchmark"] not in tex
    assert r"$^\dagger$~= baseline selected" in tex
