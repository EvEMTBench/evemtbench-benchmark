"""Validation-selection reliability diagnostic (evemtbench.evaluation.selection)."""

import pytest

from evemtbench.evaluation import selection as SEL

POOL = ("random_forest", "mlp", "gru", "cnn", "resnet")


def test_kendall_tau_b_known_values():
    assert SEL.kendall_tau_b([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert SEL.kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # one swapped pair out of six: (5 - 1) / 6
    assert SEL.kendall_tau_b([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(4 / 6)
    assert SEL.kendall_tau_b([1, 1, 1], [1, 2, 3]) is None  # constant vector
    assert SEL.kendall_tau_b([1], [1]) is None


def test_argbest_strict_tie_keeps_pool_order():
    assert SEL.argbest({"mlp": 0.9, "gru": 0.9}, POOL, +1) == "mlp"
    assert SEL.argbest({"mlp": 2.0, "gru": 1.0}, POOL, -1) == "gru"  # lower-is-better
    assert SEL.argbest({}, POOL, +1) is None


def test_cell_diagnostic_grades_the_val_pick_without_changing_it():
    scores = {
        "val": {"mlp": 0.95, "gru": 0.90, "cnn": 0.80},
        "test": {"mlp": 0.85, "gru": 0.91, "cnn": 0.70},  # val winner is not best on test
        "benchmark": {"mlp": 0.60, "gru": 0.55, "cnn": 0.50},
    }
    d = SEL.cell_diagnostic(scores, "balanced_accuracy", POOL)
    assert d["selected"] == "mlp"  # chosen on val only
    t = d["sets"]["test"]
    assert t["best"] == "gru" and not t["same_winner"]
    assert t["regret"] == pytest.approx(0.06)
    b = d["sets"]["benchmark"]
    assert b["same_winner"] and b["regret"] == pytest.approx(0.0)
    assert d["tau_test_benchmark"] == pytest.approx(1 / 3)


def test_regret_is_non_negative_for_lower_is_better_metrics():
    scores = {"val": {"mlp": 1.0, "gru": 2.0}, "test": {"mlp": 3.0, "gru": 2.5}}
    d = SEL.cell_diagnostic(scores, "mae", POOL)
    assert d["selected"] == "mlp" and d["sets"]["test"]["regret"] == pytest.approx(0.5)


def test_no_validation_signal_means_no_diagnostic():
    assert SEL.cell_diagnostic({"test": {"mlp": 0.9}}, "balanced_accuracy", POOL) is None


def test_summarize_never_pools_regret_across_metrics():
    a = SEL.cell_diagnostic(
        {"val": {"mlp": 0.9, "gru": 0.8}, "test": {"mlp": 0.7, "gru": 0.8}},
        "balanced_accuracy",
        POOL,
    )
    b = SEL.cell_diagnostic({"val": {"mlp": 1.0}, "test": {"mlp": 2.0}}, "mae", POOL)
    s = SEL.summarize([a, b])["test"]
    assert s["n_cells"] == 2 and s["n_same_winner"] == 1
    assert set(s["median_regret"]) == {"balanced_accuracy", "mae"}
    assert s["max_regret"]["balanced_accuracy"] == pytest.approx(0.1)
