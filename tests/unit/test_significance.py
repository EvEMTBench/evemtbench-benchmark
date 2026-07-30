"""Pure unit tests for evaluation.significance (t-based CI + paired-by-seed test)."""

import json
import math

import numpy as np
import pytest

from evemtbench.evaluation.significance import (
    ci95_halfwidth,
    compare_result_files,
    paired_seed_test,
)


def test_ci_undefined_below_two_finite_values():
    assert ci95_halfwidth([]) is None
    assert ci95_halfwidth([0.9]) is None
    assert ci95_halfwidth([0.9, None, float("nan")]) is None  # only one finite value


def test_ci_known_t_quantile():
    # t_{0.975, df=4} = 2.776445; halfwidth = t * s / sqrt(n)
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    expected = 2.776445 * np.std(vals, ddof=1) / math.sqrt(5)
    assert ci95_halfwidth(vals) == pytest.approx(expected, rel=1e-5)


def test_ci_ignores_non_finite_entries():
    assert ci95_halfwidth([1.0, None, 2.0]) == pytest.approx(ci95_halfwidth([1.0, 2.0]))
    assert ci95_halfwidth([1.0, float("inf"), 2.0]) == pytest.approx(ci95_halfwidth([1.0, 2.0]))


def test_ci_zero_width_for_identical_seeds():
    # deterministic baselines (majority) legitimately have a zero-width interval — not None
    assert ci95_halfwidth([0.5, 0.5, 0.5]) == 0.0


def _ps(seed_to_val, metric="balanced_accuracy"):
    return [{"seed": s, "metrics": {metric: v}} for s, v in seed_to_val.items()]


def test_paired_by_seed_id_not_list_order():
    a = _ps({0: 0.9, 1: 0.8, 2: 0.7})
    b = _ps({0: 0.5, 1: 0.6, 2: 0.4})
    assert paired_seed_test(a, b, "balanced_accuracy") == paired_seed_test(
        a, list(reversed(b)), "balanced_accuracy"
    )


def test_pairs_intersect_seed_sets_and_drop_undefined():
    a = _ps({0: 0.9, 1: 0.8, 2: 0.7, 3: 0.6, 4: 0.5})
    b = _ps({0: 0.1, 1: None, 2: 0.3})  # seed 1 undefined on side b; seeds 3,4 missing
    r = paired_seed_test(a, b, "balanced_accuracy")
    assert r["n_pairs"] == 2 and r["seeds"] == [0, 2]


def test_single_pair_degrades_without_crashing():
    r = paired_seed_test(_ps({0: 0.9}), _ps({0: 0.5}), "balanced_accuracy")
    assert r["n_pairs"] == 1
    assert r["t_stat"] is None and r["p_value"] is None and r["significant"] is False
    assert r["better"] == "a"  # direction is still reported


def test_direction_orientation_uses_metric_direction():
    hi, lo = _ps({0: 0.9, 1: 0.91}), _ps({0: 0.5, 1: 0.52})
    assert paired_seed_test(hi, lo, "balanced_accuracy")["better"] == "a"
    small = _ps({0: 1.0, 1: 1.1}, metric="mae")
    big = _ps({0: 5.0, 1: 5.3}, metric="mae")
    assert paired_seed_test(small, big, "mae")["better"] == "a"  # lower mae wins
    assert paired_seed_test(big, small, "mae")["better"] == "b"


def test_identical_sides_not_significant():
    a = _ps({0: 0.7, 1: 0.8, 2: 0.9})
    r = paired_seed_test(a, a, "balanced_accuracy")
    assert r["mean_diff"] == 0.0 and r["better"] is None and r["significant"] is False


def test_constant_shift_is_degenerate_no_pvalue():
    # zero variance in the differences: the paired t is undefined — direction only, no p-value
    r = paired_seed_test(_ps({0: 0.9, 1: 0.8}), _ps({0: 0.5, 1: 0.4}), "balanced_accuracy")
    assert r["p_value"] is None and r["significant"] is False and r["better"] == "a"


def test_clear_separation_is_significant():
    a = _ps({s: 0.90 + 0.001 * s for s in range(5)})
    b = _ps({s: 0.50 + 0.003 * s for s in range(5)})
    r = paired_seed_test(a, b, "balanced_accuracy")
    assert r["significant"] is True and r["p_value"] < 0.001 and r["better"] == "a"


def test_report_is_json_safe():
    r = paired_seed_test(
        _ps({0: 0.9, 1: 0.8, 2: 0.85}), _ps({0: 0.5, 1: 0.55, 2: 0.45}), "balanced_accuracy"
    )
    json.dumps(r, allow_nan=False)  # must not raise (no NaN/inf leaks)


def _result_doc(baseline, seed_to_val, **overrides):
    doc = {
        "task": "fault_detection_local",
        "protocol": "held_out",
        "grid": "double_line",
        "test_set": "test",
        "benchmark_version": "0.2.0-dev",
        "baseline": baseline,
        "per_seed": [
            {"seed": s, "metrics": {"balanced_accuracy": v}} for s, v in seed_to_val.items()
        ],
    }
    doc.update(overrides)
    return doc


def test_compare_result_files_defaults_to_registry_headline(tmp_path):
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(_result_doc("mlp", {0: 0.90, 1: 0.92, 2: 0.91})))
    pb.write_text(json.dumps(_result_doc("majority", {0: 0.50, 1: 0.52, 2: 0.49})))
    rep = compare_result_files(pa, pb)
    assert rep["ok"] and not rep["issues"]
    assert rep["test"]["metric"] == "balanced_accuracy"  # fault_detection_local headline
    assert rep["test"]["better"] == "a" and rep["test"]["n_pairs"] == 3


def test_compare_result_files_flags_contract_mismatch(tmp_path):
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(_result_doc("mlp", {0: 0.9})))
    pb.write_text(json.dumps(_result_doc("mlp", {0: 0.8}, grid="cigre_mv")))
    rep = compare_result_files(pa, pb)
    assert rep["ok"] is False and any("grid differs" in i for i in rep["issues"])


def test_bootstrap_ci_deterministic_and_sane():
    from evemtbench.evaluation.significance import bootstrap_ci

    vals = [0.80, 0.82, 0.84, 0.86, 0.88]
    a = bootstrap_ci(vals)
    b = bootstrap_ci(vals)
    assert a == b, "seeded bootstrap must be deterministic"
    lo, hi = a
    assert lo <= sum(vals) / len(vals) <= hi
    assert bootstrap_ci([0.5]) is None  # n<2 -> None, like ci95_halfwidth
    assert bootstrap_ci([0.5, float("nan"), 0.7]) is not None  # non-finite dropped
