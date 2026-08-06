"""The shared result-doc authority: seed aggregation, breakdown aggregation, exact rendering.

Both writers of ``result.json`` (the training runner and the offline re-evaluation pass) go through
this module, so these tests pin the contract they share.
"""

from __future__ import annotations

import json
import math

import pytest

from evemtbench.evaluation.aggregate import (
    BreakdownMismatch,
    aggregate_seed_breakdowns,
    aggregate_seed_metrics,
    build_result_doc,
    render_result_json,
)

_AGG_KEYS = {"mean", "std", "ci95", "n_defined", "n_seeds"}


def _bd(acc, n=10, extra=None):
    """One seed's breakdowns in the pre-aggregation (flat, per-seed) shape evaluate() emits."""
    d = {"accuracy": acc, "n": n}
    if extra:
        d.update(extra)
    return {"event_type": {"flt_1phg_shc": d}}


def test_aggregate_seed_metrics_shape_and_values():
    agg = aggregate_seed_metrics([{"a": 1.0}, {"a": 2.0}, {"a": 3.0}])
    assert set(agg["a"]) == _AGG_KEYS
    assert agg["a"]["mean"] == 2.0
    assert agg["a"]["std"] == pytest.approx(math.sqrt(2 / 3))  # population (ddof=0), as frozen
    assert agg["a"]["n_defined"] == 3 and agg["a"]["n_seeds"] == 3


def test_undefined_metric_is_null_not_zero():
    """The distinction the whole reporting chain depends on: undefined != 0.0."""
    agg = aggregate_seed_metrics([{"a": math.nan}, {"a": math.nan}])
    assert agg["a"]["mean"] is None and agg["a"]["n_defined"] == 0 and agg["a"]["n_seeds"] == 2
    # partially defined: averages over the defined seeds only
    agg = aggregate_seed_metrics([{"a": math.nan}, {"a": 4.0}])
    assert agg["a"]["mean"] == 4.0 and agg["a"]["n_defined"] == 1 and agg["a"]["ci95"] is None


def test_breakdowns_are_aggregated_over_all_seeds():
    """The defect this module exists to fix: breakdowns used to be the LAST seed's values."""
    agg = aggregate_seed_breakdowns([_bd(0.0), _bd(1.0), _bd(0.5)])
    g = agg["event_type"]["flt_1phg_shc"]
    assert g["metrics"]["accuracy"]["mean"] == pytest.approx(0.5)  # not 0.5-by-luck: mean of all 3
    assert g["metrics"]["accuracy"]["n_seeds"] == 3
    assert g["n_seeds"] == 3
    # last-seed semantics would have given 0.5 here too, so check a case where they differ
    agg = aggregate_seed_breakdowns([_bd(0.0), _bd(0.0), _bd(0.9)])
    assert agg["event_type"]["flt_1phg_shc"]["metrics"]["accuracy"]["mean"] == pytest.approx(0.3)


def test_breakdown_metric_entries_match_aggregate_shape():
    """A stratum metric and a headline metric must read identically."""
    agg = aggregate_seed_breakdowns([_bd(0.2), _bd(0.4)])
    entry = agg["event_type"]["flt_1phg_shc"]["metrics"]["accuracy"]
    assert set(entry) == _AGG_KEYS
    assert entry == aggregate_seed_metrics([{"accuracy": 0.2}, {"accuracy": 0.4}])["accuracy"]


def test_group_n_stays_at_group_level():
    """`n` must not move: leaderboard.py reads the group KEY set and paper_lib reads `n` — both
    would break if the count were nested under `metrics`."""
    agg = aggregate_seed_breakdowns([_bd(0.2, n=77), _bd(0.4, n=77)])
    g = agg["event_type"]["flt_1phg_shc"]
    assert g["n"] == 77
    assert "n" not in g["metrics"]


def test_undefined_stratum_metric_aggregates_to_null():
    agg = aggregate_seed_breakdowns([_bd(math.nan), _bd(math.nan)])
    entry = agg["event_type"]["flt_1phg_shc"]["metrics"]["accuracy"]
    assert entry["mean"] is None and entry["n_defined"] == 0


def test_strict_rejects_divergent_groups_and_n():
    """Strata/groups come from the eval view's metadata, identical for every seed of a cell — so a
    disagreement is a harness bug (seeds scored against different views), not data to average."""
    a, b = _bd(0.1), _bd(0.1)
    b["event_type"] = {"other_class": b["event_type"]["flt_1phg_shc"]}
    with pytest.raises(BreakdownMismatch, match="differ from seed 0"):
        aggregate_seed_breakdowns([a, b])

    with pytest.raises(BreakdownMismatch, match="n differs across seeds"):
        aggregate_seed_breakdowns([_bd(0.1, n=10), _bd(0.1, n=11)])

    c = {"grid": {"g": {"accuracy": 0.1, "n": 3}}}
    with pytest.raises(BreakdownMismatch, match="reports strata"):
        aggregate_seed_breakdowns([a, c])


def test_permissive_mode_unions_without_zero_filling():
    """With strict=False an absent seed is MISSING, never a zero."""
    a, b = _bd(0.4), _bd(0.4)
    b["event_type"]["only_in_b"] = {"accuracy": 1.0, "n": 5}
    agg = aggregate_seed_breakdowns([a, b], strict=False)
    groups = agg["event_type"]
    assert set(groups) == {"flt_1phg_shc", "only_in_b"}
    assert groups["only_in_b"]["n_seeds"] == 1
    assert groups["only_in_b"]["metrics"]["accuracy"]["mean"] == 1.0  # not 0.5


def test_empty_breakdowns_and_no_seeds():
    assert aggregate_seed_breakdowns([]) == {}
    assert aggregate_seed_breakdowns([{}, {}]) == {}


def _doc(**over):
    base = dict(
        benchmark_version="1.0.0",
        task="t",
        protocol="held_out",
        grid="g",
        baseline="b",
        pretrain_grid=None,
        test_set="test",
        n_test=3,
        seeds=[0, 1],
        n_params=None,
        compute={},
        per_seed=[],
        aggregate={},
        breakdowns={},
    )
    base.update(over)
    return build_result_doc(**base)


def test_render_is_deterministic_sorted_and_newline_free():
    """Byte-level contract: the committed corpus has sorted keys, indent=2 and NO trailing newline.
    A newline here would rewrite every file in the corpus for nothing."""
    s = render_result_json(_doc())
    assert s == render_result_json(_doc())
    assert not s.endswith("\n")
    assert json.loads(s)["task"] == "t"
    keys = list(json.loads(s))
    assert keys == sorted(keys)


def test_render_maps_non_finite_to_null():
    """JSON has no NaN/Inf literal; emitting one would produce a file consumers cannot parse."""
    s = render_result_json(_doc(aggregate={"m": {"mean": math.nan, "std": math.inf}}))
    assert json.loads(s)["aggregate"]["m"] == {"mean": None, "std": None}
    assert "NaN" not in s and "Infinity" not in s


def test_build_result_doc_is_keyword_only():
    with pytest.raises(TypeError):
        build_result_doc(
            "1.0.0", "t", "held_out", "g", "b", None, "test", 1, [0], None, {}, [], {}, {}
        )
