"""Leaderboard scans result.json artifacts and renders self-contained HTML."""

import json
import re
from pathlib import Path

import pytest

from evemtbench.leaderboard import render_html, scan_results
from evemtbench.version import results_version


def _write_result(base: Path, task, protocol, grid, baseline, bal_acc, *, test_set="test"):
    # written into the protocol's CURRENT results tree so the headline sections (coverage
    # panel / topology comparison / transfer), which are current-scoped, render these rows
    d = base / results_version(protocol) / task / protocol / grid / baseline / test_set
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(
        json.dumps(
            {
                "task": task,
                "protocol": protocol,
                "grid": grid,
                "baseline": baseline,
                "test_set": test_set,
                "benchmark_version": results_version(protocol),
                "n_test": 100,
                "seeds": [0, 1],
                "aggregate": {
                    "balanced_accuracy": {"mean": bal_acc, "std": 0.01},
                    "macro_f1": {"mean": bal_acc - 0.05, "std": 0.02},
                },
            }
        )
    )


def test_transfer_source_falls_back_to_the_pretrain_path_token(tmp_path):
    """Validation docs carry no ``pretrain_grid`` (hpc/rescore_validation.py does not write it), so
    without the path fallback every per-source val row collapses onto the legacy source-less key —
    three distinct runs sharing the key results/<version>/README.md documents as unique."""
    task, proto, grid, base = "fault_detection_line", "transfer_finetune", "cigre_mv", "gru"
    for src in ("template_110kv", "template_345kv"):
        d = tmp_path / "1.0.0-dev" / task / proto / grid / base / f"pretrain-{src}" / "val"
        d.mkdir(parents=True, exist_ok=True)
        (d / "result.json").write_text(
            json.dumps(
                {
                    "task": task,
                    "protocol": proto,
                    "grid": grid,
                    "baseline": base,
                    "test_set": "val",
                    "n_test": 10,
                    "seeds": [0, 1],
                    "aggregate": {"balanced_accuracy": {"mean": 0.9, "std": 0.01}},
                }
            )
        )
    # the source-less cell-root val doc keeps the legacy default (source=None)
    _write_result(tmp_path, task, proto, grid, base, 0.9, test_set="val")

    rows = scan_results(tmp_path)
    assert {r.source for r in rows} == {"template_110kv", "template_345kv", None}, (
        "per-source validation rows must keep their pre-training source"
    )


def test_explicit_pretrain_grid_field_wins_over_the_path(tmp_path):
    """The runner's own field stays authoritative — the path is only a fallback."""
    d = tmp_path / "1.0.0-dev" / "t" / "transfer_finetune" / "g" / "gru" / "pretrain-WRONG" / "test"
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(
        json.dumps(
            {
                "task": "t",
                "protocol": "transfer_finetune",
                "grid": "g",
                "baseline": "gru",
                "test_set": "test",
                "n_test": 10,
                "seeds": [0],
                "pretrain_grid": "template_345kv",
                "aggregate": {"balanced_accuracy": {"mean": 0.9, "std": 0.0}},
            }
        )
    )
    (row,) = scan_results(tmp_path)
    assert row.source == "template_345kv"


def test_scan_and_rank(tmp_path):
    _write_result(tmp_path, "fault_detection_local", "in_grid", "double_line", "majority", 0.50)
    _write_result(tmp_path, "fault_detection_local", "in_grid", "double_line", "random_forest", 0.9)
    _write_result(
        tmp_path, "fault_detection_local", "cross_grid_zeroshot", "double_line", "mlp", 0.5
    )

    rows = scan_results(tmp_path)
    assert len(rows) == 3
    assert {r.baseline for r in rows} == {"majority", "random_forest", "mlp"}

    html = render_html(rows, version="1.0.0-dev")
    assert "<table" in html and "random_forest" in html
    assert "fault_detection_local" in html
    assert "1.0.0-dev" in html
    # self-contained = no external *resource* fetches (CSS/JS/img). Anchor/doc links are allowed.
    assert not re.search(r'src\s*=\s*["\']https?:', html)
    assert not re.search(r'<link\b[^>]*rel=["\']?stylesheet', html, re.I)
    assert "<style" in html  # CSS stays inline


def test_document_structure_and_timestamp(tmp_path):
    _write_result(tmp_path, "fault_detection_local", "held_out", "double_line", "mlp", 0.8)
    rows = scan_results(tmp_path)
    # proper, valid HTML5 skeleton (needed for meta tags / favicon on GitLab Pages)
    html = render_html(rows, version="1.0.0-dev")
    assert "<html lang=en" in html and "<head>" in html and "</body></html>" in html
    assert "viewport" in html
    # timestamp is opt-in and deterministic by default (none unless the caller passes one)
    assert "updated" not in html
    stamped = render_html(rows, version="1.0.0-dev", generated_at="2026-01-01 00:00 UTC")
    assert "updated 2026-01-01 00:00 UTC" in stamped


def test_topology_comparison_spans_all_grids(tmp_path):
    # same task across several grids -> the comparison matrix must show a column per grid,
    # and a missing baseline×grid cell must render "—" (not a fabricated 0).
    for grid, acc in [("cigre_mv", 0.7), ("double_line", 0.8), ("ieee39", 0.6)]:
        _write_result(tmp_path, "fault_detection_local", "held_out", grid, "mlp", acc)
    # gru only ran on two grids -> ieee39 cell should be "—"
    _write_result(tmp_path, "fault_detection_local", "held_out", "cigre_mv", "gru", 0.65)
    _write_result(tmp_path, "fault_detection_local", "held_out", "double_line", "gru", 0.72)

    rows = scan_results(tmp_path)
    html = render_html(rows, version="1.0.0-dev")
    assert "Topology comparison" in html
    # coverage panel + comparison name every grid present
    for grid in ("cigre_mv", "double_line", "ieee39"):
        assert grid in html
    assert "Grids" in html  # coverage panel heading


def test_absent_class_renders_na_not_zero(tmp_path):
    # evaluate() stores an absent class's F1 as NaN -> aggregates to a null mean with n_defined=0;
    # that must render "n/a", not "0.000" (the original HIF bug). A genuine zero stays "0.000".
    # The class names here are fault_phase classes, deliberately NOT event types: presence is read
    # from the stored value, never from the event_type breakdown (whose keys only equal the class
    # names for fault_classification -- keying on them blanked every phase/category/switch F1).
    d = tmp_path.joinpath("0.1.0-dev", "fault_phase_local", "held_out", "cigre_mv", "mlp", "test")
    d.mkdir(parents=True)
    (d / "result.json").write_text(
        json.dumps(
            {
                "task": "fault_phase_local",
                "protocol": "held_out",
                "grid": "cigre_mv",
                "baseline": "mlp",
                "test_set": "test",
                "n_test": 10,
                "seeds": [0, 1],
                "aggregate": {
                    "balanced_accuracy": {"mean": 0.5, "std": 0.0},
                    "f1_A": {"mean": 0.987, "std": 0.001, "n_defined": 2, "n_seeds": 2},
                    "f1_AB": {"mean": 0.0, "std": 0.0, "n_defined": 2, "n_seeds": 2},
                    "f1_ABC": {
                        "mean": None,
                        "std": None,
                        "ci95": None,
                        "n_defined": 0,
                        "n_seeds": 2,
                    },
                },
                "breakdowns": {"event_type": {"flt_1phg": {"n": 6}, "flt_2ph": {"n": 4}}},
            }
        )
    )
    (row,) = scan_results(tmp_path)
    assert row.metrics["f1_ABC"].get("absent") is True  # no support -> flagged n/a
    assert row.metrics["f1_ABC"]["ci95"] is None  # no stale interval on an n/a cell
    assert not row.metrics["f1_A"].get("absent")  # present class keeps its score
    assert row.metrics["f1_A"]["mean"] == 0.987
    assert not row.metrics["f1_AB"].get("absent")  # genuine zero is not "absent"

    html = render_html(scan_results(tmp_path), version="0.1.0-dev")
    assert "class=na" in html and ">n/a<" in html  # absent class -> n/a
    assert "0.987" in html  # present class rendered
    assert "0.000" in html  # genuine zero (present class) stays 0.000


def test_topology_comparison_ranks_lower_is_better_by_best_grid(tmp_path):
    # FL's headline is mae (lower is better): baselines must be ordered by their BEST (lowest)
    # grid, not their worst. a: best 1.0 / worst 9.0; b: 2.0 on both -> a ranks first.
    for grid, a_mae, b_mae in [("cigre_mv", 1.0, 2.0), ("double_line", 9.0, 2.0)]:
        for base, mae in (("a_base", a_mae), ("b_base", b_mae)):
            d = tmp_path / results_version("held_out") / "fault_location_local" / "held_out"
            d = d / grid / base / "test"
            d.mkdir(parents=True)
            (d / "result.json").write_text(
                json.dumps(
                    {
                        "task": "fault_location_local",
                        "protocol": "held_out",
                        "grid": grid,
                        "baseline": base,
                        "test_set": "test",
                        "benchmark_version": results_version("held_out"),
                        "n_test": 10,
                        "seeds": [0, 1],
                        "aggregate": {"mae": {"mean": mae, "std": 0.1}},
                    }
                )
            )
    html = render_html(scan_results(tmp_path), version="1.0.0-dev")
    cmp = html[html.index("Topology comparison") :]
    assert cmp.index(">a_base<") < cmp.index(">b_base<")


def test_empty_outputs(tmp_path):
    rows = scan_results(tmp_path)
    assert rows == []
    assert "No result.json" in render_html(rows)


def test_ci95_stored_derived_and_legacy(tmp_path):
    """Three artifact generations: runner-stored ci95 renders as-is; a pre-ci95 doc WITH per_seed
    gets its CI derived at scan time; a doc with neither falls back to the legacy mean ± std."""
    common = {
        "task": "fault_detection_local",
        "protocol": "held_out",
        "grid": "double_line",
        "test_set": "test",
        "n_test": 10,
        "benchmark_version": "0.2.0-dev",
    }
    docs = [
        {
            **common,
            "baseline": "stored",
            "seeds": [0, 1, 2],
            "aggregate": {
                "balanced_accuracy": {"mean": 0.9, "std": 0.01, "ci95": 0.025, "n_defined": 3}
            },
        },
        {
            **common,
            "baseline": "derived",
            "seeds": [0, 1, 2],
            "per_seed": [
                {"seed": s, "metrics": {"balanced_accuracy": v}}
                for s, v in [(0, 0.8), (1, 0.9), (2, 1.0)]
            ],
            "aggregate": {"balanced_accuracy": {"mean": 0.9, "std": 0.081, "n_defined": 3}},
        },
        {
            **common,
            "baseline": "legacy",
            "seeds": [0, 1],
            "aggregate": {"balanced_accuracy": {"mean": 0.7, "std": 0.05}},
        },
    ]
    for doc in docs:
        d = (
            tmp_path
            / doc["benchmark_version"]
            / doc["task"]
            / doc["protocol"]
            / doc["grid"]
            / doc["baseline"]
            / doc["test_set"]
        )
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps(doc))
    rows = {r.baseline: r for r in scan_results(tmp_path)}
    assert rows["stored"].metrics["balanced_accuracy"]["ci95"] == 0.025
    # t_{0.975, df=2} = 4.302653; s([0.8,0.9,1.0], ddof=1) = 0.1 -> hw = 4.302653*0.1/sqrt(3)
    assert rows["derived"].metrics["balanced_accuracy"]["ci95"] == pytest.approx(0.24841, rel=1e-3)
    assert rows["legacy"].metrics["balanced_accuracy"]["ci95"] is None
    page = render_html(list(rows.values()), version="0.2.0-dev")
    assert "0.900 ± 0.025" in page  # stored CI rendered, mean is the leading token
    assert "0.900 ± 0.248" in page  # derived from per_seed
    assert "0.700 ± 0.050" in page  # legacy falls back to mean ± std
