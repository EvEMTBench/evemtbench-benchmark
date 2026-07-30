"""Coverage manifest: matrix enumeration, applicability, status, and the majority-floor gate."""

import json

import pytest

from evemtbench import coverage as cov
from evemtbench.coverage import Cell


def test_threshold_only_on_fault_detection_local():
    cells = cov.iter_cells(
        tasks=["fault_detection_local", "fault_detection_global", "fault_location_local"],
        grids=["double_line"],
        baselines=("threshold", "mlp"),
    )
    thr = {c.task for c in cells if c.baseline == "threshold"}
    mlp = {c.task for c in cells if c.baseline == "mlp"}
    assert thr == {"fault_detection_local"}  # overcurrent: binary + single-end only
    assert mlp == {"fault_detection_local", "fault_detection_global", "fault_location_local"}


def test_cells_span_grids_and_skip_unsupported_protocol():
    cells = cov.iter_cells(
        tasks=["fault_detection_global"],
        grids=["double_line", "cigre_mv"],
        protocols=("held_out", "cross_topology"),  # cross_topology unsupported in v0.1 registry
        baselines=("majority",),
    )
    assert {(c.grid, c.protocol) for c in cells} == {
        ("double_line", "held_out"),
        ("cigre_mv", "held_out"),
    }


def _write_result(d, test_set, valid=True):
    p = d / test_set
    p.mkdir(parents=True, exist_ok=True)
    doc = {"aggregate": {"balanced_accuracy": {"mean": 0.9}}} if valid else {"aggregate": {}}
    (p / "result.json").write_text(json.dumps(doc))


def test_status_done_pending_failed(tmp_path):
    cell = Cell("fault_detection_local", "held_out", "double_line", "majority")
    d = cov.cell_dir(tmp_path, cell, version="v")
    assert cov.cell_status(tmp_path, cell, version="v") == "pending"
    _write_result(d, "test")
    assert cov.cell_status(tmp_path, cell, version="v") == "failed"  # only one test set present
    _write_result(d, "benchmark")
    assert cov.cell_status(tmp_path, cell, version="v") == "done"


def test_scan_groups(tmp_path):
    cells = cov.iter_cells(
        tasks=["fault_detection_local"], grids=["double_line"], baselines=("majority", "mlp")
    )
    groups = cov.scan(tmp_path, cells, version="v")
    assert len(groups["pending"]) == 2 and not groups["done"] and not groups["failed"]


# ---- majority-floor sanity gate (--check-floor) ----------------------------------------------


def _write_agg(root, cell, aggregate, test_set="test", version="v"):
    d = cov.cell_dir(root, cell, version=version) / test_set
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps({"aggregate": aggregate}))


def _bacc(mean):
    return {"balanced_accuracy": {"mean": mean, "std": 0.0, "ci95": 0.0, "n_defined": 3}}


def _fd_cells(*baselines):
    return [cov.Cell("fault_detection_local", "held_out", "double_line", b) for b in baselines]


def test_floor_gate_passes_when_learned_beats_majority(tmp_path):
    cells = _fd_cells("majority", "mlp")
    _write_agg(tmp_path, cells[0], _bacc(0.5))  # binary floor = 1/2
    _write_agg(tmp_path, cells[1], _bacc(0.9))
    report = cov.check_floor(tmp_path, cells, version="v")
    assert report["ok"] and not report["violations"] and report["n_checked"] == 2


def test_floor_gate_flags_learned_below_majority(tmp_path):
    cells = _fd_cells("majority", "mlp")
    _write_agg(tmp_path, cells[0], _bacc(0.5))
    _write_agg(tmp_path, cells[1], _bacc(0.4))
    report = cov.check_floor(tmp_path, cells, version="v")
    assert not report["ok"]
    assert any("mlp" in v and "below" in v for v in report["violations"])


def test_floor_gate_flags_implausible_majority(tmp_path):
    # a constant predictor cannot have balanced_accuracy 0.8 on a binary task (floor = 0.5)
    cells = _fd_cells("majority")
    _write_agg(tmp_path, cells[0], _bacc(0.8))
    report = cov.check_floor(tmp_path, cells, version="v")
    assert not report["ok"] and any("floor" in v for v in report["violations"])


def test_floor_gate_skips_pending_and_undefined(tmp_path):
    # missing majority, then majority with an undefined mean: skipped, never a failure
    cells = _fd_cells("majority", "mlp")
    report = cov.check_floor(tmp_path, cells, version="v")
    assert report["ok"] and report["n_checked"] == 0 and report["skipped"]
    _write_agg(
        tmp_path,
        cells[0],
        {"balanced_accuracy": {"mean": None, "std": None, "ci95": None, "n_defined": 0}},
    )
    report = cov.check_floor(tmp_path, cells, version="v")
    assert report["ok"] and any("undefined" in s for s in report["skipped"])


def test_floor_gate_multiclass_uses_classes_present(tmp_path):
    # fault_classification_global: k from the DEFINED per-class F1 columns (absent class -> None)
    cells = [
        cov.Cell("fault_classification_global", "held_out", "cigre_mv", b)
        for b in ("majority", "mlp")
    ]
    agg = _bacc(1.0 / 3.0)
    agg.update(
        {
            "f1_a": {"mean": 0.5, "n_defined": 3},
            "f1_b": {"mean": 0.0, "n_defined": 3},  # present class, zero skill: still counted
            "f1_c": {"mean": 0.2, "n_defined": 3},
            "f1_absent": {"mean": None, "n_defined": 0},  # no support here: not counted
        }
    )
    _write_agg(tmp_path, cells[0], agg)
    _write_agg(tmp_path, cells[1], _bacc(0.7))
    report = cov.check_floor(tmp_path, cells, version="v")
    assert report["ok"], report["violations"]  # majority 1/3 matches the k=3 floor
    _write_agg(tmp_path, cells[0], {**agg, "balanced_accuracy": {"mean": 0.25, "n_defined": 3}})
    report = cov.check_floor(tmp_path, cells, version="v")
    assert not report["ok"]  # 0.25 is the k=4 floor, but only 3 classes are present


def test_floor_gate_regression_direction_and_relative_tol(tmp_path):
    # mae: lower is better; violation = learned ABOVE majority beyond the relative band
    cells = [
        cov.Cell("fault_location_local", "held_out", "double_line", b)
        for b in ("majority", "random_forest")
    ]
    mae = lambda m: {"mae": {"mean": m, "std": 0.0, "ci95": 0.0, "n_defined": 3}}  # noqa: E731
    _write_agg(tmp_path, cells[0], mae(10.0))
    _write_agg(tmp_path, cells[1], mae(9.0))
    assert cov.check_floor(tmp_path, cells, version="v")["ok"]
    _write_agg(tmp_path, cells[1], mae(12.0))
    report = cov.check_floor(tmp_path, cells, version="v")
    assert not report["ok"] and any("above" in v for v in report["violations"])


def test_check_floor_cli_exit_codes(tmp_path):
    args = [
        "--out",
        str(tmp_path),
        "--version",
        "v",
        "--tasks",
        "fault_detection_local",
        "--grids",
        "double_line",
        "--baselines",
        "majority",
        "mlp",
        "--check-floor",
    ]
    with pytest.raises(SystemExit) as e:  # empty tree: everything skipped -> OK
        cov.main(args)
    assert e.value.code == 0
    cells = _fd_cells("majority", "mlp")
    _write_agg(tmp_path, cells[0], _bacc(0.5))
    _write_agg(tmp_path, cells[1], _bacc(0.3))
    with pytest.raises(SystemExit) as e:
        cov.main(args)
    assert e.value.code == 1


def test_check_floor_does_not_touch_scan_semantics(tmp_path):
    cells = _fd_cells("majority")
    _write_agg(tmp_path, cells[0], _bacc(0.5))
    before = cov.scan(tmp_path, cells, version="v")
    cov.check_floor(tmp_path, cells, version="v")
    assert cov.scan(tmp_path, cells, version="v") == before


# ---------------------------------------------------------------------------------------------
# Persistent coverage snapshot (coverage_snapshot.json — committed, gate-checked)
# ---------------------------------------------------------------------------------------------
def test_snapshot_deterministic_and_matches_scan(tmp_path):
    cell = Cell("fault_detection_local", "held_out", "double_line", "resnet")
    d = cov.cell_dir(tmp_path, cell, version="v")
    for ts in cov.expected_test_sets(cell.grid, cell.protocol):
        _write_result(d, ts)

    a = cov.render_snapshot(tmp_path, version="v")
    b = cov.render_snapshot(tmp_path, version="v")
    assert a == b, "snapshot is not byte-deterministic"
    assert a.endswith("\n")

    snap = json.loads(a)
    assert snap["benchmark_version"] == "v"
    assert list(snap["protocols"]) == list(cov.SNAPSHOT_PROTOCOLS)
    # counts agree with scan() per protocol, and the totals add up
    for proto in cov.SNAPSHOT_PROTOCOLS:
        cells = cov.iter_cells(protocols=(proto,))
        groups = cov.scan(tmp_path, cells, version="v")
        c = snap["counts_by_protocol"][proto]
        assert c["total"] == len(cells)
        for k in ("done", "pending", "failed"):
            assert c[k] == len(groups[k]), (proto, k)
    assert snap["counts"]["total"] == sum(
        snap["counts_by_protocol"][p]["total"] for p in cov.SNAPSHOT_PROTOCOLS
    )
    # the seeded cell is done and carries its verbatim reproduce arg-line
    row = next(
        r
        for r in snap["cells"]
        if (r["task"], r["protocol"], r["grid"], r["baseline"])
        == ("fault_detection_local", "held_out", "double_line", "resnet")
    )
    assert row["status"] == "done"
    assert row["arg_line"] == cov._arg_line(cell)
    assert row["expected_test_sets"] == list(cov.expected_test_sets(cell.grid, cell.protocol))
    # rows are sorted (deterministic order)
    keys = [(r["task"], r["protocol"], r["grid"], r["baseline"]) for r in snap["cells"]]
    assert keys == sorted(keys)


def test_snapshot_self_check_clean_and_range_violation(tmp_path):
    """The embedded self-check: OK on a clean tree; a physically impossible aggregate flips
    value_ranges (hard fail); dominance findings never flip ok."""
    snap = cov.build_snapshot(tmp_path, version="v")
    sc = snap["self_check"]
    assert sc["ok"] and [c["id"] for c in sc["checks"]] == [
        "completeness",
        "floor_plausibility",
        "value_ranges",
    ]

    # a done cell with balanced_accuracy > 1 must fail value_ranges
    cell = Cell("fault_detection_local", "held_out", "double_line", "resnet")
    d = cov.cell_dir(tmp_path, cell, version="v")
    for ts in cov.expected_test_sets(cell.grid, cell.protocol):
        p = d / ts
        p.mkdir(parents=True, exist_ok=True)
        (p / "result.json").write_text(
            json.dumps({"aggregate": {"balanced_accuracy": {"mean": 1.7}}})
        )
    sc = cov.build_snapshot(tmp_path, version="v")["self_check"]
    assert not sc["ok"]
    bad = {c["id"]: c for c in sc["checks"]}
    assert not bad["value_ranges"]["ok"]
    assert any("1.7" in str(x) for x in bad["value_ranges"]["detail"])


def test_check_snapshot_cli_exit_codes(tmp_path):
    with pytest.raises(SystemExit) as e:
        cov.main(["--out", str(tmp_path), "--version", "v", "--check-snapshot"])
    assert e.value.code == 0
