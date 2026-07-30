"""End-to-end evemtbench-run + evemtbench-evaluate CLIs on the fixture (held_out)."""

import json

import pytest

from evemtbench import BENCHMARK_VERSION
from evemtbench.baselines import runner
from evemtbench.evaluation import cli


def _run(baseline, task, fixture_root, held_out_splits, out, seeds=("0",)):
    runner.main(
        [
            "--task",
            task,
            "--protocol",
            "held_out",
            "--grid",
            "double_line",
            "--baseline",
            baseline,
            "--seeds",
            *seeds,
            "--root",
            str(fixture_root),
            "--splits-root",
            str(held_out_splits),
            "--out",
            str(out),
        ]
    )


@pytest.mark.parametrize("baseline", ["majority", "threshold", "random_forest"])
def test_runner_writes_both_test_sets(baseline, fixture_root, held_out_splits, tmp_path):
    out = tmp_path / "out"
    _run(baseline, "fault_detection_local", fixture_root, held_out_splits, out)
    base = out / BENCHMARK_VERSION / "fault_detection_local" / "held_out" / "double_line" / baseline
    for ts in ("test", "benchmark"):
        result = json.loads((base / ts / "result.json").read_text())
        assert result["baseline"] == baseline
        assert result["test_set"] == ts
        assert "balanced_accuracy" in result["aggregate"]
        assert (base / ts / "predictions_seed0.parquet").is_file()
    assert json.loads((base / "run_record.json").read_text())["git_sha"] is not None


def test_multi_seed_ci_and_run_record_attestation(fixture_root, held_out_splits, tmp_path):
    out = tmp_path / "out"
    seeds = ("0", "1", "2")
    _run("majority", "fault_detection_local", fixture_root, held_out_splits, out, seeds=seeds)
    base = (
        out / BENCHMARK_VERSION / "fault_detection_local" / "held_out" / "double_line" / "majority"
    )
    agg = json.loads((base / "test" / "result.json").read_text())["aggregate"]["balanced_accuracy"]
    # majority is deterministic across seeds -> a zero-width but *defined* CI (not null)
    assert agg["n_defined"] == 3 and agg["ci95"] == pytest.approx(0.0)
    assert "mean" in agg and "std" in agg  # existing keys untouched
    record = json.loads((base / "run_record.json").read_text())
    att = record["attestation"]
    assert att["fit_on"] == "train" and att["held_out_untouched"] == ["test", "benchmark"]
    assert record["splits_version"]
    assert isinstance(record["lockfile_sha256"], str)
    assert record["versions"]["numpy"]
    # harness positive control (deep-audit): true labels scored the ceiling via evaluate()
    for ts in ("test", "benchmark"):
        hc = record["harness_control"][ts]
        assert hc["ok"] is True and hc["value"] == hc["ceiling"]
    # host + input-data provenance (results-audit hardening): the executing host is recorded,
    # split files are hashed, and the partitions actually loaded carry manifest fingerprints
    assert record["host"]["hostname"]
    splits_prov = record["data"]["splits"]
    assert splits_prov["protocol_dir"] == "held_out"
    assert all(len(h) == 64 for h in splits_prov["files"].values())
    assert {"train.txt", "val.txt", "test.txt", "benchmark.txt"} <= set(splits_prov["files"])
    parts = record["data"]["partitions"]
    assert parts, "no windowed-partition provenance recorded"
    for part in parts:
        assert part["grid"] == "double_line"
        assert len(part["manifest_sha256"]) == 64
        assert part["feature_hash"] is None or len(part["feature_hash"]) == 64


def test_run_record_validates_against_schema(fixture_root, held_out_splits, tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    from pathlib import Path

    out = tmp_path / "out"
    _run("majority", "fault_detection_local", fixture_root, held_out_splits, out)
    base = (
        out / BENCHMARK_VERSION / "fault_detection_local" / "held_out" / "double_line" / "majority"
    )
    record = json.loads((base / "run_record.json").read_text())
    schema_path = Path(runner.__file__).resolve().parent / "run_record.schema.json"
    jsonschema.validate(record, json.loads(schema_path.read_text()))


def test_single_seed_ci_is_null(fixture_root, held_out_splits, tmp_path):
    out = tmp_path / "out"
    _run("majority", "fault_detection_local", fixture_root, held_out_splits, out)  # seeds=("0",)
    result = json.loads(
        (
            out
            / BENCHMARK_VERSION
            / "fault_detection_local"
            / "held_out"
            / "double_line"
            / "majority"
            / "test"
            / "result.json"
        ).read_text()
    )
    # one seed -> no seed-variance estimate; ci95 must be null, never a fake ± 0
    assert result["aggregate"]["balanced_accuracy"]["ci95"] is None


def test_runner_regression_task(fixture_root, held_out_splits, tmp_path):
    out = tmp_path / "out"
    _run("random_forest", "fault_location_local", fixture_root, held_out_splits, out)
    base = (
        out
        / BENCHMARK_VERSION
        / "fault_location_local"
        / "held_out"
        / "double_line"
        / "random_forest"
        / "benchmark"
    )
    result = json.loads((base / "result.json").read_text())
    assert "mae" in result["aggregate"]


def test_evaluate_cli_audits_predictions(fixture_root, held_out_splits, tmp_path):
    out = tmp_path / "out"
    _run("majority", "fault_detection_local", fixture_root, held_out_splits, out)
    preds = (
        out
        / BENCHMARK_VERSION
        / "fault_detection_local"
        / "held_out"
        / "double_line"
        / "majority"
        / "test"
        / "predictions_seed0.parquet"
    )
    audit = tmp_path / "audit.json"
    cli.main(
        [
            "--task",
            "fault_detection_local",
            "--protocol",
            "held_out",
            "--grid",
            "double_line",
            "--predictions",
            str(preds),
            "--root",
            str(fixture_root),
            "--splits-root",
            str(held_out_splits),
            "--out",
            str(audit),
        ]
    )
    audited = json.loads(audit.read_text())
    assert audited["task"] == "fault_detection_local"
    assert audited["benchmark_version"] == BENCHMARK_VERSION
    assert "balanced_accuracy" in audited["metrics"]


def test_broken_harness_aborts_the_cell(fixture_root, held_out_splits, tmp_path, monkeypatch):
    """If the scoring harness cannot score the true labels perfectly, the run must raise and
    write NO result.json — a broken harness never persists results."""
    real = runner.evaluate

    def skewed(task, preds, *, test_set="test"):
        r = real(task, preds, test_set=test_set)
        r.metrics = {k: (v * 0.9 if isinstance(v, float) else v) for k, v in r.metrics.items()}
        return r

    # Patch evaluate on the SAME module object that _run() drives main() on — the module-level
    # `runner` imported above. A local `import ... as runner_mod` can bind a DIFFERENT module
    # object once another test resets sys.modules (e.g. test_import_isolation), so patching the
    # re-imported name would silently miss main()'s (and _harness_control's) evaluate.
    monkeypatch.setattr(runner, "evaluate", skewed)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="harness positive control FAILED"):
        _run("majority", "fault_detection_local", fixture_root, held_out_splits, out)
    base = (
        out / BENCHMARK_VERSION / "fault_detection_local" / "held_out" / "double_line" / "majority"
    )
    assert not (base / "test" / "result.json").exists()
