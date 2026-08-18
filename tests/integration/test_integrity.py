"""The artifact-integrity checker (evemtbench.integrity): clean on a freshly-produced fixture
cell, and each named corruption (missing run record, wrong seeds, truncated predictions) is
detected. The real-corpus sweep is a data_integrity test (excluded from CI; run on the loop
host / at freeze time)."""

import json

import pytest

from evemtbench import integrity, results_version
from evemtbench.baselines import runner


def _run_fixture_cell(fixture_root, held_out_splits, out) -> None:
    runner.main(
        [
            "--task", "fault_detection_local",
            "--protocol", "held_out",
            "--grid", "double_line",
            "--baseline", "majority",
            "--seeds", "0",
            "--root", str(fixture_root),
            "--splits-root", str(held_out_splits),
            "--out", str(out),
        ]
    )  # fmt: skip


@pytest.fixture()
def fixture_cell(fixture_root, held_out_splits, tmp_path):
    out = tmp_path / "out"
    _run_fixture_cell(fixture_root, held_out_splits, out)
    cell_dir = (
        out
        / results_version("held_out")
        / "fault_detection_local"
        / "held_out"
        / "double_line"
        / "majority"
    )
    return out, cell_dir


def test_clean_cell_passes(fixture_cell):
    out, _ = fixture_cell
    report = integrity.check_outputs(out, expected_seeds=[0])
    assert report["ok"], report["problems"]
    assert report["n_done"] == 1
    assert report["n_skipped"] > 0  # the rest of the matrix is pending, skipped not failed
    assert len(report["provenance_pairs"]) == 1


def test_wrong_seed_set_detected(fixture_cell):
    out, _ = fixture_cell
    report = integrity.check_outputs(out, expected_seeds=[0, 1, 2, 3, 4])
    assert not report["ok"]
    assert any("seeds" in p and "expected" in p for p in report["problems"])


def test_missing_run_record_detected(fixture_cell):
    out, cell_dir = fixture_cell
    (cell_dir / "run_record.json").unlink()
    report = integrity.check_outputs(out, expected_seeds=[0])
    assert not report["ok"]
    assert any("run_record.json missing" in p for p in report["problems"])


def test_seed_mismatch_between_record_and_result_detected(fixture_cell):
    out, cell_dir = fixture_cell
    rec = json.loads((cell_dir / "run_record.json").read_text())
    rec["seeds"] = [7]
    (cell_dir / "run_record.json").write_text(json.dumps(rec))
    report = integrity.check_outputs(out, expected_seeds=[0])
    assert not report["ok"]
    assert any("run_record.seeds" in p for p in report["problems"])


def test_null_git_sha_detected(fixture_cell):
    out, cell_dir = fixture_cell
    rec = json.loads((cell_dir / "run_record.json").read_text())
    rec["git_sha"] = None
    (cell_dir / "run_record.json").write_text(json.dumps(rec))
    report = integrity.check_outputs(out, expected_seeds=[0])
    assert not report["ok"]
    assert any("git_sha is null" in p for p in report["problems"])


def test_deep_mode_catches_truncated_predictions(fixture_cell):
    pytest.importorskip("pyarrow")
    out, cell_dir = fixture_cell
    f = cell_dir / "test" / "predictions_seed0.parquet"
    f.write_bytes(f.read_bytes()[:100])  # truncate mid-file
    report = integrity.check_outputs(out, expected_seeds=[0], deep=True)
    assert not report["ok"]
    assert any("predictions_seed0.parquet" in p for p in report["problems"])
    # fast mode stays green — the corruption is only visible to --deep
    assert integrity.check_outputs(out, expected_seeds=[0])["ok"]


def test_cli_exit_codes(fixture_cell, capsys):
    out, cell_dir = fixture_cell
    assert integrity.main(["--out", str(out), "--expect-seeds", "0"]) == 0
    (cell_dir / "run_record.json").unlink()
    assert integrity.main(["--out", str(out), "--expect-seeds", "0"]) == 1
    assert "PROBLEM" in capsys.readouterr().out


@pytest.mark.data_integrity
def test_real_corpus_integrity():
    """Fast-mode integrity over the real outputs/ tree (loop host only; excluded from CI)."""
    report = integrity.check_outputs("outputs")
    assert report["ok"], report["problems"][:20]
