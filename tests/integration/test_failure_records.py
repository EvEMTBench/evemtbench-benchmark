"""The runner records failed attempts, clears them on success, and coverage honours the retry
cap; run_record.json carries the working-tree state (git_dirty / git_diff_sha256)."""

import json

import pytest

from evemtbench import coverage as cov
from evemtbench import failures, results_version
from evemtbench.baselines import runner

HO_VER = results_version("held_out")
TASK = "fault_detection_local"


def _args(fixture_root, held_out_splits, out, *extra):
    return [
        "--task", TASK, "--protocol", "held_out", "--grid", "double_line",
        "--baseline", "majority", "--seeds", "0",
        "--root", str(fixture_root), "--splits-root", str(held_out_splits),
        "--out", str(out), *extra,
    ]  # fmt: skip


def _cell_dir(out, baseline="majority"):
    return out / HO_VER / TASK / "held_out" / "double_line" / baseline


def test_failed_run_is_recorded_with_its_stage(fixture_root, held_out_splits, tmp_path):
    out = tmp_path / "out"
    # --zero-seq is rejected for a classical baseline inside the seed loop -> a "fit" failure
    with pytest.raises(ValueError):
        runner.main(_args(fixture_root, held_out_splits, out, "--zero-seq"))
    doc = failures.load(_cell_dir(out, "majority+zero_seq"))
    (a,) = doc["attempts"]
    assert a["status"] == "failed" and a["stage"] == "fit (seed 0)"
    assert a["error_type"] == "ValueError"


def test_success_clears_the_record_and_writes_no_temp_files(
    fixture_root, held_out_splits, tmp_path
):
    out = tmp_path / "out"
    cell = _cell_dir(out)
    failures.open_attempt(cell)  # a previous attempt that was killed
    runner.main(_args(fixture_root, held_out_splits, out))
    assert failures.load(cell) is None
    assert not [p for p in cell.rglob("*") if ".tmp-" in p.name]
    rec = json.loads((cell / "run_record.json").read_text())
    assert rec["git_dirty"] in (True, False)
    assert (rec["git_diff_sha256"] is None) == (rec["git_dirty"] is not True)


def test_emit_pending_respects_the_retry_cap(tmp_path, capsys):
    out = tmp_path / "out"
    cell = cov.Cell(TASK, "held_out", "double_line", "majority")
    d = cov.cell_dir(out, cell)
    for _ in range(failures.MAX_ATTEMPTS):
        failures.open_attempt(d)
    common = ["--out", str(out), "--tasks", TASK, "--grids", "double_line",
              "--baselines", "majority"]  # fmt: skip

    cov.main([*common, "--emit-pending"])
    captured = capsys.readouterr()
    assert "--baseline majority" not in captured.out
    assert "retry cap" in captured.err

    cov.main([*common, "--emit-pending", "--max-attempts", "0"])
    assert "--baseline majority" in capsys.readouterr().out

    cov.main([*common, "--summary"])
    summary = capsys.readouterr().out
    # hpc/autonomous_loop.sh reads the line whose first field is "pending" -> exactly one
    assert [ln.split()[1] for ln in summary.splitlines() if ln.split()[:1] == ["pending"]] == ["1"]
    assert failures.NO_CLEAN_EXIT in summary

    cov.main([*common, "--failures"])
    listing = capsys.readouterr().out
    assert "[retry cap reached]" in listing and "attempts=3" in listing
