"""Loud-degradation invariants for the results/selection path (``warn_once``): a selection that
falls back from validation to the reported set, a baseline excluded from validation candidacy, and
a swallowed corrupt ``result.json`` must each print a ``[warn]`` line to stderr — while rendered
table bytes stay identical (the warnings are observability only, never content)."""

import json
from pathlib import Path

import pytest

from evemtbench.version import results_version

REPO = Path(__file__).resolve().parents[2]


def _load_modules():
    import paper_lib

    import make_results

    return make_results, paper_lib


@pytest.fixture(autouse=True)
def _fresh_warn_state(monkeypatch):
    """Each test starts with an empty dedup set (warn_once dedups per process)."""
    mr, _ = _load_modules()
    monkeypatch.setattr(mr, "_WARNED", set())


def _write_doc(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def _held_out_doc(task: str, baseline: str, test_set: str, mean: float = 0.9) -> dict:
    return {
        "task": task,
        "baseline": baseline,
        "test_set": test_set,
        "aggregate": {"balanced_accuracy": {"mean": mean, "std": 0.01, "ci95": 0.011}},
        "per_seed": [{"seed": 0, "metrics": {"balanced_accuracy": mean}}],
    }


def _held_out_path(tmp_path: Path, task: str, grid: str, baseline: str, test_set: str) -> Path:
    return (
        tmp_path
        / "outputs"
        / results_version("held_out")
        / task
        / "held_out"
        / grid
        / baseline
        / test_set
        / "result.json"
    )


def _transfer_row(task: str, grid: str, baseline: str, mean: float, source: str):
    from evemtbench.leaderboard import Row

    return Row(
        task=task,
        protocol="transfer_finetune",
        grid=grid,
        baseline=baseline,
        test_set="benchmark",
        version=results_version("held_out"),
        n_test=100,
        seeds=5,
        metrics={"balanced_accuracy": {"mean": mean, "std": 0.01, "ci95": 0.01}},
        source=source,
    )


def test_warn_once_dedups_and_targets_stderr(capsys):
    mr, _ = _load_modules()
    mr.warn_once("x")
    mr.warn_once("x")
    mr.warn_once("y")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("[warn] x") == 1
    assert captured.err.count("[warn] y") == 1


def test_empty_outputs_emits_no_warnings(tmp_path, monkeypatch, capsys):
    """All-\\pending builds stay silent: a cell with no data at all is already visible."""
    _, pl = _load_modules()
    monkeypatch.chdir(tmp_path)
    pl.build_paper_tables()
    pl.build_supplement_tables()
    pl.build_transfer_tables()
    assert capsys.readouterr().err == ""


def test_selection_on_test_fallback_warns(tmp_path, monkeypatch, capsys):
    """A learned baseline with test/benchmark docs but no val re-score renders its number AND
    announces the legacy select-on-reported-set fallback, once per (cell, test_set)."""
    _, pl = _load_modules()
    for ts in ("test", "benchmark"):
        _write_doc(
            _held_out_path(tmp_path, "fault_detection_local", "double_line", "mlp", ts),
            _held_out_doc("fault_detection_local", "mlp", ts),
        )
    monkeypatch.chdir(tmp_path)
    tables = pl.build_paper_tables()
    err = capsys.readouterr().err
    assert "0.900" in tables["tab_res_fd.tex"]  # behavior unchanged: the cell still reports
    assert "selected on the reported set" in err
    for ts in ("test", "benchmark"):
        assert err.count(f"fault_detection_local/{ts}") == 1


def test_val_rescore_silences_fallback(tmp_path, monkeypatch, capsys):
    """With a val re-score present the selection is on validation — nothing to warn about."""
    _, pl = _load_modules()
    for ts in ("val", "test", "benchmark"):
        _write_doc(
            _held_out_path(tmp_path, "fault_detection_local", "double_line", "mlp", ts),
            _held_out_doc("fault_detection_local", "mlp", ts),
        )
    monkeypatch.chdir(tmp_path)
    tables = pl.build_paper_tables()
    assert "0.900" in tables["tab_res_fd.tex"]
    assert capsys.readouterr().err == ""


def test_corrupt_result_json_warns_and_degrades(tmp_path, monkeypatch, capsys):
    mr, _ = _load_modules()
    bad_json = _held_out_path(tmp_path, "fault_detection_local", "double_line", "mlp", "test")
    bad_json.parent.mkdir(parents=True)
    bad_json.write_text("{not json")
    no_task = _held_out_path(tmp_path, "fault_detection_line", "double_line", "mlp", "test")
    _write_doc(no_task, {"baseline": "mlp", "test_set": "test"})  # valid JSON, missing "task"
    monkeypatch.chdir(tmp_path)
    rows = mr.load_rows("double_line")
    assert rows == {}  # both files skipped — graceful degradation kept
    err = capsys.readouterr().err
    # load_rows globs cwd-relative, so the warning names the offending file relative to outputs/
    assert "fault_detection_local/held_out/double_line/mlp/test/result.json" in err
    assert "JSONDecodeError" in err
    assert "KeyError" in err


def test_transfer_fallback_warns_and_keeps_winner(capsys):
    """No val doc for any baseline -> the legacy benchmark-set argmax still wins, loudly."""
    _, pl = _load_modules()
    task, grid, src = "fault_detection_line", "cigre_mv", "template_110kv"
    idx = {
        (task, grid, "transfer_finetune", src): [
            _transfer_row(task, grid, "mlp", 0.80, src),
            _transfer_row(task, grid, "gru", 0.90, src),
        ]
    }
    best = pl.best_transfer_row(idx, task, grid, "transfer_finetune", "balanced_accuracy", src, {})
    assert best is not None and best.baseline == "gru"  # pre-change semantics preserved
    err = capsys.readouterr().err
    assert "no val re-score for any" in err
    assert "legacy fallback" in err
    assert f"{task}/{grid}/transfer_finetune/src={src}" in err


def test_transfer_exclusion_warns_and_selects_on_val(capsys):
    """A baseline with a benchmark row but no val doc is excluded from candidacy — loudly — and
    the cell stays val-selected among the baselines that do have a val doc."""
    _, pl = _load_modules()
    task, grid, src = "fault_detection_line", "cigre_mv", "template_110kv"
    idx = {
        (task, grid, "transfer_finetune", src): [
            _transfer_row(task, grid, "mlp", 0.80, src),
            _transfer_row(task, grid, "gru", 0.90, src),  # better on benchmark, no val doc
        ]
    }
    val_idx = {
        ("transfer_finetune", grid, task, "mlp", src): {
            "aggregate": {"balanced_accuracy": {"mean": 0.85}}
        }
    }
    best = pl.best_transfer_row(
        idx, task, grid, "transfer_finetune", "balanced_accuracy", src, val_idx
    )
    assert best is not None and best.baseline == "mlp"  # val-selected; gru cannot win
    err = capsys.readouterr().err
    assert "baseline 'gru'" in err
    assert "excluded from validation selection" in err
    assert "legacy fallback" not in err


def test_build_deterministic_with_warnings(tmp_path, monkeypatch, capsys):
    """warn_once's dedup state never leaks into rendered bytes: two builds are byte-identical
    and the second one is silent."""
    _, pl = _load_modules()
    _write_doc(
        _held_out_path(tmp_path, "fault_detection_local", "double_line", "mlp", "test"),
        _held_out_doc("fault_detection_local", "mlp", "test"),
    )
    monkeypatch.chdir(tmp_path)
    a = pl.build_paper_tables()
    err_first = capsys.readouterr().err
    b = pl.build_paper_tables()
    err_second = capsys.readouterr().err
    assert a == b
    assert "selected on the reported set" in err_first
    assert err_second == ""
