"""Selection-evidence invariants for the results/selection path.

A model is selected on validation only. A populated cell without usable validation evidence, or
with an evaluated baseline that has none, fails with ``SelectionEvidenceError`` naming the cell; it
is never selected on the scores it reports, and no baseline leaves the candidate pool unless its
exclusion is on record. Empty cells stay ``None`` (pending). Loud degradations that remain
(``warn_once``: a swallowed corrupt ``result.json``, a recorded exclusion) print a ``[warn]`` line
to stderr while rendered table bytes stay identical."""

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


def _held_out_doc(
    task: str, baseline: str, test_set: str, mean: float = 0.9, grid: str = "double_line"
) -> dict:
    return {
        "task": task,
        "grid": grid,
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


def _transfer_row(
    task: str,
    grid: str,
    baseline: str,
    mean: float,
    source: str,
    protocol: str = "transfer_finetune",
):
    from evemtbench.leaderboard import Row

    return Row(
        task=task,
        protocol=protocol,
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


def test_held_out_without_validation_fails_and_names_the_cell(tmp_path, monkeypatch):
    """Evaluation results but no validation re-score: the publication build fails and names the
    cell. It does not render a number selected on the reported set (the former legacy fallback)."""
    _, pl = _load_modules()
    for ts in ("test", "benchmark"):
        _write_doc(
            _held_out_path(tmp_path, "fault_detection_local", "double_line", "mlp", ts),
            _held_out_doc("fault_detection_local", "mlp", ts),
        )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(pl.SelectionEvidenceError) as exc:
        pl.build_paper_tables()
    assert exc.value.missing == ["mlp"]
    assert exc.value.cell["protocol"] == "held_out"
    assert exc.value.cell["task"] == "fault_detection_local"
    assert exc.value.cell["grid"] == "double_line"
    for part in ("task=fault_detection_local", "grid=double_line", "protocol=held_out"):
        assert part in str(exc.value)


def test_val_rescore_selects_without_warnings(tmp_path, monkeypatch, capsys):
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


def _rows(task: str, scores: dict) -> dict:
    """``{baseline: {test_set: mean}}`` -> the ``load_rows`` shape ``best_cell_choice`` reads."""
    return {
        (task, b): {ts: _held_out_doc(task, b, ts, mean) for ts, mean in by_set.items()}
        for b, by_set in scores.items()
    }


def test_held_out_reports_the_validation_winner_not_the_test_winner():
    _, pl = _load_modules()
    task = "fault_detection_local"
    rows = _rows(task, {"mlp": {"val": 0.90, "test": 0.70}, "gru": {"val": 0.80, "test": 0.95}})
    choice = pl.best_cell_choice(rows, task, "test", "balanced_accuracy")
    assert choice["baseline"] == "mlp" and choice["mean"] == 0.70
    assert choice["selected_on"] == "val"


def test_held_out_incomplete_pool_blocks_unless_the_exclusion_is_recorded(monkeypatch, capsys):
    """An evaluated baseline without a validation score may not drop out of the pool silently."""
    _, pl = _load_modules()
    task = "fault_detection_local"
    rows = _rows(task, {"mlp": {"val": 0.90, "test": 0.70}, "gru": {"test": 0.95}})
    with pytest.raises(pl.SelectionEvidenceError) as exc:
        pl.best_cell_choice(rows, task, "test", "balanced_accuracy")
    assert exc.value.missing == ["gru"]
    monkeypatch.setitem(
        pl.SELECTION_EXCLUSIONS, ("held_out", task, "double_line", None, "gru"), "val re-score lost"
    )
    choice = pl.best_cell_choice(rows, task, "test", "balanced_accuracy")
    assert choice["baseline"] == "mlp"
    err = capsys.readouterr().err
    assert "baseline 'gru' excluded by record: val re-score lost" in err


def test_malformed_validation_record_blocks(tmp_path, monkeypatch, capsys):
    """A corrupt val result.json is skipped loudly by the loader; the cell then has no validation
    evidence and the build fails. A val doc without the metric counts as unusable too."""
    _, pl = _load_modules()
    task, grid = "fault_detection_local", "double_line"
    _write_doc(
        _held_out_path(tmp_path, task, grid, "mlp", "test"), _held_out_doc(task, "mlp", "test")
    )
    bad = _held_out_path(tmp_path, task, grid, "mlp", "val")
    bad.parent.mkdir(parents=True)
    bad.write_text("{not json")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(pl.SelectionEvidenceError):
        pl.build_paper_tables()
    assert "JSONDecodeError" in capsys.readouterr().err
    rows = _rows(task, {"mlp": {"test": 0.9}})
    rows[(task, "mlp")]["val"] = {
        "task": task,
        "baseline": "mlp",
        "test_set": "val",
        "aggregate": {},
    }
    with pytest.raises(pl.SelectionEvidenceError):
        pl.best_cell_choice(rows, task, "test", "balanced_accuracy")


def test_held_out_winner_without_evaluation_is_not_replaced():
    """The validation winner has no result on the reported set: the cell is pending (None). The
    runner-up, which does have one, is not reported in its place."""
    _, pl = _load_modules()
    task = "fault_detection_local"
    rows = _rows(task, {"mlp": {"val": 0.90}, "gru": {"val": 0.80, "test": 0.95}})
    assert pl.best_cell_choice(rows, task, "test", "balanced_accuracy") is None


def test_empty_cell_is_none_not_an_error():
    _, pl = _load_modules()
    task = "fault_detection_local"
    assert pl.best_cell_choice({}, task, "test", "balanced_accuracy") is None
    only_val = _rows(task, {"mlp": {"val": 0.90}})
    assert pl.best_cell_choice(only_val, task, "test", "balanced_accuracy") is None


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


_TASK, _GRID, _SRC = "fault_detection_line", "cigre_mv", "template_110kv"


def _val_key(protocol: str, baseline: str) -> tuple:
    """The validation key each transfer protocol selects on (see ``_transfer_val_index``)."""
    if protocol == "transfer_zeroshot":
        return ("multi_grid_pretrain", _SRC, _TASK, baseline)  # the SOURCE corpus val
    return ("transfer_finetune", _GRID, _TASK, baseline, _SRC)  # the TARGET fine-tune val


def _val_doc(mean: float) -> dict:
    return {"aggregate": {"balanced_accuracy": {"mean": mean}}}


def _transfer_idx(protocol: str, scores: dict) -> dict:
    rows = [_transfer_row(_TASK, _GRID, b, m, _SRC, protocol) for b, m in scores.items()]
    return {(_TASK, _GRID, protocol, _SRC): rows}


def _best(pl, protocol: str, idx: dict, val_idx: dict):
    return pl.best_transfer_row(idx, _TASK, _GRID, protocol, "balanced_accuracy", _SRC, val_idx)


@pytest.mark.parametrize("protocol", ["transfer_zeroshot", "transfer_finetune"])
def test_transfer_without_validation_fails_and_names_the_cell(protocol):
    """No val doc for any baseline: the cell fails. The benchmark argmax is not reported."""
    _, pl = _load_modules()
    idx = _transfer_idx(protocol, {"mlp": 0.80, "gru": 0.90})
    with pytest.raises(pl.SelectionEvidenceError) as exc:
        _best(pl, protocol, idx, {})
    assert exc.value.missing == ["mlp", "gru"]
    for part in (f"task={_TASK}", f"grid={_GRID}", f"protocol={protocol}", f"source={_SRC}"):
        assert part in str(exc.value)


@pytest.mark.parametrize("protocol", ["transfer_zeroshot", "transfer_finetune"])
def test_transfer_selects_on_validation_not_on_the_benchmark(protocol):
    _, pl = _load_modules()
    idx = _transfer_idx(protocol, {"mlp": 0.80, "gru": 0.90})  # gru wins on the benchmark
    val_idx = {_val_key(protocol, "mlp"): _val_doc(0.85), _val_key(protocol, "gru"): _val_doc(0.70)}
    best = _best(pl, protocol, idx, val_idx)
    assert best is not None and best.baseline == "mlp"


@pytest.mark.parametrize(
    ("protocol", "other"),
    [("transfer_zeroshot", "transfer_finetune"), ("transfer_finetune", "transfer_zeroshot")],
)
def test_transfer_protocols_do_not_borrow_each_others_validation(protocol, other):
    """Zero-shot selects on the source corpus val, fine-tuning on the target fine-tune val. With
    only the OTHER protocol's validation present, the cell fails instead of substituting it."""
    _, pl = _load_modules()
    idx = _transfer_idx(protocol, {"mlp": 0.80, "gru": 0.90})
    val_idx = {_val_key(other, "mlp"): _val_doc(0.85), _val_key(other, "gru"): _val_doc(0.70)}
    with pytest.raises(pl.SelectionEvidenceError):
        _best(pl, protocol, idx, val_idx)


def test_transfer_incomplete_pool_blocks_unless_the_exclusion_is_recorded(monkeypatch, capsys):
    """A baseline with a benchmark row but no val doc blocks the cell. With its exclusion on
    record the cell is validation-selected among the rest, and the exclusion is announced."""
    _, pl = _load_modules()
    protocol = "transfer_finetune"
    idx = _transfer_idx(protocol, {"mlp": 0.80, "gru": 0.90})  # gru: better on benchmark, no val
    val_idx = {_val_key(protocol, "mlp"): _val_doc(0.85)}
    with pytest.raises(pl.SelectionEvidenceError) as exc:
        _best(pl, protocol, idx, val_idx)
    assert exc.value.missing == ["gru"]
    monkeypatch.setitem(
        pl.SELECTION_EXCLUSIONS, (protocol, _TASK, _GRID, _SRC, "gru"), "positive control failed"
    )
    best = _best(pl, protocol, idx, val_idx)
    assert best is not None and best.baseline == "mlp"  # val-selected; gru cannot win
    assert "baseline 'gru' excluded by record: positive control failed" in capsys.readouterr().err


def test_transfer_winner_without_benchmark_row_is_not_replaced():
    _, pl = _load_modules()
    protocol = "transfer_finetune"
    idx = _transfer_idx(protocol, {"gru": 0.90})
    val_idx = {_val_key(protocol, "mlp"): _val_doc(0.85), _val_key(protocol, "gru"): _val_doc(0.70)}
    assert _best(pl, protocol, idx, val_idx) is None  # pending; gru is not reported in mlp's place


def test_transfer_empty_cell_is_none_and_unknown_protocol_fails():
    _, pl = _load_modules()
    assert _best(pl, "transfer_finetune", {}, {}) is None
    idx = _transfer_idx("held_out", {"mlp": 0.80})  # a protocol with no transfer validation rule
    with pytest.raises(pl.SelectionEvidenceError):
        _best(pl, "held_out", idx, {})


def test_build_is_deterministic_and_warnings_stay_out_of_the_bytes(tmp_path, monkeypatch, capsys):
    """warn_once's dedup state never leaks into rendered bytes: two builds are byte-identical
    and the second one is silent."""
    _, pl = _load_modules()
    task, grid = "fault_detection_local", "double_line"
    for ts in ("val", "test"):
        _write_doc(_held_out_path(tmp_path, task, grid, "mlp", ts), _held_out_doc(task, "mlp", ts))
    corrupt = _held_out_path(tmp_path, task, grid, "gru", "test")
    corrupt.parent.mkdir(parents=True)
    corrupt.write_text("{not json")
    monkeypatch.chdir(tmp_path)
    a = pl.build_paper_tables()
    err_first = capsys.readouterr().err
    b = pl.build_paper_tables()
    err_second = capsys.readouterr().err
    assert a == b
    assert "JSONDecodeError" in err_first
    assert err_second == ""
