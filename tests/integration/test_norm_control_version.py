"""The normalisation-control builder reads each cell from its CURRENT results tree.

After the v1.2.0 fault-location re-run the FL control cells live in ``<abl>/outputs/1.2.0/`` while
every other task stays in ``1.0.0``; a superseded FL copy left in ``1.0.0`` must never be read (it
was scored under the old valid-sample gate). Fixture trees only — no model is trained."""

import json
from pathlib import Path

from evemtbench.version import results_version

GRID = "double_line"


def _put(root: Path, ver: str, task: str, value: float) -> None:
    for test_set in ("val", "benchmark"):
        p = root / "outputs" / ver / task / "held_out" / GRID / "mlp" / test_set / "result.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        metric = "mae" if task.startswith("fault_location") else "balanced_accuracy"
        doc = {
            "task": task,
            "baseline": "mlp",
            "grid": GRID,
            "test_set": test_set,
            "aggregate": {metric: {"mean": value}},
        }
        p.write_text(json.dumps(doc))


def _trees(tmp_path: Path) -> tuple[Path, Path]:
    repo, abl = tmp_path / "repo", tmp_path / "abl"
    for root, base in ((repo, 0.0), (abl, 0.5)):
        _put(root, "1.0.0", "fault_detection_local", base + 0.1)
        _put(root, "1.0.0", "fault_location_local", base + 9.0)  # superseded FL copy
        _put(root, "1.2.0", "fault_location_local", base + 2.0)  # current FL cell
    return repo, abl


def test_version_api_routes_fl_to_1_2_0():
    assert results_version("held_out", "fault_location_local") == "1.2.0"
    assert results_version("held_out", "fault_detection_local") == "1.0.0"


def test_rows_read_fl_from_1_2_0_and_the_rest_from_1_0_0(tmp_path, monkeypatch):
    import build_supp_tab_norm_control as b

    repo, abl = _trees(tmp_path)
    monkeypatch.chdir(repo)
    recs = {(r["key"], r["grid"], r["view"]): r for r in b.rows(abl)}
    assert Path.cwd() == repo  # rows() restores cwd after reading the control tree
    fd, fl = recs[("fd", GRID, "local")], recs[("fl", GRID, "local")]
    assert (fd["A"], fd["C"]) == (0.1, 0.6)  # non-FL: 1.0.0, unchanged
    assert (fl["A"], fl["C"]) == (2.0, 2.5)  # FL: 1.2.0, never the superseded 1.0.0 copy


def test_provenance_names_the_per_task_trees(tmp_path, monkeypatch):
    import build_supp_tab_norm_control as b

    repo, abl = _trees(tmp_path)
    monkeypatch.chdir(repo)
    header = b.table(abl).split("\\begin{table}")[0]
    ho = "{fault_detection,fault_classification}_{local,line}/held_out/"
    assert f"outputs/1.0.0/{ho}" in header
    assert "outputs/1.2.0/fault_location_{local,line}/held_out/" in header
    assert f"<control tree>/outputs/1.0.0/{ho}" in header
    assert "<control tree>/outputs/1.2.0/fault_location_{local,line}/held_out/" in header
    tf = "_{local,line}/transfer_finetune/"
    assert f"outputs/1.1.0/{{fault_detection,fault_classification}}{tf}" in header
    assert f"outputs/1.2.0/fault_location{tf}" in header
    assert "fault_location}_{local,line}" not in header  # no FL under a 1.0.0/1.1.0 literal
