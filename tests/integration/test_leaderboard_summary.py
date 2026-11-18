"""The user-facing leaderboard: skill score, ranking rules, partial entries, page wiring."""

import json
import re

import pytest

from evemtbench import leaderboard_summary
from evemtbench.leaderboard import Row, full_path, main, render_pages
from evemtbench.leaderboard_summary import build_summary, build_table, skill
from evemtbench.version import results_version

GRIDS = ("double_line", "cigre_mv", "ieee39")


def _row(task, grid, baseline, *, ba=None, mae=None, protocol="held_out", test_set="benchmark",
         n_test=100, source=None, tier=None):  # fmt: skip
    metrics = {}
    if ba is not None:
        metrics["balanced_accuracy"] = {"mean": ba, "std": 0.01, "ci95": 0.02}
    if mae is not None:
        metrics["mae"] = {"mean": mae, "std": 0.1, "ci95": 0.2}
    return Row(task=task, protocol=protocol, grid=grid, baseline=baseline, test_set=test_set,
               version=results_version(protocol), n_test=n_test, seeds=3, metrics=metrics,
               source=source, tier=tier)  # fmt: skip


def _board(models: dict[str, float], *, grids=GRIDS, task="fault_detection_line") -> list[Row]:
    """majority at BA 0.5 (error 0.5) plus each model at the given BA on every grid."""
    rows = [_row(task, g, "majority", ba=0.5) for g in grids]
    rows += [_row(task, g, m, ba=ba) for m, ba in models.items() for g in grids]
    return rows


def test_skill_is_one_minus_geometric_mean_of_error_ratios():
    assert skill([0.5, 0.5]) == pytest.approx(0.5)
    assert skill([0.25, 1.0]) == pytest.approx(0.5)  # geometric, not arithmetic (0.375)
    assert skill([]) is None


def test_reference_scores_zero_and_halved_error_scores_half():
    t = build_table(_board({"cnn": 0.75}), "held_out", "benchmark", "line")
    (e,) = t["ranked"]
    assert e["model"] == "cnn" and e["skill"] == pytest.approx(0.5)
    assert "majority" not in {x["model"] for x in t["ranked"] + t["partial"]}


def test_existing_scores_do_not_move_when_an_entry_is_added():
    a = build_table(_board({"cnn": 0.75, "mlp": 0.6}), "held_out", "benchmark", "line")
    b = build_table(_board({"cnn": 0.75, "mlp": 0.6, "new": 0.9}), "held_out", "benchmark", "line")
    sa = {e["model"]: e["skill"] for e in a["ranked"]}
    sb = {e["model"]: e["skill"] for e in b["ranked"]}
    assert sa["cnn"] == sb["cnn"] and sa["mlp"] == sb["mlp"]
    assert [e["model"] for e in b["ranked"]] == ["new", "cnn", "mlp"]


def test_win_rate_counts_shared_cells_against_ranked_entries():
    t = build_table(_board({"cnn": 0.9, "mlp": 0.6}), "held_out", "benchmark", "line")
    wr = {e["model"]: e["win_rate"] for e in t["ranked"]}
    assert wr == {"cnn": 1.0, "mlp": 0.0}


def test_partial_entry_is_listed_apart_with_a_same_cells_comparison():
    rows = _board({"cnn": 0.75, "mlp": 0.6})
    rows.append(_row("fault_detection_line", "double_line", "probe", ba=0.8, tier="self-reported"))
    t = build_table(rows, "held_out", "benchmark", "line")
    assert [e["model"] for e in t["ranked"]] == ["cnn", "mlp"]
    (p,) = t["partial"]
    assert p["model"] == "probe" and p["tier"] == "self-reported" and p["cells"] == 1
    assert p["best_ranked_same_cells"]["model"] == "cnn"


def test_regression_uses_mae_against_the_mean_predictor():
    rows = [_row("fault_location_line", g, "majority", mae=20.0) for g in GRIDS]
    rows += [_row("fault_location_line", g, "mlp", mae=5.0) for g in GRIDS]
    t = build_table(rows, "held_out", "benchmark", "line")
    assert t["ranked"][0]["skill"] == pytest.approx(0.75)
    assert t["detail"]["fault_location"]["metric"] == "mae"


def test_transfer_reference_is_the_held_out_majority_on_the_same_eval_set():
    rows = [_row("fault_detection_line", g, "majority", ba=0.5) for g in GRIDS]
    for g in GRIDS:
        rows.append(_row("fault_detection_line", g, "cnn", ba=0.75, protocol="transfer_zeroshot",
                         source="template_20kv"))  # fmt: skip
    # a cell whose evaluation set differs in size from the reference is left out
    rows.append(_row("fault_detection_line", "ieee39", "cnn", ba=0.9, protocol="transfer_zeroshot",
                     source="template_110kv", n_test=7))  # fmt: skip
    t = build_table(rows, "transfer_zeroshot", "benchmark", "line")
    assert t["n_cells"] == 3 and t["n_mismatched"] == 1
    assert t["ranked"][0]["skill"] == pytest.approx(0.5)


def test_rank_range_is_shared_when_intervals_overlap():
    rows = _board({"a": 0.80, "b": 0.79})
    # make the cells disagree so the intervals have width
    rows.append(_row("fault_detection_line", "x", "majority", ba=0.5))
    rows.append(_row("fault_detection_line", "x", "a", ba=0.6))
    rows.append(_row("fault_detection_line", "x", "b", ba=0.95))
    t = build_table(rows, "held_out", "benchmark", "line")
    for e in t["ranked"]:
        assert e["rank_range"] == [1, 2]


def test_summary_covers_combinations_with_results_only():
    s = build_summary(_board({"cnn": 0.75}))
    assert set(s["tables"]) == {"held_out|benchmark|line"}
    assert s["default"] == "held_out|benchmark|line"


def test_pages_are_deterministic_and_link_each_other():
    rows = _board({"cnn": 0.75})
    summary, full = render_pages(rows, generated_at=None)
    assert (summary, full) == render_pages(rows, generated_at=None)
    assert 'href="leaderboard_full.html"' in summary
    assert 'href="leaderboard.html"' in full and "EvEMTBench full results" in full
    data = json.loads(re.search(r"id=data>(.*?)</script>", summary, re.S).group(1))
    assert data["tables"]["held_out|benchmark|line"]["ranked"][0]["model"] == "cnn"
    assert "<div class=v>updated" not in summary
    s2, _ = render_pages(rows, generated_at="2026-11-11 10:00 UTC")
    assert "<div class=v>updated 2026-11-11 10:00 UTC</div>" in s2


def test_embedded_json_cannot_close_the_script_tag():
    summary, _ = render_pages(_board({"x</script><b>": 0.75}), generated_at=None)
    assert "x</script><b>" not in summary


def test_main_writes_both_pages(tmp_path):
    out = tmp_path / "leaderboard.html"
    main(["--outputs", str(tmp_path / "none"), "--out", str(out),
          "--submissions", str(tmp_path / "none")])  # fmt: skip
    assert out.is_file() and full_path(out).is_file()
    assert full_path(out).name == "leaderboard_full.html"


def test_a_consistent_small_lead_separates_the_rank_ranges():
    # a beats b slightly on every cell: unpaired intervals overlap, the paired test separates them
    t = build_table(_board({"a": 0.80, "b": 0.79}), "held_out", "benchmark", "line")
    assert {e["model"]: e["rank_range"] for e in t["ranked"]} == {"a": [1, 1], "b": [2, 2]}


def test_page_without_results_says_so():
    assert build_summary([])["tables"] == {}
    summary, _ = render_pages([], generated_at=None)
    assert "No results to rank yet" in summary


def test_family_missing_from_the_label_list_does_not_break_the_build(monkeypatch):
    monkeypatch.delitem(leaderboard_summary.FAMILY_LABELS, "fault_detection")
    t = build_table(_board({"cnn": 0.75}), "held_out", "benchmark", "line")
    assert t["families"] == ["fault_detection"]


def test_non_finite_cell_is_not_scored_and_does_not_crash():
    rows = _board({"cnn": 0.75, "mlp": 0.6})
    rows.append(_row("fault_detection_line", "x", "majority", ba=0.5))
    rows.append(_row("fault_detection_line", "x", "cnn", ba=0.8))
    rows.append(_row("fault_detection_line", "x", "mlp", ba=float("nan")))
    t = build_table(rows, "held_out", "benchmark", "line")
    assert [e["model"] for e in t["ranked"]] == ["cnn"]
    assert [e["model"] for e in t["partial"]] == ["mlp"]


def test_submission_named_like_an_in_house_model_or_the_reference_stays_apart():
    rows = _board({"cnn": 0.75})
    rows += [_row("fault_detection_line", g, "cnn", ba=0.95, tier="self-reported") for g in GRIDS]
    rows += [_row("fault_detection_line", g, "majority", ba=0.99, tier="self-reported")
             for g in GRIDS]  # fmt: skip
    t = build_table(rows, "held_out", "benchmark", "line")
    by = {e["model"]: e for e in t["ranked"]}
    assert by["cnn"]["skill"] == pytest.approx(0.5) and by["cnn"]["tier"] is None
    assert by["cnn (submitted)"]["tier"] == "self-reported"
    assert by["majority (submitted)"]["skill"] == pytest.approx(0.98)


def test_drill_down_carries_the_metric_direction_and_raw_reference():
    t = build_table(_board({"cnn": 0.75}), "held_out", "benchmark", "line")
    d = t["detail"]["fault_detection"]
    assert d["higher_is_better"] is True and d["reference"] == [0.5, 0.5, 0.5]


def test_vendored_libraries_match_their_pinned_hashes():
    import hashlib
    from importlib.resources import files

    root = files("evemtbench").joinpath("_vendor")
    manifest = json.loads(root.joinpath("VENDOR.json").read_text())
    for name, meta in manifest.items():
        assert hashlib.sha256(root.joinpath(name).read_bytes()).hexdigest() == meta["sha256"], name
        assert root.joinpath(meta["license_file"]).is_file()


def test_page_is_self_contained_and_carries_the_licence_notices():
    summary, _ = render_pages(_board({"cnn": 0.75}), generated_at=None)
    assert "<script src" not in summary and "<link rel=stylesheet" not in summary
    for lib in ("tabulator-tables 6.3.1 (MIT)", "vega 5.30.0 (BSD-3-Clause)",
                "vega-lite 5.21.0 (BSD-3-Clause)", "vega-embed 6.26.0 (BSD-3-Clause)"):  # fmt: skip
        assert lib in summary
    notices = summary[summary.rindex("<!--") :]
    assert notices.count("-->") == 1  # nothing inside the notices closes the comment early


def test_libraries_load_in_dependency_order():
    from importlib.resources import files

    from evemtbench.leaderboard_summary import vendored_assets

    root = files("evemtbench").joinpath("_vendor")
    _, js, _ = vendored_assets()
    order = ("tabulator.min.js", "vega.min.js", "vega-lite.min.js", "vega-embed.min.js")
    pos = [js.index(root.joinpath(n).read_text(encoding="utf-8")) for n in order]
    assert pos == sorted(pos)  # vega-embed reads the vega and vegaLite globals when it loads


def test_page_script_is_valid_javascript(tmp_path):
    """Our hand-written JS is never executed by the Python tests; check its syntax with node
    where it exists (GitHub CI runners have it; the HPC login nodes do not)."""
    import shutil
    import subprocess

    from evemtbench.leaderboard_summary import JS

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    f = tmp_path / "page.js"
    f.write_text(JS)
    res = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
