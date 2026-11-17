"""The robustness page's validation-selection reliability section renders from held_out rows,
grades the validation pick on the held-out sets, and never re-selects on them."""

import make_audit
from evemtbench.leaderboard import Row
from evemtbench.version import results_version

TASK = "fault_detection_line"


def _row(baseline, test_set, value, grid="cigre_mv"):
    return Row(
        task=TASK,
        protocol="held_out",
        grid=grid,
        baseline=baseline,
        test_set=test_set,
        version=results_version("held_out"),
        n_test=10,
        seeds=5,
        metrics={"balanced_accuracy": {"mean": value}},
    )


def test_section_reports_the_val_pick_and_its_regret():
    rows = [
        _row("mlp", "val", 0.95),
        _row("gru", "val", 0.90),
        _row("mlp", "test", 0.85),
        _row("gru", "test", 0.91),
        _row("mlp", "benchmark", 0.60),
        _row("gru", "benchmark", 0.55),
    ]
    html = make_audit.build_selection_reliability(rows)
    assert "Validation-selection reliability" in html
    assert "<td class=base>mlp</td>" in html  # picked on val, not the test winner (gru)
    assert "<td>0.060</td>" in html  # test regret of the val pick
    assert "0</b>of 1 cells: validation pick is also best on <code>test</code>" in html
    assert "1</b>of 1 cells: validation pick is also best on <code>benchmark</code>" in html


def test_cells_without_validation_rescore_are_left_out():
    html = make_audit.build_selection_reliability([_row("mlp", "test", 0.9)])
    assert "No held_out cells with validation re-scores" in html
