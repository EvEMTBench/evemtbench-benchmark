"""Protection-grade analysis: detection latency (binary) + location MAE curve (regression)."""

from evemtbench import analysis
from evemtbench.baselines.runner import main as run_main


def _run_cell(task, baseline, out, fixture_root, held_out_splits):
    run_main(
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
            "0",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--device",
            "cpu",
            "--root",
            str(fixture_root),
            "--splits-root",
            str(held_out_splits),
            "--out",
            str(out),
        ]
    )


def test_detection_latency_on_fd(tmp_path, fixture_root, held_out_splits):
    out = tmp_path / "outputs"
    _run_cell("fault_detection_local", "majority", out, fixture_root, held_out_splits)
    res = analysis.analyze_cell(
        out,
        "fault_detection_local",
        "held_out",
        "double_line",
        "majority",
        "test",
        root=fixture_root,
    )
    lat = res["latency"]
    assert lat["n_fault_episodes"] > 0
    assert 0.0 <= lat["detection_rate"] <= 1.0
    # latency, when defined, is a non-negative time from inception (dt_start >= 0)
    assert lat["median_latency_s"] is None or lat["median_latency_s"] >= 0.0


def test_location_curve_on_fl(tmp_path, fixture_root, held_out_splits):
    out = tmp_path / "outputs"
    _run_cell("fault_location_line", "majority", out, fixture_root, held_out_splits)
    res = analysis.analyze_cell(
        out,
        "fault_location_line",
        "held_out",
        "double_line",
        "majority",
        "benchmark",
        root=fixture_root,
    )
    curve = res["location_mae_curve"]
    assert isinstance(curve, list) and curve  # an empty curve must not pass vacuously
    for row in curve:
        assert row["n"] > 0 and row["mae"] >= 0.0


def test_no_latency_for_non_detection_binary_tasks(tmp_path, fixture_root, held_out_splits):
    # fault_grounded's positive class is "grounded", not "detected": no latency is reported
    out = tmp_path / "outputs"
    _run_cell("fault_grounded_local", "majority", out, fixture_root, held_out_splits)
    res = analysis.analyze_cell(
        out, "fault_grounded_local", "held_out", "double_line", "majority", "test",
        root=fixture_root,
    )  # fmt: skip
    assert "latency" not in res
