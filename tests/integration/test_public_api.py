"""The BYO surface: top-level exports, load_arrays guard, and the baseline plugin registry."""

import json

import numpy as np
import pandas as pd
import pytest

import evemtbench
from evemtbench import TaskDefinition, get_task, list_tasks, load_arrays, results_version


def test_evaluate_export_survives_submodule_import():
    """Regression: ``evaluation.evaluate`` is BOTH a function and a submodule name; importing
    the submodule (the runner does) must not shadow the function on the package — this exact
    order crashed examples/byo_model.py with "'module' object is not callable"."""
    import types

    import evemtbench.baselines.runner  # imports evaluation.evaluate (the trigger)
    from evemtbench.evaluation import evaluate as pkg_evaluate

    assert callable(pkg_evaluate) and not isinstance(pkg_evaluate, types.ModuleType)
    assert callable(evemtbench.evaluate)


def test_top_level_exports():
    tasks = list_tasks()
    assert len(tasks) == 24
    td = get_task("fault_detection_local")
    assert isinstance(td, TaskDefinition) and td.task_id == "fault_detection_local"
    assert sorted(evemtbench.__all__) == evemtbench.__all__  # keep the API list tidy
    with pytest.raises(AttributeError):
        _ = evemtbench.not_a_symbol


def test_load_arrays_shapes_and_alignment(fixture_root, held_out_splits):
    task = evemtbench.load_task(
        "fault_detection_local",
        "held_out",
        grid="double_line",
        root=fixture_root,
        splits_root=held_out_splits,
    )
    X, y, meta = load_arrays(task, "test")
    assert X.dtype == np.float32 and X.ndim == 3
    assert X.shape[0] == len(y) == len(meta)
    assert X.shape[2] == 6  # local observability = one cubicle
    assert {"sample_id", "window_idx"} <= set(meta.columns)


def test_load_arrays_guards_big_views(fixture_root, held_out_splits):
    task = evemtbench.load_task(
        "fault_detection_local",
        "held_out",
        grid="double_line",
        root=fixture_root,
        splits_root=held_out_splits,
    )
    with pytest.raises(MemoryError, match="iter_batches"):
        load_arrays(task, "train", max_bytes=8)  # absurdly small cap -> loud, actionable error


class _ConstantBaseline:
    """Minimal third-party Baseline: predicts a constant class (registry plumbing test)."""

    name = "byo_constant"

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = seed

    def fit(self, task) -> None:
        pass

    def predict(self, view) -> pd.DataFrame:
        from evemtbench.evaluation.predictions_io import predictions_frame

        return predictions_frame(
            view.metadata["sample_id"], view.metadata["window_idx"], np.zeros(len(view), int)
        )


def test_register_baseline_runs_through_runner(fixture_root, held_out_splits, tmp_path):
    from evemtbench.baselines import runner
    from evemtbench.baselines.interface import Baseline

    assert isinstance(_ConstantBaseline(), Baseline)  # protocol is runtime-checkable
    runner.register_baseline("byo_constant", lambda *, seed, **_: _ConstantBaseline(seed=seed))
    try:
        out = tmp_path / "out"
        runner.main(
            [
                *("--task", "fault_detection_local", "--grid", "double_line"),
                *("--baseline", "byo_constant", "--seeds", "0", "1"),
                *("--root", str(fixture_root), "--splits-root", str(held_out_splits)),
                *("--out", str(out)),
            ]
        )
        base = (
            out
            / results_version("held_out")
            / "fault_detection_local"
            / "held_out"
            / "double_line"
            / "byo_constant"
        )
        result = json.loads((base / "test" / "result.json").read_text())
        # the plugin gets the SAME auditable artifact shape as the built-ins
        assert result["baseline"] == "byo_constant"
        agg = result["aggregate"]["balanced_accuracy"]
        assert {"mean", "std", "ci95", "n_defined", "n_seeds"} <= set(agg)
        assert json.loads((base / "run_record.json").read_text())["attestation"]
    finally:
        runner.BASELINE_FACTORIES.pop("byo_constant", None)  # keep the registry test-clean


def test_register_baseline_rejects_duplicates_and_unknown():
    from evemtbench.baselines import runner

    with pytest.raises(ValueError, match="already registered"):
        runner.register_baseline("majority", lambda **_: None)
    with pytest.raises(ValueError, match="unknown baseline"):
        runner._make_baseline(
            "nope", seed=0, device=None, epochs=1, batch_size=8, lr=1e-3, zero_seq=False
        )
