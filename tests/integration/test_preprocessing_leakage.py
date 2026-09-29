"""Preprocessing-fit leakage guard: every fitted preprocessing statistic a deep baseline carries
(channel standardization mean/std, regression target scaling y_mean/y_std) must be a function of
the TRAIN partition only. The check is behavioral: fit the same baseline on two split roots that
share an identical train partition but swap val and test — every fitted statistic must come out
bit-identical. (Per-window input normalization is stateless by construction; the transfer bundle
then carries these train-fitted stats to the target grid, so nothing is ever re-fit on evaluation
data.) Needs torch."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("torch")

from evemtbench.baselines.mlp import MLPBaseline
from evemtbench.splits.generator import generate_held_out, write_held_out_artifact
from evemtbench.tasks.task import load_task
from evemtbench.version import SPLITS_VERSION


@pytest.fixture(scope="module")
def swapped_split_roots(fixture_root, tmp_path_factory):
    """Two splits roots for double_line/held_out: identical train, val<->test swapped."""
    base = generate_held_out("double_line", root=fixture_root, stratify_by="event_type")
    root_a = tmp_path_factory.mktemp("splits_a")
    root_b = tmp_path_factory.mktemp("splits_b")
    write_held_out_artifact(base, root_a, version=SPLITS_VERSION)
    swapped = json.loads(json.dumps({n: base.splits[n] for n in base.splits}))  # deep copy
    swapped["val"], swapped["test"] = swapped["test"], swapped["val"]
    base.splits = swapped
    write_held_out_artifact(base, root_b, version=SPLITS_VERSION)
    return root_a, root_b


def _fit_stats(task_id: str, fixture_root, splits_root) -> dict:
    task = load_task(
        task_id, "held_out", grid="double_line", root=fixture_root, splits_root=splits_root
    )
    model = MLPBaseline(epochs=1, batch_size=8, hidden_sizes=(8,), device="cpu", seed=0)
    model.fit(task)
    return {
        "mean": np.asarray(model._mean).copy(),
        "std": np.asarray(model._std).copy(),
        "y_mean": float(model._y_mean),
        "y_std": float(model._y_std),
    }


@pytest.mark.parametrize(
    "task_id",
    ["fault_detection_local", "fault_location_local"],  # classification + regression scaling
)
def test_fitted_preprocessing_is_train_only(task_id, fixture_root, swapped_split_roots):
    root_a, root_b = swapped_split_roots
    a = _fit_stats(task_id, fixture_root, root_a)
    b = _fit_stats(task_id, fixture_root, root_b)
    np.testing.assert_array_equal(a["mean"], b["mean"])
    np.testing.assert_array_equal(a["std"], b["std"])
    assert a["y_mean"] == b["y_mean"] and a["y_std"] == b["y_std"], (
        "target scaling depends on val/test content — preprocessing leakage"
    )
