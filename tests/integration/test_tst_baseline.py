"""Time Series Transformer baseline smoke — classification (FD, FC) and regression (FL) paths on
the fixture, seed determinism, checkpoint round trip, and transfer-bundle save/restore. Needs torch.

The fixture window is 8 samples long, shorter than the default 16-sample patch, so the fixture runs
pass a small patch geometry (4 samples, stride 2 -> 3 tokens). The default 480/16/8 geometry
(59 tokens) is covered by the unit tests in test_tst_module.py only."""

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from evemtbench.baselines import _transfer
from evemtbench.baselines.tst import TSTBaseline
from evemtbench.evaluation.evaluate import evaluate
from evemtbench.tasks.task import load_task

TINY = dict(d_model=16, n_heads=2, n_layers=2, dim_ff=32, patch_len=4, stride=2)


def _load(task_id, fixture_root, held_out_splits):
    return load_task(
        task_id, "held_out", grid="double_line", root=fixture_root, splits_root=held_out_splits
    )


def _model(seed=0, epochs=2):
    return TSTBaseline(epochs=epochs, batch_size=8, device="cpu", seed=seed, **TINY)


@pytest.mark.parametrize(
    "task_id", ["fault_detection_local", "fault_detection_line", "fault_detection_global"]
)
def test_tst_classification_path(fixture_root, held_out_splits, task_id):
    task = _load(task_id, fixture_root, held_out_splits)
    model = _model()
    model.fit(task)
    assert model._mean is not None and not model._regression
    assert model.model is not None and model.model.n_tokens == 3
    assert model.n_params_ and model.n_params_ > 0  # capacity recorded
    assert model.train_stats_.get("effective_epochs") == 2  # no early stop in 2 epochs
    preds = model.predict(task.test)
    result = evaluate(task, preds, test_set="test")
    assert result.n_samples == len(task.test)
    assert set(preds["prediction"].unique()) <= {0, 1}


def test_tst_multiclass_path(fixture_root, held_out_splits):
    task = _load("fault_classification_local", fixture_root, held_out_splits)
    model = _model()
    model.fit(task)
    assert not model._regression and model._dims[1] == model.num_classes > 2
    preds = model.predict(task.test)
    assert evaluate(task, preds, test_set="test").n_samples == len(task.test)
    assert set(preds["prediction"].unique()) <= set(range(model.num_classes))


def test_tst_regression_path(fixture_root, held_out_splits):
    task = _load("fault_location_line", fixture_root, held_out_splits)
    model = _model()
    model.fit(task)
    assert model._regression and model._dims[1] == 1  # MSE head + target standardization
    preds = model.predict(task.benchmark)
    result = evaluate(task, preds, test_set="benchmark")
    assert "mae" in result.metrics
    assert np.issubdtype(preds["prediction"].to_numpy().dtype, np.floating)


def test_tst_fit_is_deterministic_under_a_fixed_seed_on_cpu(fixture_root, held_out_splits):
    task = _load("fault_detection_local", fixture_root, held_out_splits)
    a, b = _model(seed=3), _model(seed=3)
    a.fit(task)
    b.fit(task)
    for (ka, va), (kb, vb) in zip(
        a.model.state_dict().items(), b.model.state_dict().items(), strict=True
    ):
        assert ka == kb and torch.equal(va, vb), ka
    pa, pb = a.predict(task.test), b.predict(task.test)
    assert pa["prediction"].tolist() == pb["prediction"].tolist()


def test_tst_checkpoint_round_trips_into_a_fresh_model(fixture_root, held_out_splits):
    task = _load("fault_detection_global", fixture_root, held_out_splits)
    model = _model()
    model.fit(task)
    in_ch, out_dim, length = model._dims
    fresh = _model()._build_inner(in_ch, out_dim, length)
    fresh.load_state_dict(model.model.state_dict())  # exact match or this raises
    fresh.eval()
    model.model.eval()
    torch.manual_seed(0)
    xs = (torch.randn(4, length, in_ch) - model._mean_t) / model._std_t
    with torch.no_grad():
        assert torch.allclose(model.model(xs), fresh(xs), atol=1e-6)


@pytest.mark.parametrize("task_id", ["fault_detection_local", "fault_location_line"])
def test_tst_transfer_bundle_restores_identical_predictions(
    fixture_root, held_out_splits, tmp_path, task_id
):
    """The pretrain -> zero-shot mechanics: a saved bundle rebuilds the model with no training and
    reproduces the fitted model's predictions; attached as ``_pretrained_state`` it is loaded by
    ``fit()`` (fine-tune)."""
    task = _load(task_id, fixture_root, held_out_splits)
    src = _model(seed=1)
    src.fit(task)
    path = tmp_path / "seed1.pt"
    torch.save(_transfer.transfer_bundle(src), path)
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    assert bundle["baseline"] == "tst" and bundle["length"] == src._dims[2]

    zero = _model(seed=99)
    _transfer.check_bundle_preprocessing(zero, bundle)
    _transfer.restore_pretrained(zero, bundle)
    assert zero.n_params_ == src.n_params_
    got, want = zero.predict(task.test), src.predict(task.test)
    np.testing.assert_allclose(
        got["prediction"].to_numpy(), want["prediction"].to_numpy(), rtol=1e-5, atol=1e-6
    )

    # fine-tune: with zero epochs fit() only builds the model and loads the attached weights
    tuned = _model(seed=5, epochs=0)
    tuned._pretrained_state = bundle["state_dict"]
    tuned.fit(task)
    for k, v in tuned.model.state_dict().items():
        assert torch.equal(v, bundle["state_dict"][k]), k
