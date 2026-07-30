"""Checkpoints + compute telemetry: the sweep must save best-val weights and record cost so that
robustness/latency analyses can be added post-hoc (no re-training) and a GPU-hours story exists."""

import json

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from evemtbench import BENCHMARK_VERSION  # noqa: E402
from evemtbench.baselines import runner  # noqa: E402
from evemtbench.baselines.resnet import ResNetBaseline, _ResNet  # noqa: E402
from evemtbench.tasks.task import load_task  # noqa: E402


def _run(baseline, task, fixture_root, held_out_splits, out, epochs="1"):
    runner.main(
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
            epochs,
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


def test_deep_baseline_writes_checkpoint_and_telemetry(fixture_root, held_out_splits, tmp_path):
    out = tmp_path / "out"
    _run("resnet", "fault_detection_local", fixture_root, held_out_splits, out)
    base = out / BENCHMARK_VERSION / "fault_detection_local" / "held_out" / "double_line" / "resnet"

    # best-val checkpoint saved for the seed, and reloadable as a real torch state_dict
    ckpt = base / "checkpoints" / "seed0.pt"
    assert ckpt.is_file()
    assert isinstance(torch.load(ckpt, map_location="cpu"), dict)

    # run_record provenance: params + compute + checkpoint sha
    rr = json.loads((base / "run_record.json").read_text())
    assert isinstance(rr["n_params"], int) and rr["n_params"] > 0
    assert rr["compute"]["fit_seconds_mean"] > 0
    assert rr["compute"]["effective_epochs_mean"] == 1.0
    assert len(rr["checkpoints"]["0"]["sha256"]) == 64
    assert rr["checkpoints"]["0"]["path"] == "checkpoints/seed0.pt"

    # result.json carries the same capacity/compute for the builders
    res = json.loads((base / "test" / "result.json").read_text())
    assert res["n_params"] == rr["n_params"]
    assert res["compute"]["n_seeds"] == 1


def test_classical_baseline_has_no_checkpoint_or_params(fixture_root, held_out_splits, tmp_path):
    out = tmp_path / "out"
    _run("majority", "fault_detection_local", fixture_root, held_out_splits, out)
    base = (
        out / BENCHMARK_VERSION / "fault_detection_local" / "held_out" / "double_line" / "majority"
    )
    rr = json.loads((base / "run_record.json").read_text())
    assert rr["n_params"] is None  # non-parametric — reported as "—", not 0
    assert rr["checkpoints"] == {}
    assert not (base / "checkpoints").exists()
    # wall-time is still captured for every baseline
    assert rr["compute"]["fit_seconds_mean"] is not None


def test_checkpoint_round_trips_into_a_fresh_model(fixture_root, held_out_splits):
    """The saved state_dict must fully reconstruct the model (all keys/shapes) and reproduce
    outputs — the guarantee that released checkpoints enable post-hoc analysis, no re-training."""
    task = load_task(
        "fault_detection_global",
        "held_out",
        grid="double_line",
        root=fixture_root,
        splits_root=held_out_splits,
    )
    model = ResNetBaseline(epochs=2, batch_size=8, width=8, device="cpu", seed=0)
    model.fit(task)  # best-val weights restored into model.model

    in_ch = int(model._mean.shape[-1])
    length = int(task.test.feature_shape[1])
    fresh = _ResNet(in_ch, model.num_classes, width=8, dropout=0.3)
    fresh.load_state_dict(model.model.state_dict())  # exact match or this raises
    fresh.eval()
    model.model.eval()

    torch.manual_seed(0)
    xb = torch.randn(4, length, in_ch)
    xs = (xb - model._mean_t) / model._std_t
    with torch.no_grad():
        assert torch.allclose(model.model(xs), fresh(xs), atol=1e-6)
