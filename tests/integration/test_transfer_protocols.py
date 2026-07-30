"""Pre-training / cross-grid transfer protocols end-to-end on the synthetic fixture:

* ``load_task`` builds the right views per protocol (multi_grid source, portable local/line widths,
  zero-shot has no target training),
* the runner pre-trains on multi_grid, then zero-shot-evaluates and fine-tunes a reference grid from
  the saved checkpoint (exercises the checkpoint-transfer path).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evemtbench.splits.generator import generate_held_out, write_held_out_artifact
from evemtbench.tasks.task import load_task
from evemtbench.version import BENCHMARK_VERSION, SPLITS_VERSION


@pytest.fixture(scope="module")
def transfer_splits(fixture_root, tmp_path_factory) -> Path:
    """A splits root holding the multi_grid_pretrain split (template_20kv) + the held_out split of
    the double_line target the transfer protocols evaluate on."""
    root = tmp_path_factory.mktemp("transfer_splits")
    pre = generate_held_out(
        "template_20kv",
        protocol="multi_grid_pretrain",
        train_family="multi_grid",
        benchmark_family=None,
        root=fixture_root,
        stratify_by="event_type",
    )
    write_held_out_artifact(pre, root, version=SPLITS_VERSION)
    tgt = generate_held_out("double_line", root=fixture_root, stratify_by="event_type")
    write_held_out_artifact(tgt, root, version=SPLITS_VERSION)
    return root


@pytest.mark.parametrize(
    "task_id,width", [("fault_detection_line", 12), ("fault_detection_local", 6)]
)
def test_pretrain_view_widths(fixture_root, transfer_splits, task_id, width):
    t = load_task(
        task_id, "multi_grid_pretrain", grid="template_20kv",
        root=fixture_root, splits_root=transfer_splits,
    )  # fmt: skip
    assert len(t.train) > 0 and len(t.benchmark) == 0  # in-corpus: no separate benchmark family
    assert t.train.feature_shape[-1] == width  # portable double_line slice


def test_zeroshot_has_no_target_training(fixture_root, transfer_splits):
    t = load_task(
        "fault_detection_line", "transfer_zeroshot", grid="double_line",
        root=fixture_root, splits_root=transfer_splits,
    )  # fmt: skip
    assert len(t.train) == 0 and len(t.val) == 0  # target never trained on
    assert len(t.test) > 0 and len(t.benchmark) > 0
    assert t.benchmark.feature_shape[-1] == 12  # target projected to the portable line view


def test_zeroshot_records_only_eval_partitions(fixture_root, transfer_splits):
    """Provenance honesty: a zero-shot load records exactly the partitions it memory-mapped —
    the target's eval families, never a train-only family it did not read."""
    t = load_task(
        "fault_detection_line", "transfer_zeroshot", grid="double_line",
        root=fixture_root, splits_root=transfer_splits,
    )  # fmt: skip
    parts = t.provenance["partitions"]
    assert parts, "zeroshot load recorded no partitions"
    loaded_families = {p["family"] for p in parts}
    # the fixture's held_out split evaluates adapt_grid test + benchmark family only
    assert "multi_grid" not in loaded_families
    assert t.provenance["splits"]["protocol_dir"] == "held_out"


def test_finetune_trains_on_target_adapt_grid(fixture_root, transfer_splits):
    t = load_task(
        "fault_detection_line", "transfer_finetune", grid="double_line",
        root=fixture_root, splits_root=transfer_splits,
    )  # fmt: skip
    assert len(t.train) > 0 and len(t.benchmark) > 0


def _run(argv):
    from evemtbench.baselines.runner import main

    main(argv)


def _cell(out, task, protocol, grid, baseline="cnn"):
    return Path(out) / BENCHMARK_VERSION / task / protocol / grid / baseline


def test_pretrain_then_zeroshot_and_finetune_roundtrip(fixture_root, transfer_splits, tmp_path):
    """Train a tiny CNN on multi_grid, then transfer to double_line two ways from the checkpoint."""
    pytest.importorskip("torch")
    out = str(tmp_path / "outputs")
    task = "fault_detection_line"
    common = [
        "--baseline", "cnn", "--seeds", "0", "--epochs", "1", "--device", "cpu",
        "--root", str(fixture_root), "--splits-root", str(transfer_splits), "--out", out,
    ]  # fmt: skip

    # 1) pre-train on multi_grid -> writes the transfer bundle
    _run(["--task", task, "--protocol", "multi_grid_pretrain", "--grid", "template_20kv", *common])
    ckpt = _cell(out, task, "multi_grid_pretrain", "template_20kv") / "checkpoints" / "seed0.pt"
    assert ckpt.is_file(), "pretrain must save a transfer bundle"

    # 2) zero-shot on double_line: no training, prediction from the pretrained weights
    _run([
        "--task", task, "--protocol", "transfer_zeroshot", "--grid", "double_line",
        "--pretrain-grid", "template_20kv", *common,
    ])  # fmt: skip
    zdir = _cell(out, task, "transfer_zeroshot", "double_line")
    rec = json.loads((zdir / "run_record.json").read_text())
    assert rec["attestation"]["fit_on"].startswith("none")  # never trained on the target
    assert rec["normalize"] == "per_window"
    assert (zdir / "benchmark" / "result.json").is_file()

    # 3) fine-tune on double_line adapt_grid from the same checkpoint
    _run([
        "--task", task, "--protocol", "transfer_finetune", "--grid", "double_line",
        "--pretrain-grid", "template_20kv", *common,
    ])  # fmt: skip
    fdir = _cell(out, task, "transfer_finetune", "double_line")
    frec = json.loads((fdir / "run_record.json").read_text())
    assert "adapt_grid" in frec["attestation"]["fit_on"]
    assert frec["attestation"]["pretrained_from"] is not None
    assert (fdir / "benchmark" / "result.json").is_file()
