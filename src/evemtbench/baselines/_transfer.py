"""Checkpoint transfer for the deep baselines (pre-training -> cross-grid generalization).

A ``multi_grid_pretrain`` run trains a deep baseline on the pre-training corpus and saves a
**self-contained bundle** (weights + the dims/preprocessing needed to rebuild the model). The
transfer protocols reload it: ``transfer_zeroshot`` rebuilds and evaluates WITHOUT training;
``transfer_finetune`` loads the weights into a freshly built model and continues training on the
target grid. Because the portable LOCAL/LINE view fixes the input width and the class vocabulary is
frozen, the architecture is identical across grids, so ``state_dict`` loads without surgery.

Each deep baseline supplies ``_build_inner(in_ch, out_dim, length) -> nn.Module`` and stores
``self._dims = (in_ch, out_dim, length)`` in ``fit()``. This module operates on those + the shared
preprocessing attributes; it never imports a specific baseline. See ``docs/protocol.md``.
"""

from __future__ import annotations

import numpy as np
import torch

from ..tasks.task import build_input_transform


def transfer_bundle(baseline) -> dict:
    """Serialize a fitted deep baseline into a portable checkpoint bundle."""
    if baseline.model is None or getattr(baseline, "_dims", None) is None:
        raise RuntimeError("transfer_bundle() requires a fitted baseline (call fit() first)")
    in_ch, out_dim, length = baseline._dims
    return {
        "state_dict": {k: v.detach().cpu() for k, v in baseline.model.state_dict().items()},
        "in_ch": int(in_ch),
        "out_dim": int(out_dim),
        "length": int(length),
        "regression": bool(baseline._regression),
        "y_mean": float(baseline._y_mean),
        "y_std": float(baseline._y_std),
        "mean": np.asarray(baseline._mean, dtype=np.float32),
        "std": np.asarray(baseline._std, dtype=np.float32),
        "baseline": baseline.name,
        "normalize": baseline.normalize,
        "zero_seq": baseline.zero_seq,
    }


def restore_pretrained(baseline, bundle: dict) -> None:
    """Rebuild ``baseline.model`` from ``bundle`` and load its weights + preprocessing — NO training
    (zero-shot). Leaves the baseline ready for ``predict()``."""
    baseline._transform = build_input_transform(baseline.zero_seq, baseline.normalize)
    in_ch, out_dim, length = int(bundle["in_ch"]), int(bundle["out_dim"]), int(bundle["length"])
    baseline._regression = bool(bundle["regression"])
    baseline._y_mean, baseline._y_std = float(bundle["y_mean"]), float(bundle["y_std"])
    baseline._mean = np.asarray(bundle["mean"], dtype=np.float32)
    baseline._std = np.asarray(bundle["std"], dtype=np.float32)
    baseline._mean_t = torch.from_numpy(baseline._mean).float().to(baseline.device)
    baseline._std_t = torch.from_numpy(baseline._std).float().to(baseline.device)
    baseline._dims = (in_ch, out_dim, length)
    model = baseline._build_inner(in_ch, out_dim, length).to(baseline.device)
    model.load_state_dict(bundle["state_dict"])
    baseline.model = model
    baseline.n_params_ = int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def reload_held_out(baseline, task, state_dict) -> None:
    """Rebuild a deep baseline from a **bare** ``held_out`` checkpoint (a plain ``state_dict``, as
    written by ``runner._save_checkpoint``, which stores no dims/preprocessing) by recomputing the
    input transform, channel-standardisation stats, criterion/target-scaling, and dims from
    ``task.train`` exactly as ``fit()`` does --- then loading the weights, with NO training. Leaves
    the baseline ready for ``predict()``. Used by the offline validation re-scoring pass so the
    winning baseline can be selected on the validation split rather than on the reported test set.

    Correctness is guarded by the caller's positive control: a reloaded model must reproduce the
    committed ``test`` metric before its ``val`` metric is trusted (a wrong transform/dim/stat
    silently changes predictions, and the control catches it)."""
    from ._train import build_criterion

    train = task.train
    baseline._transform = build_input_transform(baseline.zero_seq, baseline.normalize)
    baseline._mean, baseline._std = train.channel_stats(transform=baseline._transform)
    baseline._mean_t = torch.from_numpy(baseline._mean).float().to(baseline.device)
    baseline._std_t = torch.from_numpy(baseline._std).float().to(baseline.device)
    _crit, out_dim, baseline._regression, baseline._y_mean, baseline._y_std = build_criterion(
        task, train, baseline.device
    )
    if not baseline._regression:
        baseline.num_classes = out_dim
    in_ch = int(baseline._mean.shape[-1])
    _n, length, _c = train.feature_shape
    model = baseline._build_inner(in_ch, out_dim, length).to(baseline.device)
    model.load_state_dict(state_dict)
    baseline.model = model
    baseline._dims = (in_ch, out_dim, length)
    baseline.n_params_ = int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def maybe_load_pretrained(baseline) -> None:
    """Called inside ``fit()`` right after the model is built: if a pretrained ``state_dict`` was
    attached (fine-tune), load it so training continues from the pre-trained weights."""
    state = getattr(baseline, "_pretrained_state", None)
    if state is not None:
        assert baseline.model is not None
        baseline.model.load_state_dict(state)
