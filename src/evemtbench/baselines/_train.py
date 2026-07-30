"""Shared training regime for the deep baselines (mlp / gru / cnn). Requires the ``torch`` extra.

Single source of truth so the three learned baselines differ ONLY in architecture, not in how they
are optimized — a benchmark-fairness requirement. Provides:

* ``build_criterion`` — the task-kind branch (class-weighted CE for classification; MSE + target
  standardization for regression), identical for every learned baseline.
* ``train_supervised`` — the canonical loop: **AdamW + weight decay**, **CosineAnnealingLR**, and
  **early stopping on the validation split** (best-val-loss checkpoint, patience). Per-batch
  standardization happens on-device via the caller's ``sx`` closure (memory-lean for the wide
  ``global`` view); the best-val weights are restored before returning.

Determinism note: seeds are set by the caller's ``fit`` (torch + numpy) before model construction;
this loop draws its epoch-shuffle order from ``np.random.default_rng(seed)`` to match the historical
MLP behavior. GPU kernels are not bit-identical (reported as mean ± std over seeds) — see
docs/determinism.md.
"""

from __future__ import annotations

import copy
import os
from contextlib import nullcontext
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn

from ..tasks.task import Task, TaskView
from ..tasks.taxonomy import LabelKind
from ._loader import (
    h2d,
    loader_reuse_enabled,
    multi_epoch_batches,
    prefetched_batches,
    reusable_batches,
)
from ._log import emit_epoch, epoch_line, log

# Opt-in NVTX ranges for Nsight Systems (EVEMTBENCH_NVTX=1). No-op otherwise (zero overhead on
# normal runs); lets `nsys profile` label data-wait vs forward/backward/step for all deep baselines.
_NVTX = os.environ.get("EVEMTBENCH_NVTX", "") not in ("", "0", "false", "False")


def _nvtx(name: str):
    return torch.cuda.nvtx.range(name) if (_NVTX and torch.cuda.is_available()) else nullcontext()


def build_criterion(task: Task, train: TaskView, device: str) -> tuple:
    """Return ``(criterion, out_dim, regression, y_mean, y_std)`` for ``task``.

    Classification: ``CrossEntropyLoss`` with inverse-frequency class weights (robust to the heavy
    negatives≫positives imbalance). Regression: ``MSELoss`` on standardized targets — the caller
    de-standardizes predictions with the returned ``(y_mean, y_std)``.
    """
    if task.definition.label_kind is LabelKind.REGRESSION:
        y_raw = np.asarray(train.labels).astype(np.float32)
        y_mean = float(y_raw.mean())
        y_std = float(y_raw.std()) + 1e-6
        return nn.MSELoss(), 1, True, y_mean, y_std
    out_dim = task.definition.num_classes or int(np.max(train.labels)) + 1
    counts = np.bincount(np.asarray(train.labels).astype(int), minlength=out_dim)
    weights = torch.tensor(
        len(train.labels) / (out_dim * np.maximum(counts, 1)), dtype=torch.float32
    ).to(device)
    return nn.CrossEntropyLoss(weight=weights), out_dim, False, 0.0, 1.0


def train_supervised(
    *,
    model: nn.Module,
    train_view: TaskView,
    val_view: TaskView,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    transform,
    pin_memory: bool,
    sx,
    y_batch,
    criterion,
    seed: int,
) -> dict:
    """Train ``model`` in place with the shared regime; restore the best-val-loss checkpoint.

    ``sx`` standardizes an on-device batch; ``y_batch`` builds the target tensor; both are the
    caller's closures (also reused at predict time). With no validation data, runs the full
    ``epochs`` (no early stop) and keeps the final weights.

    Returns compute telemetry ``{effective_epochs, best_val, train_seconds, peak_mem_bytes}`` so
    the runner can record the cost of the sweep (GPU-hours / efficiency reporting) — unrecoverable
    if not captured live. ``best_val`` is ``None`` without a validation split; ``peak_mem_bytes``
    is ``None`` off-CUDA.
    """
    rng = np.random.default_rng(seed)
    n = train_view.feature_shape[0]
    have_val = len(val_view) > 0
    _, length, c = train_view.feature_shape
    # auto: reuse loaders only when batches are small (measured win; wide batches regress — _loader)
    reuse = loader_reuse_enabled(batch_size * length * c * 4)
    n_batches = -(-n // batch_size) if n else 0
    train_iter, val_batches = None, None
    if reuse:
        # Same rng object + call sequence as the legacy path, so epoch e's order is bit-identical;
        # drawing permutations for early-stopped epochs is unobservable (rng feeds nothing else).
        orders = [rng.permutation(n) for _ in range(epochs)]
        if n_batches and epochs:
            train_iter = iter(
                multi_epoch_batches(
                    train_view,
                    batch_size,
                    orders=orders,
                    transform=transform,
                    pin_memory=pin_memory,
                )
            )
        if have_val:
            val_batches = reusable_batches(
                val_view, batch_size, transform=transform, pin_memory=pin_memory
            )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    on_cuda = "cuda" in str(device)
    move = h2d(device, pin_memory)  # staged-pin H2D unless EVEMTBENCH_PIN_MEMORY=1 (see _loader)
    if on_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    train_start = perf_counter()
    ran = 0

    @torch.no_grad()
    def _val_loss() -> float:
        model.eval()
        total, m = 0.0, 0
        it = (
            val_batches  # persistent loader, re-iterated per epoch (fixed natural order)
            if val_batches is not None
            else prefetched_batches(
                val_view, batch_size, transform=transform, pin_memory=pin_memory
            )
        )
        for xb_t, yb_np in it:
            if xb_t.shape[0] == 0:
                continue
            xb = sx(move(xb_t))
            total += float(criterion(model(xb), y_batch(yb_np))) * len(yb_np)
            m += len(yb_np)
        return total / max(m, 1)

    best_val, best_state, bad = float("inf"), None, 0
    for e in range(epochs):
        ran = e + 1
        model.train()
        if reuse:
            assert train_iter is not None  # reuse ⟹ train_iter was built above
            batch_iter, remaining = train_iter, n_batches
        else:
            order = rng.permutation(n)  # shuffle each epoch (streamed from the memmap)
            batch_iter = iter(
                prefetched_batches(
                    train_view,
                    batch_size,
                    order=order,
                    transform=transform,
                    pin_memory=pin_memory,
                )
            )
            remaining = n_batches
        ep0 = perf_counter()
        loss_sum = torch.zeros((), device=device)
        data_s, nb, prev = 0.0, 0, perf_counter()
        while True:
            with _nvtx("data_wait"):
                if remaining <= 0:  # epoch boundary of the shared fit-lifetime iterator
                    break
                try:
                    xb_t, yb_np = next(batch_iter)
                except StopIteration:
                    break
                remaining -= 1
            data_s += perf_counter() - prev
            if xb_t.shape[0] < 2:  # skip <2-sample batches (BatchNorm in mlp/cnn needs >1)
                prev = perf_counter()
                continue
            with _nvtx("h2d"):
                xb = sx(move(xb_t))
            opt.zero_grad()
            with _nvtx("forward"):
                loss = criterion(model(xb), y_batch(yb_np))
            with _nvtx("backward"):
                loss.backward()
            with _nvtx("opt_step"):
                opt.step()
            loss_sum = loss_sum + loss.detach()
            nb += 1
            prev = perf_counter()
        sched.step()
        vloss = _val_loss() if have_val else 0.0
        improved = have_val and vloss < best_val - 1e-5
        if improved:
            best_val, bad = vloss, 0
            best_state = copy.deepcopy(model.state_dict())
        elif have_val:
            bad += 1
        if "cuda" in str(device):
            torch.cuda.synchronize()
        ep_wall = perf_counter() - ep0
        ep_loss = float(loss_sum) / max(nb, 1)
        log(
            epoch_line(
                e + 1,
                epochs,
                loss=ep_loss,
                wall_s=ep_wall,
                data_s=data_s,
                n=n,
                extra=(f" val={vloss:.4f}{' *' if improved else ''}" if have_val else ""),
            )
        )
        # Feed the raw floats to the optional tracker sink (W&B); no-op unless the runner wired it.
        emit_epoch(
            {
                "seed": seed,
                "epoch": e + 1,
                "loss": ep_loss,
                "val_loss": (vloss if have_val else None),
                "wall_s": ep_wall,
                "data_pct": 100.0 * data_s / max(ep_wall, 1e-9),
                "samp_s": n / max(ep_wall, 1e-9),
                "lr": opt.param_groups[0]["lr"],
            }
        )
        if patience and have_val and bad >= patience:
            log(f"  early stop at epoch {e + 1} (no val improvement in {patience})")
            break
    if best_state is not None:
        model.load_state_dict(best_state)  # restore best-val checkpoint
    return {
        "effective_epochs": ran,
        "best_val": (best_val if have_val and best_state is not None else None),
        "train_seconds": perf_counter() - train_start,
        "peak_mem_bytes": (int(torch.cuda.max_memory_allocated(device)) if on_cuda else None),
    }
