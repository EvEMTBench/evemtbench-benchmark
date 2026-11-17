"""Time Series Transformer baseline (transformer reference). Requires the ``torch`` extra.

The encoder of Zerveas et al. 2021 ("A Transformer-based Framework for Multivariate Time Series
Representation Learning", KDD '21, arXiv:2010.02803), trained from scratch (supervised only, no
masked pre-training): a linear input projection scaled by ``sqrt(d_model)``, a learnable positional
encoding, post-norm encoder blocks that use **batch normalisation in place of layer
normalisation**, GELU activations, and a linear head on the **concatenation of all token outputs**.
Width, depth and feed-forward size are the paper's generic configuration (``d_model=128``, 3 blocks,
feed-forward 256, dropout 0.1).

Three declared deviations from the published model:

* **Strided patch embedding.** The published model makes one token per time step (480 tokens for
  the benchmark window). Here the input projection is ``Conv1d(C, d_model, kernel_size=16,
  stride=8)`` — a linear projection of overlapping 16-sample patches that span all channels — which
  gives 59 tokens for a 480-sample window (patching as in Nie et al. 2023, arXiv:2211.14730;
  channel-mixing, unlike their channel-independent model). Only this first layer depends on the
  channel count ``C``, so the encoder cost is flat from the 6-channel ``local`` view to the wide
  ``global`` view.
* **8 attention heads** instead of the generic configuration's 16.
* **Layer normalisation of the patch tokens** (``LayerNorm(d_model)`` after the scaled input
  projection, before the positional encoding). Without it, training diverged to NaN on the wide
  ``global`` views (27 seeds in 11 cells of the first 1.2.0 sweep, several by epoch 2–7) and stopped
  early on every view: the ``sqrt(d_model)`` scaling, kept from token-lookup embeddings, lets the
  token scale grow on heavy-tailed fault transients until the backward pass overflows (first
  BatchNorm running variance up to 4e5). Gradient-norm clipping did not prevent it; the token norm
  did, and gave the lowest validation loss of the variants tried (issue #26).

A window shorter than one patch (the 8-sample test fixture; never the benchmark's 480-sample
window) is embedded as a single patch of its full length, so the module stays usable there.

Like the other deep baselines it branches on the task label kind — CrossEntropy (+
inverse-frequency class weights) for classification, MSE (+ target standardization) for regression
— fits per-channel feature standardization on TRAIN only, and shares the **identical** training
regime via ``_train.train_supervised`` (AdamW + cosine schedule + val early-stop). The source
repository's examples use RAdam at batch size 128; the benchmark's shared optimiser and batch size
are kept instead so that only the architecture differs across the deep baselines.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..evaluation.predictions_io import predictions_frame
from ..tasks.task import Task, TaskView, build_input_transform
from ._loader import h2d, num_workers, prefetched_batches, projection_cache_dir
from ._log import log
from ._train import build_criterion, train_supervised
from ._transfer import maybe_load_pretrained


def patch_geometry(length: int, patch_len: int, stride: int) -> tuple[int, int, int]:
    """``(kernel, stride, n_tokens)`` of the patch embedding for a ``length``-sample window.

    For ``length >= patch_len`` this is the configured geometry (480, 16, 8 -> 59 tokens). A window
    shorter than one patch is embedded as a single full-length patch."""
    if length < 1:
        raise ValueError(f"window length must be positive, got {length}")
    kernel = min(patch_len, length)
    step = min(stride, kernel)
    return kernel, step, (length - kernel) // step + 1


class _BatchNormEncoderLayer(nn.Module):
    """A post-norm Transformer encoder block with BatchNorm1d over the feature dimension instead of
    LayerNorm (Zerveas et al. 2021, Sec. 3.1): self-attention + residual, norm, feed-forward +
    residual, norm."""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm1d(d_model, eps=1e-5)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim_ff, d_model)
        )
        self.drop2 = nn.Dropout(dropout)
        self.norm2 = nn.BatchNorm1d(d_model, eps=1e-5)

    @staticmethod
    def _bn(norm: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
        # x: (b, t, d); BatchNorm1d normalises dim 1, so move the feature dim there and back
        return norm(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self._bn(self.norm1, x + self.drop1(a))
        return self._bn(self.norm2, x + self.drop2(self.ff(x)))


class _TST(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_dim: int,
        length: int,
        *,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_ff: int,
        dropout: float,
        patch_len: int,
        stride: int,
    ):
        super().__init__()
        kernel, step, self.n_tokens = patch_geometry(length, patch_len, stride)
        self.scale = math.sqrt(d_model)
        # the only layer whose size depends on the channel count
        self.embed = nn.Conv1d(in_ch, d_model, kernel_size=kernel, stride=step)
        # bounds the token scale on heavy-tailed inputs (declared deviation, see module docstring)
        self.embed_norm = nn.LayerNorm(d_model)
        self.pos = nn.Parameter(torch.empty(1, self.n_tokens, d_model))
        nn.init.uniform_(self.pos, -0.02, 0.02)
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            _BatchNormEncoderLayer(d_model, n_heads, dim_ff, dropout) for _ in range(n_layers)
        )
        self.act = nn.GELU()
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(self.n_tokens * d_model, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, l, c) -> (b, c, l) for Conv1d -> tokens (b, t, d)
        h = self.embed_norm(self.embed(x.transpose(1, 2)).transpose(1, 2) * self.scale)
        h = self.pos_drop(h + self.pos)
        for block in self.blocks:
            h = block(h)
        return self.head(self.act(h))


class TSTBaseline:
    name = "tst"

    def __init__(
        self,
        *,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dim_ff: int = 256,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
        epochs: int = 60,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 12,
        device: str | None = None,
        seed: int = 0,
        num_classes: int = 2,
        zero_seq: bool = False,
        normalize: str = "none",
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_ff = dim_ff
        self.patch_len = patch_len
        self.stride = stride
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.num_classes = num_classes
        self.zero_seq = zero_seq
        self.normalize = normalize
        self.model: _TST | None = None
        self.n_params_: int | None = None  # trainable params, set in fit() (capacity reporting)
        self.train_stats_: dict = {}  # compute telemetry from train_supervised, set in fit()
        self._regression = False
        self._transform: Callable[[np.ndarray], np.ndarray] | None = None  # set in fit(), zero_seq
        # Per-channel standardization (over windows+time), fit on TRAIN only.
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._mean_t: torch.Tensor | None = None  # device tensors for per-batch standardization
        self._std_t: torch.Tensor | None = None
        # Target standardization (regression only).
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    def _pin(self) -> bool:
        return "cuda" in str(self.device)

    def _sx(self, xb: torch.Tensor) -> torch.Tensor:
        """Standardize a batch already on the device (per-batch -> no full standardized copy)."""
        assert self._mean_t is not None and self._std_t is not None  # set in fit()
        return (xb - self._mean_t) / self._std_t

    def _y_batch(self, y_np: np.ndarray) -> torch.Tensor:
        if self._regression:
            y = (np.asarray(y_np, dtype=np.float32) - self._y_mean) / self._y_std
            return torch.from_numpy(y).float().unsqueeze(1).to(self.device)
        return torch.from_numpy(np.asarray(y_np)).long().to(self.device)

    def _build_inner(self, in_ch: int, out_dim: int, length: int) -> _TST:
        """Construct the inner module (shared by fit() and checkpoint transfer)."""
        return _TST(
            in_ch,
            out_dim,
            length,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dim_ff=self.dim_ff,
            dropout=self.dropout,
            patch_len=self.patch_len,
            stride=self.stride,
        )

    def fit(self, task: Task) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        train = task.train
        self._transform = build_input_transform(self.zero_seq, self.normalize)
        cdir = projection_cache_dir()
        if cdir:
            _tc = perf_counter()
            train.materialize_cache(cdir)
            if len(task.val):
                task.val.materialize_cache(cdir)
            log(f"projection cache ready ({perf_counter() - _tc:.1f}s)")
        _t = perf_counter()
        self._mean, self._std = train.channel_stats(transform=self._transform)
        n, length, _in_ch = train.feature_shape
        in_ch = int(self._mean.shape[-1])  # accounts for extra zero-seq channels if enabled
        self._mean_t = torch.from_numpy(self._mean).float().to(self.device)
        self._std_t = torch.from_numpy(self._std).float().to(self.device)
        log(
            f"fit tst seed={self.seed}: n={n} ch={self._mean.shape[-1]} len={length} "
            f"dev={self.device} workers={num_workers()} epochs={self.epochs} bs={self.batch_size}"
            f" | channel_stats {perf_counter() - _t:.1f}s"
        )
        crit, out_dim, self._regression, self._y_mean, self._y_std = build_criterion(
            task, train, self.device
        )
        if not self._regression:
            self.num_classes = out_dim
        self.model = self._build_inner(in_ch, out_dim, length).to(self.device)
        self._dims = (in_ch, out_dim, length)
        maybe_load_pretrained(self)  # fine-tune: continue from the multi_grid pretrained weights
        self.n_params_ = int(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        self.train_stats_ = train_supervised(
            model=self.model,
            train_view=train,
            val_view=task.val,
            device=self.device,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            patience=self.patience,
            transform=self._transform,
            pin_memory=self._pin(),
            sx=self._sx,
            y_batch=self._y_batch,
            criterion=crit,
            seed=self.seed,
        )

    @torch.no_grad()
    def predict(self, view: TaskView) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("call fit() before predict()")
        self.model.eval()
        outs = []
        move = h2d(self.device, self._pin())
        for xb_t, _ in prefetched_batches(
            view, self.batch_size, transform=self._transform, pin_memory=self._pin()
        ):
            xb = self._sx(move(xb_t))
            outs.append(self.model(xb).cpu().numpy())
        out = np.concatenate(outs, axis=0) if outs else np.zeros((0, 1))
        preds: np.ndarray  # float (regression) or integer class ids (argmax)
        if self._regression:
            preds = out.reshape(-1) * self._y_std + self._y_mean
        else:
            preds = out.argmax(axis=1)
        return predictions_frame(view.metadata["sample_id"], view.metadata["window_idx"], preds)
