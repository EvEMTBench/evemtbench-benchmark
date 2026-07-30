"""GRU recurrent baseline (sequence model over the window). Requires the ``torch`` extra.

Unlike the MLP (which flattens the window to a vector), the GRU consumes the window as a
time series ``(l, c)`` and uses the final-timestep hidden representation for the head. Like the
MLP it branches on the task's label kind: CrossEntropy (+ inverse-frequency class weights) for
classification, MSE (+ target standardization) for regression. Per-channel feature
standardization is fit on TRAIN only (raw EMT signals are ~10^5 V / large A).
"""

from __future__ import annotations

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


class _GRU(nn.Module):
    def __init__(self, in_size: int, hidden: int, out_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            in_size,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _h = self.gru(x)  # out: (b, l, hidden)
        return self.head(out[:, -1, :])  # final-timestep representation


class GRUBaseline:
    name = "gru"

    def __init__(
        self,
        *,
        hidden: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
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
        self.hidden = hidden
        self.num_layers = num_layers
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
        self.model: _GRU | None = None
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

    def _build_inner(self, in_ch: int, out_dim: int, length: int) -> _GRU:
        """Construct the inner module (shared by fit() and checkpoint transfer)."""
        return _GRU(in_ch, self.hidden, out_dim, self.num_layers, self.dropout)

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
        n, length, _c = train.feature_shape
        in_size = int(self._mean.shape[-1])  # accounts for extra zero-seq channels if enabled
        self._mean_t = torch.from_numpy(self._mean).float().to(self.device)
        self._std_t = torch.from_numpy(self._std).float().to(self.device)
        log(
            f"fit gru seed={self.seed}: n={n} ch={self._mean.shape[-1]} len={length} "
            f"dev={self.device} workers={num_workers()} epochs={self.epochs} bs={self.batch_size}"
            f" | channel_stats {perf_counter() - _t:.1f}s"
        )
        crit, out_dim, self._regression, self._y_mean, self._y_std = build_criterion(
            task, train, self.device
        )
        if not self._regression:
            self.num_classes = out_dim
        self.model = self._build_inner(in_size, out_dim, length).to(self.device)
        self._dims = (in_size, out_dim, length)
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
        # batched inference streamed from the memmap to bound memory on the wide global view
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
