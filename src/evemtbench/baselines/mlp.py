"""MLP baseline (deep-learning reference). Requires the ``torch`` extra.

Flattens each window ``(l, c)`` to a vector and trains an MLP. v0.1 upgrades over the original
1-stack version: BatchNorm + deeper stack, AdamW with weight decay, a cosine LR schedule, and —
crucially — it uses the **validation split** for early stopping (best-val-loss checkpoint), which
the original ignored. Still a reference baseline, not a heavily tuned model.

Branches on the task label kind: CrossEntropy (+ inverse-frequency class weights) for
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


class _MLP(nn.Module):
    def __init__(
        self, input_size: int, hidden_sizes: tuple[int, ...], out_dim: int, dropout: float
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.shape[0], -1))


class MLPBaseline:
    name = "mlp"

    def __init__(
        self,
        *,
        hidden_sizes: tuple[int, ...] = (512, 256, 128),
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
        self.hidden_sizes = hidden_sizes
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
        self.model: _MLP | None = None
        self.n_params_: int | None = None  # trainable params, set in fit() (capacity reporting)
        self.train_stats_: dict = {}  # compute telemetry from train_supervised, set in fit()
        self._regression = False
        self._transform: Callable[[np.ndarray], np.ndarray] | None = None  # set in fit(), zero_seq
        # Per-channel standardization stats, fit on TRAIN only (no leakage).
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._mean_t: torch.Tensor | None = None  # device tensors for per-batch standardization
        self._std_t: torch.Tensor | None = None
        # Target standardization (regression only): location % ~1..99 -> ~N(0,1) for stable MSE.
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

    def _build_inner(self, in_ch: int, out_dim: int, length: int) -> _MLP:
        """Construct the inner module (shared by fit() and checkpoint transfer)."""
        return _MLP(length * in_ch, self.hidden_sizes, out_dim, self.dropout)

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
        log(
            f"fit mlp seed={self.seed}: n={n} ch={self._mean.shape[-1]} len={length} "
            f"dev={self.device} workers={num_workers()} epochs={self.epochs} bs={self.batch_size}"
            f" | channel_stats {perf_counter() - _t:.1f}s"
        )
        self._mean_t = torch.from_numpy(self._mean).float().to(self.device)
        self._std_t = torch.from_numpy(self._std).float().to(self.device)

        crit, out_dim, self._regression, self._y_mean, self._y_std = build_criterion(
            task, train, self.device
        )
        if not self._regression:
            self.num_classes = out_dim
        in_ch = int(self._mean.shape[-1])
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
        preds = out.reshape(-1) * self._y_std + self._y_mean if self._regression else out.argmax(1)
        return predictions_frame(view.metadata["sample_id"], view.metadata["window_idx"], preds)
