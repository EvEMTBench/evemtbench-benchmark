"""Parallel, order-preserving batch prefetching for the deep baselines.

The training loop's bottleneck is the synchronous, single-threaded read+project per batch
(`TaskView._read_positions` → memmap read of the full on-disk channels + projection), which leaves
the GPU starved (~0% util while one CPU core is pegged). This wraps the *same* batches
in a ``torch`` ``DataLoader`` so the read+project runs in worker processes and overlaps GPU compute.

Crucially it is **order-preserving**: it chunks the caller's ``order`` into batches and serves them
with ``shuffle=False``, so the batch sequence is byte-identical to ``iter_batches(order=...)`` —
workers only parallelize the fetch. ``workers=0`` reproduces exactly today's synchronous path.
The on-disk windows are read-only ``np.memmap`` (fork-safe), so ``num_workers>0`` is safe on Linux.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def projection_cache_dir() -> str | None:
    """Scratch dir for the projection cache, or None (disabled). Set by the sweep scripts."""
    return os.environ.get("EVEMTBENCH_PROJECTION_CACHE") or None


def num_workers() -> int:
    """Default DataLoader workers: ``EVEMTBENCH_NUM_WORKERS`` if set, else a share of
    ``$SLURM_CPUS_PER_TASK``, else 0. The cap depends on the projection cache: with it, per-batch
    reads are tiny (only the view's channels) so we use up to 8; without it, an uncached wide-global
    read holds a big full-channel buffer per worker, so we cap at 4 to bound host RAM."""
    env = os.environ.get("EVEMTBENCH_NUM_WORKERS")
    if env is not None and env.strip():
        return max(0, int(env))
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if not (cpus and cpus.isdigit()):
        return 0
    cap = 8 if projection_cache_dir() else 4
    return max(1, min(cap, int(cpus) - 1))


def _read_batch(view, pos: np.ndarray, transform):
    """The shared per-item body: read+project ``pos`` (the heavy step — runs in the worker
    process), apply the transform, and hand back a CPU float32 tensor with its labels."""
    xb = view._read_positions(pos)
    if transform is not None:
        xb = transform(xb)
    return torch.from_numpy(np.ascontiguousarray(xb, dtype=np.float32)), view.labels[pos]


class _PositionBatchDataset(Dataset):
    """Indexed by BATCH number: item ``i`` is the read+projected (+transform) batch for the i-th
    chunk of positions, as a CPU float32 tensor, paired with its labels."""

    def __init__(self, view, batches: list[np.ndarray], transform=None):
        self._view = view
        self._batches = batches
        self._transform = transform

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, i: int):
        return _read_batch(self._view, self._batches[i], self._transform)


class _MultiEpochBatchDataset(Dataset):
    """Indexed by GLOBAL batch number across ALL epochs: item ``g`` is batch ``g % n_batches`` of
    epoch ``g // n_batches``'s precomputed order. Items are pure functions of ``g`` (read-only
    memmap + deterministic transform), so ONE DataLoader can serve a whole fit and workers may
    prefetch across an epoch boundary without changing any batch."""

    def __init__(self, view, orders: list[np.ndarray], batch_size: int, transform=None):
        self._view = view
        self._orders = orders
        self._bs = batch_size
        self._transform = transform
        n = len(view)
        self._nb = -(-n // batch_size) if n else 0

    @property
    def batches_per_epoch(self) -> int:
        return self._nb

    def __len__(self) -> int:
        return self._nb * len(self._orders)

    def __getitem__(self, g: int):
        e, b = divmod(g, self._nb)
        pos = self._orders[e][b * self._bs : (b + 1) * self._bs]
        return _read_batch(self._view, pos, self._transform)


def _identity(batch):
    # DataLoader batch_size=1 over a batch-indexed dataset -> the single item IS the real batch.
    return batch[0]


# Fit-lifetime loaders win where fork churn dominates and lose where the still-alive train
# workers' cross-epoch prefetch contends with the val workers (A/B 3867918: tiny-view epochs
# 3.55 s -> 0.55 s, but wide-view 85 MB-batch epochs +10%). 16 MB/batch splits the measured win
# (2.9 MB local views) from the measured regression (85 MB wide-global views) and maps onto the
# local/line vs global view taxonomy.
_REUSE_MAX_BATCH_BYTES = 16 * 1024**2


def loader_reuse_enabled(batch_bytes: int | None = None) -> bool:
    """Should this fit use ONE train DataLoader + a persistent val loader instead of fresh 8-worker
    forks every epoch and every val pass?

    ``EVEMTBENCH_LOADER_REUSE``: ``1`` forces on, ``0``/``false`` forces off (kill-switch), unset or
    ``auto`` (the default) decides by batch size — reuse only when a training batch is at most
    ``_REUSE_MAX_BATCH_BYTES`` (see the measured verdict above; A/B job 3867918). Batch
    content/order are bit-identical on every path (parity-tested)."""
    env = os.environ.get("EVEMTBENCH_LOADER_REUSE", "auto")
    if env in ("1", "true", "True"):
        return True
    if env in ("", "0", "false", "False"):
        return False
    return batch_bytes is not None and batch_bytes <= _REUSE_MAX_BATCH_BYTES


def _pin_enabled(requested: bool) -> bool:
    """Gate the callers' ``pin_memory`` requests: ON by default, ``EVEMTBENCH_PIN_MEMORY=0`` opts
    into the unpinned producer + side-stream prefetcher (:class:`_PinStager`) — EXPERIMENTAL.

    Measured verdict (2026-07-17, jobs 3865948/3866000/3866138/3866296): the DataLoader pin thread
    is a serial ~20 ms/batch producer stage (41.3 ms/batch pinned vs 21.0 unpinned in isolation),
    BUT in the real training loop it also absorbs the shm first-touch + copy off the main thread —
    every unpinned consumer variant (plain sync ``.to()``, default-stream staging, side-stream
    prefetcher) lost 10-20% on the compute-bound resnet while gaining at most ~6% on data-bound
    mlp. Net across the sweep ~0: pinned stays the default. Pinning cannot move into workers
    (pinned+shared memory are mutually exclusive, pytorch/pytorch#78018). Batch VALUES are
    identical on every path (bit-identical metrics in all A/B arms)."""
    if not requested:
        return False
    return os.environ.get("EVEMTBENCH_PIN_MEMORY", "1") not in ("", "0", "false", "False")


class _PinStager:
    """Side-stream pinned H2D prefetcher (the Apex-prefetcher pattern): consume UNPINNED CPU
    batches, stage them through two reusable pinned buffers, and DMA on a DEDICATED copy stream.

    Why this exact shape (measured, 2026-07-17):
    * The DataLoader pin thread is a serial ~20 ms whole-pipeline stage per ~85 MB wide-view batch
      that caps the producer (probe 3865948: 41.3 ms/batch pinned vs 21.0 unpinned).
    * Plain unpinned ``.to()`` stalls compute instead (A/B 3866000: resnet 35->42 s epochs).
    * Enqueuing the DMA on the DEFAULT stream (A/B 3866138) serializes host progress behind
      compute — the buffer-reuse wait then spans every kernel queued before it, not just the copy.
    So: DMA on its own stream; compute orders after it with a device-side ``wait_event`` (host
    never blocks on kernels); buffer reuse host-waits only on that same DMA event (~4 ms copy,
    two batches stale — effectively always already complete); ``record_stream`` tells the caching
    allocator the output lives on the compute stream."""

    def __init__(self, device):
        self._device = device
        self._stream = torch.cuda.Stream(device=device)
        self._bufs: list = [None, None]
        self._dma: list = [None, None]  # per-buffer DMA-done event
        self._i = 0

    def __call__(self, xb: torch.Tensor) -> torch.Tensor:
        i = self._i % 2
        self._i += 1
        buf, ev = self._bufs[i], self._dma[i]
        if buf is None or buf.shape[1:] != xb.shape[1:] or buf.shape[0] < xb.shape[0]:
            self._bufs[i] = buf = torch.empty_like(xb, pin_memory=True)
            self._dma[i] = ev = torch.cuda.Event()
        ev.synchronize()  # this buffer's PREVIOUS DMA done (no-op if never recorded)
        dst = buf[: xb.shape[0]]
        dst.copy_(xb)  # pageable -> pinned, main thread
        with torch.cuda.stream(self._stream):
            out = dst.to(self._device, non_blocking=True)
        ev.record(self._stream)
        compute = torch.cuda.current_stream(self._device)
        compute.wait_event(ev)  # device-side ordering: kernels on `out` start after its DMA
        out.record_stream(compute)
        return out


def h2d(device, pin_requested: bool):
    """Return the batch H2D mover for this fit/predict pass.

    * default (DataLoader pinning on) -> batches arrive pinned; a plain ``non_blocking`` ``.to()``
      is already async.
    * ``EVEMTBENCH_PIN_MEMORY=0`` on CUDA (experimental) -> unpinned producer, stage through
      :class:`_PinStager`.
    * CPU device / no CUDA -> plain ``.to()``."""
    on_cuda = "cuda" in str(device) and torch.cuda.is_available()
    if on_cuda and not _pin_enabled(pin_requested):
        return _PinStager(device)

    def _plain(xb: torch.Tensor) -> torch.Tensor:
        return xb.to(device, non_blocking=True)

    return _plain


def prefetched_batches(
    view,
    batch_size: int,
    *,
    order: np.ndarray | None = None,
    transform=None,
    workers: int | None = None,
    pin_memory: bool = False,
):
    """Yield ``(x_cpu_float32_tensor, y_np)`` batches in ``order``, reading/projecting in worker
    procs. Same batch sequence as ``view.iter_batches(batch_size, order=order)``; ``workers=0`` =
    synchronous (today's path)."""
    n = len(view)
    order = np.arange(n) if order is None else order
    batches = [order[i : i + batch_size] for i in range(0, n, batch_size)]
    if not batches:
        return
    w = num_workers() if workers is None else workers
    ds = _PositionBatchDataset(view, batches, transform)
    if w == 0:
        for i in range(len(ds)):
            yield ds[i]
        return
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=w,
        pin_memory=_pin_enabled(pin_memory),
        collate_fn=_identity,
        persistent_workers=False,
        prefetch_factor=4,  # deeper read-ahead so the GPU isn't waiting on the next batch
    )
    yield from loader


def multi_epoch_batches(
    view,
    batch_size: int,
    *,
    orders: list[np.ndarray],
    transform=None,
    workers: int | None = None,
    pin_memory: bool = False,
):
    """Yield EVERY epoch's batches from one generator: epoch ``e``'s batches are byte-identical to
    ``prefetched_batches(view, batch_size, order=orders[e], ...)``; the consumer takes exactly
    ``ceil(len(view)/batch_size)`` items per epoch. With workers this builds a SINGLE DataLoader
    for the whole fit (one worker-fork per fit instead of per epoch); ``workers=0`` is the plain
    synchronous path. Cross-epoch prefetch is semantics-free — items are pure functions of the
    global index (see :class:`_MultiEpochBatchDataset`)."""
    ds = _MultiEpochBatchDataset(view, orders, batch_size, transform)
    if len(ds) == 0:
        return
    w = num_workers() if workers is None else workers
    if w == 0:
        for i in range(len(ds)):
            yield ds[i]
        return
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=w,
        pin_memory=_pin_enabled(pin_memory),
        collate_fn=_identity,
        persistent_workers=False,  # ONE loader lives for the whole fit; nothing to persist across
        prefetch_factor=4,
    )
    yield from loader


class _SyncBatches:
    """Re-iterable synchronous twin of the persistent val DataLoader (workers=0 path)."""

    def __init__(self, ds):
        self._ds = ds

    def __iter__(self):
        for i in range(len(self._ds)):
            yield self._ds[i]


def reusable_batches(
    view,
    batch_size: int,
    *,
    transform=None,
    workers: int | None = None,
    pin_memory: bool = False,
):
    """A RE-ITERABLE over the view's natural-order batches for the per-epoch validation pass:
    iterate it once per epoch. With workers it is a ``persistent_workers=True`` DataLoader (the
    fixed order makes every re-iteration identical; workers fork once per fit); ``workers=0``
    returns a synchronous wrapper byte-identical to ``prefetched_batches``."""
    n = len(view)
    batches = [np.arange(n)[i : i + batch_size] for i in range(0, n, batch_size)]
    ds = _PositionBatchDataset(view, batches, transform)
    w = num_workers() if workers is None else workers
    if w == 0 or len(ds) == 0:
        return _SyncBatches(ds)
    return DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=w,
        pin_memory=_pin_enabled(pin_memory),
        collate_fn=_identity,
        persistent_workers=True,
        prefetch_factor=4,
    )
