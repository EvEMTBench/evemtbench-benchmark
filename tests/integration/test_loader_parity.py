"""The parallel prefetching loader must be a drop-in for TaskView.iter_batches: same batch sequence,
same content (so it never changes the benchmark numbers), for both synchronous and worker modes."""

import numpy as np
import pytest

from evemtbench.tasks.task import add_zero_sequence

pytest.importorskip("torch")

import torch

from evemtbench.baselines._loader import _pin_enabled, num_workers, prefetched_batches


@pytest.mark.parametrize(
    "env_workers, slurm_cpus, expected",
    [
        ("4", None, 4),  # explicit override
        ("7", "8", 7),  # explicit override wins and is NOT capped
        ("0", "8", 0),  # explicit 0 (synchronous) wins over SLURM
        ("-3", None, 0),  # negative clamped to 0
        ("", "8", 4),  # blank -> fall through to SLURM: min(4, 8-1)
        (None, "8", 4),  # SLURM-derived, capped at 4
        (None, "100", 4),  # cap holds
        (None, "3", 2),  # min(4, max(1, 3-1))
        (None, "1", 1),  # max(1, 1-1) floor
        (None, "nan", 0),  # non-numeric SLURM value -> 0
        (None, None, 0),  # nothing set -> synchronous
    ],
)
def test_num_workers_env(monkeypatch, env_workers, slurm_cpus, expected):
    monkeypatch.delenv("EVEMTBENCH_NUM_WORKERS", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    if env_workers is not None:
        monkeypatch.setenv("EVEMTBENCH_NUM_WORKERS", env_workers)
    if slurm_cpus is not None:
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", slurm_cpus)
    assert num_workers() == expected


class _FakeView:
    """Minimal stand-in for TaskView: index-addressable read + labels, no memmap/cluster needed."""

    def __init__(self, n=23, length=8, channels=6, seed=0):
        rng = np.random.default_rng(seed)
        self._x = rng.standard_normal((n, length, channels)).astype(np.float32)
        self.labels = rng.integers(0, 3, size=n)

    def __len__(self):
        return self._x.shape[0]

    def _read_positions(self, positions):
        return np.ascontiguousarray(self._x[positions], dtype=np.float32)


@pytest.mark.parametrize("workers", [0, 2, 4])
@pytest.mark.parametrize("transform", [None, add_zero_sequence])
def test_prefetched_matches_iter_batches(workers, transform):
    view = _FakeView()
    n = len(view)
    order = np.random.default_rng(1).permutation(n)
    batch_size = 4

    # reference = exactly what iter_batches(order) would yield (read + same transform)
    ref = []
    for i in range(0, n, batch_size):
        pos = order[i : i + batch_size]
        x = view._read_positions(pos)
        if transform is not None:
            x = transform(x)
        ref.append((x, view.labels[pos]))

    got = list(
        prefetched_batches(view, batch_size, order=order, transform=transform, workers=workers)
    )

    assert len(got) == len(ref)
    for (xt, yt), (xr, yr) in zip(got, ref, strict=True):
        np.testing.assert_array_equal(xt.numpy(), xr)  # identical batch content + order
        np.testing.assert_array_equal(np.asarray(yt), yr)


@pytest.mark.parametrize(
    "requested, env, expected",
    [
        (False, None, False),  # caller never asked -> never pin
        (False, "0", False),  # env cannot force a path the caller didn't request
        (True, None, True),  # DEFAULT: pinned (measured best in-loop; see _pin_enabled docstring)
        (True, "1", True),
        (True, "0", False),  # experimental opt-out -> unpinned producer + side-stream prefetcher
        (True, "", False),
        (True, "false", False),
    ],
)
def test_pin_enabled_env(monkeypatch, requested, env, expected):
    monkeypatch.delenv("EVEMTBENCH_PIN_MEMORY", raising=False)
    if env is not None:
        monkeypatch.setenv("EVEMTBENCH_PIN_MEMORY", env)
    assert _pin_enabled(requested) is expected


def test_num_workers_cap_rises_with_projection_cache(monkeypatch):
    """With the projection cache, per-batch reads are small -> the SLURM-derived cap is 8, not 4."""
    monkeypatch.delenv("EVEMTBENCH_NUM_WORKERS", raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
    monkeypatch.delenv("EVEMTBENCH_PROJECTION_CACHE", raising=False)
    assert num_workers() == 4
    monkeypatch.setenv("EVEMTBENCH_PROJECTION_CACHE", "/tmp/anywhere")
    assert num_workers() == 8


@pytest.mark.parametrize("pin_env", ["0", "1"])
def test_prefetched_parity_survives_pin_request(monkeypatch, pin_env):
    """A pin_memory request must never change batch content/order, whichever way the gate resolves.
    (Without CUDA torch ignores pinning with a warning, so this is CI-safe either way.)"""
    monkeypatch.setenv("EVEMTBENCH_PIN_MEMORY", pin_env)
    view = _FakeView()
    n = len(view)
    order = np.random.default_rng(2).permutation(n)
    ref = [
        (view._read_positions(order[i : i + 4]), view.labels[order[i : i + 4]])
        for i in range(0, n, 4)
    ]
    got = list(prefetched_batches(view, 4, order=order, workers=2, pin_memory=True))
    assert len(got) == len(ref)
    for (xt, yt), (xr, yr) in zip(got, ref, strict=True):
        np.testing.assert_array_equal(xt.numpy(), xr)
        np.testing.assert_array_equal(np.asarray(yt), yr)


@pytest.mark.parametrize("pin_env", ["0", "1"])
def test_h2d_preserves_values(monkeypatch, pin_env):
    """The H2D mover (plain or staged-pin) must be value-transparent, incl. the short final batch
    that exercises the stager's buffer-slice path."""
    from evemtbench.baselines._loader import h2d

    monkeypatch.setenv("EVEMTBENCH_PIN_MEMORY", pin_env)
    move = h2d("cpu", False)  # CPU path: plain .to() is the identity here
    x = torch.randn(5, 3)
    assert torch.equal(move(x), x)
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the staged-pin path")
    move = h2d("cuda", True)
    for x in [torch.randn(8, 4, 6) for _ in range(5)] + [torch.randn(3, 4, 6)]:
        y = move(x)
        assert y.device.type == "cuda"
        torch.testing.assert_close(y.cpu(), x)


# ---- fit-lifetime loader (multi_epoch_batches / reusable_batches) ----------------------------


def _legacy_epoch_sequence(view, seed, epochs, batch_size, transform):
    """Reference: EXACTLY today's per-epoch loop — one rng, one permutation drawn per epoch."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(epochs):
        order = rng.permutation(len(view))
        for i in range(0, len(view), batch_size):
            pos = order[i : i + batch_size]
            x = view._read_positions(pos)
            if transform is not None:
                x = transform(x)
            out.append((x, view.labels[pos]))
    return out


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("transform", [None, add_zero_sequence])
def test_multi_epoch_parity_with_legacy_loop(workers, transform):
    """The fit-lifetime iterator must reproduce the legacy multi-epoch batch sequence
    bit-identically (n=23, bs=4 -> uneven final batch exercises the short-slice path)."""
    from evemtbench.baselines._loader import multi_epoch_batches

    view, seed, epochs, bs = _FakeView(), 7, 3, 4
    ref = _legacy_epoch_sequence(view, seed, epochs, bs, transform)

    rng = np.random.default_rng(seed)
    orders = [rng.permutation(len(view)) for _ in range(epochs)]
    got = list(multi_epoch_batches(view, bs, orders=orders, transform=transform, workers=workers))
    assert len(got) == len(ref)
    for (xt, yt), (xr, yr) in zip(got, ref, strict=True):
        np.testing.assert_array_equal(xt.numpy(), xr)
        np.testing.assert_array_equal(np.asarray(yt), yr)


def test_multi_epoch_early_abandonment():
    """Consuming only a prefix (early stop) yields the reference prefix; dropping the iterator
    mid-stream must tear down promptly (no worker hang)."""
    from evemtbench.baselines._loader import multi_epoch_batches

    view, seed, epochs, bs = _FakeView(), 3, 4, 4
    n_batches = -(-len(view) // bs)
    ref = _legacy_epoch_sequence(view, seed, epochs, bs, None)

    rng = np.random.default_rng(seed)
    orders = [rng.permutation(len(view)) for _ in range(epochs)]
    it = iter(multi_epoch_batches(view, bs, orders=orders, workers=2))
    k = 2 * n_batches + 1  # two full epochs + one batch into the third
    got = [next(it) for _ in range(k)]
    for (xt, _), (xr, _) in zip(got, ref[:k], strict=True):
        np.testing.assert_array_equal(xt.numpy(), xr)
    del it  # abandoned mid-epoch -> GC-driven DataLoader shutdown


@pytest.mark.parametrize("workers", [0, 2])
def test_reusable_batches_reiterates_identically(workers):
    from evemtbench.baselines._loader import reusable_batches

    view, bs = _FakeView(), 4
    rb = reusable_batches(view, bs, workers=workers)
    passes = [[(x.numpy().copy(), np.asarray(y).copy()) for x, y in rb] for _ in range(3)]
    assert len(passes[0]) == -(-len(view) // bs)
    for p in passes[1:]:
        for (x0, y0), (xp, yp) in zip(passes[0], p, strict=True):
            np.testing.assert_array_equal(x0, xp)
            np.testing.assert_array_equal(y0, yp)


def test_multi_epoch_empty_view_yields_nothing():
    from evemtbench.baselines._loader import multi_epoch_batches

    view = _FakeView(n=0)
    assert list(multi_epoch_batches(view, 4, orders=[np.array([], dtype=int)], workers=0)) == []


@pytest.mark.parametrize(
    "env, batch_bytes, expected",
    [
        (None, 2 * 1024**2, True),  # auto: small batches (local views) -> reuse
        (None, 85 * 1024**2, False),  # auto: wide-global batches -> legacy (measured regression)
        (None, None, False),  # auto with unknown size -> conservative legacy
        ("auto", 2 * 1024**2, True),
        ("1", 85 * 1024**2, True),  # force on regardless of size
        ("0", 2 * 1024**2, False),  # kill-switch
        ("", 2 * 1024**2, False),
        ("false", 2 * 1024**2, False),
    ],
)
def test_loader_reuse_gate(monkeypatch, env, batch_bytes, expected):
    from evemtbench.baselines._loader import loader_reuse_enabled

    monkeypatch.delenv("EVEMTBENCH_LOADER_REUSE", raising=False)
    if env is not None:
        monkeypatch.setenv("EVEMTBENCH_LOADER_REUSE", env)
    assert loader_reuse_enabled(batch_bytes) is expected
