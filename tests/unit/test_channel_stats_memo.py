"""TaskView.channel_stats is memoized per cell (the runner reuses one Task across seeds) — keyed by
the transform so ``None`` and ``add_zero_sequence`` don't collide, and a repeat call does NOT rerun
the full read+project pass. A bug here would silently re-scan (slow) or mis-standardize (wrong
numbers). Torch-free: drives the real method with a lightweight fake view."""

import numpy as np

from evemtbench.tasks.task import TaskView, add_zero_sequence


class _FakeView:
    """Minimal stand-in: channel_stats only needs ``self.iter_batches`` + ``self.__dict__`` (for the
    cache). ``calls`` counts how many full passes actually run."""

    def __init__(self, x):
        self._x = x
        self.calls = 0

    def iter_batches(self, batch_size):
        self.calls += 1  # one increment per channel_stats pass
        yield self._x, None


def test_channel_stats_memoized_per_transform():
    x = np.random.default_rng(0).standard_normal((4, 5, 6)).astype(np.float32)
    v = _FakeView(x)

    a = TaskView.channel_stats(v, transform=None)
    assert v.calls == 1
    b = TaskView.channel_stats(v, transform=None)  # memoized -> no second pass
    assert v.calls == 1
    assert a is b  # same cached object returned

    # a different transform is a different cache key -> recomputed, distinct entry
    c = TaskView.channel_stats(v, transform=add_zero_sequence)
    assert v.calls == 2
    assert c[0].shape[-1] == 8  # 6 channels -> 8 with per-cubicle zero-sequence
    assert c[0].shape[-1] != a[0].shape[-1]

    # the None entry is still cached and unchanged (keys don't collide)
    assert TaskView.channel_stats(v, transform=None) is a
    assert v.calls == 2


def test_channel_stats_value_matches_numpy():
    # the cached value must equal a direct numpy mean (the algorithm, not just the memoization)
    x = np.random.default_rng(1).standard_normal((3, 7, 6)).astype(np.float32)
    mean, std = TaskView.channel_stats(_FakeView(x), transform=None)
    exp_mean = x.mean(axis=(0, 1)).astype(np.float32)
    np.testing.assert_allclose(mean.reshape(-1), exp_mean, rtol=0, atol=1e-5)
    assert mean.shape == (1, 1, 6) and std.shape == (1, 1, 6)
