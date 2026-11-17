"""Fundamental-frequency phasors and symmetrical components (numpy only).

The signal layer of the impedance-based fault locator (``baselines/impedance.py``). Kept separate
and small so it can be audited and unit-tested on synthetic sinusoids alone.

**Estimator.** A full-cycle discrete Fourier transform (DFT) at the nominal frequency, evaluated on
one cycle of samples:

    X = (sqrt(2) / N) * sum_{k=0}^{N-1} x[k] * exp(-j 2 pi k / N),      N = samples per cycle

``|X|`` is the RMS value of the fundamental and ``arg X`` its angle relative to a cosine that
starts at the first sample of the cycle used. All channels of a window are evaluated on the same
samples, so the angles between channels (and between the two terminals of a line view) are
meaningful; the absolute angle is not.

**Behaviour** (asserted in ``tests/unit/test_phasor.py``):

* a constant offset and every integer harmonic are rejected exactly (they are orthogonal to the
  fundamental over one full cycle);
* a *decaying* DC offset is not a constant over the cycle and leaks into the estimate; the error
  grows with the offset and shrinks as the time constant grows. No mimic filter is applied here;
* an off-nominal frequency leaks as well (the cycle no longer holds an integer period).

**Which cycle.** ``fundamental_phasors`` uses the *last* complete cycle of the window by default:
in a fault-active window the fault starts somewhere inside the window, so the last cycle is the
one most likely to be entirely in the fault state. A window whose last cycle still contains the
inception yields a mixed pre-fault/fault phasor; the estimator does not detect that.
"""

from __future__ import annotations

import numpy as np

_A = np.exp(2j * np.pi / 3)  # the 120 degree operator "a"


def samples_per_cycle(sampling_rate_hz: float, nominal_hz: float) -> int:
    """Samples in one nominal cycle. The full-cycle DFT needs an integer number; fails loud if
    the sampling rate is not an integer multiple of the nominal frequency."""
    n = sampling_rate_hz / nominal_hz
    if abs(n - round(n)) > 1e-9 or round(n) < 4:
        raise ValueError(
            f"sampling rate {sampling_rate_hz} Hz is not an integer multiple (>= 4) of the "
            f"nominal frequency {nominal_hz} Hz; the full-cycle DFT is undefined"
        )
    return int(round(n))


def fundamental_phasors(x: np.ndarray, n_cycle: int, *, end: int | None = None) -> np.ndarray:
    """RMS fundamental phasors: ``x`` ``(..., samples, channels)`` -> complex ``(..., channels)``.

    Uses the ``n_cycle`` samples that end at sample index ``end`` (exclusive; default: the end of
    the window, i.e. the last complete cycle).
    """
    x = np.asarray(x)
    length = x.shape[-2]
    end = length if end is None else end
    if not n_cycle <= end <= length:
        raise ValueError(f"cannot take a {n_cycle}-sample cycle ending at {end} of {length}")
    cycle = x[..., end - n_cycle : end, :].astype(np.float64)
    kernel = np.exp(-2j * np.pi * np.arange(n_cycle) / n_cycle)
    return (np.sqrt(2.0) / n_cycle) * np.einsum("...kc,k->...c", cycle, kernel)


def symmetrical_components(abc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-, positive- and negative-sequence components of three-phase phasors.

    ``abc`` is complex ``(..., 3)`` in phase order L1, L2, L3 (positive rotation L1 -> L2 -> L3).
    Returns ``(X0, X1, X2)``, each ``(...,)``, referred to phase L1.
    """
    abc = np.asarray(abc)
    if abc.shape[-1] != 3:
        raise ValueError(f"expected (..., 3) phase phasors, got shape {abc.shape}")
    xa, xb, xc = abc[..., 0], abc[..., 1], abc[..., 2]
    zero = (xa + xb + xc) / 3.0
    positive = (xa + _A * xb + _A**2 * xc) / 3.0
    negative = (xa + _A**2 * xb + _A * xc) / 3.0
    return zero, positive, negative
