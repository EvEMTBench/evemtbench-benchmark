"""Full-cycle DFT phasors and symmetrical components on synthetic sinusoids (50 and 60 Hz)."""

import numpy as np
import pytest

from evemtbench.baselines._phasor import (
    fundamental_phasors,
    samples_per_cycle,
    symmetrical_components,
)

FS = 9600.0
WINDOW = 480  # 0.05 s at 9.6 kHz: 2.5 cycles at 50 Hz, 3 cycles at 60 Hz


def _wave(f0, rms, angle_deg, *, n=WINDOW, dc=0.0, tau=None, harmonic=0, harmonic_rms=0.0):
    t = np.arange(n) / FS
    x = np.sqrt(2) * rms * np.cos(2 * np.pi * f0 * t + np.radians(angle_deg))
    if harmonic:
        x = x + np.sqrt(2) * harmonic_rms * np.cos(2 * np.pi * harmonic * f0 * t + 0.3)
    if dc:
        x = x + (dc * np.exp(-t / tau) if tau else dc)
    return x


def _phasor_at_window_start(p, f0, n_cycle):
    """Re-reference a last-cycle phasor to t = 0 of the window (the cycle starts later)."""
    start = WINDOW - n_cycle
    return p * np.exp(-2j * np.pi * f0 * start / FS)


@pytest.mark.parametrize("f0, expected_n", [(50, 192), (60, 160)])
def test_samples_per_cycle(f0, expected_n):
    assert samples_per_cycle(FS, f0) == expected_n


def test_samples_per_cycle_rejects_non_integer():
    with pytest.raises(ValueError, match="integer multiple"):
        samples_per_cycle(9600.0, 55.0)


@pytest.mark.parametrize("f0", [50, 60])
def test_clean_sinusoid_magnitude_and_angle(f0):
    n = samples_per_cycle(FS, f0)
    x = np.stack([_wave(f0, 10.0, 25.0), _wave(f0, 3.0, -100.0)], axis=-1)  # (l, 2)
    p = _phasor_at_window_start(fundamental_phasors(x, n), f0, n)
    np.testing.assert_allclose(np.abs(p), [10.0, 3.0], rtol=1e-10)
    np.testing.assert_allclose(np.degrees(np.angle(p)), [25.0, -100.0], atol=1e-8)


@pytest.mark.parametrize("f0", [50, 60])
def test_constant_offset_and_harmonics_are_rejected(f0):
    n = samples_per_cycle(FS, f0)
    for h in (2, 3, 5):
        x = _wave(f0, 10.0, 40.0, dc=4.0, harmonic=h, harmonic_rms=2.0)[:, None]
        p = fundamental_phasors(x, n)
        np.testing.assert_allclose(np.abs(p), [10.0], rtol=1e-9)


@pytest.mark.parametrize("f0", [50, 60])
def test_decaying_dc_offset_leaks_and_shrinks_with_time_constant(f0):
    """A decaying DC is not orthogonal to the fundamental: the error is nonzero, bounded, and
    smaller for a slower decay. Documents the estimator; no mimic filter is applied."""
    n = samples_per_cycle(FS, f0)
    errors = []
    for tau in (0.01, 0.03, 0.1):
        x = _wave(f0, 10.0, 0.0, dc=10.0 * np.sqrt(2), tau=tau)[:, None]
        errors.append(abs(abs(fundamental_phasors(x, n)[0]) - 10.0) / 10.0)
    assert errors[0] > 1e-4  # a fully offset wave with a 10 ms time constant is visibly biased
    assert max(errors) < 0.12  # ... but stays within about a tenth of the magnitude
    assert errors[2] < errors[1]  # slower decay -> closer to a constant -> smaller leak


def test_batched_shape_and_cycle_selection():
    n = samples_per_cycle(FS, 50)
    x = np.zeros((7, WINDOW, 6))
    x[:, WINDOW - n :, 0] = _wave(50, 5.0, 0.0, n=n)  # the signal exists in the last cycle only
    p = fundamental_phasors(x, n)
    assert p.shape == (7, 6)
    np.testing.assert_allclose(np.abs(p[:, 0]), 5.0, rtol=1e-10)
    first = fundamental_phasors(x, n, end=n)  # the first cycle of the window is empty
    np.testing.assert_allclose(np.abs(first), 0.0, atol=1e-12)
    with pytest.raises(ValueError, match="cannot take"):
        fundamental_phasors(x[:, : n - 1], n)


def test_symmetrical_components():
    a = np.exp(2j * np.pi / 3)
    positive = np.array([1, a**2, a]) * (2 + 1j)  # balanced, positive rotation
    negative = np.array([1, a, a**2]) * 0.5
    zero = np.array([1, 1, 1]) * (0.25 - 0.1j)
    x0, x1, x2 = symmetrical_components(positive)
    np.testing.assert_allclose([x0, x1, x2], [0, 2 + 1j, 0], atol=1e-12)
    x0, x1, x2 = symmetrical_components(positive + negative + zero)
    np.testing.assert_allclose([x0, x1, x2], [0.25 - 0.1j, 2 + 1j, 0.5], atol=1e-12)
    batched = symmetrical_components(np.tile(positive, (4, 1)))
    assert batched[1].shape == (4,)
    with pytest.raises(ValueError, match="phase phasors"):
        symmetrical_components(np.zeros((4, 2), complex))
