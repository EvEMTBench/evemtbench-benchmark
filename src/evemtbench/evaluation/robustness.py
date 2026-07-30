"""Measurement-realism perturbations — the noise-robustness seam (designed, NOT built).

The dataset is ideal EMT simulation; tier-1 tasks are near-solved on it. The community frontier
is decay under measurement realism — but only if the perturbation taxonomy is owned by the
benchmark so levels are comparable across papers. Design: ``docs/design/robustness.md``.

This module freezes the *interface* (perturbation kinds + the curve contract); every function
raises :class:`NotImplementedError` with a pointer to the design doc. Robustness runs are
evaluation-only: frozen fitted models (released checkpoints), never retraining.
"""

from __future__ import annotations

import numpy as np

#: The frozen perturbation taxonomy (knob semantics in docs/design/robustness.md).
PERTURBATION_KINDS = (
    "awgn",  # additive white Gaussian noise (level = SNR dB)
    "harmonics",  # odd-harmonic distortion (level = THD %)
    "ct_saturation",  # current-transformer clipping (level = burden factor)
    "sensor_dropout",  # flatlined channels (level = p per channel)
    "time_skew",  # per-cubicle sampling skew (level = ms)
    "fs_mismatch",  # off-nominal sampling rate (level = resample factor)
)

_DESIGN = "designed but not built - see docs/design/robustness.md"


def perturb(x: np.ndarray, kind: str, level: float, rng: np.random.Generator) -> np.ndarray:
    """Apply one perturbation to a projected ``(n, l, c)`` batch — pure, vectorized,
    seed-deterministic, composable with every observability setting."""
    if kind not in PERTURBATION_KINDS:
        raise ValueError(f"unknown perturbation {kind!r}; known: {PERTURBATION_KINDS}")
    raise NotImplementedError(f"perturb(kind={kind!r}, level={level}): {_DESIGN}")


def curve(model, task, kind: str, levels: list[float], *, seeds: list[int]) -> dict:
    """Score a FROZEN fitted model across perturbation levels; returns the standard result-doc
    shape (``aggregate[metric]`` per level, mean/std/ci95 across seeds) so the existing
    renderers apply unchanged."""
    raise NotImplementedError(f"curve(kind={kind!r}, levels={levels}, seeds={seeds}): {_DESIGN}")
