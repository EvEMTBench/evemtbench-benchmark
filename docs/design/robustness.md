# Design: noise & measurement-realism robustness

**Status: designed, not built** — a paper Open Challenge. Code seam:
`evemtbench.evaluation.robustness`.

## Why

The dataset is ideal EMT simulation: noiseless, perfectly synchronized, infinite-resolution
sensors. Tier-1 tasks are near-solved there — the community frontier is **how fast scores decay
under measurement realism**, not another 0.001 on clean data. The benchmark should own the
perturbation taxonomy so results are comparable (everyone else's "we added 30 dB noise" is not).

## Perturbation taxonomy (applied to `x` at evaluation time; never to training data by default)

| kind | knob | models |
|---|---|---|
| `awgn` | SNR dB | thermal/quantization noise floor |
| `harmonics` | THD % (odd harmonics of f₀) | converter-rich grids |
| `ct_saturation` | remanence/burden factor | current-transformer clipping on high-current faults |
| `sensor_dropout` | p(channel) | lost/flatlined channels |
| `time_skew` | ms per cubicle | unsynchronized multi-end sampling (line/global views) |
| `fs_mismatch` | resample factor | devices at other sampling rates |

## Contract

- `robustness.perturb(x, kind, level, rng)` — pure, vectorized, seed-deterministic; operates on
  the projected `(n, l, c)` view so it composes with every observability setting.
- `robustness.curve(model, task, kind, levels, seeds)` — evaluates a **frozen** fitted model
  across levels; emits the standard result-doc shape with `aggregate[metric]` per level, so the
  existing renderers/CI plumbing (mean/std/ci95) apply unchanged.
- Robustness runs are **evaluation-only cells**: models come from the released checkpoints
  (`run_record.json.checkpoints`) — no retraining, no new training privileges.
- Curves are reported as *area-under-decay* + the level where the score crosses the majority
  floor (`coverage --check-floor` reuse).

## Open questions

- Whether a small frozen "realism suite" (one level per kind) becomes a scored v-next test set,
  vs. curves staying diagnostic.
- Physical-unit calibration per grid (SNR is relative to each grid's signal scale — pin per-grid
  reference power from the train split only, to keep the no-tuning rule).

## Acceptance criteria

- [ ] `evemtbench.evaluation.robustness` applies each perturbation as a pure eval-time transform
      (training data untouched — asserted by a test).
- [ ] Every perturbation is parameterized and its settings recorded in `result.json`-adjacent
      metadata so a robustness run is reproducible from its artifacts alone.
- [ ] A robustness sweep renders as its own protocol/section in the audit coverage matrix.
- [ ] Baseline degradation curves for one grid are committed and cited in the paper's Open
      Challenge section (no headline-table changes — additive protocol, minor bump).
