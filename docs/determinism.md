# Determinism: what is guaranteed, what is not

Being honest about reproducibility is part of the benchmark's credibility (Benchmark Plan §17).

## Guaranteed (bit-for-bit, across machines)

- **Splits** — committed `.txt` files are the source of truth and regenerate identically from a
  seeded permutation of sorted episode ids (stable sha256-derived seeds, not salted `hash()`).
  Leakage tests assert committed == regenerated (`leakage.verify_held_out_regenerates`).
- **Labels** — derivations are pure functions of the window's `y` row (`event_type` / overlap /
  `y_fault_*`), test-pinned.
- **Metrics** — deterministic given `(y_true, y_pred)`.
- **Feature/window extraction** — the window contract (length/step) is fixed and gated by the
  manifest `config_fingerprint`; the synthetic fixture is regenerated deterministically.

## Not guaranteed (and why)

- **Trained model weights / GPU kernels** — MLP/GRU training on GPU is not bit-identical across
  hardware, CUDA/cuDNN versions, or runs, even with fixed seeds (nondeterministic kernels,
  atomics). We therefore report **mean ± std over seeds** rather than a single number, and the
  `run_record.json` logs seeds, git sha, lockfile hash, torch/CUDA versions, the executing
  host + SLURM job id, and SHA-256 fingerprints of the split files and window manifests read.
- **Floating-point reductions** — large sums/means can differ in the last ULPs across BLAS
  backends; this does not affect the reported metric precision.

## Reproducibility kit

- `requirements.lock` (uv) pins the CI dependency set; CI installs it in a clean venv.
- On HPC, the `py_dl` conda base provides the interpreter + CUDA torch, and the package is run
  from source via `PYTHONPATH=src` (no per-job editable install, so concurrent jobs can't race).
- Every run emits `run_record.json` so a result can be traced to its exact code + environment.
