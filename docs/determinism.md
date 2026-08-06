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

There are **two** environments, pinned by two different files. Conflating them is the mistake this
section exists to prevent.

| File | What it pins | Who installs it |
| --- | --- | --- |
| `env/results-v<version>.lock` | the environment the **published results** were computed in — the `py_dl` conda prefix, including torch + the CUDA stack | nobody automatically; it is a *provenance record* |
| `requirements-ci-py3.10.lock`, `requirements-ci-py3.12.lock` | the **CI test gate** — a deliberately torch-free dependency set, hash-pinned | GitHub CI, with `--require-hashes` |
| `requirements.lock` | *legacy.* The CI pin as it stood during the v1.0.0 sweep | nobody |

- **Results pin.** `env/results-v1.0.0.lock` is *captured, not authored*
  (`hpc/capture_results_env.sh`, run under SLURM): a `pip list --format=freeze` of the conda prefix
  that produced `outputs/1.0.0`. Because that prefix is mutable, the file is a faithful record
  rather than a byte-reconstructable specification. The guarantee that matters is the independent
  cross-check: every `run_record.json` separately captured `numpy`/`pandas`/`scikit-learn`/`scipy`
  plus `python`/`torch` at run time, and **all 1876 records agree with this file exactly**
  (`PYTHONPATH=src python hpc/verify_results_env.py`, sidecar
  `outputs/1.0.0/results_env_verification.json`). Agreement of two independently captured records
  is stronger evidence than either alone.
- **CI pin.** GitHub CI installs its lock with `uv pip install --require-hashes`, then the package
  with `--no-deps`. One lock per interpreter, because `uv` resolves per `--python-version` and a
  single hash-pinned file cannot serve both matrix legs. The **3.12 leg is additionally constrained
  to the results environment's scoring-package versions** (`env/results-scoring.constraints`), so
  the gate exercises the same numerics the paper reports; the 3.10 leg is unconstrained and only
  proves the package still works on the oldest supported interpreter. The GitLab mirror (3.11) is
  intentionally unpinned — see the comment in `.gitlab-ci.yml`.
- **`requirements.lock` is frozen, not current.** It was compiled on a pre-3.10 interpreter, pins a
  `[dev]`-only subset, and contains **no torch**, so it never described the results environment.
  It is kept **byte-for-byte unchanged** because its sha256
  (`2685c1c2…`) is the `lockfile_sha256` recorded in all 1876 v1.0.0 run records; rewriting it
  would silently invalidate a published provenance field. New runs additionally record
  `results_env_sha256`.
- On HPC, the `py_dl` conda base provides the interpreter + CUDA torch, and the package is run
  from source via `PYTHONPATH=src` (no per-job editable install, so concurrent jobs can't race).
- Every run emits `run_record.json` so a result can be traced to its exact code + environment.
