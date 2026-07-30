# Contributing to EvEMTBench

Thanks for helping build a fair, reproducible protection benchmark. Orientation:
[docs/README.md](docs/README.md) (doc index) · [docs/extending.md](docs/extending.md)
(add a grid/task/baseline, worked examples) · [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md)
(the on-disk contract). The rules below always hold.

## Workflow

- **Branches**: `feat/<topic>`, `fix/<topic>`, `docs/<topic>` off `main`. Long-lived integration
  branches (e.g. a version-foundation branch that the sweep loop runs from) are the documented
  exception and merge to `main` via MR when their milestone lands.
- **Merge requests**: all changes reach `main` through an MR/PR with CI green and the checklist in
  [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md) ticked — that checklist is
  the Definition of Done.
  Review-before-merge is required for anything touching a frozen artifact or `src/evemtbench/`.
- **Direct-commit exception**: the autonomous sweep loop's machine-generated
  `chore(results): sweep progress snapshot` commits (regenerated, freshness-gated views only).
- **Never `--no-verify`.** The pre-commit freshness gate is part of the benchmark's credibility.
- **Quality ratchet**: the CI coverage floor (`[tool.coverage.report].fail_under`) may be raised to
  the CI-reported figure minus one point after a stable week; it is never lowered without a dated,
  reasoned note in the PR that lowers it. New `src/` modules join the strict-mypy list in
  `pyproject.toml`.
- **Maintainers**: `.github/CODEOWNERS` currently names a single owner —
  <!-- TODO(juoelhaf): nominate a second maintainer (e.g. a co-author) to lift the bus factor. -->

## What is frozen (and why)

The benchmark is versioned semantically (see [`docs/versioning.md`](docs/versioning.md)).
Within a major version, these are **frozen** and a change bumps the major version:

- test sets and the official **metrics**,
- **derived labels** (`src/evemtbench/tasks/labels/`) and the NoEvent policy,
- the committed **split files** (`src/evemtbench/splits/`).

Adding a task or a baseline is a **minor** bump and does not touch the above.

## The golden rule

A user must be able to evaluate their own model using only `data/`, `tasks/`, `splits/`, and
`evaluation/` — **never** importing `baselines/`. CI enforces this
(`tests/integration/test_import_isolation.py`). Keep baselines a leaf.

## Adding a baseline

1. Implement `evemtbench.baselines.interface.Baseline` (`fit` / `predict` → predictions file).
2. Add a config under `src/evemtbench/configs/baseline/`.
3. A **results PR** must include: the config, the seed list, and a predictions file that
   `evemtbench-evaluate` reproduces to the reported numbers, tagged with the `BENCHMARK_VERSION`.

## Dev setup

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest          # unit, golden, schema, leakage, smoke
uv run ruff check src tests
```

Use lowercase config keys, log seeds, never use benchmark data for normalization/tuning,
and never split windows independently of their simulation (`sample_id`).

**Golden metrics discipline:** `tests/golden/fixture_metrics.json` pins the exact fixture
aggregates of the deterministic baselines (majority, threshold) — any scoring drift fails
`tests/integration/test_golden_fixture_metrics.py` by name. If your change moves them
*legitimately* (label derivation, metric fix), refresh with
`GOLDEN_UPDATE=1 pytest tests/integration/test_golden_fixture_metrics.py` and commit the
golden file **in the same commit** as the change — never in a follow-up.
