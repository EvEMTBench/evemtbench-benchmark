<!-- Thanks for contributing to EvEMTBench! Keep PRs small and scoped (one concern). -->

## What & why
<!-- What does this change and why? Link any issue. -->

## Type of change
- [ ] Bug fix (score-neutral)
- [ ] New task / baseline / grid (minor benchmark bump)
- [ ] Change to a **frozen artifact** (labels / metrics / splits / window contract) — **major bump**
- [ ] Docs / tooling / CI only

## Checklist
- [ ] `ruff check` + `ruff format --check` + `mypy src` pass
- [ ] `pytest` passes (new/changed behavior has a test at the cheapest catching layer)
- [ ] If a **frozen artifact** changed: `BENCHMARK_VERSION`/`CONTRACT_VERSION` bumped + `CHANGELOG.md` updated + expectation tests updated in the same PR
- [ ] `RESULTS.md` / `leaderboard.html` are **not** hand-edited (they are derived; the freshness gate must pass — no `--no-verify`)
- [ ] For a new task/baseline: followed `docs/extending.md` / `CONTRIBUTING.md`
- [ ] Commit(s) are small and scoped to `main` (no `git add -A` grab-bags)
