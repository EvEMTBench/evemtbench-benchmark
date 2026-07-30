---
name: Feature request
about: Propose a new task, baseline, grid, metric, or capability
title: "[feat] "
labels: enhancement
---

**What & why**
What should the benchmark support, and what protection/ML question does it answer?

**Kind**
- [ ] New task (which label column / objective?)
- [ ] New baseline (model family)
- [ ] New grid / topology
- [ ] New metric or analysis
- [ ] Other

**Contract impact**
Does this touch a **frozen artifact** (labels / metrics / splits / window contract)? If so it is a
major benchmark bump — note that here. See `docs/versioning.md`.

**Data availability**
Is the data already present (e.g. an existing `y`/settings column, a graph artifact), or does it need
the producer to generate it? See `docs/DATA_LAYOUT.md` / `docs/task_suite_roadmap.md`.
