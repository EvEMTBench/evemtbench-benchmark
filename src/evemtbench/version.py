"""Semantic version of the *benchmark contract* (tasks, splits, labels, metrics).

This is distinct from the Python package version in ``pyproject.toml``. The benchmark
version is stamped into every :class:`~evemtbench.evaluation.evaluate.EvalResult` so that
results tables are always attributable to a frozen contract.

Versioning policy (see ``docs/versioning.md``):

* **major** — test sets or metrics change; a derived label changes; any split is
  regenerated; the corrupted-partition fix lands. Invalidates score comparability.
* **minor** — a task or baseline is added without touching frozen artifacts.
* **patch** — bug fixes that do not change any score.
"""

BENCHMARK_VERSION = "1.0.0"
"""Current benchmark contract version.

v1.0.0 (current): first frozen release, cut on the fully re-simulated EvEMTBench dataset. Every grid
is re-windowed to the psp_helper window-contract 1.0.0 across the ``held_out`` (adapt_grid +
benchmark) and pre-training/transfer (multi_grid) families. The multi_grid pre-training set is
Template20kV, Template110kV and Template345kV (EHV). Split files are regenerated for the new data,
so this is a **major**
bump — all pre-1.0 scores are NOT comparable. (Results land under ``outputs/1.0.0/``.)

v0.2: the in-distribution adaptation family is renamed ``fine_tune`` → ``adapt_grid``
to match the producer (``power-grid-and-ai-helper-functions``), which now writes windows under
``<root>/adapt_grid/``. This is a contract-level family relabel (split files regenerated as
``splits/v0.2/``), so it is a **major** bump: v0.1 scores are NOT comparable. Window contract,
metrics, derived labels and the task set are otherwise unchanged from v0.1.

The multi-topology **pre-training** family is likewise renamed ``train`` → ``multi_grid`` to
match the producer's on-disk directory (``<root>/multi_grid/``). ``train`` remains a read alias
(``data/roots.resolve_data_root``). This family feeds the added pre-training/transfer protocols
(``multi_grid_pretrain``, ``transfer_zeroshot``, ``transfer_finetune``) — a task/protocol
**addition**, not a change to any frozen ``held_out`` artifact, so v0.2 ``held_out`` scores stay
comparable.

v0.1 (retired): full reset onto the reconfigured data simulation (``LatestDataset/evemtbench`` is
now THE EvEMTBench). DoubleLine-only, 9 tasks {FD, FC, FL} × {local, line, global}, the
``held_out`` protocol (the in-distribution ``fine_tune`` family train/val/test + benchmark
held-out test). All prior grids/tasks/splits/results are retired and NOT comparable (see
``CHANGELOG.md``).

**Contract alignment (v0.2):** the windowed-data format the benchmark consumes is now governed
by the versioned psp_helper contract, vendored under ``evemtbench.contract`` and pinned via
``EXPECTED_CONTRACT_VERSION``. Policy: a **major** bump of that ``CONTRACT_VERSION`` forces a
**major** bump here. This release also removed an unreachable, incorrect fault-detection
``status`` fallback (``event_present``) — a **patch** with no score impact, since every dataset
carries ``overlap_ratio`` and thus always took the correct branch.
"""

# The split-file directory version (``src/evemtbench/splits/v1.0/...``). Tracks the
# major.minor of the committed split artifacts and is referenced by load_task().
# Bumped 0.2 -> 1.0 for the v1.0.0 release: splits regenerated on the re-simulated dataset
# (same methodology: stratify_by=event_type, fractions 0.7/0.15/0.15).
SPLITS_VERSION = "1.0"
