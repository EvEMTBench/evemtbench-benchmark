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

BENCHMARK_VERSION = "1.1.0"
"""Current benchmark contract version.

v1.1.0 (current): **scoped contract revision — corrected multi_grid source-corpus splits.** The
v1.0 ``multi_grid_pretrain`` splits were episode-level, and a multi_grid episode is one *line* of
one simulation (``sample_id = sim_idx*10^4 + line_idx``), so sibling lines of a simulation — and
every generated topology — straddled train/val/test. That never contaminated the scored target-grid
evaluations (reference-grid splits are simulation-level), but it weakened source-domain model
selection for the transfer protocols. v1.1 regenerates those three splits **topology-grouped**
(``splits/v1.1/multi_grid_pretrain/``; no ``graph_id``/``result_id`` crosses a partition), so:

* ``multi_grid_pretrain`` / ``transfer_zeroshot`` / ``transfer_finetune`` results are re-scored
  under the corrected splits and land in ``outputs/1.1.0/`` — NOT comparable to their 1.0.0
  counterparts (which are retained: ``outputs/1.0.0/`` + the frozen ``results/1.0.0/`` bundle).
* Every ``held_out`` artifact (splits, tasks, labels, metrics, window contract) is byte-identical
  to v1.0.0, so the published held_out results remain the current ones and are NOT re-run; they
  stay in ``outputs/1.0.0/`` and keep their ``benchmark_version: 1.0.0`` stamp.

:func:`results_version` / :func:`splits_version` encode that per-protocol scoping; see
``docs/versioning.md`` ("scoped revision") and ``CHANGELOG.md``.

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
# v1.1.0 note: the held_out splits are UNCHANGED (still read from splits/v1.0/); only the
# multi_grid_pretrain splits moved to splits/v1.1/ (topology-grouped). SPLITS_VERSION therefore
# stays the held_out artifact version; use splits_version(protocol) wherever the protocol is known.
SPLITS_VERSION = "1.0"

# The corrected multi_grid source-corpus splits (grouped; splits/v1.1/multi_grid_pretrain/).
MULTI_GRID_SPLITS_VERSION = "1.1"


def splits_version(protocol: str) -> str:
    """Committed-splits directory version for ``protocol``. ``multi_grid_pretrain`` reads the
    corrected v1.1 grouped splits; every other protocol (held_out, and the transfer protocols,
    which reuse the *target grid's* held_out split at load time) reads the frozen v1.0 tree."""
    return MULTI_GRID_SPLITS_VERSION if protocol == "multi_grid_pretrain" else SPLITS_VERSION


# The held_out results produced (and published) under the 1.0.0 contract remain the current
# results: every artifact that protocol reads is byte-identical under 1.1.0, so re-running would
# only churn published numbers. The pretrain/transfer protocols are re-scored at 1.1.0.
HELD_OUT_RESULTS_VERSION = "1.0.0"


def results_version(protocol: str) -> str:
    """The ``outputs/<version>/`` tree holding a protocol's CURRENT results (see the v1.1.0
    docstring above: held_out stays at 1.0.0; pretrain/transfer re-scored at 1.1.0)."""
    return HELD_OUT_RESULTS_VERSION if protocol == "held_out" else BENCHMARK_VERSION


# Every outputs/results version that carries current results — reporting/audit layers iterate
# these; the freshness gate owns ONLY the BENCHMARK_VERSION derived trees (older versions'
# results/<v>/ bundles and audit/provenance/<v>/ indices are frozen history, kept for comparison).
CURRENT_RESULTS_VERSIONS = tuple(dict.fromkeys((HELD_OUT_RESULTS_VERSION, BENCHMARK_VERSION)))
