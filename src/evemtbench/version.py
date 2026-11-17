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

BENCHMARK_VERSION = "1.3.0"
"""Current benchmark contract version.

v1.3.0 (current): **scoped revision, by cell — faulted-phase labels of ``flt_1phg_hif_w_arc``.**
In the 1.x windows every fault-active window of the high-impedance-with-arc single-phase-to-ground
fault carries ``y_phase_A = y_phase_B = y_phase_C = 0`` (a producer defect: the type is missing
from the one-phase rule of ``psp_helper``'s ``_phases_from_select``), so the ``fault_phase``
valid-sample rule (at least one phase bit) silently dropped the whole class. Affected: CIGRE MV
``adapt_grid`` (7,752 windows, 456 episodes) and ``benchmark`` (1,392 windows, 87 episodes) and
the ``multi_grid`` Template20kV line view (4,318 windows, 254 line episodes); no other type, grid
or family. The source label (``events/event_phase_select_1ph``) names exactly one phase for each
of these episodes. v1.3 repairs the labels WITHOUT touching the windows: a committed, sha256-pinned,
episode-keyed label patch (``data/label_patches/``, patch id ``LP-1``) restores the bits from the
source label. ``load_task`` applies it only to ``fault_phase`` tasks of a v1.3.0 cell, and only on
the partitions whose ``manifest_sha256`` it was built for. So:

* exactly the ``fault_phase`` cells that read a repaired label move to ``outputs/1.3.0/``
  (151 cells; :func:`results_version` with ``grid``/``source``): ``held_out`` on cigre_mv,
  ``multi_grid_pretrain`` on template_20kv, and the transfer cells that target cigre_mv (any
  source) or start from the template_20kv source (any target);
* every other cell, including the other ``fault_phase`` cells, keeps its tree and stamp; the
  1.0.0 / 1.1.0 records re-evaluate exactly, because the patch is off for them.

v1.2.0 (superseded by v1.3.0 for the affected ``fault_phase`` cells only): **scoped contract
revision — fault-location valid samples are line faults only.** The FL target is the distance
along the faulted *line*, but the v1.0/v1.1 gate (fault-active
AND non-null ``y_fault_location``) also admitted the **bus-target** fault episodes
(``y_fault_line`` = ``MainBus*``) of the ``adapt_grid`` family, for which the producer fills a
location that is no position on any line (``docs/known_issues.md`` §2). v1.2 adds the missing
condition to the gate — ``y_fault_line`` names a line (``MainLn*``), see
``tasks/valid_samples.py``. That changes the FL train/val/test samples drawn from a reference
grid's ``adapt_grid`` family and nothing else, so:

* the fault-location tasks' ``held_out`` / ``transfer_zeroshot`` / ``transfer_finetune`` results
  are re-run under the corrected gate and land in ``outputs/1.2.0/`` — NOT comparable to their
  1.0.0 / 1.1.0 counterparts, which are retained (``outputs/1.0.0/``, ``outputs/1.1.0/`` and the
  frozen ``results/1.0.0/`` / ``results/1.1.0/`` bundles).
* the fault-location ``multi_grid_pretrain`` source models stay current in ``outputs/1.1.0/``:
  the multi_grid corpora hold no bus-target window in the gate, so the corrected gate selects the
  same samples there.
* every other task is untouched (valid-sample rule, labels, splits, metrics and window contract
  are byte-identical): its ``held_out`` results stay current in ``outputs/1.0.0/`` and its
  pretrain/transfer results in ``outputs/1.1.0/``, each with its original ``benchmark_version``
  stamp, and none is re-run.

:func:`results_version` encodes that per-(protocol, task) scoping; see ``docs/versioning.md``
("scoped revision") and ``CHANGELOG.md``.

v1.1.0 (superseded by v1.2.0; its pretrain/transfer results for every task but fault location,
and the fault-location source models, remain current and keep their 1.1.0 identifiers by design):
**scoped contract revision — corrected multi_grid source-corpus splits.** The
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

v1.0.0 (superseded by v1.1.0; its ``held_out`` artifacts and — fault location excepted since
v1.2.0 — results remain current and keep their 1.0.0 identifiers by design): first frozen release,
cut on the fully
re-simulated EvEMTBench dataset. Every grid is re-windowed to the psp_helper window-contract 1.0.0
across the ``held_out`` (adapt_grid + benchmark) and pre-training/transfer (multi_grid) families.
The multi_grid pre-training set is Template20kV, Template110kV and Template345kV (EHV). Split files
are regenerated for the new data, so this is a **major** bump — all pre-1.0 scores are NOT
comparable. (Results land under ``outputs/1.0.0/``.)

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
# results for every task but fault location: every artifact they read is byte-identical under
# 1.1.0 and 1.2.0, so re-running would only churn published numbers.
HELD_OUT_RESULTS_VERSION = "1.0.0"

# The pretrain/transfer protocols were re-scored at 1.1.0 (corrected multi_grid source splits).
# Those results stay current for every task but fault location, and for the fault-location
# multi_grid_pretrain source models (no bus-target window in the multi_grid FL gate).
TRANSFER_RESULTS_VERSION = "1.1.0"

# Fault-location results under the corrected (line-targets-only) valid-sample gate.
FL_RESULTS_VERSION = "1.2.0"

# Faulted-phase results with the LP-1 label patch (flt_1phg_hif_w_arc phase bits restored), for
# exactly the cells that read a repaired label (see results_version).
FP_RESULTS_VERSION = "1.3.0"

# Task-id prefix of the fault-location tasks (``Function.FAULT_LOCATION.value + "_"``). The prefix
# is used instead of the task registry because ``evemtbench.tasks`` imports this module (an import
# cycle), and because reporting code also resolves versions for task ids read from result files.
# tests/unit/test_results_version.py pins that it agrees with the registry for every task.
_FL_TASK_PREFIX = "fault_location_"

# Protocols whose fault-location samples the corrected gate changes: everything that trains,
# selects or tests on a reference grid's ``adapt_grid`` family.
_FL_RESCORED_PROTOCOLS = frozenset({"held_out", "transfer_zeroshot", "transfer_finetune"})


def is_fault_location_task(task: str | None) -> bool:
    """True if ``task`` (a task id such as ``fault_location_local``) is a fault-location task."""
    return task is not None and task.startswith(_FL_TASK_PREFIX)


# Task-id prefix of the faulted-phase tasks (``Function.FAULT_PHASE.value + "_"``); a prefix for
# the same reasons as _FL_TASK_PREFIX (pinned against the registry in the results-version tests).
_FP_TASK_PREFIX = "fault_phase_"

# The grids whose 1.x windows the LP-1 label patch changes: the reference grid (held_out and the
# transfer targets) and the multi_grid pre-training source.
FP_PATCHED_REFERENCE_GRID = "cigre_mv"
FP_PATCHED_PRETRAIN_GRID = "template_20kv"

# The default transfer source (the runner's --pretrain-grid default, coverage.PRETRAIN_GRID): a
# transfer cell with ``source=None`` starts from this grid. Pinned against coverage in the tests.
DEFAULT_PRETRAIN_GRID = "template_20kv"

_TRANSFER_PROTOCOLS = frozenset({"transfer_zeroshot", "transfer_finetune"})


def is_fault_phase_task(task: str | None) -> bool:
    """True if ``task`` (a task id such as ``fault_phase_line``) is a faulted-phase task."""
    return task is not None and task.startswith(_FP_TASK_PREFIX)


def fault_phase_cell_patched(protocol: str, grid: str, source: str | None = None) -> bool:
    """Whether a ``fault_phase`` cell reads a label the LP-1 patch repairs.

    * ``held_out`` reads the target grid's own labels: patched on cigre_mv.
    * ``multi_grid_pretrain`` reads the source corpus: patched on template_20kv.
    * the transfer protocols score (and fine-tune) on the target's labels and start from the
      ``multi_grid_pretrain`` checkpoint of their source: patched when the target is cigre_mv or
      the source is template_20kv (``source=None`` = the default source, template_20kv).
    """
    if protocol == "held_out":
        return grid == FP_PATCHED_REFERENCE_GRID
    if protocol == "multi_grid_pretrain":
        return grid == FP_PATCHED_PRETRAIN_GRID
    if protocol in _TRANSFER_PROTOCOLS:
        src = DEFAULT_PRETRAIN_GRID if source is None else source
        return grid == FP_PATCHED_REFERENCE_GRID or src == FP_PATCHED_PRETRAIN_GRID
    return False


def results_version(
    protocol: str,
    task: str | None = None,
    *,
    grid: str | None = None,
    source: str | None = None,
) -> str:
    """The ``outputs/<version>/`` tree holding the CURRENT results of one cell.

    * faulted-phase task on a cell that reads an LP-1-repaired label -> 1.3.0 (see the v1.3.0
      docstring above and :func:`fault_phase_cell_patched`). ``grid`` is the cell's grid (the
      target grid for transfer, the source corpus for ``multi_grid_pretrain``) and ``source`` the
      transfer pre-training source (``None`` = the default source, template_20kv);
    * fault-location task x ``held_out`` / ``transfer_zeroshot`` / ``transfer_finetune`` -> 1.2.0
      (corrected FL gate, see the v1.2.0 docstring above);
    * otherwise ``held_out`` -> 1.0.0 and the pretrain/transfer protocols -> 1.1.0 (this includes
      fault-location x ``multi_grid_pretrain``, whose samples the corrected gate does not change).

    A faulted-phase task REQUIRES ``grid``: its version depends on the cell, and resolving it
    without one would silently pick the wrong tree, so it raises ``ValueError`` instead.

    Pass ``task`` (and ``grid``/``source``) wherever a concrete cell, row or record is resolved.
    ``task=None`` gives the protocol's tree for the tasks the scoped revisions left untouched; use
    it only for wording ("held_out results v1.0.0"). Code that walks whole trees iterates
    :data:`CURRENT_RESULTS_VERSIONS` and keeps a row only if its tree equals
    ``results_version(row.protocol, row.task, grid=row.grid, source=row.source)``.
    """
    if is_fault_phase_task(task):
        if grid is None:
            raise ValueError(
                f"results_version({protocol!r}, {task!r}) needs grid= (and source= for a transfer "
                "cell): the faulted-phase results version is scoped per cell since v1.3.0"
            )
        if fault_phase_cell_patched(protocol, grid, source):
            return FP_RESULTS_VERSION
    if is_fault_location_task(task) and protocol in _FL_RESCORED_PROTOCOLS:
        return FL_RESULTS_VERSION
    return HELD_OUT_RESULTS_VERSION if protocol == "held_out" else TRANSFER_RESULTS_VERSION


def _version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def label_patches_enabled(task: str | None, version: str) -> bool:
    """Whether the committed label patches (``data/label_patches/``) apply to a run of ``task``
    stamped ``version``: faulted-phase tasks only, and only from v1.3.0 on. A 1.0.0 / 1.1.0 record
    is therefore re-evaluated on exactly the labels it was computed with."""
    return is_fault_phase_task(task) and _version_tuple(version) >= _version_tuple(
        FP_RESULTS_VERSION
    )


# Every outputs/results version that carries current results — reporting/audit layers iterate
# these; the freshness gate owns ONLY the BENCHMARK_VERSION derived trees (older versions'
# results/<v>/ bundles and audit/provenance/<v>/ indices are frozen history, kept for comparison).
CURRENT_RESULTS_VERSIONS = tuple(
    dict.fromkeys(
        (
            HELD_OUT_RESULTS_VERSION,
            TRANSFER_RESULTS_VERSION,
            FL_RESULTS_VERSION,
            FP_RESULTS_VERSION,
        )
    )
)

# Version segment of the persistent projection / normalisation-statistics cache
# (``<data-root>/projcache/<DATA_CACHE_VERSION>/``, see ``data/staging.py``). The cache is
# content-addressed by the windowed data, the window contract and the exact rows of a view, and
# v1.2.0 changes none of the inputs (it narrows which rows a fault-location view holds; those
# views get new content keys inside the same directory). The segment therefore deliberately stays
# at "1.1.0" instead of following BENCHMARK_VERSION: bumping it would relocate the cache root and
# orphan hundreds of GB of still-valid projections. Bump it only when the windowed inputs or the
# window contract change in a way the content key does not capture.
DATA_CACHE_VERSION = "1.1.0"
