"""Shared helpers for the per-artifact paper builders (``hpc/paper_artifacts/build_*.py``).

One script → one artifact: each ``build_<x>.py`` imports from here and writes exactly one file
(``paper/manuscript/generated/*.tex`` or ``paper/manuscript/figures/*.pdf``). This module is the
single source of truth for data loading + cell selection, so the number a reviewer reads is provably
the same cell the audit dashboard traces (``best_cell_choice``). The thin shims
``hpc/make_paper_tables.py`` / ``make_supplement_tables.py`` re-export the aggregators
(``build_paper_tables`` / ``build_supplement_tables``) so the public results bundle
(``make_results_bundle.py``) and the freshness gate (``check_results_fresh.py``) keep working
byte-for-byte.

Wire numbers ONLY from a frozen benchmark version tag (see ``paper/AGENTS.md``); with ``outputs/``
absent every table renders ``\\res`` (the pre-freeze placeholder state).
"""

from __future__ import annotations

import sys
from pathlib import Path

# hpc/ is not a package; make the sibling RESULTS builder + the src/ package importable regardless
# of the cwd a standalone build_*.py is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from evemtbench.coverage import LEARNED_BASELINES, PRETRAIN_GRID, PRETRAIN_GRIDS
from evemtbench.evaluation.metrics import METRIC_DIRECTION
from make_results import BASELINES as _RESULT_BASELINES
from make_results import ci_val, load_rows, mean_val, warn_once

#: Baselines the per-baseline result tables enumerate. The conventional fault locator
#: (``impedance``, EXP-06) is never a selection candidate; its place in the result tables is an
#: open layout decision, so these builders leave it out of their set.
BASELINES = [b for b in _RESULT_BASELINES if b != "impedance"]
#: Baselines the per-baseline SI figures draw: every registered baseline, so the locator appears in
#: the local and line fault-location panels (the SI text says the figures show the complete suite).
FIGURE_BASELINES = list(_RESULT_BASELINES)

_HERE = Path(__file__).resolve()
_HPC = _HERE.parents[1]  # hpc/
_REPO = _HERE.parents[2]  # repo root

GEN_DIR = _REPO / "paper" / "manuscript" / "generated"
FIG_DIR = _REPO / "paper" / "manuscript" / "figures"


# --------------------------------------------------------------------------------------------------
# Artifact provenance header (golden rule: one artifact = one generator = one file).
# Every generated fragment opens with this block, so a reader of the .tex knows which script owns it
# and which committed inputs it was rendered from without consulting the build system.
# --------------------------------------------------------------------------------------------------
def provenance_header(generator: str, inputs: list[str] | tuple[str, ...]) -> str:
    """The mandatory ``% AUTO-GENERATED`` provenance block for one generated artifact.

    ``generator`` is the repo-relative path of the single owning script; ``inputs`` the committed
    artifacts it reads. The benchmark line names the contract version the SI reports
    (:data:`evemtbench.version.BENCHMARK_VERSION`), not the per-protocol ``outputs/<v>/`` tree —
    that scoping is visible in the input paths themselves."""
    from evemtbench.version import BENCHMARK_VERSION

    lines = [
        "% AUTO-GENERATED. DO NOT EDIT.",
        f"% Generator: {generator}",
        f"% Benchmark: EvEMTBench v{BENCHMARK_VERSION}",
        "% Inputs:" + ("" if inputs else " (none — static specification)"),
    ]
    lines += [f"%   {i}" for i in inputs]
    return "\n".join(lines) + "\n"


#: The committed held_out result tree of the tasks ``task_glob`` names: ``outputs/1.0.0/``, or
#: ``outputs/1.2.0/`` for the fault-location tasks (corrected valid-sample gate). A glob that
#: names a faulted-phase task also names ``outputs/1.3.0/`` (the LP-1-patched cigre_mv cells; the
#: tree is per grid, see ``results_version``). ``"*"`` (every task) names every tree.
def _held_out_glob(task_glob: str, sets: str = "{val,test,benchmark}") -> str:
    from evemtbench.version import FL_RESULTS_VERSION, FP_RESULTS_VERSION, results_version

    fp = task_glob == "*" or "fault_phase" in task_glob
    # a faulted-phase glob has no single tree: start from the protocol's default and add 1.3.0
    ver = results_version("held_out", None if task_glob.startswith("fault_phase") else task_glob)
    trees = {ver}
    if task_glob == "*":
        trees.add(FL_RESULTS_VERSION)
    if fp:
        trees.add(FP_RESULTS_VERSION)
    if len(trees) > 1:
        ver = "{" + ",".join(sorted(trees)) + "}"
    return f"outputs/{ver}/{task_glob}/held_out/<grid>/<baseline>/{sets}/result.json"


# --------------------------------------------------------------------------------------------------
# Shared configuration (identical to the pre-refactor make_paper_tables / make_supplement_tables).
# --------------------------------------------------------------------------------------------------
GRID_ROWS: list[tuple[str, str]] = [
    ("cigre_mv", r"CIGR\'E MV (20\,kV)"),
    ("double_line", r"Double Line (110\,kV)"),
    ("testgrid_110kv", r"Prot.\ Ref.\ (110\,kV)"),
    ("ieee39", r"IEEE 39-Bus (345\,kV)"),
]
OBS = ("local", "line", "global")
#: Compact column labels for the transposed per-baseline tables, where the view is a narrow column
#: rather than a group header. Expanded in each caption.
_OBS_SHORT = {"local": "loc", "line": "lin", "global": "glo"}
TEST_SETS = ("test", "benchmark")

_BACC = r"balanced accuracy $\uparrow$"
_MAE = r"\acs{mae}\,\% $\downarrow$"
# key -> (key, caption task name, task-id stem, headline metric, metric label w/ arrow).
PAPER_FUNCTIONS: list[tuple[str, str, str, str, str]] = [
    ("fd", r"Fault detection (\ac{fd})", "fault_detection", "balanced_accuracy", _BACC),
    (
        "fc",
        r"Fault-type classification (\ac{fc}, 9-class)",
        "fault_classification",
        "balanced_accuracy",
        _BACC,
    ),
    ("fl", r"Fault localisation (\ac{fl})", "fault_location", "mae", _MAE),
    ("ed", r"Event detection (\ac{ed})", "event_detection", "balanced_accuracy", _BACC),
    ("ec", r"Event classification (\ac{ec})", "event_classification", "balanced_accuracy", _BACC),
]
_PF_BY_KEY = {row[0]: row for row in PAPER_FUNCTIONS}

_BASELINE_LABEL = {
    "majority": "majority (floor)",
    "threshold": "threshold (overcurrent)",
    "impedance": "impedance (fault locator)",
    "random_forest": r"\ac{rf}",
    "mlp": r"\ac{mlp}",
    "gru": r"\ac{gru}",
    "cnn": r"\ac{cnn}",
    "resnet": r"\ac{resnet}",
    "tst": r"\ac{tst}",
}
# Scale-down-ONLY width backstop for the wide SI tables: shrinks a tabular that would overrun
# \textwidth, leaves a narrow one untouched (adjustbox is absent from the offline tectonic bundle,
# hence the raw \ifdim). Every user must close the extra group with `\end{tabular}`.
#: RETIRED (2026-08-17): formerly a scale-to-fit box around every wide tabular. It had no floor,
#: so a table needing 1.8x \textwidth was scaled to 0.55x and its \scriptsize type landed on the
#: page at 3.76 pt -- below the ~5 pt legibility floor, with a different effective size in every
#: table (against the journal's same-typeface-same-size rule) and rules scaled under the 1 pt
#: stroke floor. Now empty: a table that overruns \textwidth raises an Overfull \hbox at build
#: time and is fixed by layout (landscape, transpose, split) or replaced by a figure, never by
#: shrinking the type. Kept as a named constant so the emit sites stay one-line and greppable.
_SHRINK_TO_FIT = ""

# DISPLAY labels for the two held_out test sets. The manuscript calls the never-trained family
# `eval_grid` to keep it distinct from "the benchmark" (the evaluation framework as a whole); the
# dataset publication calls it Benchmark and the on-disk/code handle stays `benchmark` — which is
# why only the rendered label changes here, never TEST_SETS or any result-path key.
_TS_LABEL = {
    "test": "in-distribution (test)",
    "benchmark": "never-trained \\texttt{eval\\_grid} family",
}

# ---- F10: clip Student-t intervals to a bounded metric's support for display -------------------
# Balanced accuracy, accuracy, and the F1 family live in [0,1]; a symmetric t-interval at n=5 can
# run past the bound (e.g. 0.848 $\pm$ 0.222 -> upper 1.07). MAE/RMSE (% line length) and R^2 are
# NOT bounded to [0,1] and are never clipped.
_BOUNDED_01 = {"balanced_accuracy", "accuracy", "macro_f1"}


def _is_bounded01(metric: str) -> bool:
    return metric in _BOUNDED_01 or metric.startswith("f1_")


def ci_bounds(mean: float, ci: float | None, metric: str) -> tuple[float, float] | None:
    """The 95\\% interval to DISPLAY for a cell as ``(lo, hi)``, or ``None`` when there is none.

    A symmetric Student-t interval on a [0,1]-bounded metric can run past the bound at n=5 (F10),
    so such an interval is clipped to the metric's support --- which is why the tables print it as
    ``mean [lo, hi]`` rather than ``mean $\\pm$ ci``. MAE/RMSE (% line length) and R^2 are not
    bounded to [0,1] and are never clipped.

    Shared with the figure builders (``build_fig_transfer_overview``) so that a drawn error bar is
    definitionally the same interval the corresponding table cell prints, and the two cannot drift.
    """
    if ci is None:
        return None
    if _is_bounded01(metric):
        return max(0.0, mean - ci), min(1.0, mean + ci)
    return mean - ci, mean + ci


def _fmt_mean_ci(mean: float, ci: float | None, metric: str, compact: bool = False) -> str:
    """``mean $\\pm$ ci``; for a [0,1]-bounded metric whose symmetric interval overflows the
    bound, show the truncated interval ``mean [lo, hi]`` instead (F10). ``compact`` (used by the
    12-column per-baseline supplement tables, which otherwise overflow the two-column text width by
    >100 pt) renders the CI half-width as a subscript instead of `` $\\pm$ ci``: bounded metrics keep
    3 d.p., unbounded (MAE/RMSE) drop to 2 d.p.). A half-width below the displayed precision is
    reported as a BOUND (``$\\pm{<}0.001$``), never dropped: an omitted interval is reserved for
    ``ci is None``, i.e. no interval exists at all, and the two states must stay distinguishable."""
    if ci is None:
        return f"{mean:.3f}"
    bounded = _is_bounded01(metric)
    if bounded and (mean + ci > 1.0 or mean - ci < 0.0):
        lo, hi = ci_bounds(mean, ci, metric)
        if compact:  # narrower 2 d.p. bracket for the dense per-baseline supplement tables
            return f"{mean:.3f}$\\,{{\\scriptscriptstyle[{lo:.2f},{hi:.2f}]}}$"
        return f"{mean:.3f}$\\,{{\\scriptscriptstyle[{lo:.3f},\\,{hi:.3f}]}}$"
    if compact:
        nd = 3 if bounded else 2
        if ci < 0.5 * 10 ** (-nd):  # below display precision -> report the BOUND, not nothing
            return f"{mean:.{nd}f}$\\,{{\\scriptscriptstyle\\pm{{<}}{10**-nd:.{nd}f}}}$"
        return f"{mean:.{nd}f}$\\,{{\\scriptscriptstyle\\pm{ci:.{nd}f}}}$"
    if ci < 0.5e-3:
        # Below display precision. Report "$\\pm{<}0.001$" rather than dropping the interval: an
        # absent interval already means "no interval exists" (ci is None -- a single seed, or a
        # deterministic baseline), and collapsing the two states made a STRONG result (five seeds
        # agreeing to better than 0.001) look like an ABSENCE of information. Printing "± 0.000"
        # was rejected instead because it reads as a rounding artifact.
        return f"{mean:.3f} $\\pm{{<}}0.001$"
    return f"{mean:.3f} $\\pm$ {ci:.3f}"


# ---- F08: five-state evidence markers (P / n/a / U / ---) instead of a single \res ------------
# Event-scope functions (ED, EC) are grid-wide, so they are defined only at the *global* view;
# the overcurrent threshold baseline is defined only for binary FD at *local*. A cell that is not
# structurally defined is \notapp (n/a); a defined cell with no result yet is \pending (P); a
# fixed-width transfer cell at the grid-dependent *global* width is out of scope (\oos, ---).
_EVENT_SCOPE_STEMS = {"event_detection", "event_classification"}


def _pending_legend(body: str, clause: str) -> str:
    """``clause`` if ``body`` actually carries a ``\\pending`` cell, else "".

    Every cell of the v1.1.0 matrix is complete, so the clause is dead text today, and printing a
    key for a state the table does not contain is noise the reader has to discount. Making it
    conditional rather than deleting it keeps the table self-explaining if a future extension -- a
    new grid, baseline or protocol -- reintroduces an unrun cell: the marker would otherwise appear
    with no key. The ``\\pending`` emit sites in the cell builders are untouched for the same
    reason."""
    return clause if r"\pending" in body else ""


def _held_unsupported(stem: str, view: str, baseline: str | None = None) -> bool:
    """True if the (stem, view[, baseline]) cell is not defined by the contract (renders n/a)."""
    if stem in _EVENT_SCOPE_STEMS and view != "global":
        return True
    if baseline == "threshold" and not (stem == "fault_detection" and view == "local"):
        return True
    # the impedance-based locator is defined only for FL at the local and line views
    if baseline == "impedance" and not (stem == "fault_location" and view in ("local", "line")):
        return True
    return False


# --------------------------------------------------------------------------------------------------
# Cell selection (single source of truth, shared with the audit dashboard's paper-numbers page).
# --------------------------------------------------------------------------------------------------
def _argbest_baseline(metric: str, doc_of, pool=LEARNED_BASELINES) -> str | None:
    """The winning ``pool`` entry by a per-baseline score, or ``None`` if none scored.

    ``doc_of(baseline)`` returns that baseline's result doc (or ``None``); the winner is the argmax
    of ``mean_val(doc, metric)`` in METRIC_DIRECTION (lower-is-better metrics pick the minimum), ties
    keeping the earlier ``pool`` entry (strict ``>``). Shared kernel for the F06
    validation-selection (``best_cell_choice`` / ``best_transfer_row``). It is only ever given
    validation docs: no caller may pass the evaluation scores a table reports.

    ``pool`` is the candidate set. It defaults to :data:`LEARNED_BASELINES` — the six the held_out
    matrix fields — which is what every result table wants. A caller that compares against an arm
    with a NARROWER pool must pass that narrower pool, or the two sides of the comparison are drawn
    from different candidate sets: the cross-grid transfer protocols apply only to
    ``DEEP_BASELINES`` (``coverage.baseline_applies``), because a classical baseline has no
    transferable weights, so a from-scratch reference selected out of all six can name a model the
    transfer arm structurally cannot field. See :func:`supplement_lib.transfer_scratch_rows`."""
    direction = METRIC_DIRECTION.get(metric, 1)
    sel_base, sel_val = None, None
    for baseline in pool:
        v = mean_val(doc_of(baseline), metric)
        if v is None:
            continue
        if sel_val is None or direction * (v - sel_val) > 0:
            sel_val, sel_base = v, baseline
    return sel_base


class SelectionEvidenceError(RuntimeError):
    """A populated cell lacks the validation evidence its model selection needs.

    Raised instead of ever choosing a model by the evaluation scores a table reports (the former
    "legacy fallback"). ``cell`` identifies the cell (protocol, task, grid, source, reported set,
    metric); ``missing`` lists the baselines that have an evaluation result there but no usable
    validation score. A build that hits this must not publish: re-score validation
    (``hpc/rescore_validation.py``) or record the exclusion in :data:`SELECTION_EXCLUSIONS`."""

    def __init__(self, cell: dict, missing: list[str]):
        self.cell, self.missing = dict(cell), list(missing)
        ident = ", ".join(f"{k}={v}" for k, v in cell.items())
        super().__init__(
            f"selection blocked ({ident}): no usable validation result for {', '.join(missing)}. "
            "A model is never selected on the evaluation scores it reports; re-score validation "
            "(hpc/rescore_validation.py) or record an exclusion with its reason in "
            "paper_lib.SELECTION_EXCLUSIONS."
        )


#: Recorded exclusions from validation candidacy:
#: ``{(protocol, task_id, grid, source or None, baseline): reason}``. A baseline that has an
#: evaluation result but no usable validation score blocks its cell unless it is listed here; a
#: listed baseline cannot win the cell and is announced on every build (``warn_once``). Empty: the
#: v1.0.0 / v1.1.0 matrices carry validation evidence for every evaluated candidate.
SELECTION_EXCLUSIONS: dict[tuple[str, str, str | None, str | None, str], str] = {}


def _validated_candidates(cell: dict, pool, has_eval, val_score) -> list[str] | None:
    """The baselines of ``pool`` that may compete for ``cell``, or ``None`` for an empty cell.

    ``has_eval(b)`` says whether baseline ``b`` has a result on the reported set; ``val_score(b)``
    is its validation score or ``None``. An empty cell (no ``pool`` entry evaluated) is not an
    error: it renders as pending. In a populated cell every evaluated candidate must carry a usable
    validation score, or the cell raises :class:`SelectionEvidenceError` --- so a candidate can
    neither be selected on evaluation scores nor drop out of the pool silently. Baselines recorded
    in :data:`SELECTION_EXCLUSIONS` are removed from candidacy, loudly."""
    evaluated = [b for b in pool if has_eval(b)]
    if not evaluated:
        return None
    key = (cell["protocol"], cell["task"], cell["grid"], cell["source"])
    candidates = []
    for b in pool:
        reason = SELECTION_EXCLUSIONS.get((*key, b))
        if reason is None:
            candidates.append(b)
        else:
            ident = ", ".join(f"{k}={v}" for k, v in cell.items())
            warn_once(f"selection ({ident}): baseline '{b}' excluded by record: {reason}")
    missing = [b for b in candidates if b in evaluated and val_score(b) is None]
    if missing:
        raise SelectionEvidenceError(cell, missing)
    return candidates


def best_cell_choice(
    rows: dict,
    task_id: str,
    test_set: str,
    metric: str,
    pool=LEARNED_BASELINES,
    grid: str | None = None,
) -> dict | None:
    """The reported learned baseline for (task_id, test_set), or ``None`` if no result exists.

    F06: the reported baseline is selected on the **validation** split (``rows[...]["val"]``, the
    offline re-score written by ``hpc/rescore_validation.py``) and then its value on the requested
    ``test_set`` is reported. Selecting on validation rather than on the very test set being
    reported removes the winner's-curse (selection-on-test) bias. Direction from METRIC_DIRECTION
    (lower-is-better metrics like MAE pick the minimum); ties keep the earlier ``pool``
    entry (strict ``>``). Returns ``{baseline, mean, ci, n_seeds, selected_on}``, where
    ``selected_on`` is always ``"val"``.

    There is no fallback to the reported set. ``None`` means the cell is empty (no ``pool``
    baseline has a result on ``test_set``) or its validation winner has no result there yet --- the
    winner is never replaced by another model. A populated cell in which an evaluated candidate
    has no usable validation score raises :class:`SelectionEvidenceError`.

    ``pool`` narrows the candidate set; see :func:`_argbest_baseline` for when that is required.
    ``grid`` only names the cell in errors and exclusion records; when omitted it is read from the
    cell's result docs.

    ``rows`` is ``load_rows(grid)`` = ``{(task_id, baseline): {test_set: doc}}``."""

    def val_of(b: str):
        return rows.get((task_id, b), {}).get("val")

    def rep_of(b: str):
        return rows.get((task_id, b), {}).get(test_set)

    if grid is None:
        grid = next((d["grid"] for b in pool if (d := rep_of(b)) and d.get("grid")), None)
    cell = {
        "protocol": "held_out",
        "task": task_id,
        "grid": grid,
        "source": None,
        "reported_set": test_set,
        "metric": metric,
    }
    candidates = _validated_candidates(
        cell,
        pool,
        lambda b: mean_val(rep_of(b), metric) is not None,
        lambda b: mean_val(val_of(b), metric),
    )
    if candidates is None:
        return None  # empty cell: nothing evaluated on this set
    # 1) select the winner by its validation-split score
    sel_base = _argbest_baseline(metric, val_of, candidates)
    if sel_base is None:
        return None  # every evaluated candidate is a recorded exclusion
    # 2) report the validation-selected baseline's value on the requested test_set
    rep = rep_of(sel_base)
    mean = mean_val(rep, metric)
    if mean is None:
        return None  # selected baseline has no result on this test set yet
    return {
        "baseline": sel_base,
        "mean": mean,
        "ci": ci_val(rep, metric),
        "n_seeds": len((rep or {}).get("seeds") or []),
        "selected_on": "val",
    }


# --- what the two main-text figures cover -------------------------------------------------------
# Shared by build_fig_synthesis / build_fig_transfer_overview and by the figure-value test, so a
# change to a figure's coverage cannot silently escape verification.
#: Fig. 2 upper panel: ``(row label, task stem, observability view)``, balanced accuracy.
SYNTHESIS_BACC_ROWS: list[tuple[str, str, str]] = [
    ("FD · local", "fault_detection", "local"),
    ("FD · line", "fault_detection", "line"),
    ("FD · global", "fault_detection", "global"),
    ("FC · local", "fault_classification", "local"),
    ("FC · line", "fault_classification", "line"),
    ("FC · global", "fault_classification", "global"),
    ("ED · global", "event_detection", "global"),
    ("EC · global", "event_classification", "global"),
]
#: Fig. 2 lower panel: fault localization, its own metric (MAE % of line length) and scale.
SYNTHESIS_FL_ROWS: list[tuple[str, str, str]] = [
    ("FL · local", "fault_location", "local"),
    ("FL · line", "fault_location", "line"),
    ("FL · global", "fault_location", "global"),
]
#: Fig. 3 covers the differential two-terminal view only — transfer applies to the fixed-width
#: local/line views, and `line` is the one the manuscript reports.
TRANSFER_FIG_VIEW = "line"
#: Fig. 3 columns: only the fault-triad functions carry transferable weights (ED/EC are global-only
#: and held-out-only, so they have no transfer cell).
TRANSFER_FIG_FUNCTIONS = ("fd", "fc", "fl")


def held_out_rows() -> dict:
    """``{grid: load_rows(grid)}`` for the four reference grids --- one glob of ``outputs/`` per grid.

    Hoisted out so a caller that builds several matrices (the synthesis figure needs one for
    balanced accuracy and one for MAE) scans the tree once instead of once per matrix."""
    return {g: load_rows(g) for g, _ in GRID_ROWS}


def synthesis_matrix(
    row_specs, metric: str, per_grid: dict | None = None, test_sets=TEST_SETS
) -> list[list[dict | None]]:
    """The held-out synthesis matrix behind Fig. 2, as plain data (no matplotlib).

    ``row_specs`` is an iterable of ``(label, task stem, observability view)``. The result is one
    list per row; within a row the columns run ``test_set`` block by ``test_set`` block, each in
    :data:`GRID_ROWS` order --- so ``len(row) == len(test_sets) * len(GRID_ROWS)``. A cell is
    :func:`best_cell_choice`'s dict (plus ``grid``/``test_set``/``task_id`` for traceability), or
    ``None`` where no result exists.

    Lives here rather than in the builder so the figure's numbers are the same object the tables
    are built from, and so a test can assert figure-vs-table agreement without importing matplotlib.
    """
    grids = [g for g, _ in GRID_ROWS]
    if per_grid is None:
        per_grid = held_out_rows()
    matrix: list[list[dict | None]] = []
    for _label, stem, view in row_specs:
        task_id = f"{stem}_{view}"
        row: list[dict | None] = []
        for test_set in test_sets:
            for grid in grids:
                choice = best_cell_choice(per_grid[grid], task_id, test_set, metric)
                if choice is not None:
                    choice = {**choice, "grid": grid, "test_set": test_set, "task_id": task_id}
                row.append(choice)
        matrix.append(row)
    return matrix


# --------------------------------------------------------------------------------------------------
# Cross-grid transfer tables — mirror the held_out layout but cols = obs x {zero-shot, fine-tune}
# on the never-trained benchmark family. Data source: the same scan the leaderboard's transfer
# section uses (leaderboard.scan_results), version-scoped, best learned baseline per cell.
# --------------------------------------------------------------------------------------------------
_TRANSFER_PROTOS = [("transfer_zeroshot", "Zero-shot"), ("transfer_finetune", "Fine-tune")]


def transfer_series(
    idx: dict,
    task_id: str,
    source: str,
    metric: str,
    val_idx: dict | None = None,
    grids=None,
) -> dict[str, list[dict | None]]:
    """The cross-grid transfer cells behind one Fig. 3 panel, as plain data (no matplotlib).

    ``{protocol: [cell | None per target grid]}`` in :data:`GRID_ROWS` order, for a single
    ``(task_id, source)``. Each cell carries ``mean``, the raw ``ci`` half-width, and the
    ``(lo, hi)`` the table displays via :func:`ci_bounds` --- so the figure's error bars and the
    table's printed interval are the same numbers by construction. Selection is
    :func:`best_transfer_row`, i.e. identical to ``tab_transfer_*.tex``.
    """
    grids = [g for g, _ in GRID_ROWS] if grids is None else list(grids)
    out: dict[str, list[dict | None]] = {}
    for protocol, _label in _TRANSFER_PROTOS:
        cells: list[dict | None] = []
        for grid in grids:
            row = best_transfer_row(idx, task_id, grid, protocol, metric, source, val_idx)
            if row is None:
                cells.append(None)
                continue
            mean = row.mean(metric)
            ci = (row.metrics.get(metric) or {}).get("ci95")
            lo, hi = ci_bounds(mean, ci, metric) or (mean, mean)
            cells.append(
                {
                    "grid": grid,
                    "protocol": protocol,
                    "source": source,
                    "task_id": task_id,
                    "baseline": row.baseline,
                    "mean": mean,
                    "ci": ci,
                    "lo": lo,
                    "hi": hi,
                }
            )
        out[protocol] = cells
    return out


def per_baseline_matrix(
    stem: str,
    metric: str,
    per_grid: dict | None = None,
    views=OBS,
    test_sets=TEST_SETS,
    baselines=BASELINES,
) -> dict[tuple[str, str, str, str], dict | None]:
    """Every per-baseline held_out cell of one function, as plain data (no matplotlib).

    Keyed ``(grid, view, baseline, test_set)`` over :data:`GRID_ROWS` x ``views`` x
    ``baselines`` x ``test_sets``; the value is ``{mean, ci, lo, hi, n_seeds}`` or ``None``.
    ``None`` covers both states the S2 tables render as markers: a (view, baseline) combination the
    function does not define (``\\notapp``), and a defined cell with no result yet (``\\pending``).

    This is the seam behind the S2 per-baseline figures. It deliberately reuses
    :func:`load_rows` / :func:`mean_val` / :func:`ci_val` --- the same objects
    the retired S2 tables were rendered from --- so a figure and the published CSV cannot disagree, and so the
    parity test can assert that without importing matplotlib.
    """
    if per_grid is None:
        per_grid = held_out_rows()
    out: dict[tuple[str, str, str, str], dict | None] = {}
    for grid, _ in GRID_ROWS:
        for view in views:
            for baseline in baselines:
                unsupported = _held_unsupported(stem, view, baseline)
                for test_set in test_sets:
                    key = (grid, view, baseline, test_set)
                    if unsupported:
                        out[key] = None
                        continue
                    doc = per_grid[grid].get((f"{stem}_{view}", baseline), {}).get(test_set)
                    mean = mean_val(doc, metric)
                    if mean is None:
                        out[key] = None
                        continue
                    ci = ci_val(doc, metric)
                    # None for a deterministic baseline (one seed) or a half-width that rounds
                    # away -- the figures then draw a bare marker, as the tables print a bare mean.
                    bounds = ci_bounds(mean, ci, metric)
                    out[key] = {
                        "mean": mean,
                        "ci": ci,
                        "lo": None if bounds is None else bounds[0],
                        "hi": None if bounds is None else bounds[1],
                        "n_seeds": (doc.get("aggregate", {}).get(metric) or {}).get("n_seeds"),
                    }
    return out


def _best_cell(rows: dict, stem: str, view: str, test_set: str, metric: str) -> str:
    """Best learned baseline for (stem, view, test_set): a clipped ``mean $\\pm$ ci`` value, or
    a five-state marker --- ``\\notapp`` (view not defined for this function) / ``\\pending``
    (defined but no result yet)."""
    if _held_unsupported(stem, view):
        return r"\notapp"
    choice = best_cell_choice(rows, f"{stem}_{view}", test_set, metric)
    if choice is None:
        return r"\pending"
    return _fmt_mean_ci(choice["mean"], choice["ci"], metric)


def _supp_cell(rows: dict, stem: str, view: str, baseline: str, test_set: str, metric: str) -> str:
    if _held_unsupported(stem, view, baseline):
        return r"\notapp"
    doc = rows.get((f"{stem}_{view}", baseline), {}).get(test_set)
    mean = mean_val(doc, metric)
    if mean is None:
        return r"\pending"
    # compact CI rendering keeps the 12-column per-baseline supplement tables within the two-column
    # text width (they otherwise overflow it by >100 pt for every primary function).
    return _fmt_mean_ci(mean, ci_val(doc, metric), metric, compact=True)


# --------------------------------------------------------------------------------------------------
# Full per-baseline supplement tables (supp_tab_<fn>_{test,benchmark}).
#
# RESTORED 2026-08-20, TRANSPOSED. The retired version laid these out as
# ``baseline x (grid x view)`` = 1 + 4*3 = 13 columns, which overran \textwidth and only fitted
# inside the scale-to-fit box that rendered it at 3.76-4.98 pt -- under the ~5 pt legibility floor.
# The fix is layout, not type size (the rule recorded at _SHRINK_TO_FIT): transposing to
# ``(grid, view) x baseline`` gives 2 + 7 = 9 columns, which fits at \scriptsize with normal
# tabcolsep, so no scaling is involved and every table in the SI is set at the same size.
#
# Reading direction is also better this way round: a reader compares BASELINES against each other
# for a fixed (grid, view), which is now a row-wise scan across adjacent columns instead of a
# hop across four column groups.
#
# The body must contain exactly ONE ``\midrule`` before it, because
# tests/integration/test_supplement_tables.py counts the five-state markers in
# ``latex.split("\midrule")[1]`` and expects N_BODY_CELLS = 8 * 12. Grid groups are therefore
# separated with \addlinespace, never with a rule.
# --------------------------------------------------------------------------------------------------
def supp_table(key: str, test_set: str) -> str:
    r"""Full per-baseline results for one function on one held_out test set.

    Rows are ``(grid, observability view)`` in :data:`GRID_ROWS` x :data:`OBS` order; columns are the
    eight :data:`BASELINES`. Cells come from :func:`_supp_cell`, so a (view, baseline) combination
    the function does not define renders ``\notapp`` and a defined-but-unrun cell ``\pending`` --
    the same five-state marker set the rest of the SI uses."""
    _, name, stem, metric, metric_label = _PF_BY_KEY[key]
    per_grid = {g: load_rows(g) for g, _ in GRID_ROWS}
    # Column HEADERS, so the compact _BASELINE_ACS forms rather than _BASELINE_LABEL: transposing
    # turned these from row labels into headers, and "threshold (overcurrent)" / "majority (floor)"
    # as headers set the column width for the whole table -- measured at up to 53.9 pt over
    # \textwidth on the first restored draft. The long names are given in the caption instead.
    heads = " & ".join(f"\\textbf{{{_BASELINE_ACS[b]}}}" for b in BASELINES)
    body: list[str] = []
    for gi, (grid, glabel) in enumerate(GRID_ROWS):
        for vi, view in enumerate(OBS):
            cells = [
                _supp_cell(per_grid[grid], stem, view, baseline, test_set, metric)
                for baseline in BASELINES
            ]
            head = f"\\multirow{{{len(OBS)}}}{{*}}{{{glabel}}}" if vi == 0 else ""
            body.append(f"{head:<44} & {_OBS_SHORT[view]:<5} & " + " & ".join(cells) + r" \\")
        if gi < len(GRID_ROWS) - 1:
            # \addlinespace, NOT \midrule: a second rule would break the marker-count test above
            body.append(r"\addlinespace")
    ci_note = (
        "the 95\\% CI half-width is shown as a compact subscript ($\\pm h$); an interval that would "
        "exceed $[0,1]$ is instead shown truncated as $[\\text{lo},\\text{hi}]$"
        if _is_bounded01(metric)
        else "the 95\\% CI half-width is shown as a compact subscript ($\\pm h$) at 2 d.p"
    )
    return (
        provenance_header(
            f"hpc/paper_artifacts/build_supp_tab_{key}_{test_set}.py",
            [_held_out_glob(f"{stem}_{{local,line,global}}", "{val," + test_set + "}")],
        )
        + "\\begin{table*}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{4pt}\n"
        f"\\caption{{{name} --- full per-baseline results on the {_TS_LABEL[test_set]} "
        f"(\\texttt{{held\\_out}} protocol), headline metric ({metric_label}). Rows are grid and "
        "observability view (loc/lin/glo = local/line/global); columns are the reference baselines, "
        "where maj.\\ is the trivial majority floor and thr.\\ the conventional overcurrent "
        "threshold. Cells are "
        f"mean\\,$\\pm$\\,95\\% CI over seeds (Student's $t$); {ci_note}. "
        + _pending_legend(body, "\\pending{}~= pending; ")
        + "\\notapp{}~= not defined for this (function, view) or baseline.}\n"
        f"\\label{{tab:supp_{key}_{test_set}}}\n"
        f"\\begin{{tabular}}{{l l {'c' * len(BASELINES)}}}\n\\toprule\n"
        f"\\textbf{{Grid}} & \\textbf{{View}} & {heads} \\\\\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )


# --------------------------------------------------------------------------------------------------
# Main result tables (held_out) — verbatim layout of the pre-refactor make_paper_tables._table.
# --------------------------------------------------------------------------------------------------
def result_table(key: str) -> str:
    """Full LaTeX table for one paper function (rows = grids, cols = obs × held_out test-set)."""
    _, name, stem, metric, metric_label = _PF_BY_KEY[key]
    per_grid = {g: load_rows(g) for g, _ in GRID_ROWS}
    body_lines = []
    for grid, label in GRID_ROWS:
        rows = per_grid[grid]
        # columns = observability × test-set; ED/EC have only a *_global task id, so their
        # local/line cells find no (task_id, baseline) key and render \res.
        cells = [_best_cell(rows, stem, obs, ts, metric) for ts in TEST_SETS for obs in OBS]
        body_lines.append(f"{label:<26} & " + " & ".join(cells) + r" \\")
    body = "\n".join(body_lines)
    # The [0,1]-truncation note only applies to bounded metrics (balanced accuracy); FL is MAE %.
    ci_note = (
        "an interval exceeding the metric's $[0,1]$ range is shown truncated as "
        "$[\\text{lo},\\text{hi}]$. "
        if _is_bounded01(metric)
        else ""
    )
    return (
        provenance_header(
            f"hpc/paper_artifacts/build_tab_res_{key}.py",
            [_held_out_glob(f"{stem}_{{local,line,global}}")],
        )
        + "\\begin{table}[!htbp]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.05}\n"
        f"\\caption{{{name} --- reference baseline (best of RF/MLP/GRU/CNN/ResNet/TST, "
        "selected on the validation split; a beatable reference, full per-baseline breakdown "
        "in the Supplementary), headline metric "
        f"({metric_label}) per grid and observability view, for the two \\texttt{{held\\_out}} "
        "test sets (in-distribution vs.\\ never-trained \\texttt{eval\\_grid} family). Cells are "
        "mean\\,$\\pm$\\,95\\% CI (Student's $t$); "
        + ci_note
        + _pending_legend(body, "\\pending{}~= pending (no result yet); ")
        + "\\notapp{}~= not defined for this view. Full per-baseline tables in the "
        "Supplementary.}\n"
        f"\\label{{tab:res_{key}}}\n" + _SHRINK_TO_FIT + "\\begin{tabular}{l ccc ccc}\n\\toprule\n"
        " & \\multicolumn{3}{c}{\\textbf{In-distribution (test)}} & "
        "\\multicolumn{3}{c}{\\textbf{Eval-Grid (never trained)}} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
        "\\textbf{Grid} & \\textbf{local} & \\textbf{line} & \\textbf{global} & "
        "\\textbf{local} & \\textbf{line} & \\textbf{global} \\\\\n\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


# --------------------------------------------------------------------------------------------------
# Supplement tables — verbatim layout of the pre-refactor make_supplement_tables._table.
# --------------------------------------------------------------------------------------------------
def _source_short(source: str) -> str:
    r"""LaTeX short label for a transfer pre-training SOURCE grid: ``template_20kv`` -> ``20\,kV``."""
    kv = source.replace("template_", "").replace("kv", "")
    return f"{kv}\\,kV"


def _transfer_index(outputs: str | Path = "outputs") -> dict:
    """``{(task_id, grid, protocol, source): [Row, ...]}`` for the CURRENT transfer runs on
    benchmark (``Row.version == results_version(protocol, task)``: the fault-location transfer
    cells live in the 1.2.0 tree, every other task's in 1.1.0; a superseded copy in an older tree
    is never indexed). ``source`` is the pre-training grid the transfer was seeded from
    (``Row.source``; legacy source-less results default to ``PRETRAIN_GRID`` = the 20 kV template)."""
    from evemtbench.leaderboard import scan_results
    from evemtbench.version import results_version

    idx: dict = {}
    for r in scan_results(outputs):
        if r.version != results_version(r.protocol, r.task, grid=r.grid, source=r.source):
            continue
        if r.protocol not in ("transfer_zeroshot", "transfer_finetune"):
            continue
        if r.test_set != "benchmark":
            continue
        src = r.source or PRETRAIN_GRID
        idx.setdefault((r.task, r.grid, r.protocol, src), []).append(r)
    return idx


def _transfer_val_index(outputs: str | Path = "outputs") -> dict:
    """Validation-split docs from the offline re-score, source-aware. Keys:

    * zero-shot selection: ``("multi_grid_pretrain", source, task, baseline)`` — the pre-training
      corpus val of the SOURCE grid (grid == source for multi_grid_pretrain);
    * fine-tune selection: ``("transfer_finetune", target, task, baseline, source)`` — the fine-tune
      val on the target grid per source (legacy source-less path = the 20 kV default; a
      ``pretrain-<src>`` subdir = that source).

    A populated transfer cell whose key is missing here fails in :func:`best_transfer_row`; it is
    never selected on the benchmark it reports (val artifacts under ``**/val/`` are gitignored, so
    a checkout without ``outputs/`` has empty cells, not failures). Produced by
    ``hpc/rescore_validation.py``.

    Each doc is read from the CURRENT tree of its (protocol, task): the ``multi_grid_pretrain``
    source-model val docs from 1.1.0 for every task (fault location included — its source corpora
    are unaffected by the v1.2.0 gate), the ``transfer_finetune`` val docs from 1.1.0, or from
    1.2.0 for the fault-location tasks."""
    import glob
    import json

    from evemtbench.coverage import cell_from_key, cell_results_version
    from evemtbench.version import CURRENT_RESULTS_VERSIONS

    idx: dict = {}

    def _warn_bad(f: str, e: Exception) -> None:
        warn_once(f"skipping unreadable/malformed val result.json: {f} ({type(e).__name__}: {e})")

    def _cur(protocol: str, tail: str) -> list[str]:
        """Current-tree files ``<outputs>/<ver>/<task>/<protocol>/<tail>``: a file is kept when
        its tree is the current one of its cell (task, protocol, grid, source; parsed from the
        path below the tree root)."""
        n = len(Path(outputs).parts)  # <outputs>/<ver>/<task>/... -> the task dir is parts[n + 1]

        def _current(f: str, ver: str) -> bool:
            cell = cell_from_key(Path(*Path(f).parts[n + 1 :]))
            return cell is not None and cell_results_version(cell) == ver

        return [
            f
            for ver in CURRENT_RESULTS_VERSIONS
            for f in glob.glob(f"{outputs}/{ver}/*/{protocol}/{tail}")
            if _current(f, ver)
        ]

    for f in _cur("multi_grid_pretrain", "*/*/val/result.json"):
        try:
            d = json.load(open(f))
            idx[("multi_grid_pretrain", d["grid"], d["task"], d["baseline"])] = d
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
            _warn_bad(f, e)
    # transfer_finetune val: legacy source-less path (=> 20 kV) and source-tokened (pretrain-<src>)
    for f in _cur("transfer_finetune", "*/*/val/result.json"):
        try:
            d = json.load(open(f))
            idx[("transfer_finetune", d["grid"], d["task"], d["baseline"], PRETRAIN_GRID)] = d
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
            _warn_bad(f, e)
    for f in _cur("transfer_finetune", "*/*/pretrain-*/val/result.json"):
        try:
            d = json.load(open(f))
            src = Path(f).parent.parent.name.replace("pretrain-", "")  # .../pretrain-<src>/val/…
            idx[("transfer_finetune", d["grid"], d["task"], d["baseline"], src)] = d
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
            _warn_bad(f, e)
    return idx


def best_transfer_row(
    idx: dict,
    task_id: str,
    grid: str,
    protocol: str,
    metric: str,
    source: str,
    val_idx: dict | None = None,
):
    """Winning learned baseline Row for a transfer cell seeded from pre-training ``source`` (or
    ``None``). F06 consistency: the winner is selected on a VALIDATION signal, never on the benchmark
    it reports — ``transfer_zeroshot`` on the ``source``'s ``multi_grid`` pretrain val
    (grid-independent: one pretrained model applied to every target), ``transfer_finetune`` on that
    source's fine-tune val on the target grid. Neither signal substitutes for the other, and there
    is no fallback to the benchmark set: a populated cell in which an evaluated baseline has no
    usable validation score (or whose protocol has no validation rule) raises
    :class:`SelectionEvidenceError`. ``None`` means the cell is empty or its validation winner has
    no benchmark row yet (rendered ``\\pending``); the winner is never replaced by another model."""
    bench = {
        r.baseline: r
        for r in idx.get((task_id, grid, protocol, source), [])
        if r.baseline in LEARNED_BASELINES and r.mean(metric) is not None
    }
    if val_idx is None:
        val_idx = _transfer_val_index()
    if protocol == "transfer_zeroshot":

        def val_of(b: str):
            return val_idx.get(("multi_grid_pretrain", source, task_id, b))

    elif protocol == "transfer_finetune":

        def val_of(b: str):
            return val_idx.get(("transfer_finetune", grid, task_id, b, source))

    else:

        def val_of(b: str):
            return None

    cell = {
        "protocol": protocol,
        "task": task_id,
        "grid": grid,
        "source": source,
        "reported_set": "benchmark",
        "metric": metric,
    }
    # a baseline with a benchmark row but no val doc (e.g. a re-score rejected by the positive
    # control) blocks the cell unless its exclusion is recorded: it may not shrink the pool mutely
    candidates = _validated_candidates(
        cell, LEARNED_BASELINES, lambda b: b in bench, lambda b: mean_val(val_of(b), metric)
    )
    if candidates is None:
        return None  # empty cell: no learned baseline has a benchmark row
    sel_base = _argbest_baseline(metric, val_of, candidates)
    return bench.get(sel_base)  # None -> \pending if the winner lacks a benchmark cell here


def _transfer_cell(
    idx: dict,
    stem: str,
    view: str,
    grid: str,
    protocol: str,
    metric: str,
    source: str,
    val_idx: dict | None = None,
) -> str:
    if view == "global":
        # fixed-width transfer cannot ingest the grid-dependent global channel count -> out of scope
        return r"\oos"
    best = best_transfer_row(idx, f"{stem}_{view}", grid, protocol, metric, source, val_idx)
    if best is None:
        return r"\pending"
    ci = (best.metrics.get(metric) or {}).get("ci95")
    return _fmt_mean_ci(best.mean(metric), ci, metric)


def transfer_table(
    key: str,
    outputs: str | Path = "outputs",
    idx: dict | None = None,
    val_idx: dict | None = None,
) -> str:
    """Full LaTeX transfer table for one paper function (rows = grids, cols = {zero-shot,
    fine-tune} × obs on the benchmark family). ``idx``/``val_idx`` may be prebuilt and shared across
    the five functions (see ``build_transfer_tables``); each is built from ``outputs`` when omitted."""
    _, name, stem, metric, metric_label = _PF_BY_KEY[key]
    if idx is None:
        idx = _transfer_index(outputs)
    if val_idx is None:
        val_idx = _transfer_val_index(outputs)
    body_lines = []
    for gi, (grid, label) in enumerate(GRID_ROWS):
        for si, source in enumerate(PRETRAIN_GRIDS):  # source axis: 20/110/345 kV pre-training
            cells = [
                _transfer_cell(idx, stem, obs, grid, proto, metric, source, val_idx)
                for proto, _ in _TRANSFER_PROTOS
                for obs in OBS
            ]
            gcol = label if si == 0 else ""  # grid label only on the first source row
            body_lines.append(
                f"{gcol:<22} & {_source_short(source):<8} & " + " & ".join(cells) + r" \\"
            )
        if gi < len(GRID_ROWS) - 1:
            body_lines.append(r"\midrule")
    body = "\n".join(body_lines)
    from evemtbench.version import results_version

    # the trees this function's transfer cells / selection docs are current in (fault location:
    # transfer cells in 1.2.0, its multi_grid_pretrain source models still in 1.1.0)
    _tr_ver = results_version("transfer_finetune", f"{stem}_local")
    _src_ver = results_version("multi_grid_pretrain", f"{stem}_local")

    header = provenance_header(
        f"hpc/paper_artifacts/build_tab_transfer_{key}.py",
        [
            f"outputs/{_tr_ver}/{stem}_{{local,line}}/transfer_{{zeroshot,finetune}}/"
            "<grid>/<baseline>/[pretrain-<source>/]benchmark/result.json",
            f"outputs/{_src_ver}/{stem}_{{local,line}}/multi_grid_pretrain/<source>/"
            "<baseline>/val/result.json  (zero-shot selection)",
            f"outputs/{_tr_ver}/{stem}_{{local,line}}/transfer_finetune/<grid>/"
            "<baseline>/[pretrain-<source>/]val/result.json  (fine-tune selection)",
        ],
    )
    return (
        header
        + "\\begin{table}[!htbp]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.05}\n"
        f"\\caption{{{name} --- cross-grid transfer reference baseline (best of "
        "MLP/GRU/CNN/ResNet/TST, selected on the validation split, never on the reported "
        "\\texttt{eval\\_grid} set: \\emph{fine-tune} on the target grid's validation split, "
        "\\emph{zero-shot} on the "
        "pre-training source's \\texttt{multi\\_grid} validation --- so the zero-shot choice is "
        f"grid-independent), headline metric ({metric_label}), pre-trained on each "
        "\\texttt{multi\\_grid} source (20/110/345\\,kV) then \\emph{zero-shot} vs.\\ "
        "\\emph{fine-tuned} on each target grid, scored on the never-trained "
        "\\texttt{eval\\_grid} family per "
        "observability view. Cells are mean\\,$\\pm$\\,95\\% CI; "
        + _pending_legend(body, "\\pending{}~= pending (no result yet); ")
        + "\\oos{}~= out of scope (fixed-width transfer cannot use the grid-dependent "
        "\\emph{global} width).}\n"
        f"\\label{{tab:transfer_{key}}}\n"
        + _SHRINK_TO_FIT
        + "\\begin{tabular}{ll ccc ccc}\n\\toprule\n"
        " & & \\multicolumn{3}{c}{\\textbf{Zero-shot}} & "
        "\\multicolumn{3}{c}{\\textbf{Fine-tune}} \\\\\n"
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
        "\\textbf{Grid} & \\textbf{Src} & \\textbf{local} & \\textbf{line} & \\textbf{global} & "
        "\\textbf{local} & \\textbf{line} & \\textbf{global} \\\\\n\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


# --------------------------------------------------------------------------------------------------
# Static specification tables (authored content lifted verbatim out of main.tex). These are NOT
# data-derived, so they are exempt from the pre-freeze \res policy (see check_results_fresh.py).
# task_spec, dataset_families and baselines are byte-identical to the table environments of the
# revised manuscript (Tables 2-4, 23 Nov 2026); update them here when the manuscript text changes.
# --------------------------------------------------------------------------------------------------
def _spec_header(name: str) -> str:
    """Provenance block for a static specification table (authored content, no data inputs)."""
    return provenance_header(f"hpc/paper_artifacts/build_tab_{name}.py", [])


SPEC_TABLES: dict[str, str] = {
    "task_spec": r"""\begin{table*}[t]
\centering
%\small

\caption{Target and valid-sample definitions for the five primary EvEMTBench functions. L, Li, and G denote local, line, and global observability, respectively.}
\label{tab:task_spec}

\begin{tabular}{@{}
  >{\raggedright\arraybackslash}p{2.7cm}
  >{\raggedright\arraybackslash}p{3.0cm}
  >{\raggedright\arraybackslash}p{8.3cm}
  >{\centering\arraybackslash}p{1.5cm}
@{}}
\toprule
\textbf{Function} &
\textbf{Target} &
\textbf{Valid evaluation windows} &
\textbf{Views} \\
\midrule

Fault detection (\acs{fd})
& Fault active / inactive
& All windows from fault episodes; fault-active windows are positive, while windows before inception or after the annotated fault interval are negative
& L, Li, G \\

Fault classification (\acs{fc})
& Fault class
& Fault-active windows with labels in the frozen nine-class fault vocabulary
& L, Li, G \\

Fault localization (\acs{fl})
& Position along line (\% of line length)
& Fault-active windows from line-fault episodes; busbar faults are excluded because no unique line position is defined
& L, Li, G \\

Event detection (\acs{ed})
& Event active / inactive
& All valid event-analysis windows; fault- or switching-active windows are positive, while inactive windows are negative
& G \\

Event classification (\acs{ec})
& Event class
& All valid event-analysis windows over the frozen 23-class vocabulary; inactive windows are labeled \texttt{no\_event}
& G \\

\bottomrule
\end{tabular}
\end{table*}
""",
    "task_suite": r"""\begin{table*}[t]
\centering
\small
\caption{The full scored task suite (benchmark v1.0.0): twelve
protection/event-analysis functions instantiated across the observability views,
giving \emph{24 scored (function, view) tasks} in three difficulty tiers. The
five \emph{primary} functions foregrounded in the main text are marked~$\dagger$; the
remaining seven are scored diagnostics reported in the Supplementary and the online
leaderboard. Views: L~= local, Li~= line, G~= global. Event-scope functions are
grid-wide (global-only) by construction, because a benign or quiescent window has no
faulted line on which to centre the local/line oracle. Classification tasks report
balanced accuracy as the headline metric; \ac{fl} reports \acs{mae} in \% of line
length.}
\label{tab:task_suite}
\begin{tabular}{L{4.4cm} c L{2.4cm} c L{4.0cm}}
\toprule
\textbf{Function} & \textbf{Tier} & \textbf{Label kind} & \textbf{Views} & \textbf{Interpretation} \\
\midrule
Fault detection (\ac{fd})\,$\dagger$      & 1 & binary      & L, Li, G & fault-active vs.\ pre-fault window \\
Event detection (\ac{ed})\,$\dagger$      & 1 & binary      & G        & any active event vs.\ quiescent \\
Switch detection                          & 1 & binary      & G        & benign switching vs.\ else \\
Fault grounded                            & 1 & binary      & L, Li, G & ground element engaged? \\
\midrule
Event state                               & 2 & 3-class     & G        & normal / switching / fault \\
Fault-type classification (\ac{fc})\,$\dagger$ & 2 & 9-class & L, Li, G & short-circuit / incipient / \ac{hif} subtype \\
Faulted-phase identification              & 2 & 7-class     & L, Li, G & A\,/\,B\,/\,C\,/\,AB\,/\,AC\,/\,BC\,/\,ABC \\
Fault category                            & 2 & 3-class     & L, Li, G & short-circuit / incipient / \ac{hif} \\
Switch type                               & 2 & 8-class     & G        & which benign switching event \\
Fault origin                              & 2 & binary      & G        & line vs.\ busbar fault \\
\midrule
Event classification (\ac{ec})\,$\dagger$ & 3 & 23-class    & G        & full event-type taxonomy \\
Fault localisation (\ac{fl})\,$\dagger$   & 3 & regression  & L, Li, G & position along the faulted line \\
\bottomrule
\end{tabular}
\end{table*}
""",
    "topology_overview": r"""\begin{table}[!htbp]
\caption{Reference grids used to instantiate the benchmark, as released with the
EvEMTBench dataset. The \emph{global} view width (number of channels)
is grid-dependent, which motivates topology-agnostic models for the planned
cross-topology protocol.}
\label{tab:topology_overview}
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.05}
\begin{tabular}{lccc}
\toprule
\textbf{Grid} & \textbf{Voltage} & \textbf{$f_0$} & \textbf{\emph{global} width} \\
\midrule
CIGR\'E MV Grid          & 20\,kV  & 50\,Hz & 174\,ch \\
Double Line Grid         & 110\,kV & 50\,Hz & 48\,ch \\
Protection Reference Grid& 110\,kV & 50\,Hz & 108\,ch \\
IEEE 39-Bus System       & 345\,kV & 60\,Hz & 420\,ch \\
\bottomrule
\end{tabular}
\end{table}
""",
    "dataset_families": r"""\begin{table*}[t]
\centering
\caption{Dataset families defining the EvEMTBench generalization conditions. The families differ in topology, operating state, and event-parameter variation.}
\label{tab:dataset_families}

\begin{tabular}{@{}
  >{\raggedright\arraybackslash}p{2.4cm}
  >{\raggedright\arraybackslash}p{6.1cm}
  >{\raggedright\arraybackslash}p{5.3cm}
@{}}
\toprule
\textbf{Family} &
\textbf{Variation} &
\textbf{Role} \\
\midrule

\texttt{multi\_grid}
& Topology, operating state, and event parameters vary
& Cross-grid pre-training and source-domain learning \\

\texttt{adapt\_grid}
& Topology is fixed; operating state and event parameters vary
& Target-grid training, validation, and in-distribution testing \\

\texttt{eval\_grid}
& Topology and operating state are fixed; event parameters follow predefined sweeps
& Evaluation-only shifted condition for held-out and transfer protocols \\

\bottomrule
\end{tabular}
\end{table*}
""",
    "observability": r"""\begin{table}[t]
\centering
\caption{Observability views (nested oracle on the faulted line).}
\label{tab:observability}
\begin{tabular}{lll}
\toprule
\textbf{View} & \textbf{Channels} & \textbf{Protection analogue} \\
\midrule
local  & 6  & single-ended relay (near terminal) \\
line   & 12 & two-terminal current-differential \\
global & grid-dependent & centralised / wide-area \\
\bottomrule
\end{tabular}
\end{table}
""",
    "baselines": r"""\begin{table}[t]
\centering
\caption{Reference baseline suite. The threshold baseline is defined only for local \ac{fd}, and the impedance-based locator only for \ac{fl} under local and line observability. Complete configurations are reported in the Supplementary Information.}
\label{tab:baselines}
\begin{tabular}{@{}lll@{}}
\toprule
\textbf{Baseline} &
\textbf{Type} &
\textbf{Input representation} \\
\midrule
Majority
& Trivial
& Training labels or regression targets \\

Threshold
& Conventional
& Local phase currents (\ac{fd}) \\

Impedance locator
& Conventional
& Fundamental phasors and line impedance (\ac{fl}) \\

\Ac{rf}
& Feature-based
& Per-channel waveform statistics \\

\Ac{mlp}
& Deep
& Flattened waveform window \\

\Ac{tst}
& Deep
& Patch-embedded waveform sequence \\

\Ac{gru}
& Deep
& Waveform sequence \\

\Ac{cnn}
& Deep
& Waveform sequence \\

\Ac{resnet}
& Deep
& Waveform sequence \\
\bottomrule
\end{tabular}
\end{table}
""",
}


def spec_table(name: str) -> str:
    """The static specification table ``name`` with a generated-by header (LaTeX comment)."""
    return _spec_header(name) + SPEC_TABLES[name]


# --------------------------------------------------------------------------------------------------
# F03: protection-grade error rates + per-class F1 (the "protection-aware metrics" contribution,
# reported for the validation-selected reference baseline, from the same result.json aggregates).
# --------------------------------------------------------------------------------------------------
# binary function -> (dependability = missed-rate key, security = false-alarm-rate key)
_BINARY_RATE_KEYS = {
    "fault_detection": ("missed_fault_rate", "false_alarm_rate"),
    "event_detection": ("missed_event_rate", "false_event_trip_rate"),
}
# FC 9-class vocabulary: (per-class f1 aggregate key, compact column label). Frozen order.
_FC_F1 = [
    ("f1_flt_1phg_shc", "1pg-SHC"),
    ("f1_flt_1phg_shc_w_arc", "1pg-SHC-arc"),
    ("f1_flt_2ph_shc", "2p-SHC"),
    ("f1_flt_2phg_shc", "2pg-SHC"),
    ("f1_flt_3ph_shc", "3p-SHC"),
    ("f1_flt_1phg_incipient", "1pg-inc"),
    ("f1_flt_1phg_incipient_w_arc", "1pg-inc-arc"),
    ("f1_flt_1phg_hif", "1pg-HIF"),
    ("f1_flt_1phg_hif_w_arc", "1pg-HIF-arc"),
]


def _sel_metric_cell(
    rows: dict, task_id: str, test_set: str, metric: str, compact: bool = False
) -> str:
    """The validation-selected reference baseline's value of ``metric`` on ``test_set`` (the same
    baseline chosen for the headline), as clipped mean$\\pm$ci; ``\\pending``/``\\undefm`` markers
    otherwise. The headline selection uses balanced accuracy; here we read a secondary metric of
    that selected baseline."""
    choice = best_cell_choice(rows, task_id, test_set, "balanced_accuracy")
    if choice is None:
        return r"\pending"
    doc = rows.get((task_id, choice["baseline"]), {}).get(test_set)
    m = mean_val(doc, metric)
    if m is None:
        return r"\undefm"  # metric undefined for this cell (e.g. class absent) -- never zero
    return _fmt_mean_ci(m, ci_val(doc, metric), metric, compact=compact)


#: Short LaTeX label for a selected baseline, used where the table names the reference identity.
_BASELINE_ACS = {
    "random_forest": r"\acs{rf}",
    "mlp": r"\acs{mlp}",
    "gru": r"\acs{gru}",
    "cnn": r"\acs{cnn}",
    "resnet": r"\acs{resnet}",
    "tst": r"\acs{tst}",
    "majority": "maj.",
    "threshold": "thr.",
    "impedance": "imp.",
}


def _sel_baseline_cell(rows: dict, task_id: str, test_set: str) -> str:
    """Identity of the validation-selected reference reported for (task_id, test_set)."""
    choice = best_cell_choice(rows, task_id, test_set, "balanced_accuracy")
    if choice is None:
        return r"\pending"
    return _BASELINE_ACS.get(choice["baseline"], choice["baseline"])


def protection_rate_table() -> str:
    """SI table: the protection-grade confusion pair for the binary functions (FD, ED) of the
    validation-selected reference, alongside that reference's identity and headline balanced
    accuracy, per (function, view, grid) on the in-distribution and never-trained sets.

    Carrying balanced accuracy and the selected architecture in the SAME row as MDR/FAR is what
    makes the ``eval_grid`` collapse cells self-evident: a row reading ``0.50`` balanced accuracy
    with MDR $\\approx 0$ and FAR $\\approx 1$ is an always-positive decision rule, not a random
    one. No test against random prediction is performed or implied --- the behaviour is read
    directly off the confusion-derived rates.

    RESTORED 2026-08-20. The two test-set blocks stay SIDE BY SIDE, because comparing a cell's
    in-distribution behaviour against its ``eval_grid`` behaviour is the entire point of the table
    and stacking them would put the pair four rows apart. What changed is the cell rendering: the
    six numeric columns now use the compact subscript CI form (as the per-baseline tables do)
    instead of ``mean $\\pm$ ci``, which is what let the old 10-column layout fit only inside the
    scale-to-fit box that rendered it at 3.76--4.98 pt. Compact cells fit at \\scriptsize with no
    scaling."""
    per_grid = {g: load_rows(g) for g, _ in GRID_ROWS}
    body = []
    specs = [("FD", "fault_detection", OBS), ("ED", "event_detection", ("global",))]
    for flabel, stem, views in specs:
        mk, fk = _BINARY_RATE_KEYS[stem]
        for v in views:
            for g, glabel in GRID_ROWS:
                r = per_grid[g]
                tid = f"{stem}_{v}"
                cells = []
                for ts in TEST_SETS:
                    cells.append(_sel_baseline_cell(r, tid, ts))
                    cells.append(_sel_metric_cell(r, tid, ts, "balanced_accuracy", compact=True))
                    cells += [_sel_metric_cell(r, tid, ts, key, compact=True) for key in (mk, fk)]
                # ASCII-safe separator: a literal U+00B7 in the .tex renders as mojibake under the
                # T1/inputenc combination this document uses.
                body.append(f"{flabel}\\,/\\,{v} & {glabel} & " + " & ".join(cells) + r" \\")
            body.append(r"\midrule")
    body = body[:-1]  # drop the trailing rule
    return (
        provenance_header(
            "hpc/paper_artifacts/build_supp_tab_protection.py",
            [
                _held_out_glob("{fault_detection_{local,line,global},event_detection_global}"),
                "  metrics: balanced_accuracy, missed_fault_rate / missed_event_rate,",
                "           false_alarm_rate / false_event_trip_rate (aggregate over five seeds)",
            ],
        )
        + "\\begin{table*}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3.4pt}\n"
        "\\caption{Protection-grade behaviour of the binary functions (\\ac{fd}, \\ac{ed}) for the "
        "\\emph{validation-selected} reference baseline: its identity, headline balanced accuracy "
        "($\\uparrow$), dependability (\\ac{mdr}, missed-fault/-event rate $\\downarrow$) and "
        "security (\\ac{far}, false-alarm/-trip rate $\\downarrow$), on the in-distribution test "
        "split and the evaluation-only \\texttt{eval\\_grid} family. \\ac{fd} is scored on windows of "
        "fault episodes, so its \\ac{far} is measured on the pre-/post-fault negatives of those "
        "episodes and does not describe security against benign switching. A row with balanced "
        "accuracy $\\approx 0.5$, \\ac{mdr}~$\\approx 0$ and \\ac{far}~$\\approx 1$ is a degenerate "
        "\\emph{always-positive} decision rule rather than random prediction; no statistical test "
        "against a random predictor is performed. Cells are mean with the 95\\% CI half-width as a "
        "compact subscript ($\\pm h$) over five seeds."
        + _pending_legend(body, " \\pending{}~= pending.")
        + "}\n"
        "\\label{tab:supp_protection}\n"
        "\\begin{tabular}{l l l ccc l ccc}\n\\toprule\n"
        " & & \\multicolumn{4}{c}{\\textbf{In-distribution (test)}} & "
        "\\multicolumn{4}{c}{\\textbf{Evaluation-only (\\texttt{eval\\_grid})}} \\\\\n"
        "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\n"
        "\\textbf{Function\\,/\\,view} & \\textbf{Grid} & \\textbf{ref.} & "
        "\\textbf{bal.\\ acc.} & "
        "\\textbf{\\acs{mdr}} & \\textbf{\\acs{far}} & \\textbf{ref.} & \\textbf{bal.\\ acc.} & "
        "\\textbf{\\acs{mdr}} & \\textbf{\\acs{far}} \\\\\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )


def fc_per_class_f1_table() -> str:
    """SI table: per-class F1 for fault-type classification (FC, 9-class), validation-selected
    reference, in-distribution test set, per grid. Absent classes (e.g. HIF off MV) render \\undefm
    (undefined, not zero). The wider EC per-class breakdown is in the leaderboard."""
    per_grid = {g: load_rows(g) for g, _ in GRID_ROWS}
    heads = " & ".join(f"\\textbf{{{lab}}}" for _g, lab in GRID_ROWS)
    body = []
    for key, clabel in _FC_F1:
        # rows = class, columns = grid (narrow); line view, in-distribution test
        cells = [
            _sel_metric_cell(per_grid[g], "fault_classification_line", "test", key)
            for g, _ in GRID_ROWS
        ]
        body.append(f"{clabel} & " + " & ".join(cells) + r" \\")
    return (
        provenance_header(
            "hpc/paper_artifacts/build_supp_tab_fc_perclass.py",
            [_held_out_glob("fault_classification_line", "{val,test}")],
        )
        + "\\begin{table}[!htbp]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.05}\n"
        "\\caption{Per-class $\\text{F}_1$ for fault-type classification (FC, 9-class, "
        "\\emph{line} view, in-distribution test), validation-selected reference baseline. "
        "\\undefm{}~= $\\text{F}_1$ undefined (no supported \\emph{and} no predicted samples); "
        "the two HIF subtypes are \\ac{mv}-only, so their per-grid support (Table~"
        "\\ref{tab:supp_fc_support}) is zero on the \\ac{hv}/\\ac{ehv} grids. Cells are "
        "mean\\,$\\pm$\\,95\\% CI over seeds.}\n"
        "\\label{tab:supp_fc_perclass}\n" + _SHRINK_TO_FIT + "\\begin{tabular}{l cccc}\n\\toprule\n"
        f"\\textbf{{Fault class}} & {heads} \\\\\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


# --------------------------------------------------------------------------------------------------
# Aggregators (preserve the entry points make_results_bundle.py + check_results_fresh.py import).
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# F13: dataset composition (episode counts per grid x family; per-class FC support). Split counts are
# repo-derived (committed split files); per-class support is read from the result.json breakdowns.
# --------------------------------------------------------------------------------------------------
def _episode_counts(grid: str) -> dict:
    from evemtbench.version import splits_version

    base = (
        _REPO
        / "src"
        / "evemtbench"
        / "splits"
        / f"v{splits_version('held_out')}"
        / "held_out"
        / grid
    )
    out = {}
    for split in ("train", "val", "test", "benchmark"):
        f = base / f"{split}.txt"
        out[split] = (
            sum(1 for line in f.read_text().splitlines() if line.strip()) if f.is_file() else None
        )
    return out


def dataset_split_table() -> str:
    """SI table: episode counts per grid for the held_out split families (train/val/test from the
    Fine-tune/adapt_grid family, plus the never-trained eval_grid family)."""
    body = []
    for g, glabel in GRID_ROWS:
        c = _episode_counts(g)
        cells = [
            (f"{c[s]:,}" if c[s] is not None else r"\pending")
            for s in ("train", "val", "test", "benchmark")
        ]
        body.append(f"{glabel} & " + " & ".join(cells) + r" \\")
    return (
        provenance_header(
            "hpc/paper_artifacts/build_supp_tab_splits.py",
            ["src/evemtbench/splits/v1.0/held_out/<grid>/{train,val,test,benchmark}.txt"],
        )
        + "\\begin{table}[!htbp]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.05}\n"
        "\\caption{Held-out split composition: number of \\emph{episodes} per grid. Under the frozen "
        "window contract each episode yields $23$ windows ($50\\,$ms length, $5\\,$ms step, over "
        "the event-centred extraction interval), uniformly across every grid and family; the "
        "number of windows actually \\emph{scored} by each function is smaller and is given in "
        "Table~\\ref{tab:supp_primary_valid_counts}. Train/validation/test are the $70/15/15$ "
        "episode-level split of the \\texttt{adapt\\_grid} family; \\texttt{eval\\_grid} is the "
        "entire never-trained family.}\n"
        "\\label{tab:supp_splits}\n"
        "\\begin{tabular}{l rrrr}\n\\toprule\n"
        "\\textbf{Grid} & \\textbf{train} & \\textbf{val} & \\textbf{test} & "
        "\\textbf{\\texttt{eval\\_grid}} \\\\\n"
        "\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def _fc_class_support(grid: str, test_set: str = "test") -> dict:
    """{FC class name -> support count n} on the eval set, from any FC cell's event_type breakdown
    (support is a property of the set, not the model)."""
    import glob
    import json

    from evemtbench.version import results_version

    ver = results_version("held_out", "fault_classification_line")
    pat = f"outputs/{ver}/fault_classification_line/held_out/{grid}/*/{test_set}/result.json"
    for f in sorted(glob.glob(pat)):
        try:
            bd = json.load(open(f)).get("breakdowns", {}).get("event_type", {})
        except (OSError, json.JSONDecodeError, AttributeError) as e:
            warn_once(
                f"skipping unreadable/malformed FC-support result.json: {f} "
                f"({type(e).__name__}: {e})"
            )
            continue
        if bd:
            return {name: (bd.get(name) or {}).get("n") for _k, name in _FC_SUPPORT_CLASSES}
    return {}


# (per-class f1 key -> event_type breakdown class name) = the key without the f1_ prefix
_FC_SUPPORT_CLASSES = [(k, k[len("f1_") :]) for k, _lab in _FC_F1]


def fc_support_table() -> str:
    """SI table: per-class support (window count) for the 9 FC classes per grid (in-distribution
    test), so that a per-class $F_1$ of \\undefm{} can be read as genuine zero support."""
    per_grid = {g: _fc_class_support(g, "test") for g, _ in GRID_ROWS}
    heads = " & ".join(f"\\textbf{{{lab}}}" for _g, lab in GRID_ROWS)
    body = []
    for (_key, cname), (_k2, clabel) in zip(_FC_SUPPORT_CLASSES, _FC_F1, strict=True):
        cells = []
        for g, _ in GRID_ROWS:
            n = per_grid[g].get(cname)
            cells.append(f"{n:,}" if n else "0")  # 0 support = the class is absent on that grid
        body.append(f"{clabel} & " + " & ".join(cells) + r" \\")
    return (
        provenance_header(
            "hpc/paper_artifacts/build_supp_tab_fc_support.py",
            [_held_out_glob("fault_classification_line", "test") + "  (breakdowns.event_type.n)"],
        )
        + "\\begin{table}[!htbp]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.05}\n"
        "\\caption{Per-class support (test-set window count) for the 9 FC classes per grid. A count "
        "of $0$ means the class is absent on that grid (the two HIF subtypes are \\ac{mv}-only), so "
        "its per-class $\\text{F}_1$ is undefined rather than zero.}\n"
        "\\label{tab:supp_fc_support}\n" + _SHRINK_TO_FIT + "\\begin{tabular}{l rrrr}\n\\toprule\n"
        f"\\textbf{{Fault class}} & {heads} \\\\\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


# --------------------------------------------------------------------------------------------------
# F04: the seven extended (diagnostic) functions, per-baseline (no selection), balanced accuracy on
# the in-distribution test set at each function's natural view. Per-baseline (not a single "best")
# keeps this consistent with the winner's-curse fix without a separate validation re-score.
# --------------------------------------------------------------------------------------------------
_DIAGNOSTICS = [  # (row label, task stem, natural view, class kind)
    ("Fault grounded", "fault_grounded", "line", "binary"),
    ("Faulted phase", "fault_phase", "line", "7-class"),
    ("Fault category", "fault_category", "line", "3-class"),
    ("Fault origin", "fault_origin", "global", "binary"),
    ("Event state", "event_state", "global", "3-class"),
    ("Switch detection", "switch_detection", "global", "binary"),
    ("Switch type", "switch_type", "global", "8-class"),
]
_DIAG_BASELINES = ["majority", "random_forest", "mlp", "gru", "cnn", "resnet", "tst"]


def _diag_cell(rows: dict, task_id: str, baseline: str) -> str:
    doc = rows.get((task_id, baseline), {}).get("test")
    m = mean_val(doc, "balanced_accuracy")
    if m is None:
        return r"\pending"
    return _fmt_mean_ci(m, ci_val(doc, "balanced_accuracy"), "balanced_accuracy")


def si_panels(tabulars: list[str]) -> str:
    """Stack tabulars vertically under one caption and label.

    An SI table that does not fit the text width at the minimum table font is split into panels;
    each panel repeats the identifying columns and the column headings."""
    return "\n\\par\\medskip\n".join(tabulars) + "\n"


def diagnostic_overview_table() -> str:
    """SI table: per-baseline balanced accuracy for the seven diagnostic functions (in-distribution
    test, each at its natural view), so the extended suite is reported rather than only summarised.

    Each function is a group heading above its four grid rows, so the table needs no function
    column and fits the text width at \\footnotesize."""
    per_grid = {g: load_rows(g) for g, _ in GRID_ROWS}
    heads = " & ".join(f"\\textbf{{{_BASELINE_LABEL[b].split()[0]}}}" for b in _DIAG_BASELINES)
    ncol = 1 + len(_DIAG_BASELINES)
    rows = []
    for label, stem, view, kind in _DIAGNOSTICS:
        rows.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{label} ({kind}, {view})}}}} \\\\")
        for g, glabel in GRID_ROWS:
            cells = [_diag_cell(per_grid[g], f"{stem}_{view}", b) for b in _DIAG_BASELINES]
            rows.append(f"{glabel} & " + " & ".join(cells) + r" \\")
        rows.append("\\midrule")
    body = "\n".join(rows[:-1])  # drop trailing midrule
    return (
        provenance_header(
            "hpc/paper_artifacts/build_supp_tab_diagnostics.py",
            [
                _held_out_glob("{fault_grounded,fault_phase,fault_category}_line"),
                _held_out_glob("{fault_origin,event_state,switch_detection,switch_type}_global"),
            ],
        )
        + "\\begin{table}[!htbp]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
        "\\renewcommand{\\arraystretch}{1.05}\n"
        "\\caption{Extended (diagnostic) functions: per-baseline balanced accuracy ($\\uparrow$) on "
        "the in-distribution test set, each function at its natural observability view. Reported "
        "per baseline (no single `best') for the full picture."
        + _pending_legend(body, " \\pending{}~= pending.")
        + " The never-trained, per-class, and protection-grade breakdowns are in the "
        "leaderboard.}\n"
        "\\label{tab:supp_diagnostics}\n"
        + f"\\begin{{tabular}}{{l {'c ' * len(_DIAG_BASELINES)}}}\n\\toprule\n"
        f"\\textbf{{Grid}} & {heads} \\\\\n\\midrule\n"
        + body
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def build_paper_tables() -> dict[str, str]:
    """``{filename: latex}`` for the five held_out result tables (pure, deterministic)."""
    return {
        f"tab_res_{key}.tex": manuscript_caption(f"tab_res_{key}.tex", result_table(key))
        for key, *_ in PAPER_FUNCTIONS
    }


def _val_audit_cell(rows: dict, stem: str, view: str, metric: str) -> str:
    """S10 audit cell: the winner's validation score *plus the winner's identity* — auditing a
    best-of-six selection needs at least who won (evemt-v3 V3-082/041), not only the score."""
    if _held_unsupported(stem, view):
        return r"\notapp"
    choice = best_cell_choice(rows, f"{stem}_{view}", "val", metric)
    if choice is None:
        return r"\pending"
    short = {
        "random_forest": r"\acs{rf}",
        "mlp": r"\acs{mlp}",
        "gru": r"\acs{gru}",
        "cnn": r"\acs{cnn}",
        "resnet": r"\acs{resnet}",
        "tst": r"\acs{tst}",
    }
    label = short.get(choice["baseline"], choice["baseline"])
    return f"{_fmt_mean_ci(choice['mean'], choice['ci'], metric)} ({label})"


def val_selection_table() -> str:
    """SI selection-audit table (V2-019 + V3-082): the validation-split headline AND identity of the
    *validation-selected* reference baseline for each primary function x grid x view -- i.e. the
    score that drives the F06 model selection reported in the S1 held_out tables, with the winner
    named so the selection is auditable against the per-baseline S2 tables."""
    per_grid = {g: load_rows(g) for g, _ in GRID_ROWS}
    short = {
        "fd": r"\ac{fd} $\uparrow$",
        "fc": r"\ac{fc} $\uparrow$",
        "fl": r"\ac{fl}\,\% $\downarrow$",
        "ed": r"\ac{ed} $\uparrow$",
        "ec": r"\ac{ec} $\uparrow$",
    }
    groups = []
    for key, _name, stem, metric, _label in PAPER_FUNCTIONS:
        rows_out = []
        for gi, (grid, glabel) in enumerate(GRID_ROWS):
            cells = [_val_audit_cell(per_grid[grid], stem, view, metric) for view in OBS]
            head = f"\\multirow{{{len(GRID_ROWS)}}}{{*}}{{{short[key]}}}" if gi == 0 else ""
            rows_out.append(f"{head:<30} & {glabel:<24} & " + " & ".join(cells) + r" \\")
        groups.append("\n".join(rows_out))
    body = "\n\\midrule\n".join(groups)
    return (
        provenance_header(
            "hpc/paper_artifacts/build_supp_tab_val_selection.py",
            [
                _held_out_glob(
                    "{fault_detection,fault_classification,fault_location}_{local,line,global}",
                    "val",
                ),
                _held_out_glob("{event_detection,event_classification}_global", "val"),
            ],
        )
        + "\\begin{table}[!htbp]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.05}\n"
        "\\caption{Validation-split selection audit: the headline score \\emph{and identity} (in "
        "parentheses) of the validation-selected reference baseline that drives the model selection "
        "reported in Tables~\\ref{tab:res_fd}--\\ref{tab:res_ec}, per primary function, grid, and "
        "observability view. Each cell is the winner's validation mean\\,$\\pm$\\,95\\% CI over five "
        "seeds; its held-out \\emph{test} and \\emph{benchmark} values are in "
        "Tables~\\ref{tab:res_fd}--\\ref{tab:res_ec}, and the competing baselines' held-out values "
        "are in the S2 per-baseline tables. Arrows give the metric direction (balanced "
        "accuracy $\\uparrow$; \\acs{mae}\\,\\% $\\downarrow$ for \\ac{fl}). \\notapp{}~= not defined "
        "for this view (the global-only event-scope functions \\ac{ed}/\\ac{ec} at \\emph{local}/"
        "\\emph{line}).}\n"
        "\\label{tab:supp_val_selection}\n"
        "\\begin{tabular}{ll ccc}\n\\toprule\n"
        "\\textbf{Function} & \\textbf{Grid} & \\textbf{local} & \\textbf{line} & \\textbf{global} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def build_supplement_tables() -> dict[str, str]:
    """``{filename: latex}`` for the supplementary tables (pure, deterministic)."""
    # The ten per-baseline supp_tab_<fn>_{test,benchmark} fragments and supp_tab_protection were
    # retired on 2026-08-17 (they only fitted inside a scale-to-fit box, which rendered them at
    # 3.76-4.98 pt) and RESTORED on 2026-08-20 with layouts that fit \textwidth unscaled: the
    # per-baseline tables transposed to (grid, view) x baseline = 9 columns, and the protection
    # table keeping its side-by-side test-set blocks but rendering cells in the compact subscript
    # CI form. The fig_supp_<fn> / fig_supp_protection figures remain; tables and figures now show
    # the same numbers in the two registers (scannable vs at-a-glance).
    tables = {}
    for k in ("fd", "fc", "fl", "ed", "ec"):
        for ts in TEST_SETS:
            tables[f"supp_tab_{k}_{ts}.tex"] = supp_table(k, ts)
    tables["supp_tab_protection.tex"] = protection_rate_table()
    tables["supp_tab_fc_perclass.tex"] = fc_per_class_f1_table()
    tables["supp_tab_splits.tex"] = dataset_split_table()
    tables["supp_tab_fc_support.tex"] = fc_support_table()
    tables["supp_tab_diagnostics.tex"] = diagnostic_overview_table()
    tables["supp_tab_val_selection.tex"] = val_selection_table()
    return {name: manuscript_caption(name, latex) for name, latex in tables.items()}


def build_transfer_tables(outputs: str | Path = "outputs") -> dict[str, str]:
    """``{filename: latex}`` for the three cross-grid transfer tables (indexes built once, shared).
    ED/EC are global-only and transfer is fixed-width local/line only, so their transfer tables
    would be structurally out of scope (all \\oos) — they are not built (evemt-v3 V3-009)."""
    idx = _transfer_index(outputs)
    val_idx = _transfer_val_index(outputs)
    return {
        f"tab_transfer_{key}.tex": manuscript_caption(
            f"tab_transfer_{key}.tex", transfer_table(key, outputs, idx, val_idx)
        )
        for key, *_ in PAPER_FUNCTIONS
        if key not in ("ed", "ec")
    }


def build_spec_tables() -> dict[str, str]:
    """``{filename: latex}`` for the five static specification tables."""
    return {
        f"tab_{name}.tex": manuscript_caption(f"tab_{name}.tex", spec_table(name))
        for name in SPEC_TABLES
    }


def manuscript_caption(name: str, content: str) -> str:
    """Replace the generated caption with the manuscript's wording (si_captions.CAPTIONS), if any.

    The caption text is maintained in the manuscript; the generator keeps the numbers and layout."""
    from si_captions import CAPTIONS

    caption = CAPTIONS.get(name)
    if caption is None:
        return content
    start = content.index("\\caption{") + len("\\caption{")
    depth, i = 1, start
    while depth:
        ch = content[i]
        if ch == "\\":
            i += 2
            continue
        depth += {"{": 1, "}": -1}.get(ch, 0)
        i += 1
    return content[:start] + caption + content[i - 1 :]


def run(name: str, content: str, argv: list[str] | None = None) -> None:
    """Write (or, with ``--check``, verify) exactly ONE artifact — the standalone entry point every
    ``build_<x>.py`` calls in ``__main__``.

    This is the write *primitive* the owning script drives; it never decides which artifacts exist,
    so the golden rule ("one artifact = one generator") is enforced at the call site. ``--check``
    regenerates in memory and exits non-zero if the committed file is stale or missing, and never
    touches the file — the mode ``make check-supp-artifacts`` and CI use."""
    import argparse
    import difflib

    content = manuscript_caption(name, content)
    ap = argparse.ArgumentParser(description=f"Build (or --check) {name}.")
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify the committed artifact matches a fresh rebuild; write nothing",
    )
    args = ap.parse_args(argv)
    dest = GEN_DIR / name
    if args.check:
        if not dest.is_file():
            sys.exit(f"MISSING {dest} — run this generator without --check")
        committed = dest.read_text()
        if committed == content:
            print(f"ok {dest}")
            return
        diff = list(
            difflib.unified_diff(
                committed.splitlines(),
                content.splitlines(),
                fromfile=f"committed/{name}",
                tofile=f"fresh/{name}",
                lineterm="",
            )
        )
        print("\n".join(diff[:60]))
        sys.exit(f"STALE {dest} — regenerate it (see paper/manuscript/generated/README.md)")
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    dest.write_text(content)
    n_res = content.count(r"\res")
    print(f"wrote {dest} ({n_res} empty cells -> \\res)")
