"""Coverage manifest: the expected experiment matrix vs. what's on disk.

A *cell* is one runner invocation = (task, protocol, grid, baseline). One cell produces a
``result.json`` for each held-out test set (``test`` + ``benchmark``) under
``outputs/<version>/<task>/<protocol>/<grid>/<baseline>/<test_set>/``.

This module is the single source of truth for both the SLURM **skip-existing** filter (only
submit cells that are not yet ``done``) and **progress tracking** (done / pending / failed). It
enumerates the expected matrix from the registry + grid registry, applying applicability rules
(e.g. the ``threshold`` overcurrent baseline only fits a binary single-end detection task), and
diffs it against the ``result.json`` files actually present.

    python -m evemtbench.coverage --summary
    python -m evemtbench.coverage --emit-pending      # runner arg-lines for the sweep driver
    python -m evemtbench.coverage --check-floor       # majority-floor sanity gate (validity)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from . import failures
from .data.roots import GRIDS
from .splits.loader import read_held_out_splits
from .splits.protocols import PRETRAIN_TRANSFER_PROTOCOLS, TRANSFER_PROTOCOLS
from .tasks.registry import get_task, list_tasks
from .tasks.taxonomy import Function, LabelKind, Observability, TaskDefinition
from .version import BENCHMARK_VERSION, is_fault_phase_task, results_version

TEST_SETS = ("test", "benchmark")

# The multi_grid pre-training grids vs. the reference grids the transfer protocols evaluate on.
PRETRAIN_GRIDS = ("template_20kv", "template_110kv", "template_345kv")
PRETRAIN_GRID = PRETRAIN_GRIDS[0]  # primary pre-training grid (default transfer source)
REFERENCE_GRIDS = ("double_line", "cigre_mv", "testgrid_110kv", "ieee39")


@cache
def expected_test_sets(grid: str, protocol: str = "held_out") -> tuple[str, ...]:
    """Test sets a cell must produce. ``multi_grid_pretrain`` is in-corpus (``test`` only, no
    separate benchmark family). Otherwise: in-dist ``test`` always; ``benchmark`` only when that
    grid's benchmark family is split (a not-yet-windowed benchmark -> in-dist only)."""
    if protocol == "multi_grid_pretrain":
        return ("test",)
    try:
        splits = read_held_out_splits(grid)
        return ("test", "benchmark") if splits.get("benchmark") else ("test",)
    except Exception:
        return TEST_SETS


# gnn is added with the cross_topology protocol; threshold is the conventional overcurrent anchor.
# resnet is the modern learned reference (1D ResNet) added in v0.2 alongside mlp/gru/cnn.
# tst is the transformer reference (Time Series Transformer encoder with a patch embedding). It is
# appended LAST in every tuple: the validation selection keeps the earlier entry on a tie, so the
# order of the existing baselines, and with it every existing tie-break, is unchanged.
# impedance is the conventional impedance-based fault locator: the fault-location counterpart of
# threshold, a deterministic physical reference outside the learned-baseline selection pool (so its
# position here touches no tie-break).
ALL_BASELINES = (
    "majority",
    "threshold",
    "impedance",
    "random_forest",
    "mlp",
    "gru",
    "cnn",
    "resnet",
    "tst",
)
# The conventional (non-learned, physics-based) references and where each is defined; see
# ``baseline_applies``. They are reported next to the learned baselines, never selected among.
CONVENTIONAL_BASELINES = ("threshold", "impedance")
# The floor gate checks only the learned baselines: majority IS the floor, and threshold/impedance
# are conventional rules whose failures are findings about the rule, not plumbing bugs.
LEARNED_BASELINES = ("random_forest", "mlp", "gru", "cnn", "resnet", "tst")
# Deep baselines carry transferable weights — the only baselines the pre-training/transfer
# protocols apply to (classical baselines have no pretrained state to transfer; the target's
# in-distribution held_out majority provides the reference floor).
DEEP_BASELINES = ("mlp", "gru", "cnn", "resnet", "tst")
FAULT_TRIAD = ("fault_detection", "fault_classification", "fault_location")


@dataclass(frozen=True)
class Cell:
    task: str
    protocol: str
    grid: str
    baseline: str
    source: str | None = None  # transfer pre-training source grid; None for non-transfer cells


def baseline_applies(baseline: str, td: TaskDefinition, protocol: str = "held_out") -> bool:
    """Whether a baseline is meaningful for a task under a protocol.

    ``threshold`` is a single-end overcurrent rule: only binary detection at LOCAL observability.
    ``impedance`` is an impedance-based fault locator: only fault location, at LOCAL (one-terminal
    reactance method) and LINE (two-terminal method) observability; GLOBAL adds no physical
    information for a per-line locator.
    The pre-training/transfer protocols apply only to the deep baselines (transferable weights).
    All learned baselines apply to every task (they branch on label_kind internally).
    """
    if protocol in PRETRAIN_TRANSFER_PROTOCOLS:
        return baseline in DEEP_BASELINES
    if baseline == "impedance":
        return td.function is Function.FAULT_LOCATION and td.observability in (
            Observability.LOCAL,
            Observability.LINE,
        )
    if baseline == "threshold":
        return (
            td.label_kind is LabelKind.BINARY
            and td.observability is Observability.LOCAL
            and td.function in (Function.FAULT_DETECTION,)
        )
    return True


def protocol_grid_applies(protocol: str, grid: str) -> bool:
    """Which grids a protocol is scored on. ``multi_grid_pretrain`` runs only on the pre-training
    grids; ``held_out`` and the transfer protocols run only on the reference grids (the pre-training
    grids have no adapt_grid/benchmark family to score against)."""
    if protocol == "multi_grid_pretrain":
        return grid in PRETRAIN_GRIDS
    if protocol in ("held_out", *PRETRAIN_TRANSFER_PROTOCOLS):
        return grid in REFERENCE_GRIDS
    return True


def iter_cells(
    *,
    tasks: list[str] | None = None,
    grids: list[str] | None = None,
    protocols: tuple[str, ...] = ("held_out",),
    baselines: tuple[str, ...] = ALL_BASELINES,
) -> list[Cell]:
    """Enumerate the applicable cells of the matrix (applicability rules pre-applied)."""
    tasks = tasks if tasks is not None else list_tasks()
    grids = grids if grids is not None else list(GRIDS)
    cells: list[Cell] = []
    for task in tasks:
        td = get_task(task)
        for protocol in protocols:
            if protocol not in td.protocols:
                continue
            for grid in grids:
                if not protocol_grid_applies(protocol, grid):
                    continue
                for baseline in baselines:
                    if not baseline_applies(baseline, td, protocol):
                        continue
                    if protocol in TRANSFER_PROTOCOLS:
                        # transfer cells fan out over the pre-training source grids (20/110/345 kV);
                        # each (target, task, baseline) is scored once per source.
                        for source in PRETRAIN_GRIDS:
                            cells.append(Cell(task, protocol, grid, baseline, source=source))
                    else:
                        cells.append(Cell(task, protocol, grid, baseline))
    return cells


def cell_results_version(cell: Cell) -> str:
    """The current results tree of one cell: :func:`evemtbench.version.results_version` with the
    cell's grid and (transfer) source, which the v1.3.0 fault_phase scoping depends on."""
    return results_version(cell.protocol, cell.task, grid=cell.grid, source=cell.source)


def cell_from_key(key: str | Path) -> Cell | None:
    """The :class:`Cell` a cell path names, relative to its results tree:
    ``<task>/<protocol>/<grid>/<baseline>[/pretrain-<source>]`` (the layout :func:`cell_dir`
    writes; a transfer cell without the token is from the default source). ``None`` for anything
    shorter. Tree walkers use it to resolve a path's current version per cell
    (``cell_results_version(cell_from_key(...))``) instead of re-parsing the layout themselves."""
    parts = Path(key).parts
    if len(parts) < 4:
        return None
    task, protocol, grid, baseline = parts[:4]
    source = None
    if len(parts) > 4 and parts[4].startswith("pretrain-"):
        source = parts[4][len("pretrain-") :]
    elif protocol in TRANSFER_PROTOCOLS:
        source = PRETRAIN_GRID
    return Cell(task, protocol, grid, baseline, source=source)


def cell_dir(out: Path, cell: Cell, *, version: str | None = None) -> Path:
    """A cell's output directory. ``version=None`` (default) resolves per protocol, task, grid and
    source (:func:`cell_results_version`): held_out results live in — and stay in — the 1.0.0
    tree and the pretrain/transfer protocols re-scored under the corrected v1.1 source splits in
    the 1.1.0 tree, except the fault-location cells re-run under the corrected v1.2 valid-sample
    gate, which live in the 1.2.0 tree, and the fault_phase cells that read an LP-1-repaired
    label, which live in the 1.3.0 tree. An explicit ``version`` always wins (historical
    inspection)."""
    if version is None:
        version = cell_results_version(cell)
    d = Path(out) / version / cell.task / cell.protocol / cell.grid / cell.baseline
    # A transfer cell from a NON-default source gets a source-qualified subdir so it never collides
    # with the default (PRETRAIN_GRID) source, which keeps the legacy source-less path — so the
    # frozen 20 kV-sourced results stay byte-identical.
    if cell.source is not None and cell.source != PRETRAIN_GRID:
        d = d / f"pretrain-{cell.source}"
    return d


def _load_result(path: Path) -> dict | None:
    """The parsed result.json when present + well-formed (non-empty aggregate), else None."""
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(doc.get("aggregate"), dict) and doc["aggregate"]:
        return doc
    return None


def _valid_result(path: Path) -> bool:
    return _load_result(path) is not None


def cell_status(out: Path, cell: Cell, *, version: str | None = None) -> str:
    """``done`` (all expected test sets valid), ``failed`` (partial), or ``pending`` (none)."""
    d = cell_dir(out, cell, version=version)
    expected = expected_test_sets(cell.grid, cell.protocol)
    present = sum(_valid_result(d / ts / "result.json") for ts in expected)
    if present == len(expected):
        return "done"
    return "failed" if present else "pending"


def scan(out: Path, cells: list[Cell], *, version: str | None = None) -> dict[str, list[Cell]]:
    groups: dict[str, list[Cell]] = {"done": [], "pending": [], "failed": []}
    for c in cells:
        groups[cell_status(out, c, version=version)].append(c)
    return groups


def failure_records(
    out: Path, cells: list[Cell], *, version: str | None = None
) -> dict[Cell, dict]:
    """``{cell: failure record}`` for the given (non-done) cells that have one
    (:mod:`evemtbench.failures`)."""
    recs: dict[Cell, dict] = {}
    for c in cells:
        doc = failures.load(cell_dir(out, c, version=version))
        if doc and doc["attempts"]:
            recs[c] = doc
    return recs


def exhausted_cells(recs: dict[Cell, dict], max_attempts: int) -> set[Cell]:
    """Cells whose failed attempts reached the retry cap (``max_attempts <= 0`` = no cap)."""
    if max_attempts <= 0:
        return set()
    return {c for c, doc in recs.items() if failures.n_attempts(doc) >= max_attempts}


def _arg_line(cell: Cell) -> str:
    line = (
        f"--task {cell.task} --protocol {cell.protocol} "
        f"--grid {cell.grid} --baseline {cell.baseline}"
    )
    if cell.source is not None:  # transfer: pin the pre-training source grid for the runner
        line += f" --pretrain-grid {cell.source}"
    return line


# Every protocol the sweep can run — the snapshot covers the full applicable matrix, with the
# familiar held_out sub-matrix reported first-class in ``counts_by_protocol``.
SNAPSHOT_PROTOCOLS = ("held_out", "multi_grid_pretrain", "transfer_zeroshot", "transfer_finetune")


def build_snapshot(
    out: Path,
    *,
    version: str | None = None,
    protocols: tuple[str, ...] = SNAPSHOT_PROTOCOLS,
) -> dict:
    """The persistent coverage manifest: per-cell status + reproduce arg-line for every applicable
    cell of the matrix, committed at the repo root as ``coverage_snapshot.json``.

    Deterministic — a pure function of ``outputs/`` and the registry (no timestamp, no git sha),
    so the freshness gate byte-compares it; history/timing live in the git commits that carry it.
    """
    counts_by_protocol: dict[str, dict[str, int]] = {}
    cell_rows: list[dict] = []
    for protocol in protocols:
        cells = iter_cells(protocols=(protocol,))
        groups = scan(out, cells, version=version)
        counts_by_protocol[protocol] = {
            "total": len(cells),
            "done": len(groups["done"]),
            "pending": len(groups["pending"]),
            "failed": len(groups["failed"]),
        }
        for status, group in groups.items():
            for c in group:
                cell_rows.append(
                    {
                        "task": c.task,
                        "protocol": c.protocol,
                        "grid": c.grid,
                        "baseline": c.baseline,
                        "source": c.source,
                        "results_version": (
                            cell_results_version(c) if version is None else version
                        ),
                        "status": status,
                        "expected_test_sets": list(expected_test_sets(c.grid, c.protocol)),
                        "arg_line": _arg_line(c),
                    }
                )
    cell_rows.sort(
        key=lambda r: (r["task"], r["protocol"], r["grid"], r["baseline"], r["source"] or "")
    )
    counts = {
        k: sum(c[k] for c in counts_by_protocol.values())
        for k in ("total", "done", "pending", "failed")
    }
    return {
        "benchmark_version": BENCHMARK_VERSION if version is None else version,
        # which outputs/<v>/ tree each protocol's CURRENT results live in (None = the per-protocol
        # default: held_out stays at its published 1.0.0 tree, pretrain/transfer at 1.1.0) ...
        "results_versions": {
            p: (results_version(p) if version is None else version) for p in protocols
        },
        # ... and the "<task>/<protocol>" pairs that deviate from that default as a whole (the
        # fault-location cells re-run under the corrected v1.2.0 valid-sample gate). Empty for an
        # explicit version. The fault_phase tasks are scoped per CELL since v1.3.0 (grid and
        # source), so they are not listed here: each cell row carries its own "results_version".
        "results_versions_by_task": {
            f"{r['task']}/{r['protocol']}": results_version(r["protocol"], r["task"])
            for r in cell_rows
            if version is None
            and not is_fault_phase_task(r["task"])
            and results_version(r["protocol"], r["task"]) != results_version(r["protocol"])
        },
        "protocols": list(protocols),
        "self_check": snapshot_self_check(out, cell_rows, counts, version=version),
        "counts": counts,
        "counts_by_protocol": counts_by_protocol,
        "cells": cell_rows,
    }


_RANGE_01 = ("balanced_accuracy", "accuracy", "macro_f1")
_RANGE_NONNEG = ("mae", "rmse", "median_abs_error")


def snapshot_self_check(
    out: Path, cell_rows: list[dict], counts: dict, *, version: str | None = None
) -> dict:
    """The snapshot's embedded invariant set (UnderstandingVCS leaderboard-self-check idiom).

    Hard checks (any failure => ok:false, and the freshness gate refuses to pass):
      * completeness — every applicable cell appears exactly once;
      * floor plausibility — the majority baseline sits on the analytic 1/k floor (plumbing);
      * value ranges — every done cell's aggregates are physically possible.
    Dominance violations (a learned baseline under the floor) are FINDINGS: reported, never
    blocking — they are results about methods, not about the plumbing. Deterministic: a pure
    function of outputs/ + the registry, so the byte-compare freshness gate still holds."""
    checks: list[dict] = []

    keys = [(r["task"], r["protocol"], r["grid"], r["baseline"], r["source"]) for r in cell_rows]
    complete = len(keys) == counts["total"] and len(set(keys)) == len(keys)
    checks.append(
        {
            "id": "completeness",
            "ok": complete,
            "detail": f"{len(set(keys))} unique cells listed of {counts['total']} applicable",
        }
    )

    floor = check_floor(out, iter_cells(protocols=("held_out",)), version=version)
    checks.append(
        {
            "id": "floor_plausibility",
            "ok": not floor["plausibility_violations"],
            "detail": floor["plausibility_violations"]
            or f"majority == 1/k in all {floor['n_groups']} checkable groups",
        }
    )

    range_problems: list[str] = []
    for r in cell_rows:
        if r["status"] != "done":
            continue
        # source=... is load-bearing: cell_dir appends `pretrain-<source>` for transfer cells whose
        # source is not the default, so dropping it silently redirected every 110 kV / 345 kV row to
        # the 20 kV directory — two thirds of all transfer cells were never range-checked, while the
        # 20 kV cell was re-checked up to three times.
        cell = Cell(r["task"], r["protocol"], r["grid"], r["baseline"], source=r["source"])
        label = f"{r['task']}/{r['protocol']}/{r['grid']}/{r['baseline']}" + (
            f"/pretrain-{r['source']}" if r["source"] else ""
        )
        for ts in r["expected_test_sets"]:
            doc = _load_result(cell_dir(out, cell, version=version) / ts / "result.json")
            if doc is None:
                # Invariant guard, not an expected branch: `cell_status` calls a cell done only when
                # _load_result succeeds for every expected test set at THIS path, so reaching here
                # means status and the range check disagree about where the cell lives. That is
                # exactly the failure the source= bug above produced, and the old bare `continue`
                # is why it produced no symptom — a misdirected lookup was indistinguishable from a
                # clean pass. Report it rather than skip it.
                range_problems.append(
                    f"{label}/{ts}: status=done but result.json missing/unreadable/empty "
                    f"(status/range-check path disagreement)"
                )
                continue
            for metric, agg in doc.get("aggregate", {}).items():
                mean = agg.get("mean") if isinstance(agg, dict) else None
                if mean is None:
                    continue
                bad = (
                    (metric in _RANGE_01 or metric.startswith("f1_"))
                    and not (0.0 <= mean <= 1.0)
                    or metric in _RANGE_NONNEG
                    and mean < 0.0
                    or metric == "r2"
                    and mean > 1.0 + 1e-9
                )
                if bad:
                    range_problems.append(f"{label}/{ts}: {metric}={mean!r} out of range")
    checks.append(
        {
            "id": "value_ranges",
            "ok": not range_problems,
            "detail": range_problems or "all done-cell aggregates within physical ranges",
        }
    )

    return {
        "ok": all(c["ok"] for c in checks),
        "checks": checks,
        "findings": floor["dominance_violations"],
    }


def render_snapshot(out: str | Path = "outputs", *, version: str | None = None) -> str:
    """The byte-canonical committed form — the single call site the freshness gate and every
    regen path share, so no two writers can disagree on formatting or parameters."""
    return json.dumps(build_snapshot(Path(out), version=version), indent=2, sort_keys=True) + "\n"


def _agg_mean(doc: dict, metric: str) -> float | None:
    entry = doc.get("aggregate", {}).get(metric)
    return entry.get("mean") if isinstance(entry, dict) else None


def _n_classes_present(doc: dict, td: TaskDefinition) -> int | None:
    """Classes with support in the eval set: 2 for a (defined) binary task; for multiclass, the
    per-class F1 columns that are defined (an absent class aggregates to ``mean=None``)."""
    if td.label_kind is LabelKind.BINARY:
        return 2  # a defined balanced_accuracy ⇒ both classes present
    f1_defined = [
        k
        for k, v in doc.get("aggregate", {}).items()
        if k.startswith("f1_") and isinstance(v, dict) and v.get("mean") is not None
    ]
    return len(f1_defined) or None


def check_floor(
    out: Path,
    cells: list[Cell],
    *,
    version: str | None = None,
    tol: float = 0.02,
    test_set: str = "test",
) -> dict:
    """Majority-floor sanity gate (validity plumbing) over finished results.

    Two checks per ``(task, protocol, grid)`` group, on the in-distribution ``test`` set only —
    below-floor scores under transfer (``benchmark``) are *findings*, not plumbing bugs:

    * **floor plausibility** — a constant predictor has recall 1 on its class and 0 elsewhere,
      so the majority baseline's balanced accuracy must sit at the analytic floor ``1/k`` over
      the ``k`` classes present in the eval set. Drift beyond ``tol`` means the labels, class
      vocabulary, or eval alignment are broken.
    * **dominance** — no learned baseline may score below the majority floor on the task's
      headline metric (direction-aware; relative tolerance for regression scales).

    Pending/undefined cells are recorded as ``skipped`` — the gate never fails on an incomplete
    matrix. Iterates coverage ``cells`` (never raw dirs), so '+zero_seq'-style variant outputs
    are out of scope by construction. Read-only: never touches done/pending semantics.
    """
    from .evaluation.metrics import METRIC_DIRECTION  # lazy: keep `--summary` import-light

    groups: dict[tuple[str, str, str], dict[str, Cell]] = {}
    for c in cells:
        groups.setdefault((c.task, c.protocol, c.grid), {})[c.baseline] = c
    plausibility_violations: list[str] = []
    dominance_violations: list[str] = []
    skipped: list[str] = []
    n_checked = 0
    for (task, protocol, grid), by_baseline in sorted(groups.items()):
        key = f"{task}/{protocol}/{grid}"
        td = get_task(task)
        headline = td.metrics[0]
        direction = METRIC_DIRECTION.get(headline, +1)
        maj_cell = by_baseline.get("majority")
        if maj_cell is None:
            skipped.append(f"{key}: no majority cell in the matrix")
            continue
        maj_doc = _load_result(cell_dir(out, maj_cell, version=version) / test_set / "result.json")
        if maj_doc is None:
            skipped.append(f"{key}: majority result not on disk yet")
            continue
        maj = _agg_mean(maj_doc, headline)
        if maj is None:
            skipped.append(f"{key}: majority {headline} undefined (n_defined=0)")
            continue
        is_regression = td.label_kind is LabelKind.REGRESSION
        if headline == "balanced_accuracy":
            k = _n_classes_present(maj_doc, td)
            if k is None:
                skipped.append(f"{key}: cannot infer classes present (no per-class F1)")
            else:
                n_checked += 1
                floor = 1.0 / k
                if abs(maj - floor) > tol:
                    plausibility_violations.append(
                        f"{key}: majority {headline}={maj:.3f} far from the 1/{k}={floor:.3f} "
                        f"floor (labels/vocab/alignment suspect)"
                    )
        # dominance: learned baselines must not sit below the majority floor
        band = tol * abs(maj) if is_regression else tol  # scale-free tol on regression metrics
        for baseline in LEARNED_BASELINES:
            cell = by_baseline.get(baseline)
            if cell is None:
                continue
            doc = _load_result(cell_dir(out, cell, version=version) / test_set / "result.json")
            if doc is None:
                skipped.append(f"{key}/{baseline}: result not on disk yet")
                continue
            val = _agg_mean(doc, headline)
            if val is None:
                skipped.append(f"{key}/{baseline}: {headline} undefined")
                continue
            n_checked += 1
            if direction * (val - maj) < -band:
                side = "above" if direction < 0 else "below"
                dominance_violations.append(
                    f"{key}: {baseline} {headline}={val:.3f} {side} the majority floor "
                    f"{maj:.3f} (tol {band:.3g})"
                )
    violations = plausibility_violations + dominance_violations
    return {
        "ok": not violations,
        "n_groups": len(groups),
        "n_checked": n_checked,
        "violations": violations,
        # structural split (additive): plausibility = plumbing must-hold; dominance = findings
        "plausibility_violations": plausibility_violations,
        "dominance_violations": dominance_violations,
        "skipped": skipped,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Coverage manifest for the experiment matrix.")
    p.add_argument("--out", default="outputs", help="outputs root")
    p.add_argument(
        "--version",
        default=None,
        help="explicit outputs/<version> tree (default: per-protocol current-results resolution)",
    )
    p.add_argument("--tasks", nargs="*", default=None, help="task ids (default: all registered)")
    p.add_argument("--grids", nargs="*", default=None, help="grid keys (default: all registered)")
    p.add_argument("--protocols", nargs="*", default=["held_out"])
    p.add_argument("--baselines", nargs="*", default=list(ALL_BASELINES))
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--summary", action="store_true", help="print done/pending/failed counts")
    mode.add_argument("--emit-pending", action="store_true", help="print runner arg-lines to run")
    mode.add_argument("--emit-failed", action="store_true", help="print arg-lines for failed cells")
    mode.add_argument(
        "--check-floor",
        action="store_true",
        help="majority-floor sanity gate: majority ≈ 1/k floor + learned ≥ floor (test set)",
    )
    mode.add_argument(
        "--check-snapshot",
        action="store_true",
        help="build the snapshot and exit 1 if its embedded self-check fails (hard invariants: "
        "completeness, floor plausibility, value ranges; dominance findings reported only)",
    )
    mode.add_argument(
        "--write-snapshot",
        nargs="?",
        const="coverage_snapshot.json",
        default=None,
        metavar="PATH",
        help="write the persistent coverage manifest (all protocols) and exit",
    )
    mode.add_argument(
        "--failures",
        action="store_true",
        help="list not-done cells that have a failure record: attempts, last stage, error",
    )
    p.add_argument("--floor-tol", type=float, default=0.02, help="tolerance for --check-floor")
    p.add_argument(
        "--max-attempts",
        type=int,
        default=failures.MAX_ATTEMPTS,
        help="retry cap: --emit-pending/--emit-failed skip a cell after this many recorded failed "
        "attempts (0 = no cap; deleting a cell's failure.json also resets it)",
    )
    args = p.parse_args(argv)

    if args.check_snapshot:
        snap = build_snapshot(Path(args.out), version=args.version)
        sc = snap["self_check"]
        for c in sc["checks"]:
            print(f"{'PASS' if c['ok'] else 'FAIL'}  {c['id']}: {c['detail']}")
        for f in sc["findings"]:
            print(f"FINDING  {f}")
        print(f"snapshot self-check: {'OK' if sc['ok'] else 'FAILED'}")
        raise SystemExit(0 if sc["ok"] else 1)

    if args.write_snapshot:
        Path(args.write_snapshot).write_text(render_snapshot(args.out, version=args.version))
        snap = json.loads(Path(args.write_snapshot).read_text())
        c = snap["counts"]
        print(
            f"wrote {args.write_snapshot}: {c['done']} done / {c['pending']} pending / "
            f"{c['failed']} failed of {c['total']} cells across {len(snap['protocols'])} protocols"
        )
        return

    cells = iter_cells(
        tasks=args.tasks,
        grids=args.grids,
        protocols=tuple(args.protocols),
        baselines=tuple(args.baselines),
    )

    if args.check_floor:
        report = check_floor(Path(args.out), cells, version=args.version, tol=args.floor_tol)
        for v in report["violations"]:
            print(f"VIOLATION: {v}")
        status = "OK" if report["ok"] else "FAIL"
        print(
            f"floor gate: {status} — {report['n_checked']} comparisons, "
            f"{len(report['violations'])} violations, {len(report['skipped'])} skipped"
        )
        raise SystemExit(0 if report["ok"] else 1)

    groups = scan(Path(args.out), cells, version=args.version)
    recs = failure_records(
        Path(args.out), groups["pending"] + groups["failed"], version=args.version
    )
    exhausted = exhausted_cells(recs, args.max_attempts)

    if args.emit_pending or args.emit_failed:
        # pending mode re-runs both not-yet-started and failed cells (idempotent skip);
        # failed mode targets only cells that produced no/invalid result. Cells at the retry cap
        # are left out (stdout carries only arg-lines; the note goes to stderr).
        want = groups["failed"] if args.emit_failed else groups["pending"] + groups["failed"]
        skipped = [c for c in want if c in exhausted]
        for c in want:
            if c not in exhausted:
                print(_arg_line(c))
        if skipped:
            print(
                f"coverage: skipped {len(skipped)} cell(s) at the retry cap "
                f"({args.max_attempts} failed attempts) — see `coverage --failures`",
                file=sys.stderr,
            )
        return

    if args.failures:
        for c, doc in sorted(recs.items(), key=lambda kv: _arg_line(kv[0])):
            last = doc["attempts"][-1]
            cap = "  [retry cap reached]" if c in exhausted else ""
            print(f"{_arg_line(c)}{cap}")
            print(f"    attempts={failures.n_attempts(doc)}  last: {failures.reason(last)}")
            if last.get("message"):
                print(f"    message: {last['message'].splitlines()[0][:200]}")
            rec_path = failures.failure_path(cell_dir(Path(args.out), c, version=args.version))
            print(f"    record: {rec_path}")
        print(f"{len(recs)} cell(s) with a failure record, {len(exhausted)} at the retry cap")
        return

    total = len(cells)
    ver = (
        args.version
        or "per-cell ("
        + ", ".join(
            f"{p}=" + "/".join(sorted({cell_results_version(c) for c in cells if c.protocol == p}))
            for p in sorted({c.protocol for c in cells})
        )
        + ")"
    )
    print(f"version={ver}  matrix={total} cells")
    for k in ("done", "pending", "failed"):
        print(f"  {k:8s} {len(groups[k]):4d}")
    # NB: hpc/autonomous_loop.sh reads the count from the line whose first field is the status
    # word — keep the lines below free of it.
    if recs:
        cap_note = (
            f"cells with ≥{args.max_attempts} failed attempts; not resubmitted"
            if args.max_attempts > 0
            else "no retry cap set"
        )
        print(f"  {'retry-cap':8s} {len(exhausted):4d}  ({cap_note})")
        by_reason: dict[str, int] = {}
        for doc in recs.values():
            r = failures.last_reason(doc) or "?"
            by_reason[r] = by_reason.get(r, 0) + 1
        print("failure reasons (last attempt per cell):")
        for r, n in sorted(by_reason.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {n:4d}  {r}")
    if groups["failed"]:
        print("failed cells:")
        for c in groups["failed"]:
            print(f"  {c.task}/{c.protocol}/{c.grid}/{c.baseline}")


if __name__ == "__main__":
    main()
