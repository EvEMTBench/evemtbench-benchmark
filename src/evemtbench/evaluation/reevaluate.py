"""Rebuild every cell's ``result.json`` from the stored per-seed predictions. No retraining.

Why this can exist at all: ``evaluate()`` needs only labels + metadata, never features (it reads
``task.view(ts).labels`` / ``.metadata`` and aligns predictions by key), and the sweep persisted
``predictions_seed{n}.parquet`` for every seed of every cell. So a metric or breakdown change is
recomputable offline for the price of reading label tables — no GPU, no projection cache, no raw
window bytes.

Why it is needed: ``breakdowns`` used to be the LAST seed's values next to a multi-seed
``aggregate`` (see ``evaluation/aggregate.py``), and ``macro_f1_supported`` did not exist. Both are
functions of the predictions, so both can be fixed without touching a model.

What makes it trustworthy is the controls, not the arithmetic. Before anything is written, each cell
must satisfy:

* **provenance equality** — the recomputed ``task.provenance`` (split-file sha256s + per-partition
  ``manifest_sha256`` / ``feature_hash``) must equal what the cell's ``run_record.json`` recorded,
  so we know the labels are the same labels;
* **environment equality** — the record's ``versions`` must match the running interpreter, since the
  metrics are only reproducible under the library versions that produced them;
* **exact per-seed agreement** — every metric already stored in ``per_seed`` must recompute bit-for-
  bit (``--tol 0``). This is the real test: it re-derives 5 × |metrics| numbers per cell from the
  predictions and compares them to what the sweep wrote.

A cell failing any control is skipped **unwritten** and reported. Nothing is ever partially written:
the document is rendered in full and compared before replacing the file.

    evemtbench-reevaluate --check                # compute + diff, write nothing (exit 1 on drift)
    evemtbench-reevaluate --shard 3 --num-shards 228

Carried over verbatim (unrecoverable without retraining): ``compute``, ``n_params``, ``seeds``,
``baseline``. Never touched: ``run_record.json``, the predictions, checkpoints, and ``*/val/*``
(rescored by ``hpc/rescore_validation.py``, which stores no predictions).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..coverage import Cell, cell_dir, iter_cells
from ..splits.protocols import PRETRAIN_TRANSFER_PROTOCOLS, TRANSFER_PROTOCOLS
from ..tasks.task import load_task
from ..version import results_version
from .aggregate import (
    aggregate_seed_breakdowns,
    aggregate_seed_metrics,
    build_result_doc,
    render_result_json,
)
from .evaluate import evaluate

# Only these move a metric value; the rest of the environment does not enter a score.
_SCORING_PKGS = ("numpy", "pandas", "scikit-learn", "scipy")

# Every protocol the matrix defines, derived rather than restated so a new protocol is covered
# automatically (matches the ordering hpc/make_audit.py uses).
ALL_PROTOCOLS = ("held_out", *PRETRAIN_TRANSFER_PROTOCOLS)


@dataclass
class Report:
    """Outcome of a pass.

    ``skipped`` and ``excluded`` are deliberately different things. A *skip* means a control fired —
    the cell could have been rebuilt but something about it did not check out, so it must fail the
    run loudly. An *exclusion* is structural and expected (``val`` has no stored predictions), so it
    is reported but must not make the run non-zero; otherwise the idempotency gate could never go
    green and would stop meaning anything.
    """

    written: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    changed_in_check: list[str] = field(default_factory=list)

    def skip(self, cell: str, reason: str, detail: object = None) -> None:
        self.skipped.append({"cell": cell, "reason": reason, "detail": detail})

    def exclude(self, cell: str) -> None:
        self.excluded.append(cell)

    def as_dict(self) -> dict:
        return {
            "n_written": len(self.written),
            "n_unchanged": len(self.unchanged),
            "n_skipped": len(self.skipped),
            "n_excluded": len(self.excluded),
            "n_changed_in_check": len(self.changed_in_check),
            "written": self.written,
            "skipped": self.skipped,
            "excluded": self.excluded,
            "changed_in_check": self.changed_in_check,
        }


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _dist_versions() -> dict[str, str | None]:
    import importlib.metadata

    out: dict[str, str | None] = {}
    for d in _SCORING_PKGS:
        try:
            out[d] = importlib.metadata.version(d)
        except importlib.metadata.PackageNotFoundError:
            out[d] = None
    return out


def _provenance_matches(recomputed: dict, recorded: dict | None) -> str | None:
    """``None`` when the inputs provably match, else a human-readable difference.

    Compares only content-addressed fields. ``x_path`` is excluded on purpose: it differs between
    the vault master and a ``$TMPDIR`` staged copy of identical bytes.
    """
    if not recorded:
        return "run_record has no data provenance block (pre-hardening record)"
    rec_splits = (recorded.get("splits") or {}).get("files") or {}
    new_splits = (recomputed.get("splits") or {}).get("files") or {}
    if rec_splits and new_splits and rec_splits != new_splits:
        differing = sorted(
            k for k in set(rec_splits) | set(new_splits) if rec_splits.get(k) != new_splits.get(k)
        )
        return f"split-file hashes differ: {differing}"

    def _key(parts):
        return sorted(
            (p.get("grid"), p.get("family"), p.get("manifest_sha256"), p.get("feature_hash"))
            for p in parts or []
        )

    rec_parts, new_parts = _key(recorded.get("partitions")), _key(recomputed.get("partitions"))
    if rec_parts and new_parts and rec_parts != new_parts:
        return "window-manifest/feature hashes differ from the recorded partitions"
    return None


def _seed_metrics_agree(recomputed: dict[str, float], recorded: dict[str, float], tol: float):
    """Difference report for one seed, or ``None`` when every recorded metric reproduces.

    Only metrics the original recorded are compared — a metric added since (``macro_f1_supported``)
    has nothing to disagree with. NaN must still match NaN: a metric that was undefined then and
    defined now is a real change, not a tolerance question.
    """
    import math

    bad = {}
    for k, old in recorded.items():
        if k not in recomputed:
            bad[k] = {"recorded": old, "recomputed": "<absent>"}
            continue
        new = recomputed[k]
        old_f = float("nan") if old is None else float(old)
        new_f = float("nan") if new is None else float(new)
        if math.isnan(old_f) and math.isnan(new_f):
            continue
        if math.isnan(old_f) != math.isnan(new_f) or abs(old_f - new_f) > tol:
            bad[k] = {"recorded": old_f, "recomputed": new_f}
    return bad or None


def reevaluate_unit(
    task_id: str,
    protocol: str,
    grid: str,
    cells: list[Cell],
    *,
    out: Path,
    version: str | None,
    report: Report,
    tol: float = 0.0,
    check_only: bool = False,
    strict_versions: bool = True,
    root=None,
    splits_root=None,
) -> None:
    """Rebuild every result.json of one (task, protocol, grid) work unit.

    One ``load_task`` is amortised over the unit's cells and test sets — that call is the expensive
    part (label tables + split hashing), and it is identical for every baseline and source.
    """
    try:
        task = load_task(task_id, protocol, grid=grid, root=root, splits_root=splits_root)
    except Exception as exc:  # a grid/protocol combination with no data on this machine
        for c in cells:
            report.skip(f"{task_id}/{protocol}/{grid}/{c.baseline}", "load_task failed", str(exc))
        return

    env = _dist_versions()

    for cell in cells:
        # None = each protocol's own results tree (held_out stays in 1.0.0), as coverage resolves it
        ver = version or results_version(protocol)
        cdir = cell_dir(out, cell, version=ver)
        root = out / ver
        label = str(cdir.relative_to(root)) if cdir.is_relative_to(root) else str(cdir)
        record = _load_json(cdir / "run_record.json")
        if record is None:
            report.skip(label, "run_record.json missing/unreadable")
            continue
        if strict_versions:
            rec_env = {k: (record.get("versions") or {}).get(k) for k in _SCORING_PKGS}
            if rec_env != env:
                report.skip(label, "environment differs", {"recorded": rec_env, "running": env})
                continue
        prov_problem = _provenance_matches(task.provenance, record.get("data"))
        if prov_problem:
            report.skip(label, "input provenance mismatch", prov_problem)
            continue

        for ts_dir in sorted(p for p in cdir.iterdir() if p.is_dir()) if cdir.is_dir() else []:
            ts = ts_dir.name
            if ts in ("checkpoints",) or ts.startswith("pretrain-"):
                continue  # not a test set (nested source dirs are separate cells)
            old_path = ts_dir / "result.json"
            old = _load_json(old_path)
            if old is None:
                continue
            if ts == "val":
                # Structural exclusion, not a control failure: hpc/rescore_validation.py wrote these
                # and stores no predictions, so there is nothing here to rebuild from.
                report.exclude(f"{label}/{ts}")
                continue

            seeds = old.get("seeds") or []
            per_seed_old = {int(d["seed"]): d.get("metrics", {}) for d in old.get("per_seed") or []}
            pred_paths = {s: ts_dir / f"predictions_seed{s}.parquet" for s in seeds}
            absent = [s for s, p in pred_paths.items() if not p.is_file()]
            if absent:
                report.skip(f"{label}/{ts}", "missing stored predictions", {"seeds": absent})
                continue

            if not seeds:
                report.skip(f"{label}/{ts}", "result.json lists no seeds")
                continue
            per_seed_metrics, per_seed_results, per_seed_breakdowns = [], [], []
            failed: tuple[str, object] | None = None
            n_test = 0
            for s in seeds:
                try:
                    res = evaluate(task, pd.read_parquet(pred_paths[s]), test_set=ts)
                except Exception as exc:
                    failed = ("evaluate failed", f"seed {s}: {exc}")
                    break
                diff = _seed_metrics_agree(res.metrics, per_seed_old.get(s, {}), tol)
                if diff is not None:
                    failed = ("per-seed metrics do not reproduce", {"seed": s, "deltas": diff})
                    break
                per_seed_metrics.append(res.metrics)
                per_seed_results.append({"seed": s, "metrics": res.metrics})
                per_seed_breakdowns.append(res.breakdowns)
                n_test = res.n_samples
            if failed is not None:
                report.skip(f"{label}/{ts}", failed[0], failed[1])
                continue

            doc = build_result_doc(
                benchmark_version=ver,
                task=task.definition.task_id,
                protocol=task.protocol,
                grid=task.grid,
                # carried: not derivable without the fitted model / the original invocation
                baseline=old.get("baseline") or cell.baseline,
                # derived from the directory, not carried: ~half the corpus predates the key
                pretrain_grid=(cell.source if protocol in TRANSFER_PROTOCOLS else None),
                test_set=ts,
                n_test=n_test,
                seeds=seeds,
                n_params=old.get("n_params"),
                compute=old.get("compute") or {},
                per_seed=per_seed_results,
                aggregate=aggregate_seed_metrics(per_seed_metrics),
                breakdowns=aggregate_seed_breakdowns(per_seed_breakdowns),
            )
            rendered = render_result_json(doc)
            if rendered == old_path.read_text():
                report.unchanged.append(f"{label}/{ts}")
            elif check_only:
                report.changed_in_check.append(f"{label}/{ts}")
            else:
                tmp = old_path.with_suffix(".json.tmp")
                tmp.write_text(rendered)
                tmp.replace(old_path)
                report.written.append(f"{label}/{ts}")


def _work_units(cells: list[Cell]) -> list[tuple[tuple[str, str, str], list[Cell]]]:
    """Group cells by (task, protocol, grid) so one ``load_task`` serves all of them."""
    units: dict[tuple[str, str, str], list[Cell]] = {}
    for c in cells:
        units.setdefault((c.task, c.protocol, c.grid), []).append(c)
    return sorted(units.items())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Rebuild result.json from stored per-seed predictions (no retraining)."
    )
    ap.add_argument("--out", default="outputs")
    ap.add_argument(
        "--version",
        default=None,
        help="results tree to rebuild (default: each protocol's own, version.results_version)",
    )
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--protocols", nargs="*", default=list(ALL_PROTOCOLS))
    ap.add_argument("--grids", nargs="*", default=None)
    ap.add_argument("--root", default=None, help="data root override (tests/fixtures)")
    ap.add_argument("--splits-root", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument(
        "--check",
        action="store_true",
        help="compute and diff without writing; exit 1 if any document would change",
    )
    ap.add_argument(
        "--tol", type=float, default=0.0, help="per-seed agreement tolerance (default 0)"
    )
    ap.add_argument(
        "--no-strict-versions",
        dest="strict_versions",
        action="store_false",
        help="do not require the running library versions to match each run record",
    )
    ap.add_argument("--report", default=None, help="write the full JSON report here")
    args = ap.parse_args(argv)

    cells = iter_cells(tasks=args.tasks, grids=args.grids, protocols=tuple(args.protocols))
    units = _work_units(cells)
    if args.num_shards > 1:
        units = [u for i, u in enumerate(units) if i % args.num_shards == args.shard]

    out = Path(args.out)
    report = Report()
    for (task_id, protocol, grid), unit_cells in units:
        reevaluate_unit(
            task_id,
            protocol,
            grid,
            unit_cells,
            out=out,
            version=args.version,
            report=report,
            tol=args.tol,
            check_only=args.check,
            strict_versions=args.strict_versions,
            root=args.root,
            splits_root=args.splits_root,
        )

    d = report.as_dict()
    mode = "check" if args.check else "write"
    print(
        f"reevaluate[{mode}] shard {args.shard}/{args.num_shards}: "
        f"{d['n_written']} written, {d['n_unchanged']} unchanged, "
        f"{d['n_changed_in_check']} would change, {d['n_skipped']} skipped "
        f"(+{d['n_excluded']} excluded: val)"
    )
    for s in d["skipped"][:20]:
        print(f"  SKIP {s['cell']}: {s['reason']} {json.dumps(s['detail'], default=str)[:200]}")
    if len(d["skipped"]) > 20:
        print(f"  ... and {len(d['skipped']) - 20} more skips")
    if args.report:
        Path(args.report).write_text(json.dumps(d, indent=2, sort_keys=True, default=str))
        print(f"wrote {args.report}")

    # A skip is a control firing — never silent. In --check mode a would-change document is also a
    # failure, which is what makes the idempotency gate meaningful.
    if d["n_skipped"] or (args.check and d["n_changed_in_check"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
