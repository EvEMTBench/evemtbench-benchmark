"""Fault-location (FL) line-target audit: which stored FL results include bus-target windows, and
what the stored predictions score on line targets only. No retraining, no model is loaded, and
nothing under ``outputs/`` or ``results/`` is written.

Background (``docs/known_issues.md`` section 2): the official FL gate keeps every fault-active
window with a non-null ``y_fault_location``. In the ``adapt_grid`` family the producer also fills a
location for bus-target episodes (``y_fault_line`` = ``MainBus*``), so those windows pass the gate
although they describe no position on a line.

The audit reads label tables (through ``load_task``; features are never read), the committed split
files and the per-seed prediction files, and writes to ``--out``:

* ``counts.csv`` -- gate windows and episodes per task, protocol, grid and split, by target origin
  (``line`` = ``MainLn*``, ``bus`` = ``MainBus*``, ``unrecognised``, ``missing``), with the label
  range per origin and the same counts for the fault-classification task as a cross-check.
* ``controls.csv`` -- per cell and test set: do the split hashes match the run record, does every
  recorded per-seed metric recompute within ``--tol``, do the window counts reconcile with
  ``n_test`` and the stored origin breakdown. A cell that fails a control is listed, never scored.
* ``rescore_per_seed.csv`` -- per cell, test set, seed and subset (``all``, ``line``, ``bus``):
  n and the task's official regression metrics.
* ``original_vs_line_only.csv`` -- the seed aggregate (mean, std, ci95, as
  ``aggregate_seed_metrics``) per subset, with the paper's validation-selected baseline marked.
  Model choices are the original ones; nothing is selected on a subset.
* ``cell_provenance.csv`` -- per cell: results tree, version, git commit and date of the run,
  seeds found, prediction files found.
* ``inputs.sha256`` and ``summary.json``.

    PYTHONPATH=src python hpc/audit_fl_line_targets.py --out <dir>
    PYTHONPATH=src python hpc/audit_fl_line_targets.py --out <dir> --tree control=<path>/outputs

Exit status is non-zero if any cell fails a control.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "paper_artifacts"))

import numpy as np
import pandas as pd

from evemtbench.coverage import CONVENTIONAL_BASELINES, Cell, cell_dir, iter_cells
from evemtbench.evaluation.aggregate import aggregate_seed_metrics
from evemtbench.evaluation.metrics import compute_regression_metrics
from evemtbench.evaluation.predictions_io import align_to_view
from evemtbench.evaluation.reevaluate import (
    ALL_PROTOCOLS,
    _load_json,
    _provenance_matches,
    _seed_metrics_agree,
    _work_units,
)
from evemtbench.evaluation.stratify import _with_origin
from evemtbench.tasks.line_view import LINE_LABEL_PREFIX
from evemtbench.tasks.task import load_task

FL_TASKS = ("fault_location_local", "fault_location_line", "fault_location_global")
FC_OF = {t: t.replace("fault_location", "fault_classification") for t in FL_TASKS}
SPLITS = ("train", "val", "test", "benchmark")
SUBSETS = ("all", "line", "bus")
BUS_LABEL_PREFIX = "MainBus"
DETERMINISTIC = ("majority", *CONVENTIONAL_BASELINES)


def target_origin(fault_line: pd.Series) -> np.ndarray:
    """``line`` / ``bus`` / ``unrecognised`` / ``missing`` per window, by label prefix."""
    s = fault_line.astype("string")
    out = np.full(len(s), "unrecognised", dtype=object)
    out[s.str.startswith(LINE_LABEL_PREFIX).fillna(False).to_numpy()] = "line"
    out[s.str.startswith(BUS_LABEL_PREFIX).fillna(False).to_numpy()] = "bus"
    out[s.isna().to_numpy()] = "missing"
    return out


def _view(task, split: str):
    """A task view, or ``None`` when the protocol does not define that split."""
    try:
        v = task.view(split)
    except Exception:
        return None
    if v is None or v.metadata is None or len(v.metadata) == 0:
        return None
    return v


def count_rows(task_id: str, protocol: str, grid: str, task, fc_task) -> list[dict]:
    rows = []
    for split in SPLITS:
        v = _view(task, split)
        if v is None:
            continue
        md = v.metadata
        origin = target_origin(md["y_fault_line"])
        # the official ``origin`` stratum (substring rule) and the prefix rule used here must
        # name the same windows, or the stored breakdowns and this audit talk about different sets
        official = _with_origin(md)["origin"].to_numpy()
        fc_n = None
        if fc_task is not None:
            fv = _view(fc_task, split)
            fc_n = None if fv is None else len(fv.metadata)
        loc = pd.to_numeric(pd.Series(np.asarray(v.labels, dtype=float)), errors="coerce")
        for o in ("line", "bus", "unrecognised", "missing"):
            m = origin == o
            sub = md.loc[m]
            per_episode = (
                pd.DataFrame({"sid": sub["sample_id"].to_numpy(), "loc": loc[m].to_numpy()})
                .groupby("sid")["loc"]
                .nunique()
            )
            rows.append(
                {
                    "task": task_id,
                    "protocol": protocol,
                    "grid": grid,
                    "split": split,
                    "origin": o,
                    "windows": int(m.sum()),
                    "episodes": int(sub["sample_id"].nunique()),
                    "gate_windows_total": len(md),
                    "share_of_gate": round(float(m.mean()), 6) if len(md) else "",
                    "null_location_in_gate": int(loc[m].isna().sum()),
                    "location_min": _num(loc[m].min()),
                    "location_max": _num(loc[m].max()),
                    "location_mean": _num(loc[m].mean()),
                    "max_distinct_locations_per_episode": _num(per_episode.max()),
                    "fc_gate_windows_total": "" if fc_n is None else fc_n,
                    "official_origin_agrees": (
                        bool(((official == o) == m).all()) if o in ("line", "bus") else ""
                    ),
                }
            )
    return rows


def _num(x) -> float | str:
    return "" if x is None or pd.isna(x) else float(x)


def _selected_baselines(outputs: Path) -> dict[tuple, str]:
    """``{(protocol, task, grid, source, test_set): baseline}``: the paper's validation-selected
    learned baseline per FL cell, from the same kernel the tables use. Only for the main tree."""
    import paper_lib as pl

    from evemtbench.coverage import PRETRAIN_GRIDS
    from make_results import load_rows

    sel: dict[tuple, str] = {}
    grids = [g for g, _ in pl.GRID_ROWS]
    for grid in grids:
        rows = load_rows(grid)
        for task_id in FL_TASKS:
            for ts in pl.TEST_SETS:
                c = pl.best_cell_choice(rows, task_id, ts, "mae")
                if c is not None:
                    sel[("held_out", task_id, grid, None, ts)] = c["baseline"]
    idx, val_idx = pl._transfer_index(str(outputs)), pl._transfer_val_index(str(outputs))
    for task_id in FL_TASKS:
        for grid in grids:
            for source in PRETRAIN_GRIDS:
                for protocol, _label in pl._TRANSFER_PROTOS:
                    r = pl.best_transfer_row(idx, task_id, grid, protocol, "mae", source, val_idx)
                    if r is not None:
                        sel[(protocol, task_id, grid, source, "benchmark")] = r.baseline
    return sel


def audit_cell(tree: str, out: Path, cell: Cell, task, tol: float, version: str | None):
    """Controls, per-seed re-scores and provenance for one cell. Returns three row lists."""
    td = task.definition
    cdir = cell_dir(out, cell, version=version)
    ident = {
        "tree": tree,
        "task": cell.task,
        "protocol": cell.protocol,
        "grid": cell.grid,
        "source": cell.source or "",
        "baseline": cell.baseline,
    }
    record = _load_json(cdir / "run_record.json")
    prov = {
        **ident,
        "cell_dir": str(cdir),
        "exists": cdir.is_dir(),
        "benchmark_version": (record or {}).get("benchmark_version", ""),
        "splits_version": (record or {}).get("splits_version", ""),
        "git_sha": (record or {}).get("git_sha", ""),
        "run_utc": (
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record["timestamp"]))
            if record and record.get("timestamp")
            else ""
        ),
        "seeds": " ".join(str(s) for s in (record or {}).get("seeds") or []),
        "unresolved": "",
    }
    controls, rescored = [], []
    if not cdir.is_dir():
        prov["unresolved"] = "cell directory missing"
        return controls, rescored, [prov]
    if record is None:
        prov["unresolved"] = "run_record.json missing or unreadable"
    split_problem = _provenance_matches(task.provenance, (record or {}).get("data"))
    for ts_dir in sorted(p for p in cdir.iterdir() if p.is_dir()):
        ts = ts_dir.name
        if ts in ("checkpoints", "val") or ts.startswith("pretrain-"):
            continue
        old = _load_json(ts_dir / "result.json")
        ctl = {**ident, "test_set": ts, "status": "ok", "detail": ""}
        if old is None:
            controls.append({**ctl, "status": "FAILED", "detail": "result.json missing"})
            continue
        v = _view(task, ts)
        if v is None:
            controls.append({**ctl, "status": "FAILED", "detail": f"no '{ts}' view for this task"})
            continue
        seeds = old.get("seeds") or []
        origin = target_origin(v.metadata["y_fault_line"])
        y_true = np.asarray(v.labels, dtype=float)
        recorded = {int(d["seed"]): d.get("metrics", {}) for d in old.get("per_seed") or []}
        stored_origin = (old.get("breakdowns") or {}).get("origin") or {}
        ctl.update(
            n_test_recorded=old.get("n_test"),
            n_gate=len(y_true),
            n_line=int((origin == "line").sum()),
            n_bus=int((origin == "bus").sum()),
            n_other=int(((origin != "line") & (origin != "bus")).sum()),
            stored_origin_n=json.dumps({k: g.get("n") for k, g in stored_origin.items()}),
            split_hashes=split_problem or "match",
            seeds_recorded=len(seeds),
            deterministic=cell.baseline in DETERMINISTIC,
        )
        problems = []
        if split_problem:
            problems.append(split_problem)
        if old.get("n_test") != len(y_true):
            problems.append(f"n_test {old.get('n_test')} != gate {len(y_true)}")
        for name, key in (("line", "n_line"), ("bus", "n_bus")):
            sn = (stored_origin.get(name) or {}).get("n")
            if stored_origin and (sn or 0) != ctl[key]:
                problems.append(f"stored origin.{name}.n {sn} != {ctl[key]}")
        per_subset: dict[str, list[dict]] = {s: [] for s in SUBSETS}
        missing_seeds = []
        for s in seeds:
            pp = ts_dir / f"predictions_seed{s}.parquet"
            if not pp.is_file():
                missing_seeds.append(s)
                continue
            y_pred = np.asarray(align_to_view(v.metadata, pd.read_parquet(pp)), dtype=float)
            full = compute_regression_metrics(y_true, y_pred, td.metrics)
            diff = _seed_metrics_agree(full, recorded.get(int(s), {}), tol)
            if diff is not None:
                problems.append(f"seed {s} does not reproduce: {json.dumps(diff)[:160]}")
                continue
            for subset in SUBSETS:
                m = np.ones(len(y_true), bool) if subset == "all" else origin == subset
                metrics = compute_regression_metrics(y_true[m], y_pred[m], td.metrics)
                per_subset[subset].append(metrics)
                rescored.append(
                    {**ident, "test_set": ts, "seed": s, "subset": subset, "n": int(m.sum()),
                     **metrics}
                )  # fmt: skip
        if missing_seeds:
            problems.append(f"missing prediction files for seeds {missing_seeds}")
        ctl["seeds_rescored"] = len(per_subset["all"])
        # second positive control: the stored seed-aggregated line breakdown must reproduce
        stored_line = ((stored_origin.get("line") or {}).get("metrics") or {}).get("mae") or {}
        if per_subset["line"] and isinstance(stored_line, dict) and "mean" in stored_line:
            new = aggregate_seed_metrics(per_subset["line"])["mae"]["mean"]
            if new is None or abs(new - stored_line["mean"]) > max(tol, 1e-9):
                problems.append(f"stored origin.line mae {stored_line['mean']} != recomputed {new}")
        if problems:
            ctl.update(status="FAILED", detail=" | ".join(problems))
            rescored = [r for r in rescored if r["test_set"] != ts]
        controls.append(ctl)
    return controls, rescored, [prov]


def aggregate_rows(rescored: list[dict], selected: dict[tuple, str]) -> list[dict]:
    df = pd.DataFrame(rescored)
    if df.empty:
        return []
    keys = ["tree", "task", "protocol", "grid", "source", "baseline", "test_set", "subset"]
    metric_cols = [c for c in df.columns if c not in (*keys, "seed", "n")]
    out = []
    for key, g in df.groupby(keys, sort=True, dropna=False):
        row = dict(zip(keys, key, strict=True))
        agg = aggregate_seed_metrics(g[metric_cols].to_dict("records"))
        row.update(n=int(g["n"].iloc[0]), n_seeds=len(g))
        for m in metric_cols:
            for stat in ("mean", "std", "ci95"):
                row[f"{m}_{stat}"] = _num(agg[m].get(stat))
        sel_key = (
            row["protocol"],
            row["task"],
            row["grid"],
            row["source"] or None,
            row["test_set"],
        )
        row["paper_selected"] = (
            row["tree"] == "main" and selected.get(sel_key) == row["baseline"]
            # held_out cells have no source; transfer cells are keyed by theirs
        )
        out.append(row)
    return out


def _write(path: Path, rows: list[dict]) -> None:
    cols: list[str] = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--outputs", default="outputs", help="the main results tree")
    ap.add_argument(
        "--tree",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="an additional results tree with the same layout (e.g. a control run); repeatable",
    )
    ap.add_argument("--tree-version", default="1.0.0", help="version dir inside --tree trees")
    ap.add_argument("--out", required=True, help="directory for the audit files (created)")
    ap.add_argument("--tol", type=float, default=0.0, help="per-seed agreement tolerance")
    ap.add_argument("--protocols", nargs="*", default=list(ALL_PROTOCOLS))
    ap.add_argument("--grids", nargs="*", default=None)
    ap.add_argument("--root", default=None, help="data root override")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    trees: list[tuple[str, Path, str | None]] = [("main", Path(args.outputs), None)]
    for spec in args.tree:
        name, _, path = spec.partition("=")
        trees.append((name, Path(path), args.tree_version))

    cells = iter_cells(tasks=list(FL_TASKS), grids=args.grids, protocols=tuple(args.protocols))
    counts, controls, rescored, provenance = [], [], [], []
    for (task_id, protocol, grid), unit_cells in _work_units(cells):
        t0 = time.time()
        try:
            task = load_task(task_id, protocol, grid=grid, root=args.root)
        except Exception as exc:
            controls.append(
                {"tree": "main", "task": task_id, "protocol": protocol, "grid": grid,
                 "status": "FAILED", "detail": f"load_task failed: {exc}"}
            )  # fmt: skip
            continue
        try:
            fc_task = load_task(FC_OF[task_id], protocol, grid=grid, root=args.root)
        except Exception:
            fc_task = None
        counts += count_rows(task_id, protocol, grid, task, fc_task)
        for tree, path, version in trees:
            for cell in unit_cells:
                if tree != "main" and not cell_dir(path, cell, version=version).is_dir():
                    continue  # an extra tree holds only the cells its experiment ran
                c, r, p = audit_cell(tree, path, cell, task, args.tol, version)
                controls += c
                rescored += r
                provenance += p
        print(f"{task_id}/{protocol}/{grid}: {len(unit_cells)} cells, {time.time() - t0:.0f}s",
              file=sys.stderr, flush=True)  # fmt: skip

    try:
        selected = _selected_baselines(Path(args.outputs))
    except Exception as exc:  # selection marks are a convenience; the re-scores stand without them
        print(f"[warn] could not resolve the paper's selected baselines: {exc}", file=sys.stderr)
        selected = {}
    aggregated = aggregate_rows(rescored, selected)

    _write(out_dir / "counts.csv", counts)
    _write(out_dir / "controls.csv", controls)
    _write(out_dir / "rescore_per_seed.csv", rescored)
    _write(out_dir / "original_vs_line_only.csv", aggregated)
    _write(out_dir / "cell_provenance.csv", provenance)

    repo = Path(__file__).resolve().parents[1]
    hashed = sorted({f for t in ("held_out", "multi_grid_pretrain") for f in
                     (repo / "src/evemtbench/splits").glob(f"v1.*/{t}/*/*.txt")})  # fmt: skip
    (out_dir / "inputs.sha256").write_text(
        "".join(f"{_sha256(f)}  {f.relative_to(repo)}\n" for f in hashed)
    )
    failed = [c for c in controls if c.get("status") != "ok"]
    git = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    summary = {
        "audit_commit": git,
        "tol": args.tol,
        "trees": {name: str(path) for name, path, _ in trees},
        "cells_with_provenance": len(provenance),
        "cell_test_sets_checked": len(controls),
        "cell_test_sets_failed": len(failed),
        "rescored_rows": len(rescored),
        "versions": {m: sys.modules[m].__version__ for m in ("numpy", "pandas")},
        "python": sys.version.split()[0],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    for c in failed[:20]:
        print(f"  FAILED {c.get('tree')}/{c.get('task')}/{c.get('protocol')}/{c.get('grid')}/"
              f"{c.get('baseline', '')}/{c.get('test_set', '')}: {c.get('detail')}")  # fmt: skip
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
