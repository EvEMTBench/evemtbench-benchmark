"""Per-cell model-selection audit for the paper's headline and transfer cells. Generates NO artifact
unless ``--out`` is given.

For every held_out primary cell (function x view x grid x reported set) and every cross-grid
transfer cell (function x view x target grid x source x protocol) it prints which baseline the
paper reports, the validation score it was selected on, and every candidate's validation and
evaluation score, as tab-separated rows. A populated cell without usable validation evidence is
listed as ``BLOCKED`` with the reason (``paper_lib.SelectionEvidenceError``) and makes the script
exit non-zero, so it can gate a release next to ``check_supplement_consistency.py``.

    PYTHONPATH=src python hpc/audit_selection.py                 # TSV on stdout
    PYTHONPATH=src python hpc/audit_selection.py --out audit.tsv

``eval_argbest`` is informational only: the baseline that would have won on the reported set. It
never takes part in selection; where it differs from ``selected`` the cell shows what selecting on
the reported set would have changed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "paper_artifacts"))

import paper_lib as pl
import supplement_lib as sl

from check_supplement_consistency import GRIDS, PRIMARY
from evemtbench.coverage import LEARNED_BASELINES, PRETRAIN_GRIDS
from evemtbench.evaluation.metrics import METRIC_DIRECTION
from make_results import load_rows, mean_val

COLUMNS = (
    "protocol",
    "task",
    "grid",
    "source",
    "reported_set",
    "metric",
    "status",
    "selected",
    "val_score",
    "reported",
    "eval_argbest",
    "val_scores",
    "eval_scores",
    "note",
)


def _fmt(scores: dict[str, float | None]) -> str:
    return " ".join(f"{b}={'-' if v is None else format(v, '.6g')}" for b, v in scores.items())


def _argbest(scores: dict[str, float | None], metric: str) -> str:
    direction = METRIC_DIRECTION.get(metric, 1)
    best = None
    for b, v in scores.items():
        if v is not None and (best is None or direction * (v - scores[best]) > 0):
            best = b
    return best or ""


def _row(cell: dict, val: dict, ev: dict, select) -> dict:
    """One audit row. ``select()`` runs the paper's own selection and returns (baseline, value)."""
    row = {**cell, "source": cell["source"] or "", "val_scores": _fmt(val), "eval_scores": _fmt(ev)}
    row["eval_argbest"] = _argbest(ev, cell["metric"])
    row.update(selected="", val_score="", reported="", note="")
    try:
        chosen = select()
    except pl.SelectionEvidenceError as e:
        row.update(status="BLOCKED", note=str(e))
        return row
    if chosen is None:
        row["status"] = "pending" if any(v is not None for v in ev.values()) else "empty"
        return row
    baseline, value = chosen
    row.update(
        status="selected",
        selected=baseline,
        val_score=format(val[baseline], ".6g"),
        reported=format(value, ".6g"),
    )
    return row


def audit(outputs: str = "outputs") -> list[dict]:
    rows = []
    per_grid = {g: load_rows(g) for g in GRIDS}
    for stem, (metric, views) in PRIMARY.items():
        for view in views:
            task = f"{stem}_{view}"
            for grid in GRIDS:
                docs = per_grid[grid]
                val = {
                    b: mean_val(docs.get((task, b), {}).get("val"), metric)
                    for b in LEARNED_BASELINES
                }
                for ts in pl.TEST_SETS:
                    ev = {
                        b: mean_val(docs.get((task, b), {}).get(ts), metric)
                        for b in LEARNED_BASELINES
                    }
                    cell = dict(
                        protocol="held_out",
                        task=task,
                        grid=grid,
                        source=None,
                        reported_set=ts,
                        metric=metric,
                    )

                    def select(docs=docs, task=task, ts=ts, metric=metric, grid=grid):
                        c = pl.best_cell_choice(docs, task, ts, metric, grid=grid)
                        return None if c is None else (c["baseline"], c["mean"])

                    rows.append(_row(cell, val, ev, select))
    idx = pl._transfer_index(outputs)
    val_idx = pl._transfer_val_index(outputs)
    for _key, _lab, stem, metric, _arrow in sl.TRANSFER_FUNCTIONS:
        for view in sl.TRANSFER_VIEWS:
            task = f"{stem}_{view}"
            for grid in GRIDS:
                for source in PRETRAIN_GRIDS:
                    for protocol, _label in pl._TRANSFER_PROTOS:
                        bench = {r.baseline: r for r in idx.get((task, grid, protocol, source), [])}
                        ev = {
                            b: bench[b].mean(metric) if b in bench else None
                            for b in LEARNED_BASELINES
                        }
                        if protocol == "transfer_zeroshot":
                            keys = {
                                b: ("multi_grid_pretrain", source, task, b)
                                for b in LEARNED_BASELINES
                            }
                        else:
                            keys = {
                                b: ("transfer_finetune", grid, task, b, source)
                                for b in LEARNED_BASELINES
                            }
                        val = {b: mean_val(val_idx.get(k), metric) for b, k in keys.items()}
                        cell = dict(
                            protocol=protocol,
                            task=task,
                            grid=grid,
                            source=source,
                            reported_set="benchmark",
                            metric=metric,
                        )

                        def select(
                            task=task, grid=grid, protocol=protocol, metric=metric, source=source
                        ):
                            r = pl.best_transfer_row(
                                idx, task, grid, protocol, metric, source, val_idx
                            )
                            return None if r is None else (r.baseline, r.mean(metric))

                        rows.append(_row(cell, val, ev, select))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", help="write the TSV here instead of stdout")
    args = ap.parse_args()
    rows = audit()
    lines = ["\t".join(COLUMNS)] + ["\t".join(str(r[c]) for c in COLUMNS) for r in rows]
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    else:
        sys.stdout.write(text)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    differ = sum(
        1 for r in rows if r["status"] == "selected" and r["selected"] != r["eval_argbest"]
    )
    print(
        f"{len(rows)} cells: {counts}; validation winner differs from the reported-set argbest "
        f"in {differ}",
        file=sys.stderr,
    )
    return 1 if counts.get("BLOCKED") else 0


if __name__ == "__main__":
    sys.exit(main())
