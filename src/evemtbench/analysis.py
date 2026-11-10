"""Protection-grade analysis on saved predictions: detection latency + difficulty curves.

A protection benchmark is judged on dependability, security, AND **speed**, and on how
difficulty varies with the swept fault parameters. This module derives, post-hoc, from the
per-cell ``predictions_seed{n}.parquet`` + the windows ``y`` (joined on sample_id/window_idx,
never loading ``X`` — memory-safe on the big grids):

* ``detection_latency`` — for the fault/event detection tasks (FD, ED), the time from fault
  inception (``dt_start`` = 0) to the first correctly-flagged active window, per fault episode;
  reports detection rate + median/mean latency over detected episodes. (Other binary tasks label
  something other than "detected" as 1, so a latency would be meaningless there.)
* ``location_mae_curve`` — for fault-location regression, MAE binned by distance-along-line
  (the benchmark systematically sweeps location), exposing where localization degrades.

Resistance-stratified curves need the per-episode fault resistance, which is not in the windows
``y`` (it lives in the simulation ``settings.csv``); that is a dataset-side dependency, noted but
not computed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .coverage import Cell, cell_dir
from .data.windows import load_windows
from .tasks.registry import get_task
from .tasks.taxonomy import Function, LabelKind

_FAMILY = {"test": "adapt_grid", "benchmark": "benchmark"}
_TRUTH_COLS = [
    "sample_id",
    "window_idx",
    "dt_start",
    "y_fault_present",
    "y_fault_line",
    "y_fault_location",
]


# Tasks whose positive class means "the fault/event was detected", so a time-to-first-positive is a
# detection latency. Other binary tasks (grounded, origin, switching) label something else as 1.
_LATENCY_FUNCTIONS = frozenset({Function.FAULT_DETECTION, Function.EVENT_DETECTION})


def _cell_dir(
    out: Path, task: str, protocol: str, grid: str, baseline: str, source: str | None = None
) -> Path:
    # the same layout as coverage.cell_dir: per-protocol results tree (held_out stays at its frozen
    # version), and a pretrain-<src> subdir for a transfer cell from a non-default source
    return cell_dir(Path(out), Cell(task, protocol, grid, baseline, source))


def load_joined(
    out: Path,
    task: str,
    protocol: str,
    grid: str,
    baseline: str,
    test_set: str,
    *,
    seed: int = 0,
    root=None,
    source: str | None = None,
) -> pd.DataFrame:
    """Join one cell's predictions with the truth columns (dt_start, location, fault flags)."""
    cell = _cell_dir(out, task, protocol, grid, baseline, source)
    preds = pd.read_parquet(cell / test_set / f"predictions_seed{seed}.parquet")
    y = load_windows(grid, _FAMILY[test_set], root=root).y
    cols = [c for c in _TRUTH_COLS if c in y.columns]
    return preds.merge(y[cols], on=["sample_id", "window_idx"], how="left")


def detection_latency(joined: pd.DataFrame) -> dict:
    """Per fault episode, latency (s) from inception to the first correctly-flagged active window.

    Active windows have dt_start >= 0 and y_fault_present == 1; an episode is *detected* if any
    of them is predicted positive. Returns detection rate + median/mean latency over detected.
    """
    f = joined[joined["y_fault_line"].notna()]
    detected, latencies = 0, []
    episodes = 0
    for _sid, g in f.groupby("sample_id"):
        active = g[(g["dt_start"] >= 0) & (g["y_fault_present"] == 1)].sort_values("dt_start")
        if active.empty:
            continue
        episodes += 1
        hit = active[active["prediction"] == 1]
        if not hit.empty:
            detected += 1
            latencies.append(float(hit["dt_start"].iloc[0]))
    return {
        "n_fault_episodes": episodes,
        "detection_rate": (detected / episodes) if episodes else None,
        "median_latency_s": float(np.median(latencies)) if latencies else None,
        "mean_latency_s": float(np.mean(latencies)) if latencies else None,
    }


def location_mae_curve(
    joined: pd.DataFrame, *, edges: tuple[float, ...] = (0, 10, 30, 50, 70, 90, 100)
) -> list[dict]:
    """MAE of predicted vs true fault location, binned by distance-along-line (%)."""
    d = joined[joined["y_fault_location"].notna()].copy()
    if d.empty:
        return []
    d["abs_err"] = (d["prediction"].astype(float) - d["y_fault_location"].astype(float)).abs()
    d["bin"] = pd.cut(d["y_fault_location"], bins=list(edges), include_lowest=True)
    out = []
    for b, g in d.groupby("bin", observed=True):
        out.append({"location_bin": str(b), "n": int(len(g)), "mae": float(g["abs_err"].mean())})
    return out


def analyze_cell(
    out: Path,
    task: str,
    protocol: str,
    grid: str,
    baseline: str,
    test_set: str,
    *,
    seed: int = 0,
    root=None,
    source: str | None = None,
) -> dict:
    """Compute the protection-grade analysis appropriate to the task's label kind."""
    td = get_task(task)
    joined = load_joined(
        out, task, protocol, grid, baseline, test_set, seed=seed, root=root, source=source
    )
    res: dict = {"task": task, "grid": grid, "baseline": baseline, "test_set": test_set}
    if td.label_kind is LabelKind.BINARY and td.function in _LATENCY_FUNCTIONS:
        res["latency"] = detection_latency(joined)
    if td.label_kind is LabelKind.REGRESSION:
        res["location_mae_curve"] = location_mae_curve(joined)
    return res


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Protection-grade analysis for one result cell.")
    p.add_argument("--out", default="outputs")
    p.add_argument("--task", required=True)
    p.add_argument("--protocol", default="held_out")
    p.add_argument("--grid", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--test-set", default="test", choices=["test", "benchmark"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", default=None)
    p.add_argument(
        "--pretrain-grid",
        default=None,
        help="transfer cells: the pre-training source grid (default: the default source)",
    )
    args = p.parse_args(argv)
    res = analyze_cell(
        Path(args.out),
        args.task,
        args.protocol,
        args.grid,
        args.baseline,
        args.test_set,
        seed=args.seed,
        root=args.root,
        source=args.pretrain_grid,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
