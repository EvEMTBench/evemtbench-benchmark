"""The single authority for turning per-seed evaluations into a cell's ``result.json``.

Two writers produce ``result.json``: the training runner (:mod:`evemtbench.baselines.runner`) and
the offline re-evaluation pass (:mod:`evemtbench.evaluation.reevaluate`). If they each built the
document themselves, byte-identity between "what the sweep wrote" and "what a rebuild produces"
would be an assertion to re-verify rather than a property of the code. So both call
:func:`build_result_doc` and :func:`render_result_json` here, and identity is structural.

This module also fixes the defect that motivated the re-evaluation pass: headline metrics were
aggregated over all seeds while ``breakdowns`` were taken from whichever seed happened to run last,
so one document mixed a 5-seed mean with a 1-seed diagnostic and said nothing about it.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from .significance import ci95_halfwidth


def aggregate_seed_metrics(
    per_seed_metrics: list[dict[str, float]],
) -> dict[str, dict[str, float | None]]:
    """``{metric: {mean, std, ci95, n_defined, n_seeds}}`` over seeds.

    Moved verbatim from ``runner._aggregate`` (which now delegates here) so the runner's numbers are
    unchanged: population std (ddof=0) and a t-based ci95 half-width over the DEFINED seeds only
    (ddof=1). Note ``hpc/rescore_validation.py`` carries a separate ddof=1 mirror for the validation
    rescoring pass — do not unify them without re-deriving those numbers.
    """
    keys = per_seed_metrics[0].keys()
    summary: dict[str, dict[str, float | None]] = {}
    for k in keys:
        vals = np.array([m[k] for m in per_seed_metrics], dtype=float)
        finite = vals[np.isfinite(vals)]
        n_defined, n_seeds = int(finite.size), int(vals.size)
        if n_defined == 0:
            # Metric is structurally undefined for this whole partition across every seed
            # (e.g. balanced_accuracy on a single-class subset). Record nulls + n_defined=0
            # so the result distinguishes "undefined" from a genuine 0.0, and skip the
            # all-NaN nanmean/nanstd that would otherwise warn "Mean of empty slice".
            summary[k] = {
                "mean": None,
                "std": None,
                "ci95": None,
                "n_defined": 0,
                "n_seeds": n_seeds,
            }
        else:
            # Average over the defined seeds only (no nan-aware reducer, hence no warning).
            # ci95: t-based half-width over the defined seeds (ddof=1 — the stored std stays
            # population/ddof=0). None when < 2 seeds: no seed-variance estimate, never a fake ±0.
            summary[k] = {
                "mean": float(finite.mean()),
                "std": float(finite.std()),
                "ci95": ci95_halfwidth(finite),
                "n_defined": n_defined,
                "n_seeds": n_seeds,
            }
    return summary


class BreakdownMismatch(ValueError):
    """Per-seed breakdowns disagree on strata, groups, or group size.

    Strata and groups come from ``stratified_metrics`` over the eval view's *metadata*, which is
    identical for every seed of a cell — so a disagreement is a harness bug (seeds scored against
    different views), not a result to be averaged.
    """


def aggregate_seed_breakdowns(
    per_seed_breakdowns: list[dict[str, dict[str, dict[str, Any]]]],
    *,
    strict: bool = True,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Aggregate stratified breakdowns across seeds.

    Returns ``{stratum: {group: {"n": int, "n_seeds": int, "metrics": {metric: {...}}}}}``. Each
    entry under ``metrics`` has the SAME shape as a top-level ``aggregate`` entry (built by the same
    :func:`aggregate_seed_metrics`), so a stratum metric and a headline metric read identically.

    ``n`` stays at group level, where it already was: the two in-repo consumers
    (``leaderboard.scan_results`` reads the group KEY set, ``hpc/paper_artifacts/paper_lib`` reads
    ``n``) therefore keep working unchanged.

    With ``strict`` (the default) a stratum/group/``n`` disagreement raises
    :class:`BreakdownMismatch`. ``strict=False`` takes the union of groups and aggregates each over
    only the seeds that reported it — an absent seed is then *missing*, never a zero, mirroring the
    ``n_defined`` convention.
    """
    seeds = [b for b in per_seed_breakdowns if b]
    if not seeds:
        return {}

    strata = list(seeds[0])
    if strict:
        for i, b in enumerate(seeds[1:], start=1):
            if list(b) != strata:
                raise BreakdownMismatch(
                    f"seed {i} reports strata {list(b)}, seed 0 reports {strata}"
                )
    else:
        seen: dict[str, None] = {}
        for b in seeds:
            for col in b:
                seen.setdefault(col, None)
        strata = list(seen)

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for col in strata:
        per_seed_groups = [b.get(col, {}) for b in seeds]
        if strict:
            g0 = set(per_seed_groups[0])
            for i, g in enumerate(per_seed_groups[1:], start=1):
                if set(g) != g0:
                    raise BreakdownMismatch(
                        f"stratum {col!r}: seed {i} groups {sorted(set(g) ^ g0)} differ from seed 0"
                    )
            group_keys = list(per_seed_groups[0])
        else:
            keys: dict[str, None] = {}
            for g in per_seed_groups:
                for k in g:
                    keys.setdefault(k, None)
            group_keys = list(keys)

        groups: dict[str, dict[str, Any]] = {}
        for gk in group_keys:
            present = [g[gk] for g in per_seed_groups if gk in g]
            ns = {int(d["n"]) for d in present if d.get("n") is not None}
            if strict and len(ns) > 1:
                raise BreakdownMismatch(
                    f"stratum {col!r} group {gk!r}: n differs across seeds ({sorted(ns)}) — "
                    "the seeds did not score the same eval view"
                )
            # Metric names, excluding the group-level count. Union so a metric that is absent for
            # one seed still aggregates (it becomes NaN for that seed -> n_defined reflects it).
            names: dict[str, None] = {}
            for d in present:
                for k in d:
                    if k != "n":
                        names.setdefault(k, None)
            rows = [{k: float(d.get(k, math.nan)) for k in names} for d in present]
            groups[gk] = {
                # One value under `strict`; in the permissive path take the largest so `n` cannot
                # understate the group's support.
                "n": max(ns) if ns else 0,
                "n_seeds": len(present),
                "metrics": aggregate_seed_metrics(rows) if rows and names else {},
            }
        out[col] = groups
    return out


def build_result_doc(
    *,
    benchmark_version: str,
    task: str,
    protocol: str,
    grid: str,
    baseline: str,
    pretrain_grid: str | None,
    test_set: str,
    n_test: int,
    seeds: list[int],
    n_params: int | None,
    compute: dict,
    per_seed: list[dict],
    aggregate: dict,
    breakdowns: dict,
) -> dict:
    """Assemble a cell's ``result.json`` payload. Keyword-only: this is a published schema, and the
    two callers must not be able to drift by argument position."""
    return {
        "benchmark_version": benchmark_version,
        "task": task,
        "protocol": protocol,
        "grid": grid,
        "baseline": baseline,
        # pre-training source grid for transfer cells (None otherwise) — lets the leaderboard /
        # paper builders distinguish 20/110/345 kV-sourced transfer at the same target.
        "pretrain_grid": pretrain_grid,
        "test_set": test_set,
        "n_test": n_test,
        "seeds": seeds,
        "n_params": n_params,  # capacity: trainable params (None for classical)
        "compute": compute,  # per-cell training cost (GPU-hours reporting)
        "per_seed": per_seed,
        "aggregate": aggregate,
        # Seed-aggregated, same machinery as `aggregate` (see aggregate_seed_breakdowns). Until the
        # 2026-08 re-evaluation pass this field held the LAST seed's breakdowns beside a multi-seed
        # `aggregate`, undocumented.
        "breakdowns": breakdowns,
    }


def render_result_json(doc: dict) -> str:
    """The exact bytes of a ``result.json``.

    Both writers go through this, so a rebuild is byte-comparable to the original. NaN/Inf become
    ``null`` (JSON has no literal for them, and ``allow_nan=False`` then guarantees we never emit
    invalid JSON). Deliberately returns no trailing newline: the corpus was written this way and a
    newline here would rewrite all 3608 files for nothing.
    """
    from .evaluate import _nan_to_none

    sanitized = _nan_to_none(json.loads(json.dumps(doc, default=float)))
    return json.dumps(sanitized, indent=2, sort_keys=True, allow_nan=False)
