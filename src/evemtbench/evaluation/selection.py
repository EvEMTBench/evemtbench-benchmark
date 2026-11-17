"""How reliable is validation-based model selection? A diagnostic, never a selector.

EvEMTBench reports, per (task, grid), the learned baseline that wins on the **validation** split,
then quotes its score on the held-out sets. This module measures how that choice holds up
afterwards. It never changes it:

* **same winner**: whether the validation winner is also the best baseline on the reported set;
* **selection regret**: how far the validation winner falls short of the best baseline on the
  reported set, in the headline metric's units (``≥ 0``; ``0`` when the winners agree);
* **Kendall τ-b**: agreement between two rankings of the same baselines (validation vs a
  held-out set, and in-distribution ``test`` vs the cross-distribution ``benchmark``).

The held-out scores only *grade* the selection after the fact: nothing here feeds back into
which model is reported. The idea comes from akmaier/Agent4CT
(``scripts/breast_board_rank_corr.py``; its ``docs/findings.md`` records a validation champion
that did not generalise to test). With five learned baselines a rank correlation is coarse, so
regret and the same-winner rate are the primary numbers.

Selection rule = the paper kernel (``hpc/paper_artifacts/paper_lib.py::_argbest_baseline``):
argbest in ``METRIC_DIRECTION`` over the candidate pool in pool order, ties keep the earlier
entry (strict ``>``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import median

from .metrics import METRIC_DIRECTION

#: the held-out sets a validation choice is graded on
GRADED_SETS = ("test", "benchmark")


def argbest(scores: Mapping[str, float], pool: Sequence[str], direction: int) -> str | None:
    """The best ``pool`` entry with a score (strict ``>``: ties keep the earlier entry)."""
    best, best_v = None, None
    for b in pool:
        v = scores.get(b)
        if v is None:
            continue
        if best_v is None or direction * (v - best_v) > 0:
            best, best_v = b, v
    return best


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Kendall's τ-b of two paired score vectors (tie-corrected); ``None`` below 2 pairs or when
    either vector is constant."""
    n = len(x)
    if n != len(y) or n < 2:
        return None
    concordant = discordant = ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue  # tied in both: counts in neither denominator term
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    denom = ((concordant + discordant + ties_x) * (concordant + discordant + ties_y)) ** 0.5
    return None if denom == 0 else (concordant - discordant) / denom


def _tau_between(a: Mapping[str, float], b: Mapping[str, float], pool: Sequence[str]):
    common = [k for k in pool if a.get(k) is not None and b.get(k) is not None]
    return kendall_tau_b([a[k] for k in common], [b[k] for k in common]), len(common)


def cell_diagnostic(
    scores: Mapping[str, Mapping[str, float]], metric: str, pool: Sequence[str]
) -> dict | None:
    """Diagnostic for one (task, grid). ``scores[split][baseline]`` holds the headline mean for
    ``split`` in ``{"val", "test", "benchmark"}``. ``None`` without a validation winner."""
    direction = METRIC_DIRECTION.get(metric, 1)
    val = scores.get("val") or {}
    selected = argbest(val, pool, direction)
    if selected is None:
        return None
    out: dict = {"metric": metric, "selected": selected, "sets": {}}
    for ts in GRADED_SETS:
        rep = scores.get(ts) or {}
        best = argbest(rep, pool, direction)
        if best is None or rep.get(selected) is None:
            continue
        tau, n = _tau_between(val, rep, pool)
        out["sets"][ts] = {
            "best": best,
            "same_winner": best == selected,
            # + 0.0 turns the -0.0 of a lower-is-better tie into 0.0 (no "-0.000" in the tables)
            "regret": direction * (rep[best] - rep[selected]) + 0.0,
            "tau_val": tau,
            "n_baselines": n,
        }
    tau_tb, n_tb = _tau_between(scores.get("test") or {}, scores.get("benchmark") or {}, pool)
    out["tau_test_benchmark"] = tau_tb
    out["n_test_benchmark"] = n_tb
    return out


def summarize(diagnostics: Sequence[dict]) -> dict:
    """Roll cell diagnostics up per graded set: cell count, same-winner rate, median and max
    regret per metric (regret is in metric units, so it is never pooled across metrics), mean
    Kendall τ-b."""
    summary: dict = {}
    for ts in GRADED_SETS:
        cells = [d["sets"][ts] for d in diagnostics if ts in d["sets"]]
        if not cells:
            continue
        taus = [c["tau_val"] for c in cells if c["tau_val"] is not None]
        regrets: dict[str, list[float]] = {}
        for d in diagnostics:
            if ts in d["sets"]:
                regrets.setdefault(d["metric"], []).append(d["sets"][ts]["regret"])
        summary[ts] = {
            "n_cells": len(cells),
            "n_same_winner": sum(c["same_winner"] for c in cells),
            "median_regret": {m: median(v) + 0.0 for m, v in sorted(regrets.items())},
            # the median hides the tail: a pick that collapses on one cell is the case that matters
            "max_regret": {m: max(v) + 0.0 for m, v in sorted(regrets.items())},
            "mean_tau_val": (sum(taus) / len(taus)) if taus else None,
        }
    tb = [d["tau_test_benchmark"] for d in diagnostics if d["tau_test_benchmark"] is not None]
    summary["mean_tau_test_benchmark"] = (sum(tb) / len(tb)) if tb else None
    summary["n_tau_test_benchmark"] = len(tb)
    return summary
