"""Seed-level uncertainty and paired significance (validity plumbing).

The benchmark reports every aggregate metric as *mean over seeds*; these helpers add the
uncertainty and comparison machinery the paper cites:

* :func:`ci95_halfwidth` — half-width of the t-based confidence interval over per-seed values
  (n=5–10 seeds → Student t, not a normal approximation; ``ddof=1``). ``None`` when fewer than
  two finite values exist — a CI over one seed is undefined, never a fake ``± 0``.
* :func:`paired_seed_test` — paired-across-seeds t-test between two baselines' per-seed metric
  vectors, paired **by seed id** (never by list order; cells may be run with different
  ``--seeds``). Descriptive at 5 seeds (df=4, low power): a large p-value means "not
  distinguishable at this seed count", not "equal".
* :func:`compare_result_files` / ``python -m evemtbench.evaluation.significance`` — compare two
  runner-written ``result.json`` artifacts on a task's headline metric.

Direction comes from :data:`~evemtbench.evaluation.metrics.METRIC_DIRECTION` so ``better`` is
meaningful for lower-is-better metrics (mae, missed_fault_rate, …). All emitted values are
JSON-native (finite floats or ``None`` — never NaN/inf; result docs are written with
``allow_nan=False``).
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.stats import t as _student_t

from .metrics import METRIC_DIRECTION


def _finite(values: Sequence[float | None] | np.ndarray) -> np.ndarray:
    """Float array of the finite entries (None/NaN/inf dropped) — the ``_aggregate`` convention."""
    vals = np.array(list(values), dtype=float)  # None -> nan
    return vals[np.isfinite(vals)]


def ci95_halfwidth(
    values: Sequence[float | None] | np.ndarray, confidence: float = 0.95
) -> float | None:
    """Half-width of the t-based CI over per-seed metric values (``mean ± halfwidth``).

    Non-finite entries are dropped first (an undefined seed is missing, not zero). Returns
    ``None`` when fewer than two finite values remain — the seed-variance estimate needs n ≥ 2.
    """
    finite = _finite(values)
    n = int(finite.size)
    if n < 2:
        return None
    quantile = float(_student_t.ppf(0.5 + confidence / 2.0, df=n - 1))
    return float(quantile * finite.std(ddof=1) / math.sqrt(n))


def _seed_values(per_seed: list[dict], metric: str) -> dict[int, float]:
    """``seed id -> finite metric value`` from a result.json ``per_seed`` list."""
    out: dict[int, float] = {}
    for rec in per_seed:
        v = (rec.get("metrics") or {}).get(metric)
        if v is not None and math.isfinite(float(v)):
            out[int(rec["seed"])] = float(v)
    return out


def bootstrap_ci(
    values, n_boot: int = 2000, seed: int = 0, confidence: float = 0.95
) -> tuple[float, float] | None:
    """Seeded percentile-bootstrap CI of the mean — deterministic (fixed default_rng seed), so
    pages built from it stay byte-stable under the freshness gate. None for n < 2 (like
    :func:`ci95_halfwidth`); non-finite values are dropped."""
    import numpy as np

    xs = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if xs.size < 2:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(xs, size=(n_boot, xs.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def paired_seed_test(
    per_seed_a: list[dict],
    per_seed_b: list[dict],
    metric: str,
    *,
    alpha: float = 0.05,
    confidence: float = 0.95,
) -> dict:
    """Paired t-test of side *a* vs side *b* on ``metric`` across their common seeds.

    Inputs are ``result.json`` ``per_seed`` lists (``[{"seed": int, "metrics": {...}}]``). Pairs
    are formed on the **intersection of seed ids** with a finite value on both sides. With fewer
    than two pairs, or zero variance in the differences (a degenerate paired t), the test fields
    are ``None`` and ``significant`` is ``False`` — ``mean_diff``/``better`` still describe the
    direction. ``better`` is oriented by ``METRIC_DIRECTION`` (so lower mae ⇒ better).
    """
    a, b = _seed_values(per_seed_a, metric), _seed_values(per_seed_b, metric)
    seeds = sorted(a.keys() & b.keys())
    n = len(seeds)
    diffs = np.array([a[s] - b[s] for s in seeds], dtype=float)
    direction = METRIC_DIRECTION.get(metric, +1)
    mean_diff = float(diffs.mean()) if n else None
    report: dict = {
        "metric": metric,
        "n_pairs": n,
        "seeds": seeds,
        "mean_a": float(np.mean([a[s] for s in seeds])) if n else None,
        "mean_b": float(np.mean([b[s] for s in seeds])) if n else None,
        "mean_diff": mean_diff,
        "better": (
            None
            if mean_diff is None or mean_diff == 0.0
            else ("a" if direction * mean_diff > 0 else "b")
        ),
        "t_stat": None,
        "p_value": None,
        "ci95_diff": None,
        "significant": False,
    }
    if n < 2:
        return report
    sd = float(diffs.std(ddof=1))
    if sd == 0.0:
        # Identical per-seed deltas: the paired t is degenerate; claim direction, not a p-value.
        return report
    assert mean_diff is not None
    se = sd / math.sqrt(n)
    t_stat = mean_diff / se
    p_value = float(2.0 * _student_t.sf(abs(t_stat), df=n - 1))
    quantile = float(_student_t.ppf(0.5 + confidence / 2.0, df=n - 1))
    report.update(
        t_stat=float(t_stat),
        p_value=p_value,
        ci95_diff=float(quantile * se),
        significant=bool(p_value < alpha),
    )
    return report


def compare_result_files(
    path_a: str | Path,
    path_b: str | Path,
    metric: str | None = None,
    *,
    alpha: float = 0.05,
) -> dict:
    """Paired-seed comparison of two runner-written ``result.json`` artifacts.

    The sides must describe the same (task, protocol, grid, test_set, benchmark_version) —
    mismatches are reported as ``issues`` and flip ``ok`` (comparing across contracts is
    meaningless). ``metric`` defaults to the task's registry headline (``metrics[0]``).
    """
    doc_a = json.loads(Path(path_a).read_text())
    doc_b = json.loads(Path(path_b).read_text())
    issues = [
        f"{key} differs: {doc_a.get(key)!r} vs {doc_b.get(key)!r}"
        for key in ("task", "protocol", "grid", "test_set", "benchmark_version")
        if doc_a.get(key) != doc_b.get(key)
    ]
    if metric is None:
        from ..tasks.registry import get_task  # lazy: keep module import light

        metric = get_task(doc_a["task"]).metrics[0]
    test = paired_seed_test(
        doc_a.get("per_seed") or [], doc_b.get("per_seed") or [], metric, alpha=alpha
    )
    return {
        "ok": not issues,
        "issues": issues,
        "a": {"path": str(path_a), "baseline": doc_a.get("baseline")},
        "b": {"path": str(path_b), "baseline": doc_b.get("baseline")},
        "test": test,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Paired-across-seeds significance test between two result.json artifacts."
    )
    p.add_argument("result_a", help="first result.json (side 'a')")
    p.add_argument("result_b", help="second result.json (side 'b')")
    p.add_argument("--metric", default=None, help="metric name (default: the task's headline)")
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args(argv)
    report = compare_result_files(args.result_a, args.result_b, args.metric, alpha=args.alpha)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
