"""Regenerate RESULTS.md (narrative + auto tables) from outputs/<version>/.../result.json.

Registry-driven and tier-grouped: every task in ``evemtbench.tasks.registry`` is rendered (grouped
by difficulty ``tier`` then objective), so adding a task needs no change here. Idempotent — re-run
after more baselines land and the tables refresh.

    python hpc/make_results.py
"""

from __future__ import annotations

import collections
import glob
import json
import sys

from evemtbench.data.roots import GRIDS
from evemtbench.evaluation.significance import ci95_halfwidth
from evemtbench.tasks.registry import get_task, tasks_by_tier
from evemtbench.tasks.taxonomy import Function
from evemtbench.version import BENCHMARK_VERSION, results_version

# Short "vMAJOR.MINOR" tag for the title, derived from the contract version so it never goes stale.
SHORT_VERSION = "v" + ".".join(BENCHMARK_VERSION.split(".")[:2])

# RESULTS.md renders the held_out matrix; its current results live in the per-protocol tree
# (1.0.0 — the v1.1.0 revision changed only the multi_grid source splits, see version.py).
HELD_OUT_VERSION = results_version("held_out")

NARRATIVE = """\
# EvEMTBench {short} — results ({ver})

The `held_out` results below were produced under contract v{ho_ver} and remain current under
v{ver}: the v1.1 revision regenerated only the `multi_grid` source-corpus splits (pre-training /
transfer protocols — see the leaderboard and `results/{ver}/`), leaving every `held_out` artifact
byte-identical.

Protocol `held_out`: train/val/test on the `adapt_grid` family; **benchmark** = the held-out
`benchmark` family (never trained on). 5 seeds — cells show **mean ± 95% CI half-width** (Student
t across seeds; single-seed cells show the mean alone). Deep baselines trained to a 60-epoch
cap with validation early-stopping. Baselines: majority, threshold (FD-local only), random_forest,
mlp, gru, cnn, resnet (1D-ResNet). Deep baselines share one training regime (only the architecture
differs); `params` = trainable-parameter count (capacity is reported, **not** equalized). Grids are
reported **per grid** (never pooled — class spaces differ, e.g. HIF is MV-only).

Tasks are grouped by **difficulty tier** (see `docs/task_suite_roadmap.md`):
- **Tier 1 — easy anchors:** near-solved on ideal data; the community frontier is added realism.
- **Tier 2 — medium:** learnable in-distribution, harder on MV / under transfer.
- **Tier 3 — hard frontier:** hard even on ideal data; the community frontier is better models.

> **Status**: tables are accumulated as jobs complete; `—` marks a pending cell. This file is
> auto-generated from `outputs/` by `hpc/make_results.py` — do not hand-edit.

> **Known issues**: dataset properties that affect some of these numbers are listed, with scope,
> in [`docs/known_issues.md`](docs/known_issues.md) (flagged; the published behaviour is unchanged).
"""

BASELINES = ["majority", "threshold", "random_forest", "mlp", "gru", "cnn", "resnet"]

GRID_LABEL = {
    "double_line": "DoubleLine (110 kV HV)",
    "cigre_mv": "CigreMV (20 kV MV)",
    "testgrid_110kv": "TestGrid (110 kV HV)",
    "ieee39": "IEEE-39 (345 kV EHV)",
}

TIER_TITLE = {
    1: "Tier 1 — easy anchors",
    2: "Tier 2 — medium",
    3: "Tier 3 — hard frontier",
    None: "Untiered",
}

FUNCTION_LABEL = {
    Function.FAULT_DETECTION: "FD — fault detection",
    Function.FAULT_CLASS: "FC — fault classification",
    Function.FAULT_LOCATION: "FL — fault location (regression)",
    Function.EVENT_STATE: "ES — event state (3-class)",
    Function.EVENT_DETECTION: "ED — event detection",
    Function.SWITCH_DETECTION: "SD — switch detection",
    Function.SWITCH_TYPE: "ST — switch type",
    Function.EVENT_CLASSIFICATION: "EC — event classification (23-class)",
    Function.FAULT_GROUNDED: "grounded vs ungrounded",
    Function.FAULT_PHASE: "faulted phase (7-class)",
    Function.FAULT_CATEGORY: "fault category (shc/incipient/hif)",
    Function.FAULT_ORIGIN: "fault origin (line vs bus)",
}

# Metrics where lower is better (for the header arrow); everything else is higher-better.
_LOWER_BETTER = {
    "mae",
    "rmse",
    "median_abs_error",
    "missed_fault_rate",
    "false_alarm_rate",
    "missed_event_rate",
    "false_event_trip_rate",
}
_OBS_ORDER = {"local": 0, "line": 1, "global": 2}


_WARNED: set[str] = set()


def warn_once(msg: str) -> None:
    """Print ``[warn] <msg>`` to stderr once per process (dedup by exact message).

    Silent degradations in the results/selection path (a swallowed corrupt result.json, a
    selection falling back from validation to the reported set) must be loud enough to show up
    in build/CI/SLURM logs without touching stdout or any byte-compared rendered artifact."""
    if msg in _WARNED:
        return
    _WARNED.add(msg)
    print(f"[warn] {msg}", file=sys.stderr, flush=True)


def load_rows(grid: str) -> dict:
    """(task_id, baseline) -> {test_set: doc} for one grid, from outputs/<version>/..."""
    rows: dict = collections.defaultdict(dict)
    for f in glob.glob(f"outputs/{HELD_OUT_VERSION}/*/held_out/{grid}/*/*/result.json"):
        try:
            d = json.load(open(f))
            rows[(d["task"], d["baseline"])][d["test_set"]] = d
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
            warn_once(f"skipping unreadable/malformed result.json: {f} ({type(e).__name__}: {e})")
    return rows


def mean_val(doc, metric):
    if not doc:
        return None
    a = doc.get("aggregate", {}).get(metric)
    return None if not a else a.get("mean")


def ci_val(doc, metric):
    """95% CI half-width for a cell: the runner-stored ``ci95`` when present, else derived from
    the persisted ``per_seed`` vector (so CIs render for artifacts written before ci95 existed)."""
    if not doc:
        return None
    a = doc.get("aggregate", {}).get(metric)
    if a and a.get("ci95") is not None:
        return a["ci95"]
    vals = [(r.get("metrics") or {}).get(metric) for r in doc.get("per_seed") or []]
    return ci95_halfwidth(vals) if vals else None


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def fmt_mci(doc, metric, nd=3):
    """``mean ± ci95`` cell text; bare mean when the CI is undefined (< 2 seeds); ``—`` pending."""
    mean = mean_val(doc, metric)
    if mean is None:
        return "—"
    ci = ci_val(doc, metric)
    return f"{mean:.{nd}f}" if ci is None else f"{mean:.{nd}f} ± {ci:.{nd}f}"


def params_str(doc) -> str:
    """Humanize a cell's trainable-parameter count (1.6M / 52k / — for classical/None)."""
    n = (doc or {}).get("n_params")
    if not n:
        return "—"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.0f}k"
    return str(n)


def train_str(doc) -> str:
    """Humanize a cell's mean per-seed training wall-time (1.1h / 3.2m / 12s / —)."""
    s = ((doc or {}).get("compute") or {}).get("fit_seconds_mean")
    if not s:
        return "—"
    if s >= 3600:
        return f"{s / 3600:.1f}h"
    if s >= 60:
        return f"{s / 60:.1f}m"
    return f"{s:.0f}s"


def _arrow(metric: str) -> str:
    return "↓" if metric in _LOWER_BETTER else "↑"


def objective_table(fn: Function, task_ids: list[str], rows: dict) -> list[str]:
    """One table for an objective: its observability variants × baselines. Empty if no data."""
    present = [t for t in task_ids if any((t, b) in rows for b in BASELINES)]
    if not present:
        return []
    metrics = get_task(present[0]).metrics
    headline = metrics[0]
    secondary = metrics[1] if len(metrics) > 1 else None
    lines = [f"\n**{FUNCTION_LABEL.get(fn, fn.value)}** ({headline} {_arrow(headline)})\n"]
    cols = ["observability", "baseline", f"{headline} (test)", f"{headline} (bench)"]
    if secondary:
        cols.append(f"{secondary} (bench)")
    cols += ["params", "train"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for t in present:
        obs = get_task(t).observability.value
        for base in BASELINES:
            if (t, base) not in rows:
                continue
            te = rows[(t, base)].get("test")
            be = rows[(t, base)].get("benchmark")
            cells = [obs, base, fmt_mci(te, headline), fmt_mci(be, headline)]
            if secondary:
                cells.append(fmt_mci(be, secondary))
            cells += [params_str(te or be), train_str(te or be)]
            lines.append("| " + " | ".join(cells) + " |")
    return lines


def grid_section(grid: str, rows: dict) -> list[str]:
    """A grid's results: tier subsections, one table per objective present in this grid."""
    lines = [f"\n### {GRID_LABEL.get(grid, grid)}\n"]
    for tier, tids in tasks_by_tier().items():
        # objective functions in this tier, first-seen order preserved
        fns: list[Function] = []
        for t in tids:
            fn = get_task(t).function
            if fn not in fns:
                fns.append(fn)
        rendered: list[str] = []
        for fn in fns:
            fn_ids = sorted(
                (t for t in tids if get_task(t).function is fn),
                key=lambda t: _OBS_ORDER.get(get_task(t).observability.value, 9),
            )
            rendered += objective_table(fn, fn_ids, rows)
        if rendered:
            lines.append(f"\n#### {TIER_TITLE.get(tier, 'Untiered')}\n")
            lines += rendered
    return lines


def build_results() -> str:
    """Deterministically build RESULTS.md from outputs/ (pure: no I/O, no timestamps).

    Single source of truth for RESULTS.md; side-effect-free so the freshness gate
    (hpc/check_results_fresh.py) can compare it byte-for-byte against the committed file."""
    out = [
        NARRATIVE.format(short=SHORT_VERSION, ver=BENCHMARK_VERSION, ho_ver=HELD_OUT_VERSION),
        "\n## Results by grid\n",
    ]
    for grid in GRIDS:
        rows = load_rows(grid)
        if sum(len(v) for v in rows.values()):
            out.extend(grid_section(grid, rows))
    return "\n".join(out) + "\n"


def main():
    content = build_results()
    with open("RESULTS.md", "w") as f:
        f.write(content)
    total_files = sum(sum(len(v) for v in load_rows(g).values()) for g in GRIDS)
    print(f"RESULTS.md regenerated from {total_files} result.json files across {len(GRIDS)} grids")


if __name__ == "__main__":
    main()
