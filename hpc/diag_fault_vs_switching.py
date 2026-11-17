"""Fault-versus-switching diagnostic (revision ticket EXP-04) from stored event-state predictions.

Reads the per-seed predictions the held_out sweep stored for ``event_state_global`` and asks, per
grid, baseline, test set and seed, how often the fault class is confused with switching. No model
is loaded or trained, no threshold is set, and nothing under ``outputs/`` is written.

Frozen definitions (fixed before any result was inspected; see ``MAPPING`` below):

* source task: ``event_state_global`` (``normal`` / ``switching`` / ``fault``). Event-detection
  outputs are not used: active switching is a positive there.
* class mapping: predicted ``fault`` = fault prediction; predicted ``normal`` or ``switching`` =
  non-fault. The true state is the task's own label (``derive_event_state_series``): active fault
  window, active switching window, inactive (``normal``) window.
* reported rates, separately:
  (a) ``switch_to_fault`` -- fault predictions among active switching windows;
  (b) ``missed_fault`` -- non-fault predictions among active fault windows;
  (c) ``inactive_to_fault`` -- fault predictions among inactive windows.
  A group without support for a rate has an undefined rate (empty in the CSVs), never 0.
* candidates: the baseline the paper's kernel (``paper_lib.best_cell_choice``, headline metric
  ``balanced_accuracy``) selects on VALIDATION for this task and grid, plus every learned baseline
  in a supplementary table. Nothing is selected on ``test`` or ``benchmark``. When the validation
  evidence is missing the kernel refuses (``SelectionEvidenceError``); the refusal is recorded and
  no substitute is chosen.
* evaluation sets: ``test`` and ``benchmark`` as stored, all five seeds. Seed mean and spread are
  over the five seeds; windows overlap and are not treated as independent replicates (no
  window-level confidence intervals; window and episode supports are reported).

Positive control: per cell and test set, the split hashes must match the run record, the stored
seeds must be the five seeds with a prediction file each, and the headline metric recomputed from
each seed's predictions must equal the stored ``per_seed`` value (``--tol``, default 0). A cell
that fails is listed in ``controls.csv`` and not scored.

Writes to ``--out``: ``mapping.json``, ``controls.csv``, ``selection.csv``,
``confusion_per_seed.csv`` (3x3 true x predicted state per seed), ``confusion_summary.csv``
(row shares, seed mean/std/min/max), ``rates_per_seed.csv``,
``rates_summary.csv`` (seed mean, std, min, max), ``si_table.tex``, ``inputs.sha256`` and
``summary.json``.

    PYTHONPATH=src python hpc/diag_fault_vs_switching.py --out <dir>
    PYTHONPATH=src python hpc/diag_fault_vs_switching.py --out <dir> --grids double_line

Run from the repository root (the selection kernel reads ``outputs/`` relative to it). Exit status
is non-zero if any cell fails a control.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "paper_artifacts"))

import numpy as np
import pandas as pd

from evemtbench.coverage import LEARNED_BASELINES
from evemtbench.evaluation.metrics import compute_multiclass_metrics
from evemtbench.evaluation.predictions_io import align_to_view, check_prediction_values
from evemtbench.evaluation.reevaluate import _load_json, _provenance_matches, _seed_metrics_agree
from evemtbench.tasks.labels.event_state import CLASS_TO_IDX, EVENT_STATE_CLASSES
from evemtbench.tasks.labels.switch_type import SWITCH_TYPE_CLASSES
from evemtbench.version import results_version

TASK = "event_state_global"
PROTOCOL = "held_out"
GRIDS = ("double_line", "testgrid_110kv", "cigre_mv", "ieee39")
TEST_SETS = ("test", "benchmark")
SEEDS = (0, 1, 2, 3, 4)
HEADLINE = "balanced_accuracy"

NORMAL, SWITCHING, FAULT = (CLASS_TO_IDX[c] for c in ("normal", "switching", "fault"))
#: rate -> (true state that forms the support, predicate on the predicted state counted as an event)
RATES: dict[str, tuple[int, str]] = {
    "switch_to_fault": (SWITCHING, "fault"),
    "missed_fault": (FAULT, "non_fault"),
    "inactive_to_fault": (NORMAL, "fault"),
}
#: breakdown column -> fixed vocabulary (groups listed even at zero support) or None (observed)
BREAKDOWNS: dict[str, tuple[str, ...] | None] = {
    "all": ("all",),
    "event_family": ("normal", "switching", "fault"),
    "event_type": None,
    "y_switch_category": SWITCH_TYPE_CLASSES,
}
NONE_GROUP = "<none>"  # null descriptor (e.g. y_switch_category on a fault or no_event window)

MAPPING = {
    "ticket": "EXP-04",
    "source_task": TASK,
    "protocol": PROTOCOL,
    "classes": list(EVENT_STATE_CLASSES),
    "fault_prediction": ["fault"],
    "non_fault_prediction": ["normal", "switching"],
    "true_state": {
        "fault": "overlap_ratio > 0 and event_family == 'fault'",
        "switching": "overlap_ratio > 0 and event_family == 'switching'",
        "normal": "every other window (inactive)",
    },
    "rates": {
        "switch_to_fault": "fault predictions / active switching windows",
        "missed_fault": "non-fault predictions / active fault windows",
        "inactive_to_fault": "fault predictions / inactive (normal) windows",
    },
    "zero_support": "undefined (empty), never 0",
    "selection": "paper_lib.best_cell_choice on validation, headline balanced_accuracy; no "
    "selection on test/benchmark; every learned baseline reported in the supplementary table",
    "evaluation_sets": list(TEST_SETS),
    "seeds": list(SEEDS),
    "event_detection_used": False,
    "threshold_tuning": False,
    "scope": "global view, window level; no trip-security or fixed-relay selectivity claim",
}


# ------------------------------------------------------------------------------------------------
# pure kernels (unit-tested on synthetic frames)
# ------------------------------------------------------------------------------------------------


def is_fault_prediction(pred: np.ndarray) -> np.ndarray:
    """The frozen mapping: only a predicted ``fault`` is a fault prediction."""
    return np.asarray(pred).astype(int) == FAULT


def conditional_rates(
    true_state: np.ndarray, pred: np.ndarray, episodes: np.ndarray | None = None
) -> dict[str, dict]:
    """The three conditional rates on one set of windows.

    Returns ``{rate: {support, support_episodes, count, rate}}``; ``rate`` is ``None`` when the
    support is zero (undefined, not zero)."""
    true_state = np.asarray(true_state).astype(int)
    fault_pred = is_fault_prediction(pred)
    out = {}
    for name, (state, event) in RATES.items():
        support = true_state == state
        hit = support & (fault_pred if event == "fault" else ~fault_pred)
        n, k = int(support.sum()), int(hit.sum())
        out[name] = {
            "support": n,
            "support_episodes": (
                int(pd.unique(np.asarray(episodes)[support]).size) if episodes is not None else None
            ),
            "count": k,
            "rate": (k / n) if n else None,
        }
    return out


def _group_values(meta: pd.DataFrame, col: str) -> np.ndarray:
    if col == "all":
        return np.full(len(meta), "all", dtype=object)
    if col not in meta.columns:
        return np.full(len(meta), NONE_GROUP, dtype=object)
    s = meta[col].astype("string")
    return s.fillna(NONE_GROUP).to_numpy(dtype=object)


def rate_rows(
    meta: pd.DataFrame,
    true_state: np.ndarray,
    pred: np.ndarray,
    breakdowns: dict[str, tuple[str, ...] | None] = BREAKDOWNS,
) -> list[dict]:
    """One row per (breakdown, group, rate). Fixed-vocabulary groups appear even without windows;
    observed groups are those present in ``meta``. Episodes are ``sample_id`` values."""
    true_state = np.asarray(true_state).astype(int)
    pred = np.asarray(pred).astype(int)
    episodes = meta["sample_id"].to_numpy()
    rows = []
    for col, vocab in breakdowns.items():
        vals = _group_values(meta, col)
        observed = sorted(set(vals.tolist()))
        groups = list(vocab) + [g for g in observed if g not in vocab] if vocab else observed
        for g in groups:
            m = vals == g
            for name, r in conditional_rates(true_state[m], pred[m], episodes[m]).items():
                rows.append({"breakdown": col, "group": g, "rate_name": name, **r})
    return rows


def confusion(true_state: np.ndarray, pred: np.ndarray) -> list[dict]:
    """3x3 true x predicted event-state counts, as long rows."""
    t, p = np.asarray(true_state).astype(int), np.asarray(pred).astype(int)
    return [
        {"true_state": tc, "pred_state": pc, "count": int(((t == ti) & (p == pi)).sum())}
        for ti, tc in enumerate(EVENT_STATE_CLASSES)
        for pi, pc in enumerate(EVENT_STATE_CLASSES)
    ]


def summarise_seeds(per_seed: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Seed aggregate per ``keys``: mean / std (ddof=1) / min / max of the defined per-seed rates.

    Supports are window/episode counts of the evaluation set and must agree across seeds (the same
    windows are scored); a disagreement raises. A group whose rate is undefined in every seed keeps
    an undefined (NaN) mean."""
    out = []
    for key, g in per_seed.groupby(keys, sort=True, dropna=False):
        if g["support"].nunique() != 1:
            raise ValueError(f"support differs across seeds for {key}: {sorted(g['support'])}")
        r = pd.to_numeric(g["rate"], errors="coerce").dropna()
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True))
        row.update(
            support=int(g["support"].iloc[0]),
            support_episodes=g["support_episodes"].iloc[0],
            n_seeds=len(g),
            n_defined=len(r),
            count_mean=float(g["count"].mean()),
            mean=float(r.mean()) if len(r) else math.nan,
            std=float(r.std(ddof=1)) if len(r) > 1 else math.nan,
            min=float(r.min()) if len(r) else math.nan,
            max=float(r.max()) if len(r) else math.nan,
            per_seed=" ".join("" if pd.isna(v) else f"{float(v):.6g}" for v in g["rate"]),
        )
        out.append(row)
    return pd.DataFrame(out)


def summarise_confusion(conf: pd.DataFrame) -> pd.DataFrame:
    """Seed aggregate of the 3x3 confusion: per cell and true state, the share of windows
    predicted as each state (row-normalised per seed), mean / std (ddof=1) / min / max over seeds.
    Shows where missed faults go (``normal`` vs ``switching``) without adding a rate."""
    keys = ["grid", "baseline", "test_set", "true_state"]
    c = conf.copy()
    row_n = c.groupby([*keys, "seed"])["count"].transform("sum")
    c["share"] = (c["count"] / row_n).where(row_n > 0)
    out = []
    for key, g in c.groupby([*keys, "pred_state"], sort=True):
        r = g["share"].dropna()
        out.append(
            {
                **dict(zip([*keys, "pred_state"], key, strict=True)),
                "support": int(row_n.loc[g.index].iloc[0]),
                "n_seeds": len(g),
                "count_mean": float(g["count"].mean()),
                "share_mean": float(r.mean()) if len(r) else math.nan,
                "share_std": float(r.std(ddof=1)) if len(r) > 1 else math.nan,
                "share_min": float(r.min()) if len(r) else math.nan,
                "share_max": float(r.max()) if len(r) else math.nan,
            }
        )
    return pd.DataFrame(out)


def headline_control(
    recomputed: dict[str, float], recorded: dict[str, float], tol: float
) -> str | None:
    """``None`` when the recomputed headline equals the stored per-seed value within ``tol``."""
    old, new = recorded.get(HEADLINE), recomputed.get(HEADLINE)
    if old is None or new is None:
        return f"{HEADLINE} missing (recorded={old}, recomputed={new})"
    if not (abs(float(old) - float(new)) <= tol):
        return f"{HEADLINE} recorded={old!r} recomputed={new!r}"
    return None


# ------------------------------------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _view_with_switch_category(task, split: str):
    """The task view plus the ``y_switch_category`` descriptor, which the view's metadata does not
    carry: read from each source window set at the same rows, in the same order as the metadata
    (``task._build_view`` concatenates sources and metadata in one loop). Keys are checked."""
    v = task.view(split)
    meta = v.metadata.copy()
    parts, keys = [], []
    for s in v._sources:
        y = s.ws.y
        col = (
            y["y_switch_category"].to_numpy()[s.rows]
            if "y_switch_category" in y.columns
            else np.full(len(s.rows), None, dtype=object)
        )
        parts.append(col)
        keys.append(y[["sample_id", "window_idx"]].to_numpy()[s.rows])
    if parts:
        k = np.concatenate(keys)
        if not (
            np.array_equal(k[:, 0], meta["sample_id"].to_numpy())
            and np.array_equal(k[:, 1], meta["window_idx"].to_numpy())
        ):
            raise RuntimeError(f"{split}: source rows do not line up with the view metadata")
        meta["y_switch_category"] = np.concatenate(parts)
    else:
        meta["y_switch_category"] = pd.Series([], dtype=object)
    return v, meta


def select_baseline(grid: str, test_set: str) -> tuple[str, str]:
    """``(baseline or "", status)`` from the validation-selection kernel over ``outputs/``."""
    import paper_lib as pl

    from make_results import load_rows

    rows = load_rows(grid)
    try:
        c = pl.best_cell_choice(rows, TASK, test_set, HEADLINE, grid=grid)
    except pl.SelectionEvidenceError as exc:
        return "", f"unavailable: {exc}"
    if c is None:
        return "", "unavailable: empty cell or validation winner without a result"
    return c["baseline"], f"selected on {c.get('selected_on', 'val')}"


def score_cell(outputs: Path, grid: str, baseline: str, ts: str, task, v, meta, tol: float):
    """Controls and per-seed rows for one (grid, baseline, test set). Returns
    ``(control_row, rate_rows, confusion_rows, hashed_files)``."""
    td = task.definition
    cdir = outputs / results_version(PROTOCOL, TASK) / TASK / PROTOCOL / grid / baseline
    ident = {"grid": grid, "baseline": baseline, "test_set": ts}
    ctl = {**ident, "status": "ok", "detail": "", "cell_dir": str(cdir)}
    hashed: list[Path] = []
    problems = []
    record = _load_json(cdir / "run_record.json")
    doc = _load_json(cdir / ts / "result.json")
    if record is None:
        problems.append("run_record.json missing or unreadable")
    else:
        hashed.append(cdir / "run_record.json")
        sp = _provenance_matches(task.provenance, record.get("data"))
        if sp:
            problems.append(sp)
    if doc is None:
        ctl.update(status="FAILED", detail=" | ".join([*problems, "result.json missing"]))
        return ctl, [], [], hashed
    hashed.append(cdir / ts / "result.json")
    y_true = np.asarray(v.labels).astype(int)
    seeds = [int(s) for s in doc.get("seeds") or []]
    recorded = {int(d["seed"]): d.get("metrics", {}) for d in doc.get("per_seed") or []}
    ctl.update(
        n_test_recorded=doc.get("n_test"),
        n_view=len(y_true),
        seeds_recorded=" ".join(map(str, seeds)),
        benchmark_version=doc.get("benchmark_version"),
        git_sha=(record or {}).get("git_sha", ""),
    )
    if doc.get("n_test") != len(y_true):
        problems.append(f"n_test {doc.get('n_test')} != view {len(y_true)}")
    if sorted(seeds) != list(SEEDS):
        problems.append(f"seeds {seeds} != {list(SEEDS)}")
    rates, conf, all_agree = [], [], True
    episodes = meta["sample_id"].to_numpy()
    for s in seeds:
        pp = cdir / ts / f"predictions_seed{s}.parquet"
        if not pp.is_file():
            problems.append(f"seed {s}: prediction file missing")
            continue
        hashed.append(pp)
        try:
            raw = align_to_view(v.metadata, pd.read_parquet(pp))
            check_prediction_values(raw, regression=False, n_classes=len(EVENT_STATE_CLASSES))
        except ValueError as exc:
            problems.append(f"seed {s}: {exc}")
            continue
        pred = np.round(np.asarray(raw, dtype=float)).astype(int)
        recomputed = compute_multiclass_metrics(
            y_true, pred, td.metrics, classes=EVENT_STATE_CLASSES
        )
        bad = headline_control(recomputed, recorded.get(s, {}), tol)
        if bad:
            problems.append(f"seed {s}: {bad}")
            continue
        all_agree &= _seed_metrics_agree(recomputed, recorded.get(s, {}), tol) is None
        ctl[f"{HEADLINE}_seed{s}"] = recomputed[HEADLINE]
        for r in rate_rows(meta, y_true, pred):
            rates.append({**ident, "seed": s, **r})
        for r in confusion(y_true, pred):
            conf.append({**ident, "seed": s, **r})
    ctl["all_recorded_metrics_agree"] = all_agree
    ctl["n_episodes"] = int(pd.unique(episodes).size)
    if problems:
        ctl.update(status="FAILED", detail=" | ".join(problems))
        return ctl, [], [], hashed  # a failing cell is reported, never scored
    return ctl, rates, conf, hashed


def _write(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    cols: list[str] = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if _isnan(x) else x) for k, x in r.items()})


def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


# ------------------------------------------------------------------------------------------------
# SI table
# ------------------------------------------------------------------------------------------------

_GRID_LABEL = {
    "double_line": r"Double Line",
    "testgrid_110kv": r"Prot.\ Ref.",
    "cigre_mv": r"CIGR\'E MV",
    "ieee39": r"IEEE 39-Bus",
}
_BASE_LABEL = {
    "random_forest": r"\acs{rf}",
    "mlp": r"\acs{mlp}",
    "gru": r"\acs{gru}",
    "cnn": r"\acs{cnn}",
    "resnet": r"\acs{resnet}",
    "tst": r"\acs{tst}",
}
_SET_LABEL = {"test": "test", "benchmark": r"\texttt{eval\_grid}"}


def _pct(row) -> str:
    if row is None or _isnan(row["mean"]):
        return r"\undefm"
    sd = row["std"]
    sd_s = "" if _isnan(sd) else f"$\\pm${100 * sd:.1f}"
    return f"{100 * row['mean']:.1f}{sd_s}"


def si_table(summary: pd.DataFrame, selected: dict[tuple[str, str], str]) -> str:
    """One LaTeX table: per grid, test set and learned baseline, the three overall rates (seed
    mean $\\pm$ sd, %) with window supports. The validation-selected baseline, where the kernel
    resolves one, is marked with a dagger."""
    s = summary[summary["breakdown"] == "all"].set_index(
        ["grid", "test_set", "baseline", "rate_name"]
    )
    lines = []
    for grid in [g for g in GRIDS if g in set(summary["grid"])]:
        for ts in TEST_SETS:
            if (grid, ts) not in s.index.droplevel([2, 3]):
                continue  # every cell of this set failed its controls
            supports = []
            for name in RATES:
                sub = s.xs((grid, ts), level=("grid", "test_set"))
                sup = sub.xs(name, level="rate_name")["support"]
                supports.append(str(int(sup.iloc[0])) if len(sup) else "--")
            first = True
            for b in LEARNED_BASELINES:
                key = (grid, ts, b)
                cells = []
                for name in RATES:
                    row = s.loc[(*key, name)] if (*key, name) in s.index else None
                    cells.append(_pct(row))
                mark = r"$^\dagger$" if selected.get((grid, ts)) == b else ""
                g_lab = _GRID_LABEL.get(grid, grid) if first and ts == "test" else ""
                ts_lab = (
                    f"{_SET_LABEL[ts]} ({supports[0]}/{supports[1]}/{supports[2]})" if first else ""
                )
                lines.append(
                    f"{g_lab} & {ts_lab} & {_BASE_LABEL[b]}{mark} & " + " & ".join(cells) + r" \\"
                )
                first = False
        lines.append(r"\midrule")
    body = "\n".join(lines[:-1])
    return (
        "% generated by hpc/diag_fault_vs_switching.py (EXP-04); do not edit by hand\n"
        "\\begin{table*}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\caption{Fault-versus-switching diagnostic from the stored event-state "
        "(\\texttt{event\\_state\\_global}, \\texttt{held\\_out}) predictions: (a) fault "
        "predictions among active switching windows, (b) missed faults (predicted normal or "
        "switching) among active fault windows, (c) fault predictions among inactive windows, in "
        "\\% as mean $\\pm$ s.d.\\ over five seeds. Supports in brackets are window counts "
        "(a/b/c); windows overlap and are not independent. Global view, window level; not a "
        "trip-security or relay-selectivity measure. \\undefm{}~= undefined (no support)."
        + (
            " $^\\dagger$~= baseline selected by the validation rule of the headline tables."
            if r"$^\dagger$" in body
            else ""
        )
        + "}\n"
        "\\label{tab:supp_fault_vs_switching}\n"
        "\\begin{tabular}{l l l c c c}\n\\toprule\n"
        "\\textbf{Grid} & \\textbf{Set (support a/b/c)} & \\textbf{Baseline} & "
        "\\textbf{(a) switch$\\to$fault} & \\textbf{(b) missed fault} & "
        "\\textbf{(c) inactive$\\to$fault} \\\\\n\\midrule\n" + body + "\n\\bottomrule\n"
        "\\end{tabular}\n\\end{table*}\n"
    )


# ------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="output directory (created)")
    ap.add_argument("--grids", nargs="*", default=list(GRIDS))
    ap.add_argument("--tol", type=float, default=0.0, help="per-seed headline tolerance")
    ap.add_argument("--root", default=None, help="data root override")
    args = ap.parse_args(argv)
    from evemtbench.tasks.task import load_task

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = Path("outputs")  # the selection kernel (make_results.load_rows) reads this tree too
    (out_dir / "mapping.json").write_text(json.dumps(MAPPING, indent=2) + "\n")

    controls, rates, conf, selection, hashed, provenance = [], [], [], [], [], {}
    selected: dict[tuple[str, str], str] = {}
    for grid in args.grids:
        t0 = time.time()
        for ts in TEST_SETS:
            b, status = select_baseline(grid, ts)
            selection.append({"grid": grid, "test_set": ts, "selected": b, "status": status})
            if b:
                selected[(grid, ts)] = b
        try:
            task = load_task(TASK, PROTOCOL, grid=grid, root=args.root)
        except Exception as exc:
            controls.append(
                {"grid": grid, "status": "FAILED", "detail": f"load_task failed: {exc}"}
            )
            continue
        provenance[grid] = task.provenance
        for ts in TEST_SETS:
            v, meta = _view_with_switch_category(task, ts)
            for b in LEARNED_BASELINES:
                c, r, k, h = score_cell(outputs, grid, b, ts, task, v, meta, args.tol)
                controls.append(c)
                rates += r
                conf += k
                hashed += h
        print(f"{grid}: {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

    per_seed = pd.DataFrame(rates)
    keys = ["grid", "baseline", "test_set", "breakdown", "group", "rate_name"]
    summary = summarise_seeds(per_seed, keys) if len(per_seed) else pd.DataFrame(columns=keys)
    if len(summary):
        cols = (summary["grid"], summary["test_set"], summary["baseline"])
        summary["validation_selected"] = [
            selected.get((g, t)) == b for g, t, b in zip(*cols, strict=True)
        ]

    _write(out_dir / "controls.csv", controls)
    _write(out_dir / "selection.csv", selection)
    _write(out_dir / "confusion_per_seed.csv", conf)
    if conf:
        _write(
            out_dir / "confusion_summary.csv",
            summarise_confusion(pd.DataFrame(conf)).to_dict("records"),
        )
    _write(out_dir / "rates_per_seed.csv", rates)
    _write(out_dir / "rates_summary.csv", summary.to_dict("records"))
    if len(summary):
        (out_dir / "si_table.tex").write_text(si_table(summary, selected))

    repo = Path(__file__).resolve().parents[1]
    (out_dir / "inputs.sha256").write_text(
        "".join(f"{_sha256(f)}  {f}\n" for f in sorted(set(hashed)))
    )

    def _git(*a: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *a], capture_output=True, text=True
        ).stdout.strip()

    failed = [c for c in controls if c.get("status") != "ok"]
    summary_doc = {
        "ticket": "EXP-04",
        "command": "PYTHONPATH=src python "
        + shlex.join([f"hpc/{Path(__file__).name}", *sys.argv[1:]]),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty_paths": _git("status", "--porcelain").splitlines(),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "outputs": str(outputs.resolve()),
        "grids": args.grids,
        "tol": args.tol,
        "cells_checked": len(controls),
        "cells_failed": len(failed),
        "selection": selection,
        "split_provenance": provenance,
        "n_hashed_inputs": len(set(hashed)),
        "versions": {m: sys.modules[m].__version__ for m in ("numpy", "pandas")},
        "python": sys.version.split()[0],
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary_doc, indent=2, sort_keys=True, default=str) + "\n"
    )
    for c in failed:
        print(f"  FAILED {c.get('grid')}/{c.get('baseline', '')}/{c.get('test_set', '')}: "
              f"{c.get('detail')}", file=sys.stderr)  # fmt: skip
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
