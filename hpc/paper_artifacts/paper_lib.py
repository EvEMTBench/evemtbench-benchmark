"""Shared helpers for the per-artifact paper builders (``hpc/paper_artifacts/build_*.py``).

One script → one artifact: each ``build_<x>.py`` imports from here and writes exactly one file
(``paper/manuscript/generated/*.tex`` or ``paper/manuscript/figures/*.pdf``). This module is the
single source of truth for data loading + cell selection, so the number a reviewer reads is provably
the same cell the audit dashboard traces (``best_cell_choice``). The thin shims
``hpc/make_paper_tables.py`` / ``make_supplement_tables.py`` re-export the aggregators
(``build_paper_tables`` / ``build_supplement_tables``) so the public results bundle
(``make_results_bundle.py``) and the freshness gate (``check_results_fresh.py``) keep working
byte-for-byte.

Wire numbers ONLY from a frozen benchmark version tag (see ``paper/AGENTS.md``); with ``outputs/``
absent every table renders ``\\res`` (the pre-freeze placeholder state).
"""

from __future__ import annotations

import sys
from pathlib import Path

# hpc/ is not a package; make the sibling RESULTS builder + the src/ package importable regardless
# of the cwd a standalone build_*.py is launched from.
_HERE = Path(__file__).resolve()
_HPC = _HERE.parents[1]  # hpc/
_REPO = _HERE.parents[2]  # repo root
for _p in (_HPC, _REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from evemtbench.coverage import LEARNED_BASELINES, PRETRAIN_GRID, PRETRAIN_GRIDS  # noqa: E402
from evemtbench.evaluation.metrics import METRIC_DIRECTION  # noqa: E402
from make_results import BASELINES, ci_val, load_rows, mean_val  # noqa: E402

GEN_DIR = _REPO / "paper" / "manuscript" / "generated"
FIG_DIR = _REPO / "paper" / "manuscript" / "figures"

# --------------------------------------------------------------------------------------------------
# Shared configuration (identical to the pre-refactor make_paper_tables / make_supplement_tables).
# --------------------------------------------------------------------------------------------------
GRID_ROWS: list[tuple[str, str]] = [
    ("cigre_mv", r"CIGR\'E MV (20\,kV)"),
    ("double_line", r"Double Line (110\,kV)"),
    ("testgrid_110kv", r"Prot.\ Ref.\ (110\,kV)"),
    ("ieee39", r"IEEE 39-Bus (345\,kV)"),
]
OBS = ("local", "line", "global")
TEST_SETS = ("test", "benchmark")

_BACC = r"balanced accuracy $\uparrow$"
_MAE = r"\acs{mae}\,\% $\downarrow$"
# key -> (key, caption task name, task-id stem, headline metric, metric label w/ arrow).
PAPER_FUNCTIONS: list[tuple[str, str, str, str, str]] = [
    ("fd", r"Fault detection (\ac{fd})", "fault_detection", "balanced_accuracy", _BACC),
    (
        "fc",
        r"Fault-type classification (\ac{fc}, 9-class)",
        "fault_classification",
        "balanced_accuracy",
        _BACC,
    ),
    ("fl", r"Fault localisation (\ac{fl})", "fault_location", "mae", _MAE),
    ("ed", r"Event detection (\ac{ed})", "event_detection", "balanced_accuracy", _BACC),
    ("ec", r"Event classification (\ac{ec})", "event_classification", "balanced_accuracy", _BACC),
]
_PF_BY_KEY = {row[0]: row for row in PAPER_FUNCTIONS}

_BASELINE_LABEL = {
    "majority": "majority (floor)",
    "threshold": "threshold (overcurrent)",
    "random_forest": r"\ac{rf}",
    "mlp": r"\ac{mlp}",
    "gru": r"\ac{gru}",
    "cnn": r"\ac{cnn}",
    "resnet": r"\ac{resnet}",
}
_TS_LABEL = {"test": "in-distribution (test)", "benchmark": "never-trained benchmark family"}

# ---- F10: clip Student-t intervals to a bounded metric's support for display -------------------
# Balanced accuracy, accuracy, and the F1 family live in [0,1]; a symmetric t-interval at n=5 can
# run past the bound (e.g. 0.848 $\pm$ 0.222 -> upper 1.07). MAE/RMSE (% line length) and R^2 are
# NOT bounded to [0,1] and are never clipped.
_BOUNDED_01 = {"balanced_accuracy", "accuracy", "macro_f1"}


def _is_bounded01(metric: str) -> bool:
    return metric in _BOUNDED_01 or metric.startswith("f1_")


def _fmt_mean_ci(mean: float, ci: float | None, metric: str, compact: bool = False) -> str:
    """``mean $\\pm$ ci``; for a [0,1]-bounded metric whose symmetric interval overflows the
    bound, show the truncated interval ``mean [lo, hi]`` instead (F10). ``compact`` (used by the
    12-column per-baseline supplement tables, which otherwise overflow the two-column text width by
    >100 pt) renders the CI half-width as a subscript instead of `` $\\pm$ ci``: bounded metrics keep
    3 d.p., unbounded (MAE/RMSE) drop to 2 d.p.; a half-width that rounds to zero is omitted."""
    if ci is None:
        return f"{mean:.3f}"
    bounded = _is_bounded01(metric)
    if bounded and (mean + ci > 1.0 or mean - ci < 0.0):
        lo, hi = max(0.0, mean - ci), min(1.0, mean + ci)
        if compact:  # narrower 2 d.p. bracket for the dense per-baseline supplement tables
            return f"{mean:.3f}$\\,{{\\scriptscriptstyle[{lo:.2f},{hi:.2f}]}}$"
        return f"{mean:.3f}$\\,{{\\scriptscriptstyle[{lo:.3f},\\,{hi:.3f}]}}$"
    if compact:
        nd = 3 if bounded else 2
        if ci < 0.5 * 10 ** (-nd):  # half-width rounds to zero at nd d.p. -> show mean only
            return f"{mean:.{nd}f}"
        return f"{mean:.{nd}f}$\\,{{\\scriptscriptstyle\\pm{ci:.{nd}f}}}$"
    if ci < 0.5e-3:  # half-width rounds to 0.000 -> show mean only (seeds agree; no "± 0.000")
        return f"{mean:.3f}"
    return f"{mean:.3f} $\\pm$ {ci:.3f}"


# ---- F08: five-state evidence markers (P / n/a / U / ---) instead of a single \res ------------
# Event-scope functions (ED, EC) are grid-wide, so they are defined only at the *global* view;
# the overcurrent threshold baseline is defined only for binary FD at *local*. A cell that is not
# structurally defined is \notapp (n/a); a defined cell with no result yet is \pending (P); a
# fixed-width transfer cell at the grid-dependent *global* width is out of scope (\oos, ---).
_EVENT_SCOPE_STEMS = {"event_detection", "event_classification"}


def _held_unsupported(stem: str, view: str, baseline: str | None = None) -> bool:
    """True if the (stem, view[, baseline]) cell is not defined by the contract (renders n/a)."""
    if stem in _EVENT_SCOPE_STEMS and view != "global":
        return True
    if baseline == "threshold" and not (stem == "fault_detection" and view == "local"):
        return True
    return False


# --------------------------------------------------------------------------------------------------
# Cell selection (single source of truth, shared with the audit dashboard's paper-numbers page).
# --------------------------------------------------------------------------------------------------
def _argbest_baseline(metric: str, doc_of) -> str | None:
    """The winning ``LEARNED_BASELINES`` entry by a per-baseline score, or ``None`` if none scored.

    ``doc_of(baseline)`` returns that baseline's result doc (or ``None``); the winner is the argmax
    of ``mean_val(doc, metric)`` in METRIC_DIRECTION (lower-is-better metrics pick the minimum), ties
    keeping the earlier ``LEARNED_BASELINES`` entry (strict ``>``). Shared kernel for the F06
    validation-selection (``best_cell_choice`` / ``best_transfer_row``) and the legacy
    select-on-reported-set fallback."""
    direction = METRIC_DIRECTION.get(metric, 1)
    sel_base, sel_val = None, None
    for baseline in LEARNED_BASELINES:
        v = mean_val(doc_of(baseline), metric)
        if v is None:
            continue
        if sel_val is None or direction * (v - sel_val) > 0:
            sel_val, sel_base = v, baseline
    return sel_base


def best_cell_choice(rows: dict, task_id: str, test_set: str, metric: str) -> dict | None:
    """The reported learned baseline for (task_id, test_set), or ``None`` if no result exists.

    F06: the reported baseline is selected on the **validation** split (``rows[...]["val"]``, the
    offline re-score written by ``hpc/rescore_validation.py``) and then its value on the requested
    ``test_set`` is reported. Selecting on validation rather than on the very test set being
    reported removes the winner's-curse (selection-on-test) bias. Direction from METRIC_DIRECTION
    (lower-is-better metrics like MAE pick the minimum); ties keep the earlier LEARNED_BASELINES
    entry (strict ``>``). Falls back to legacy selection-on-``test_set`` when no ``val`` doc exists
    for the cell (e.g. protocols not re-scored), so the function degrades gracefully. Returns
    ``{baseline, mean, ci, n_seeds, selected_on}``.

    ``rows`` is ``load_rows(grid)`` = ``{(task_id, baseline): {test_set: doc}}``."""
    # 1) select the winner by its validation-split score
    sel_base = _argbest_baseline(metric, lambda b: rows.get((task_id, b), {}).get("val"))
    if sel_base is not None:
        # 2) report the validation-selected baseline's value on the requested test_set
        rep = rows.get((task_id, sel_base), {}).get(test_set)
        mean = mean_val(rep, metric)
        if mean is None:
            return None  # selected baseline has no result on this test set yet
        return {
            "baseline": sel_base,
            "mean": mean,
            "ci": ci_val(rep, metric),
            "n_seeds": len((rep or {}).get("seeds") or []),
            "selected_on": "val",
        }
    # 3) fallback: no validation re-score for this cell -> legacy selection on the reported set
    best_base = _argbest_baseline(metric, lambda b: rows.get((task_id, b), {}).get(test_set))
    best_doc = rows.get((task_id, best_base), {}).get(test_set) if best_base is not None else None
    if best_doc is None:
        return None
    return {
        "baseline": best_base,
        "mean": mean_val(best_doc, metric),
        "ci": ci_val(best_doc, metric),
        "n_seeds": len(best_doc.get("seeds") or []),
        "selected_on": "test",
    }


def _best_cell(rows: dict, stem: str, view: str, test_set: str, metric: str) -> str:
    """Best learned baseline for (stem, view, test_set): a clipped ``mean $\\pm$ ci`` value, or
    a five-state marker --- ``\\notapp`` (view not defined for this function) / ``\\pending``
    (defined but no result yet)."""
    if _held_unsupported(stem, view):
        return r"\notapp"
    choice = best_cell_choice(rows, f"{stem}_{view}", test_set, metric)
    if choice is None:
        return r"\pending"
    return _fmt_mean_ci(choice["mean"], choice["ci"], metric)


def _supp_cell(rows: dict, stem: str, view: str, baseline: str, test_set: str, metric: str) -> str:
    if _held_unsupported(stem, view, baseline):
        return r"\notapp"
    doc = rows.get((f"{stem}_{view}", baseline), {}).get(test_set)
    mean = mean_val(doc, metric)
    if mean is None:
        return r"\pending"
    # compact CI rendering keeps the 12-column per-baseline supplement tables within the two-column
    # text width (they otherwise overflow it by >100 pt for every primary function).
    return _fmt_mean_ci(mean, ci_val(doc, metric), metric, compact=True)


# --------------------------------------------------------------------------------------------------
# Main result tables (held_out) — verbatim layout of the pre-refactor make_paper_tables._table.
# --------------------------------------------------------------------------------------------------
def result_table(key: str) -> str:
    """Full LaTeX table for one paper function (rows = grids, cols = obs × held_out test-set)."""
    _, name, stem, metric, metric_label = _PF_BY_KEY[key]
    per_grid = {g: load_rows(g) for g, _ in GRID_ROWS}
    body_lines = []
    for grid, label in GRID_ROWS:
        rows = per_grid[grid]
        # columns = observability × test-set; ED/EC have only a *_global task id, so their
        # local/line cells find no (task_id, baseline) key and render \res.
        cells = [_best_cell(rows, stem, obs, ts, metric) for ts in TEST_SETS for obs in OBS]
        body_lines.append(f"{label:<26} & " + " & ".join(cells) + r" \\")
    body = "\n".join(body_lines)
    # The [0,1]-truncation note only applies to bounded metrics (balanced accuracy); FL is MAE %.
    ci_note = (
        "an interval exceeding the metric's $[0,1]$ range is shown truncated as "
        "$[\\text{lo},\\text{hi}]$. "
        if _is_bounded01(metric)
        else ""
    )
    return (
        "% AUTO-GENERATED by hpc/make_paper_tables.py from outputs/ — do not edit by hand.\n"
        "\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4.5pt}\n"
        f"\\caption{{{name} --- reference baseline (best of RF/MLP/GRU/CNN/ResNet, "
        "selected on the validation split; a beatable reference, full per-baseline breakdown "
        "in the Supplementary), headline metric "
        f"({metric_label}) per grid and observability view, for the two \\texttt{{held\\_out}} "
        "test sets (in-distribution vs.\\ never-trained benchmark family). Cells are "
        "mean\\,$\\pm$\\,95\\% CI (Student's $t$); " + ci_note + "\\pending{}~= pending (no result "
        "yet); \\notapp{}~= not defined for this view. Full per-baseline tables in the "
        "Supplementary.}\n"
        f"\\label{{tab:res_{key}}}\n"
        "\\begin{tabular}{l ccc ccc}\n\\toprule\n"
        " & \\multicolumn{3}{c}{\\textbf{In-distribution (test)}} & "
        "\\multicolumn{3}{c}{\\textbf{Benchmark (transfer)}} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
        "\\textbf{Grid} & \\textbf{local} & \\textbf{line} & \\textbf{global} & "
        "\\textbf{local} & \\textbf{line} & \\textbf{global} \\\\\n\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )


# --------------------------------------------------------------------------------------------------
# Supplement tables — verbatim layout of the pre-refactor make_supplement_tables._table.
# --------------------------------------------------------------------------------------------------
def supp_table(key: str, test_set: str) -> str:
    _, name, stem, metric, metric_label = _PF_BY_KEY[key]
    per_grid = {g: load_rows(g) for g, _ in GRID_ROWS}
    grid_heads = " & ".join(
        f"\\multicolumn{{3}}{{c}}{{\\textbf{{{label}}}}}" for _, label in GRID_ROWS
    )
    cmids = "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(GRID_ROWS)))
    obs_heads = " & ".join(["\\textbf{loc} & \\textbf{lin} & \\textbf{glo}"] * len(GRID_ROWS))
    body_lines = []
    for baseline in BASELINES:
        cells = [
            _supp_cell(per_grid[grid], stem, obs, baseline, test_set, metric)
            for grid, _ in GRID_ROWS
            for obs in OBS
        ]
        body_lines.append(f"{_BASELINE_LABEL[baseline]:<28} & " + " & ".join(cells) + r" \\")
    body = "\n".join(body_lines)
    ci_note = (
        "the 95\\% CI half-width is shown as a compact subscript ($\\pm h$); an interval that would "
        "exceed $[0,1]$ is instead shown truncated as $[\\text{lo},\\text{hi}]$"
        if _is_bounded01(metric)
        else "the 95\\% CI half-width is shown as a compact subscript ($\\pm h$) at 2 d.p"
    )
    return (
        "% AUTO-GENERATED by hpc/make_supplement_tables.py from outputs/ — do not edit by hand.\n"
        "\\begin{table*}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{2.6pt}\n"
        f"\\caption{{{name} --- full per-baseline results on the {_TS_LABEL[test_set]} "
        f"(\\texttt{{held\\_out}} protocol), headline metric ({metric_label}). Cells are "
        f"mean\\,$\\pm$\\,95\\% CI over seeds (Student's $t$); {ci_note}. \\pending{{}}~= pending; "
        "\\notapp{}~= not defined for this (function, view) or baseline.}\n"
        f"\\label{{tab:supp_{key}_{test_set}}}\n"
        # scale down to the text width only if the natural table is wider (graphicx; no-op otherwise)
        "\\resizebox{\\ifdim\\width>\\textwidth \\textwidth\\else \\width\\fi}{!}{%\n"
        f"\\begin{{tabular}}{{l {'ccc ' * len(GRID_ROWS)}}}\n\\toprule\n"
        f" & {grid_heads} \\\\\n{cmids}\n"
        f"\\textbf{{Baseline}} & {obs_heads} \\\\\n\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n\\end{tabular}}\n\\end{table*}\n"
    )


# --------------------------------------------------------------------------------------------------
# Cross-grid transfer tables (NEW) — mirror the held_out layout but cols = obs × {zero-shot,
# fine-tune} on the never-trained benchmark family. Data source: the same scan the leaderboard's
# transfer section uses (leaderboard.scan_results), version-scoped, best learned baseline per cell.
# --------------------------------------------------------------------------------------------------
_TRANSFER_PROTOS = [("transfer_zeroshot", "Zero-shot"), ("transfer_finetune", "Fine-tune")]


def _source_short(source: str) -> str:
    r"""LaTeX short label for a transfer pre-training SOURCE grid: ``template_20kv`` -> ``20\,kV``."""
    kv = source.replace("template_", "").replace("kv", "")
    return f"{kv}\\,kV"


def _transfer_index(outputs: str | Path = "outputs") -> dict:
    """``{(task_id, grid, protocol, source): [Row, ...]}`` for the version's transfer runs on
    benchmark. ``source`` is the pre-training grid the transfer was seeded from (``Row.source``;
    legacy source-less results default to ``PRETRAIN_GRID`` = the 20 kV template)."""
    from evemtbench.leaderboard import scan_results
    from evemtbench.version import BENCHMARK_VERSION

    idx: dict = {}
    for r in scan_results(outputs):
        if r.version != BENCHMARK_VERSION:
            continue
        if r.protocol not in ("transfer_zeroshot", "transfer_finetune"):
            continue
        if r.test_set != "benchmark":
            continue
        src = r.source or PRETRAIN_GRID
        idx.setdefault((r.task, r.grid, r.protocol, src), []).append(r)
    return idx


def _transfer_val_index(outputs: str | Path = "outputs") -> dict:
    """Validation-split docs from the offline re-score, source-aware. Keys:

    * zero-shot selection: ``("multi_grid_pretrain", source, task, baseline)`` — the pre-training
      corpus val of the SOURCE grid (grid == source for multi_grid_pretrain);
    * fine-tune selection: ``("transfer_finetune", target, task, baseline, source)`` — the fine-tune
      val on the target grid per source (legacy source-less path = the 20 kV default; a
      ``pretrain-<src>`` subdir = that source).

    Missing keys degrade to legacy select-on-benchmark (val artifacts under ``**/val/`` are
    gitignored). Produced by ``hpc/rescore_validation.py``."""
    import glob
    import json

    from evemtbench.version import BENCHMARK_VERSION

    idx: dict = {}
    base = f"{outputs}/{BENCHMARK_VERSION}"
    for f in glob.glob(f"{base}/*/multi_grid_pretrain/*/*/val/result.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        idx[("multi_grid_pretrain", d["grid"], d["task"], d["baseline"])] = d
    # transfer_finetune val: legacy source-less path (=> 20 kV) and source-tokened (pretrain-<src>)
    for f in glob.glob(f"{base}/*/transfer_finetune/*/*/val/result.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        idx[("transfer_finetune", d["grid"], d["task"], d["baseline"], PRETRAIN_GRID)] = d
    for f in glob.glob(f"{base}/*/transfer_finetune/*/*/pretrain-*/val/result.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        src = Path(f).parent.parent.name.replace("pretrain-", "")  # .../pretrain-<src>/val/…
        idx[("transfer_finetune", d["grid"], d["task"], d["baseline"], src)] = d
    return idx


def best_transfer_row(
    idx: dict,
    task_id: str,
    grid: str,
    protocol: str,
    metric: str,
    source: str,
    val_idx: dict | None = None,
):
    """Winning learned baseline Row for a transfer cell seeded from pre-training ``source`` (or
    ``None``). F06 consistency: the winner is selected on a VALIDATION signal, never on the benchmark
    it reports — ``transfer_zeroshot`` on the ``source``'s ``multi_grid`` pretrain val
    (grid-independent: one pretrained model applied to every target), ``transfer_finetune`` on that
    source's fine-tune val on the target grid. Falls back to legacy select-on-benchmark when the
    relevant val re-score is absent (e.g. the 110/345 kV fine-tune val is not yet scored)."""
    direction = METRIC_DIRECTION.get(metric, 1)
    bench = {
        r.baseline: r
        for r in idx.get((task_id, grid, protocol, source), [])
        if r.baseline in LEARNED_BASELINES and r.mean(metric) is not None
    }
    if val_idx is None:
        val_idx = _transfer_val_index()
    if protocol == "transfer_zeroshot":
        sel_base = _argbest_baseline(
            metric, lambda b: val_idx.get(("multi_grid_pretrain", source, task_id, b))
        )
    elif protocol == "transfer_finetune":
        sel_base = _argbest_baseline(
            metric, lambda b: val_idx.get(("transfer_finetune", grid, task_id, b, source))
        )
    else:
        sel_base = None
    if sel_base is not None:
        return bench.get(sel_base)  # None -> \pending if the winner lacks a benchmark cell here
    # fallback: no val re-score for this (source, protocol) -> legacy selection on the benchmark set
    if not bench:
        return None
    return max(bench.values(), key=lambda r: direction * r.mean(metric))


def _transfer_cell(
    idx: dict,
    stem: str,
    view: str,
    grid: str,
    protocol: str,
    metric: str,
    source: str,
    val_idx: dict | None = None,
) -> str:
    if view == "global":
        # fixed-width transfer cannot ingest the grid-dependent global channel count -> out of scope
        return r"\oos"
    best = best_transfer_row(idx, f"{stem}_{view}", grid, protocol, metric, source, val_idx)
    if best is None:
        return r"\pending"
    ci = (best.metrics.get(metric) or {}).get("ci95")
    return _fmt_mean_ci(best.mean(metric), ci, metric)


def transfer_table(
    key: str,
    outputs: str | Path = "outputs",
    idx: dict | None = None,
    val_idx: dict | None = None,
) -> str:
    """Full LaTeX transfer table for one paper function (rows = grids, cols = {zero-shot,
    fine-tune} × obs on the benchmark family). ``idx``/``val_idx`` may be prebuilt and shared across
    the five functions (see ``build_transfer_tables``); each is built from ``outputs`` when omitted."""
    _, name, stem, metric, metric_label = _PF_BY_KEY[key]
    if idx is None:
        idx = _transfer_index(outputs)
    if val_idx is None:
        val_idx = _transfer_val_index(outputs)
    body_lines = []
    for gi, (grid, label) in enumerate(GRID_ROWS):
        for si, source in enumerate(PRETRAIN_GRIDS):  # source axis: 20/110/345 kV pre-training
            cells = [
                _transfer_cell(idx, stem, obs, grid, proto, metric, source, val_idx)
                for proto, _ in _TRANSFER_PROTOS
                for obs in OBS
            ]
            gcol = label if si == 0 else ""  # grid label only on the first source row
            body_lines.append(
                f"{gcol:<22} & {_source_short(source):<8} & " + " & ".join(cells) + r" \\"
            )
        if gi < len(GRID_ROWS) - 1:
            body_lines.append(r"\midrule")
    body = "\n".join(body_lines)
    header = (
        f"% AUTO-GENERATED by hpc/paper_artifacts/build_tab_transfer_{key}.py from outputs/ "
        "— do not edit by hand.\n"
    )
    return (
        header + "\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4.5pt}\n"
        f"\\caption{{{name} --- cross-grid transfer reference baseline (best of "
        "MLP/GRU/CNN/ResNet, selected on the validation split, never on the reported benchmark "
        "set: \\emph{fine-tune} on the target grid's validation split, \\emph{zero-shot} on the "
        "pre-training source's \\texttt{multi\\_grid} validation --- so the zero-shot choice is "
        f"grid-independent), headline metric ({metric_label}), pre-trained on each "
        "\\texttt{multi\\_grid} source (20/110/345\\,kV) then \\emph{zero-shot} vs.\\ "
        "\\emph{fine-tuned} on each target grid, scored on the never-trained benchmark family per "
        "observability view. Cells are mean\\,$\\pm$\\,95\\% CI; \\pending{}~= pending (no result "
        "yet); \\oos{}~= out of scope (fixed-width transfer cannot use the grid-dependent "
        "\\emph{global} width).}\n"
        f"\\label{{tab:transfer_{key}}}\n"
        "\\begin{tabular}{ll ccc ccc}\n\\toprule\n"
        " & & \\multicolumn{3}{c}{\\textbf{Zero-shot}} & "
        "\\multicolumn{3}{c}{\\textbf{Fine-tune}} \\\\\n"
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
        "\\textbf{Grid} & \\textbf{Src} & \\textbf{local} & \\textbf{line} & \\textbf{global} & "
        "\\textbf{local} & \\textbf{line} & \\textbf{global} \\\\\n\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )


# --------------------------------------------------------------------------------------------------
# Static specification tables (authored content lifted verbatim out of main.tex). These are NOT
# data-derived, so they are exempt from the pre-freeze \res policy (see check_results_fresh.py).
# --------------------------------------------------------------------------------------------------
_SPEC_HEADER = "% AUTO-GENERATED by hpc/paper_artifacts/build_tab_{name}.py — static spec table.\n"

SPEC_TABLES: dict[str, str] = {
    "task_spec": r"""\begin{table*}[t]
\centering
\caption{Benchmark task specification. Each function is scored under each
observability view (Table~\ref{tab:observability}); a window is scored by exactly one
rule per task. FC uses a frozen 9-class fault vocabulary.}
\label{tab:task_spec}
\begin{tabular}{L{2.2cm} L{2.0cm} L{4.6cm} L{4.4cm}}
\toprule
\textbf{Function} & \textbf{Label kind} & \textbf{Valid samples (windows scored)} & \textbf{Primary metrics} \\
\midrule
\ac{fd} & binary & all windows of fault episodes: pre-fault negatives + fault-active positives & balanced accuracy, macro $\text{F}_1$, missed-fault rate, false-alarm rate \\
\ac{fc} & 9-class & fault-active windows whose type is in the 9-class vocabulary & balanced accuracy, macro / per-class $\text{F}_1$ \\
\ac{fl} & regression & fault-active windows of \emph{line}-fault episodes (location defined; bus faults excluded) & \acs{mae}, \acs{rmse}, $R^2$ \\
\ac{ed} & binary & every window: fault- or switching-active positives vs.\ quiescent negatives & balanced accuracy, missed-detection rate, false-trip rate; detection latency \\
\ac{ec} & 23-class & every window over the full event-type vocabulary (non-active windows labelled \texttt{no\_event}; out-of-vocabulary types excluded) & balanced accuracy, macro / per-class $\text{F}_1$ \\
\bottomrule
\end{tabular}
\end{table*}
""",
    "task_suite": r"""\begin{table*}[t]
\centering
\small
\caption{The full scored task suite (benchmark v1.0.0): twelve
protection/event-analysis functions instantiated across the observability views,
giving \emph{24 scored (function, view) tasks} in three difficulty tiers. The
five \emph{primary} functions foregrounded in the main text are marked~$\dagger$; the
remaining seven are scored diagnostics reported in the Supplementary and the online
leaderboard. Views: L~= local, Li~= line, G~= global. Event-scope functions are
grid-wide (global-only) by construction, because a benign or quiescent window has no
faulted line on which to centre the local/line oracle. Classification tasks report
balanced accuracy as the headline metric; \ac{fl} reports \acs{mae} in \% of line
length.}
\label{tab:task_suite}
\begin{tabular}{L{4.4cm} c L{2.4cm} c L{4.0cm}}
\toprule
\textbf{Function} & \textbf{Tier} & \textbf{Label kind} & \textbf{Views} & \textbf{Interpretation} \\
\midrule
Fault detection (\ac{fd})\,$\dagger$      & 1 & binary      & L, Li, G & fault-active vs.\ pre-fault window \\
Event detection (\ac{ed})\,$\dagger$      & 1 & binary      & G        & any active event vs.\ quiescent \\
Switch detection                          & 1 & binary      & G        & benign switching vs.\ else \\
Fault grounded                            & 1 & binary      & L, Li, G & ground element engaged? \\
\midrule
Event state                               & 2 & 3-class     & G        & normal / switching / fault \\
Fault-type classification (\ac{fc})\,$\dagger$ & 2 & 9-class & L, Li, G & short-circuit / incipient / \ac{hif} subtype \\
Faulted-phase identification              & 2 & 7-class     & L, Li, G & A\,/\,B\,/\,C\,/\,AB\,/\,AC\,/\,BC\,/\,ABC \\
Fault category                            & 2 & 3-class     & L, Li, G & short-circuit / incipient / \ac{hif} \\
Switch type                               & 2 & 8-class     & G        & which benign switching event \\
Fault origin                              & 2 & binary      & G        & line vs.\ busbar fault \\
\midrule
Event classification (\ac{ec})\,$\dagger$ & 3 & 23-class    & G        & full event-type taxonomy \\
Fault localisation (\ac{fl})\,$\dagger$   & 3 & regression  & L, Li, G & position along the faulted line \\
\bottomrule
\end{tabular}
\end{table*}
""",
    "topology_overview": r"""\begin{table}[t]
\caption{Reference grids used to instantiate the benchmark
\cite{kordowich_evemtbench_2026}. The \emph{global} view width (number of channels)
is grid-dependent, which motivates topology-agnostic models for the planned
cross-topology protocol.}
\label{tab:topology_overview}
\centering
\begin{tabular}{lccc}
\toprule
\textbf{Grid} & \textbf{Voltage} & \textbf{$f_0$} & \textbf{\emph{global} width} \\
\midrule
CIGR\'E MV Grid          & 20\,kV  & 50\,Hz & 174\,ch \\
Double Line Grid         & 110\,kV & 50\,Hz & 48\,ch \\
Protection Reference Grid& 110\,kV & 50\,Hz & 108\,ch \\
IEEE 39-Bus System       & 345\,kV & 60\,Hz & 420\,ch \\
\bottomrule
\end{tabular}
\end{table}
""",
    "dataset_families": r"""\begin{table*}[t]
\caption{Dataset families as a generalisation ladder (dataset-paper name; benchmark code
handle in \texttt{typewriter}; \texttt{train} and \texttt{fine\_tune} are accepted as
legacy read-aliases of \texttt{multi\_grid} and \texttt{adapt\_grid}). Rows increase in
what is held fixed between training and test. The Benchmark family is
\emph{evaluation-only}: it is never used for training, validation, model selection,
early stopping, normalisation, or threshold/hyperparameter tuning.}
\label{tab:dataset_families}
\centering
\begin{tabular}{L{3.4cm} L{5.2cm} L{5.4cm}}
\toprule
\textbf{Family (handle)} & \textbf{Randomised / fixed} & \textbf{Generalisation role} \\
\midrule
Train (\texttt{multi\_grid}) & all three parameter classes; diverse random topologies & cross-topology training + evaluation \planned \\
Fine-tune (\texttt{adapt\_grid}) & fixed topology; random events + operating states & in-grid training + in-distribution test (\texttt{held\_out}) \\
Benchmark (\texttt{benchmark}) & fixed topology + operating point; discrete event-parameter sweep & never-trained systematic transfer evaluation \\
\bottomrule
\end{tabular}
\end{table*}
""",
    "observability": r"""\begin{table}[t]
\centering
\caption{Observability views (nested oracle on the faulted line).}
\label{tab:observability}
\begin{tabular}{lll}
\toprule
\textbf{View} & \textbf{Channels} & \textbf{Protection analogue} \\
\midrule
local  & 6  & single-ended relay (near terminal) \\
line   & 12 & two-terminal current-differential \\
global & grid-dependent & centralised / wide-area \\
\bottomrule
\end{tabular}
\end{table}
""",
    "baselines": r"""\begin{table*}[t]
\centering
\small
\caption{Reference baselines: a trivial floor, a conventional-protection anchor,
classical \ac{ml}, and deep learning. The \ac{gnn} is planned for the cross-topology
protocol.}
\label{tab:baselines}
\begin{tabular}{L{2.2cm} L{2.8cm} L{3.0cm} L{4.6cm}}
\toprule
\textbf{Baseline} & \textbf{Family} & \textbf{Input} & \textbf{Notes} \\
\midrule
majority & trivial floor & labels only & per-task chance level \\
threshold & conventional (overcurrent) & local current & binary \ac{fd}, \emph{local} only; the conventional-protection anchor \\
\ac{rf} & classical \ac{ml} & per-channel summary features & tree-based ensemble \\
\ac{mlp} & deep (flattened window) & raw window & no hand-crafted features \\
\ac{gru} & deep (sequence) & raw window & sequence model \\
\ac{cnn} & deep (1-D waveform) & raw window & waveform model \\
\ac{resnet} & deep (1-D residual) & raw window & residual waveform model \\
\ac{gnn} \planned & deep (topology-aware) & grid graph & cross-topology contender \\
\bottomrule
\end{tabular}
\end{table*}
""",
}


def spec_table(name: str) -> str:
    """The static specification table ``name`` with a generated-by header (LaTeX comment)."""
    return _SPEC_HEADER.format(name=name) + SPEC_TABLES[name]


# --------------------------------------------------------------------------------------------------
# F03: protection-grade error rates + per-class F1 (the "protection-aware metrics" contribution,
# reported for the validation-selected reference baseline, from the same result.json aggregates).
# --------------------------------------------------------------------------------------------------
# binary function -> (dependability = missed-rate key, security = false-alarm-rate key)
_BINARY_RATE_KEYS = {
    "fault_detection": ("missed_fault_rate", "false_alarm_rate"),
    "event_detection": ("missed_event_rate", "false_event_trip_rate"),
}
# FC 9-class vocabulary: (per-class f1 aggregate key, compact column label). Frozen order.
_FC_F1 = [
    ("f1_flt_1phg_shc", "1pg-SHC"),
    ("f1_flt_1phg_shc_w_arc", "1pg-SHC-arc"),
    ("f1_flt_2ph_shc", "2p-SHC"),
    ("f1_flt_2phg_shc", "2pg-SHC"),
    ("f1_flt_3ph_shc", "3p-SHC"),
    ("f1_flt_1phg_incipient", "1pg-inc"),
    ("f1_flt_1phg_incipient_w_arc", "1pg-inc-arc"),
    ("f1_flt_1phg_hif", "1pg-HIF"),
    ("f1_flt_1phg_hif_w_arc", "1pg-HIF-arc"),
]


def _sel_metric_cell(rows: dict, task_id: str, test_set: str, metric: str) -> str:
    """The validation-selected reference baseline's value of ``metric`` on ``test_set`` (the same
    baseline chosen for the headline), as clipped mean$\\pm$ci; ``\\pending``/``\\undefm`` markers
    otherwise. The headline selection uses balanced accuracy; here we read a secondary metric of
    that selected baseline."""
    choice = best_cell_choice(rows, task_id, test_set, "balanced_accuracy")
    if choice is None:
        return r"\pending"
    doc = rows.get((task_id, choice["baseline"]), {}).get(test_set)
    m = mean_val(doc, metric)
    if m is None:
        return r"\undefm"  # metric undefined for this cell (e.g. class absent) -- never zero
    return _fmt_mean_ci(m, ci_val(doc, metric), metric)


def protection_rate_table() -> str:
    """SI table: dependability (missed-fault/-event rate) and security (false-alarm/-trip rate) for
    the binary functions FD and ED, validation-selected reference, per (function, view, grid) on the
    in-distribution and never-trained sets."""
    per_grid = {g: load_rows(g) for g, _ in GRID_ROWS}
    body = []
    specs = [("FD", "fault_detection", OBS), ("ED", "event_detection", ("global",))]
    for flabel, stem, views in specs:
        mk, fk = _BINARY_RATE_KEYS[stem]
        for v in views:
            for g, glabel in GRID_ROWS:
                r = per_grid[g]
                tid = f"{stem}_{v}"
                cells = [_sel_metric_cell(r, tid, ts, key) for ts in TEST_SETS for key in (mk, fk)]
                body.append(f"{flabel}\\,·\\,{v} & {glabel} & " + " & ".join(cells) + r" \\")
    return (
        "% AUTO-GENERATED by hpc/paper_artifacts/build_supp_tab_protection.py — do not edit.\n"
        "\\begin{table*}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\caption{Protection-grade error rates for the binary functions (FD, ED), "
        "validation-selected reference baseline: dependability = missed-fault/-event rate "
        "($\\downarrow$), security = false-alarm/-trip rate ($\\downarrow$). Cells are "
        "mean\\,$\\pm$\\,95\\% CI over seeds; \\pending{}~= pending.}\n"
        "\\label{tab:supp_protection}\n"
        "\\begin{tabular}{l l cc cc}\n\\toprule\n"
        " & & \\multicolumn{2}{c}{\\textbf{In-distribution (test)}} & "
        "\\multicolumn{2}{c}{\\textbf{Never-trained (benchmark)}} \\\\\n"
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
        "\\textbf{Function·view} & \\textbf{Grid} & \\textbf{missed} & \\textbf{false-alarm} & "
        "\\textbf{missed} & \\textbf{false-alarm} \\\\\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )


def fc_per_class_f1_table() -> str:
    """SI table: per-class F1 for fault-type classification (FC, 9-class), validation-selected
    reference, in-distribution test set, per grid. Absent classes (e.g. HIF off MV) render \\undefm
    (undefined, not zero). The wider EC per-class breakdown is in the leaderboard."""
    per_grid = {g: load_rows(g) for g, _ in GRID_ROWS}
    heads = " & ".join(f"\\textbf{{{lab}}}" for _g, lab in GRID_ROWS)
    body = []
    for key, clabel in _FC_F1:
        # rows = class, columns = grid (narrow); line view, in-distribution test
        cells = [
            _sel_metric_cell(per_grid[g], "fault_classification_line", "test", key)
            for g, _ in GRID_ROWS
        ]
        body.append(f"{clabel} & " + " & ".join(cells) + r" \\")
    return (
        "% AUTO-GENERATED by hpc/paper_artifacts/build_supp_tab_fc_perclass.py — do not edit.\n"
        "\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{5pt}\n"
        "\\caption{Per-class $\\text{F}_1$ for fault-type classification (FC, 9-class, "
        "\\emph{line} view, in-distribution test), validation-selected reference baseline. "
        "\\undefm{}~= $\\text{F}_1$ undefined (no supported \\emph{and} no predicted samples); "
        "the two HIF subtypes are \\ac{mv}-only, so their per-grid support (Table~"
        "\\ref{tab:supp_fc_support}) is zero on the \\ac{hv}/\\ac{ehv} grids. Cells are "
        "mean\\,$\\pm$\\,95\\% CI over seeds.}\n"
        "\\label{tab:supp_fc_perclass}\n"
        "\\begin{tabular}{l cccc}\n\\toprule\n"
        f"\\textbf{{Fault class}} & {heads} \\\\\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


# --------------------------------------------------------------------------------------------------
# Aggregators (preserve the entry points make_results_bundle.py + check_results_fresh.py import).
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# F13: dataset composition (episode counts per grid x family; per-class FC support). Split counts are
# repo-derived (committed split files); per-class support is read from the result.json breakdowns.
# --------------------------------------------------------------------------------------------------
def _episode_counts(grid: str) -> dict:
    base = _REPO / "src" / "evemtbench" / "splits" / "v1.0" / "held_out" / grid
    out = {}
    for split in ("train", "val", "test", "benchmark"):
        f = base / f"{split}.txt"
        out[split] = (
            sum(1 for line in f.read_text().splitlines() if line.strip()) if f.is_file() else None
        )
    return out


def dataset_split_table() -> str:
    """SI table: episode counts per grid for the held_out split families (train/val/test from the
    Fine-tune/adapt_grid family, plus the never-trained Benchmark family)."""
    body = []
    for g, glabel in GRID_ROWS:
        c = _episode_counts(g)
        cells = [
            (f"{c[s]:,}" if c[s] is not None else r"\pending")
            for s in ("train", "val", "test", "benchmark")
        ]
        body.append(f"{glabel} & " + " & ".join(cells) + r" \\")
    return (
        "% AUTO-GENERATED by hpc/paper_artifacts/build_supp_tab_splits.py — do not edit.\n"
        "\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{6pt}\n"
        "\\caption{Held-out split composition: number of \\emph{episodes} per grid (each episode "
        "yields $91$ windows at $50\\,\\text{ms}$ / $5\\,\\text{ms}$). Train/validation/test "
        "are the $70/15/15$ episode-level split of the Fine-tune (adapt-grid) family; Benchmark is "
        "the entire never-trained family.}\n"
        "\\label{tab:supp_splits}\n"
        "\\begin{tabular}{l rrrr}\n\\toprule\n"
        "\\textbf{Grid} & \\textbf{train} & \\textbf{val} & \\textbf{test} & \\textbf{benchmark} \\\\\n"
        "\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def _fc_class_support(grid: str, test_set: str = "test") -> dict:
    """{FC class name -> support count n} on the eval set, from any FC cell's event_type breakdown
    (support is a property of the set, not the model)."""
    import glob
    import json

    from evemtbench.version import BENCHMARK_VERSION

    pat = f"outputs/{BENCHMARK_VERSION}/fault_classification_line/held_out/{grid}/*/{test_set}/result.json"
    for f in sorted(glob.glob(pat)):
        try:
            bd = json.load(open(f)).get("breakdowns", {}).get("event_type", {})
        except Exception:
            continue
        if bd:
            return {name: (bd.get(name) or {}).get("n") for _k, name in _FC_SUPPORT_CLASSES}
    return {}


# (per-class f1 key -> event_type breakdown class name) = the key without the f1_ prefix
_FC_SUPPORT_CLASSES = [(k, k[len("f1_") :]) for k, _lab in _FC_F1]


def fc_support_table() -> str:
    """SI table: per-class support (window count) for the 9 FC classes per grid (in-distribution
    test), so that a per-class $F_1$ of \\undefm{} can be read as genuine zero support."""
    per_grid = {g: _fc_class_support(g, "test") for g, _ in GRID_ROWS}
    heads = " & ".join(f"\\textbf{{{lab}}}" for _g, lab in GRID_ROWS)
    body = []
    for (_key, cname), (_k2, clabel) in zip(_FC_SUPPORT_CLASSES, _FC_F1, strict=True):
        cells = []
        for g, _ in GRID_ROWS:
            n = per_grid[g].get(cname)
            cells.append(f"{n:,}" if n else "0")  # 0 support = the class is absent on that grid
        body.append(f"{clabel} & " + " & ".join(cells) + r" \\")
    return (
        "% AUTO-GENERATED by hpc/paper_artifacts/build_supp_tab_fc_support.py — do not edit.\n"
        "\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{5pt}\n"
        "\\caption{Per-class support (test-set window count) for the 9 FC classes per grid. A count "
        "of $0$ means the class is absent on that grid (the two HIF subtypes are \\ac{mv}-only), so "
        "its per-class $\\text{F}_1$ is undefined rather than zero.}\n"
        "\\label{tab:supp_fc_support}\n"
        "\\begin{tabular}{l rrrr}\n\\toprule\n"
        f"\\textbf{{Fault class}} & {heads} \\\\\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


# --------------------------------------------------------------------------------------------------
# F04: the seven extended (diagnostic) functions, per-baseline (no selection), balanced accuracy on
# the in-distribution test set at each function's natural view. Per-baseline (not a single "best")
# keeps this consistent with the winner's-curse fix without a separate validation re-score.
# --------------------------------------------------------------------------------------------------
_DIAGNOSTICS = [  # (row label, task stem, natural view, class kind)
    ("Fault grounded", "fault_grounded", "line", "binary"),
    ("Faulted phase", "fault_phase", "line", "7-class"),
    ("Fault category", "fault_category", "line", "3-class"),
    ("Fault origin", "fault_origin", "global", "binary"),
    ("Event state", "event_state", "global", "3-class"),
    ("Switch detection", "switch_detection", "global", "binary"),
    ("Switch type", "switch_type", "global", "8-class"),
]
_DIAG_BASELINES = ["majority", "random_forest", "mlp", "gru", "cnn", "resnet"]


def _diag_cell(rows: dict, task_id: str, baseline: str) -> str:
    doc = rows.get((task_id, baseline), {}).get("test")
    m = mean_val(doc, "balanced_accuracy")
    if m is None:
        return r"\pending"
    return _fmt_mean_ci(m, ci_val(doc, "balanced_accuracy"), "balanced_accuracy")


def diagnostic_overview_table() -> str:
    """SI table: per-baseline balanced accuracy for the seven diagnostic functions (in-distribution
    test, each at its natural view), so the extended suite is reported rather than only summarised."""
    per_grid = {g: load_rows(g) for g, _ in GRID_ROWS}
    heads = " & ".join(f"\\textbf{{{_BASELINE_LABEL[b].split()[0]}}}" for b in _DIAG_BASELINES)
    body = []
    for label, stem, view, kind in _DIAGNOSTICS:
        for i, (g, glabel) in enumerate(GRID_ROWS):
            tid = f"{stem}_{view}"
            cells = [_diag_cell(per_grid[g], tid, b) for b in _DIAG_BASELINES]
            fn = f"{label} ({kind}, {view})" if i == 0 else ""
            body.append(f"{fn} & {glabel} & " + " & ".join(cells) + r" \\")
        body.append("\\midrule")
    body = "\n".join(body[:-1])  # drop trailing midrule
    return (
        "% AUTO-GENERATED by hpc/paper_artifacts/build_supp_tab_diagnostics.py — do not edit.\n"
        "\\begin{table*}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\caption{Extended (diagnostic) functions: per-baseline balanced accuracy ($\\uparrow$) on "
        "the in-distribution test set, each function at its natural observability view. Reported "
        "per baseline (no single `best') for the full picture; \\pending{}~= pending. The "
        "never-trained, per-class, and protection-grade breakdowns are in the leaderboard.}\n"
        "\\label{tab:supp_diagnostics}\n"
        f"\\begin{{tabular}}{{L{{3.4cm}} l {'c ' * len(_DIAG_BASELINES)}}}\n\\toprule\n"
        f"\\textbf{{Function}} & \\textbf{{Grid}} & {heads} \\\\\n\\midrule\n"
        + body
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )


def build_paper_tables() -> dict[str, str]:
    """``{filename: latex}`` for the five held_out result tables (pure, deterministic)."""
    return {f"tab_res_{key}.tex": result_table(key) for key, *_ in PAPER_FUNCTIONS}


def _val_audit_cell(rows: dict, stem: str, view: str, metric: str) -> str:
    """S10 audit cell: the winner's validation score *plus the winner's identity* — auditing a
    best-of-five selection needs at least who won (evemt-v3 V3-082/041), not only the score."""
    if _held_unsupported(stem, view):
        return r"\notapp"
    choice = best_cell_choice(rows, f"{stem}_{view}", "val", metric)
    if choice is None:
        return r"\pending"
    short = {
        "random_forest": r"\acs{rf}",
        "mlp": r"\acs{mlp}",
        "gru": r"\acs{gru}",
        "cnn": r"\acs{cnn}",
        "resnet": r"\acs{resnet}",
    }
    label = short.get(choice["baseline"], choice["baseline"])
    return f"{_fmt_mean_ci(choice['mean'], choice['ci'], metric)} ({label})"


def val_selection_table() -> str:
    """SI selection-audit table (V2-019 + V3-082): the validation-split headline AND identity of the
    *validation-selected* reference baseline for each primary function x grid x view -- i.e. the
    score that drives the F06 model selection reported in the S1 held_out tables, with the winner
    named so the selection is auditable against the per-baseline S2 tables."""
    per_grid = {g: load_rows(g) for g, _ in GRID_ROWS}
    short = {
        "fd": r"\ac{fd} $\uparrow$",
        "fc": r"\ac{fc} $\uparrow$",
        "fl": r"\ac{fl}\,\% $\downarrow$",
        "ed": r"\ac{ed} $\uparrow$",
        "ec": r"\ac{ec} $\uparrow$",
    }
    groups = []
    for key, _name, stem, metric, _label in PAPER_FUNCTIONS:
        rows_out = []
        for gi, (grid, glabel) in enumerate(GRID_ROWS):
            cells = [_val_audit_cell(per_grid[grid], stem, view, metric) for view in OBS]
            head = f"\\multirow{{{len(GRID_ROWS)}}}{{*}}{{{short[key]}}}" if gi == 0 else ""
            rows_out.append(f"{head:<30} & {glabel:<24} & " + " & ".join(cells) + r" \\")
        groups.append("\n".join(rows_out))
    body = "\n\\midrule\n".join(groups)
    return (
        "% AUTO-GENERATED by hpc/make_supplement_tables.py from outputs/ — do not edit by hand.\n"
        "\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4.5pt}\n"
        "\\caption{Validation-split selection audit: the headline score \\emph{and identity} (in "
        "parentheses) of the validation-selected reference baseline that drives the model selection "
        "reported in Tables~\\ref{tab:res_fd}--\\ref{tab:res_ec}, per primary function, grid, and "
        "observability view. Each cell is the winner's validation mean\\,$\\pm$\\,95\\% CI over five "
        "seeds; its held-out \\emph{test} and \\emph{benchmark} values are in "
        "Tables~\\ref{tab:res_fd}--\\ref{tab:res_ec}, and the competing baselines' held-out values "
        "are in the S2 per-baseline tables. Arrows give the metric direction (balanced "
        "accuracy $\\uparrow$; \\acs{mae}\\,\\% $\\downarrow$ for \\ac{fl}). \\notapp{}~= not defined "
        "for this view (the global-only event-scope functions \\ac{ed}/\\ac{ec} at \\emph{local}/"
        "\\emph{line}).}\n"
        "\\label{tab:supp_val_selection}\n"
        "\\begin{tabular}{ll ccc}\n\\toprule\n"
        "\\textbf{Function} & \\textbf{Grid} & \\textbf{local} & \\textbf{line} & \\textbf{global} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )


def build_supplement_tables() -> dict[str, str]:
    """``{filename: latex}`` for the supplementary tables (pure, deterministic): the ten
    per-baseline held_out tables plus the F03 protection-grade + FC per-class F1 tables."""
    tables = {
        f"supp_tab_{key}_{ts}.tex": supp_table(key, ts)
        for key, *_ in PAPER_FUNCTIONS
        for ts in TEST_SETS
    }
    tables["supp_tab_protection.tex"] = protection_rate_table()
    tables["supp_tab_fc_perclass.tex"] = fc_per_class_f1_table()
    tables["supp_tab_splits.tex"] = dataset_split_table()
    tables["supp_tab_fc_support.tex"] = fc_support_table()
    tables["supp_tab_diagnostics.tex"] = diagnostic_overview_table()
    tables["supp_tab_val_selection.tex"] = val_selection_table()
    return tables


def build_transfer_tables(outputs: str | Path = "outputs") -> dict[str, str]:
    """``{filename: latex}`` for the three cross-grid transfer tables (indexes built once, shared).
    ED/EC are global-only and transfer is fixed-width local/line only, so their transfer tables
    would be structurally out of scope (all \\oos) — they are not built (evemt-v3 V3-009)."""
    idx = _transfer_index(outputs)
    val_idx = _transfer_val_index(outputs)
    return {
        f"tab_transfer_{key}.tex": transfer_table(key, outputs, idx, val_idx)
        for key, *_ in PAPER_FUNCTIONS
        if key not in ("ed", "ec")
    }


def build_spec_tables() -> dict[str, str]:
    """``{filename: latex}`` for the five static specification tables."""
    return {f"tab_{name}.tex": spec_table(name) for name in SPEC_TABLES}


def write_all_tables(outputs: str | Path = "outputs") -> int:
    """Regenerate every committed table fragment (held_out + supplement + transfer + spec) into
    GEN_DIR — used by the paper build for a full preview (``hpc/build_paper.py``). The figure PDF is
    built separately by ``build_fig_transfer_overview.py``. Returns the number of fragments written.
    Populates from ``outputs/`` (cwd-relative for the held_out/supplement tables); commit populated
    fragments ONLY from a frozen tag (``git restore`` after a preview otherwise)."""
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    frags = {
        **build_paper_tables(),
        **build_supplement_tables(),
        **build_transfer_tables(outputs),
        **build_spec_tables(),
    }
    for name, content in frags.items():
        (GEN_DIR / name).write_text(content)
    return len(frags)


def run(name: str, content: str) -> None:
    """Write exactly one artifact (``name``) under GEN_DIR and report — the standalone entry point
    every ``build_<x>.py`` calls in ``__main__``."""
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    (GEN_DIR / name).write_text(content)
    n_res = content.count(r"\res")
    print(f"wrote {GEN_DIR / name} ({n_res} empty cells -> \\res)")
