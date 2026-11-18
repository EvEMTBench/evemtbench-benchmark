"""Shared pure helpers for the Supplementary-Information artifact builders.

Companion to :mod:`paper_lib`: that module owns the held_out / transfer / spec tables the
manuscript and the SI share; this one owns the SI-only evidence tables added to close the
reproducibility gaps (partition + leakage audit, valid-sample counts, ED negative composition,
EC class support, transfer-vs-scratch reference, raw current-scale diagnostic, baseline
configuration).

**This module writes no artifact.** Each table is a pure ``() -> str`` function; exactly one thin
``hpc/paper_artifacts/build_<name>.py`` entry point calls it and hands the string to
``paper_lib.run("<name>.tex", ...)``, which is the only writer. See
``paper/manuscript/generated/README.md`` for the artifact → generator → input map.

Two of the builders read the *released dataset* (the windowed ``y`` label frames) rather than only
``outputs/``: the EC class-support table, the ED negative-composition table and the current-scale
diagnostic are properties of the evaluation sets, not of any model. They need
``$EVEMT_ROOT`` / ``--root`` to point at the mounted dataset; :data:`DATASET_BACKED` names them so
the freshness gate can skip them where the dataset is not mounted (CI) instead of failing.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import paper_lib as pl

from evemtbench.coverage import DEEP_BASELINES
from evemtbench.version import BENCHMARK_VERSION, splits_version
from make_results import load_rows

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]

#: Builders that need the mounted EvEMTBench dataset (not just ``outputs/``).
DATASET_BACKED = (
    "supp_tab_ec_support.tex",
    "supp_tab_ed_negative_support.tex",
    "supp_tab_current_shift.tex",
    "supp_tab_fl_locator_breakdown.tex",
)

_SPLITS_ROOT = _REPO / "src" / "evemtbench" / "splits"
_HASH_CHARS = 12  # stable short prefix shown in tables; full digests live in the JSON artifacts


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _short(digest: str) -> str:
    return f"\\texttt{{{digest[:_HASH_CHARS]}}}"


# ==================================================================================================
# S: Partition and leakage audit  ->  supp_tab_partition_audit.tex
# ==================================================================================================
def _held_out_partition_audit(grid: str) -> dict:
    """Audit one reference grid's committed held_out partition, from the split files themselves.

    The benchmark stores no ``audit.json`` for the reference grids (their audit gate reads strata
    from the mounted dataset), so the overlap counts are computed here directly from the committed
    episode-id lists with the same set arithmetic ``evemtbench.splits.audit`` uses — a measurement
    of the artifacts, not a reconstruction of them. One episode is one EMT simulation
    (``splits.protocols``), so the episode and simulation partitions coincide by construction."""
    from evemtbench.splits.loader import read_held_out_splits, read_meta, split_dir

    parts = read_held_out_splits(grid)
    d = split_dir("held_out", grid)
    sets = {k: set(v) for k, v in parts.items()}
    adapt = sets["train"] | sets["val"] | sets["test"]
    return {
        "label": None,
        "protocol": "held_out",
        "grouping": "simulation episode",
        "group_column": "sample_id",
        "counts": {k: len(sets[k]) for k in ("train", "val", "test", "benchmark")},
        "overlaps": {
            "episode": {
                "train&val": len(sets["train"] & sets["val"]),
                "train&test": len(sets["train"] & sets["test"]),
                "val&test": len(sets["val"] & sets["test"]),
            },
            # one episode == one simulation for a reference grid, so the two partitions coincide
            "simulation": None,
            "topology": None,  # a reference grid is a single fixed topology by construction
        },
        "adapt_vs_eval_overlap": len(adapt & sets["benchmark"]),
        "seed": read_meta("held_out", grid)["seed"],
        "hashes": {f.name: _sha256(f) for f in sorted(d.glob("*.txt")) if f.stat().st_size},
        "train_hash": _sha256(d / "train.txt"),
        "splits_version": splits_version("held_out"),
    }


def _corpus_partition_audit(source: str) -> dict:
    """One transfer source corpus's audit, read from its COMMITTED ``audit.json`` (authoritative:
    written by ``evemtbench.splits.audit.write_multi_grid_audits`` when the v1.1 topology-grouped
    splits were generated). Nothing is recomputed here."""
    d = _SPLITS_ROOT / f"v{splits_version('multi_grid_pretrain')}" / "multi_grid_pretrain" / source
    audit = json.loads((d / "audit.json").read_text())
    meta = json.loads((d / "meta.json").read_text())
    return {
        "protocol": "multi_grid_pretrain",
        "grouping": audit["grouping"],
        "group_column": audit["group_column"],
        "counts": {k: audit["counts"][k]["episodes"] for k in ("train", "val", "test")},
        "topologies": {k: audit["counts"][k]["topologies"] for k in ("train", "val", "test")},
        "simulations": {k: audit["counts"][k]["simulations"] for k in ("train", "val", "test")},
        "overlaps": audit["overlaps"],
        "seed": meta["seed"],
        "hashes": audit["partition_hashes"],
        "train_hash": audit["partition_hashes"]["train.txt"],
        "ok": audit["ok"],
        "issues": audit["issues"],
        "splits_version": splits_version("multi_grid_pretrain"),
    }


def partition_audit_rows() -> list[dict]:
    """Every audited partition set: the four target grids' held_out partitions, then the three
    transfer source corpora. Shared with ``hpc/check_supplement_consistency.py`` so the table and
    the leakage assertions read the same records."""
    rows = []
    for grid, label in pl.GRID_ROWS:
        rec = _held_out_partition_audit(grid)
        rec["label"] = label
        rec["key"] = grid
        rows.append(rec)
    from evemtbench.coverage import PRETRAIN_GRIDS

    for src in PRETRAIN_GRIDS:
        rec = _corpus_partition_audit(src)
        rec["label"] = f"\\texttt{{multi\\_grid}} {pl._source_short(src)}"
        rec["key"] = src
        rows.append(rec)
    return rows


def _ov3(ov: dict | None) -> str:
    """The three pairwise overlap counts as one compact ``a/b/c`` cell (train-val / train-test /
    val-test), or ``\\notapp`` where the level does not apply to that partition set."""
    if ov is None:
        return r"\notapp"
    return "/".join(str(ov[k]) for k in ("train&val", "train&test", "val&test"))


def partition_audit_table() -> str:
    """SI table: the committed partitions behind every reported number, with their grouping unit,
    sizes, machine-checked zero-overlap counts, split seed and partition hash."""
    rows = partition_audit_rows()
    body_a: list[str] = []
    body_b: list[str] = []
    for i, r in enumerate(rows):
        if r["protocol"] == "held_out":
            counts = [f"{r['counts'][k]:,}" for k in ("train", "val", "test", "benchmark")]
        else:
            counts = [f"{r['counts'][k]:,}" for k in ("train", "val", "test")] + [r"\notapp"]
        ident = f"{r['label']} & {r['grouping']} & "
        body_a.append(
            ident + " & ".join(counts + [str(r["seed"]), _short(r["train_hash"])]) + r" \\"
        )
        overlaps = [_ov3(r["overlaps"][k]) for k in ("episode", "simulation", "topology")]
        body_b.append(ident + " & ".join(overlaps) + r" \\")
        if i == len(pl.GRID_ROWS) - 1:
            body_a.append(r"\midrule")
            body_b.append(r"\midrule")
    return (
        pl.provenance_header(
            "hpc/paper_artifacts/build_supp_tab_partition_audit.py",
            [
                "src/evemtbench/splits/v1.0/held_out/<grid>/{train,val,test,benchmark}.txt",
                "src/evemtbench/splits/v1.0/held_out/<grid>/meta.json",
                "src/evemtbench/splits/v1.1/multi_grid_pretrain/<source>/audit.json  (committed"
                " leakage audit; not recomputed)",
                "src/evemtbench/splits/v1.1/multi_grid_pretrain/<source>/meta.json",
            ],
        )
        + "\\begin{table}[!htbp]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\renewcommand{\\arraystretch}{1.05}\n"
        "\\caption{Partition and leakage audit of the committed splits used for every reported "
        "result. \\emph{Top}: the four target grids' \\texttt{held\\_out} partitions of the "
        "\\texttt{adapt\\_grid} family, plus the never-trained \\texttt{eval\\_grid} family. "
        "\\emph{Bottom}: the three \\texttt{multi\\_grid} pre-training source corpora, whose v1.1 "
        "splits are grouped by generated topology. Counts are \\emph{episodes}. Overlap columns "
        "give the three pairwise intersection sizes train\\,$\\cap$\\,val / train\\,$\\cap$\\,test / "
        "val\\,$\\cap$\\,test at each grouping level; every one is zero. For a reference grid one "
        "episode is one \\ac{emt} simulation, so the simulation partition is the episode partition "
        "(\\notapp{}), and all partitions share a single fixed topology (\\notapp{}); the "
        "\\texttt{eval\\_grid} family is disjoint from the whole \\texttt{adapt\\_grid} family by "
        "construction (verified separately, see text). Source-corpus rows reproduce the committed "
        "\\texttt{audit.json} written when the splits were generated; they are not a new "
        "resampling. Hashes are the leading $12$ hex digits of the SHA-256 of \\texttt{train.txt}; "
        "the complete digests of every partition file are retained in each run's provenance record "
        "and in the source corpora's \\texttt{audit.json}.}\n"
        "\\label{tab:supp_partition_audit}\n"
        + pl.si_panels(
            [
                "\\begin{tabular}{l l rrrr r l}\n\\toprule\n"
                " & & \\multicolumn{4}{c}{\\textbf{Episodes}} & & \\\\\n"
                "\\cmidrule(lr){3-6}\n"
                "\\textbf{Partition set} & \\textbf{Grouping unit} & \\textbf{train} & "
                "\\textbf{val} & \\textbf{test} & \\textbf{\\texttt{eval\\_grid}} & \\textbf{seed} & "
                "\\textbf{hash (\\texttt{train.txt})} \\\\\n\\midrule\n"
                + "\n".join(body_a)
                + "\n\\bottomrule\n\\end{tabular}",
                "\\begin{tabular}{l l ccc}\n\\toprule\n"
                " & & \\multicolumn{3}{c}{\\textbf{Overlaps} (tr$\\cap$va / tr$\\cap$te / va$\\cap$te)} "
                "\\\\\n\\cmidrule(lr){3-5}\n"
                "\\textbf{Partition set} & \\textbf{Grouping unit} & \\textbf{episode} & "
                "\\textbf{simulation} & \\textbf{topology} \\\\\n\\midrule\n"
                + "\n".join(body_b)
                + "\n\\bottomrule\n\\end{tabular}",
            ]
        )
        + "\\end{table}\n"
    )


# ==================================================================================================
# S: Transfer vs. training from scratch  ->  supp_tab_transfer_scratch.tex
# ==================================================================================================
#: The transferable fault-triad functions, in manuscript order: (key, label, stem, metric, arrow).
TRANSFER_FUNCTIONS = [
    ("fd", r"\ac{fd}", "fault_detection", "balanced_accuracy", r"bal.\ acc.\ $\uparrow$"),
    ("fc", r"\ac{fc}", "fault_classification", "balanced_accuracy", r"bal.\ acc.\ $\uparrow$"),
    ("fl", r"\ac{fl}", "fault_location", "mae", r"\acs{mae}\,\% $\downarrow$"),
]
#: Transfer is fixed-width, so only these two views carry a from-scratch counterpart.
TRANSFER_VIEWS = ("local", "line")


def transfer_scratch_rows() -> list[dict]:
    """The source-independent from-scratch reference for every transferable (function, grid, view).

    The reference is the ``held_out`` cell: fitted on the TARGET grid's ``adapt_grid`` train
    partition, architecture chosen on the TARGET grid's validation split, reported on that grid's
    never-trained ``eval_grid`` family — the same protocol and the same evaluation set as the
    transfer tables, differing only in the initialisation (random instead of pre-trained). It
    therefore does not depend on the transfer source, which is exactly why it is tabulated once
    here instead of repeated in every source row of S3.

    Selected out of ``DEEP_BASELINES``, NOT the full ``LEARNED_BASELINES``, because this reference is
    read against the transfer tables and those apply only to the deep baselines
    (``coverage.baseline_applies``: a classical baseline has no transferable weights, so no
    ``random_forest`` transfer cell exists). Selecting out of all learned baselines drew the two sides of the
    comparison from different candidate sets and could name a model the transfer arm cannot field --
    it published ``rf`` at 0.499 balanced accuracy for FD/local/Double Line, where the random forest
    wins the in-distribution validation split (0.889) and then collapses on the never-trained
    ``eval_grid`` family while every deep baseline reaches 0.981--0.989."""
    per_grid = {g: load_rows(g) for g, _ in pl.GRID_ROWS}
    out = []
    for key, _lab, stem, metric, _arrow in TRANSFER_FUNCTIONS:
        for grid, glabel in pl.GRID_ROWS:
            for view in TRANSFER_VIEWS:
                choice = pl.best_cell_choice(
                    per_grid[grid], f"{stem}_{view}", "benchmark", metric, DEEP_BASELINES
                )
                out.append(
                    {
                        "function": key,
                        "stem": stem,
                        "metric": metric,
                        "grid": grid,
                        "grid_label": glabel,
                        "view": view,
                        "choice": choice,
                    }
                )
    return out


def _scratch_cell(rec: dict) -> tuple[str, str]:
    c = rec["choice"]
    if c is None:
        return r"\pending", r"\pending"
    return (
        pl._BASELINE_ACS.get(c["baseline"], c["baseline"]),
        pl._fmt_mean_ci(c["mean"], c["ci"], rec["metric"]),
    )


def transfer_scratch_table() -> str:
    """SI companion to the S3 transfer tables: the from-scratch target-grid reference each transfer
    cell must be compared against, once per (function, grid, view)."""
    recs = {(r["function"], r["grid"], r["view"]): r for r in transfer_scratch_rows()}
    body: list[str] = []
    for fi, (key, flabel, _stem, _metric, arrow) in enumerate(TRANSFER_FUNCTIONS):
        for grid, glabel in pl.GRID_ROWS:
            cells: list[str] = []
            for view in TRANSFER_VIEWS:
                cells += list(_scratch_cell(recs[(key, grid, view)]))
            head = (
                f"\\multirow{{{len(pl.GRID_ROWS)}}}{{*}}{{{flabel}\\,({arrow})}}"
                if grid == pl.GRID_ROWS[0][0]
                else ""
            )
            body.append(f"{head:<44} & {glabel:<24} & " + " & ".join(cells) + r" \\")
        if fi < len(TRANSFER_FUNCTIONS) - 1:
            body.append(r"\midrule")
    return (
        pl.provenance_header(
            "hpc/paper_artifacts/build_supp_tab_transfer_scratch.py",
            [
                pl._held_out_glob(
                    "{fault_detection,fault_classification,fault_location}_{local,line}",
                    "{val,benchmark}",
                )
            ],
        )
        + "\\begin{table}[!htbp]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.05}\n"
        "\\caption{From-scratch target-grid reference for the cross-grid transfer tables "
        "(Tables~\\ref{tab:transfer_fd}--\\ref{tab:transfer_fl}). Each cell is the "
        "\\texttt{held\\_out} baseline trained on the \\emph{target} grid's \\texttt{adapt\\_grid} "
        "training partition with random initialisation, its architecture selected on the "
        "\\emph{target} grid's validation split, and evaluated on that grid's never-trained "
        "\\texttt{eval\\_grid} family --- the same evaluation set the transfer tables report. The "
        "reference is independent of the pre-training source, so it is tabulated once here rather "
        "than repeated in every source row. The architecture is selected out of the five deep "
        "baselines only, matching the candidate set the transfer protocols can field (a classical "
        "baseline carries no transferable weights), so both sides of the comparison are drawn from "
        "the same pool; the unrestricted per-baseline \\texttt{held\\_out} results are in "
        "Tables~\\ref{tab:res_fd}--\\ref{tab:res_fl}. \\textbf{ref.} names the selected architecture; the "
        "\\texttt{eval\\_grid} family is prediction-only and takes no part in any selection. Cells "
        "are mean\\,$\\pm$\\,95\\% CI over five seeds."
        + pl._pending_legend("\n".join(body), " \\pending{}~= pending.")
        + "}\n"
        "\\label{tab:supp_transfer_scratch}\n"
        + pl._SHRINK_TO_FIT
        + "\\begin{tabular}{ll ll ll}\n\\toprule\n"
        " & & \\multicolumn{2}{c}{\\textbf{local}} & \\multicolumn{2}{c}{\\textbf{line}} \\\\\n"
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
        "\\textbf{Function} & \\textbf{Target grid} & \\textbf{ref.} & \\textbf{from scratch} & "
        "\\textbf{ref.} & \\textbf{from scratch} \\\\\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


# ==================================================================================================
# S: Primary-function valid-sample counts  ->  supp_tab_primary_valid_counts.tex
# ==================================================================================================
def valid_counts() -> dict:
    """``{(task_id, grid, test_set): n}`` — the evaluated window count of every primary cell.

    Read from the committed ``result.json`` ``n_test`` field, which the evaluator writes as the
    length of the task's frozen valid-sample mask on that split. The mask is a function of the
    window label frame alone (``tasks.valid_samples.valid_sample_mask``) and never of the
    observability view, so a cell's ``n`` is shared by every baseline and every view of the same
    function; the builder asserts that agreement rather than assuming it."""
    out: dict = {}
    for grid, _ in pl.GRID_ROWS:
        rows = load_rows(grid)
        for (task_id, _baseline), by_set in rows.items():
            for test_set, doc in by_set.items():
                n = doc.get("n_test")
                if n is None:
                    continue
                key = (task_id, grid, test_set)
                prev = out.setdefault(key, int(n))
                if prev != int(n):
                    raise AssertionError(
                        f"valid-sample count disagrees between baselines for {key}: "
                        f"{prev} vs {n} — the mask is not model-dependent, so this is a defect"
                    )
    return out


def primary_valid_counts_table() -> str:
    """SI table: evaluated window count (the reported ``n``) for the five primary functions, per
    grid and evaluation family."""
    counts = valid_counts()
    body: list[str] = []
    for fi, (_key, _name, stem, _metric, _label) in enumerate(pl.PAPER_FUNCTIONS):
        views = ("global",) if stem in pl._EVENT_SCOPE_STEMS else pl.OBS
        label = {
            "fault_detection": r"\ac{fd}",
            "fault_classification": r"\ac{fc}",
            "fault_location": r"\ac{fl}",
            "event_detection": r"\ac{ed}",
            "event_classification": r"\ac{ec}",
        }[stem]
        scope = "windows of fault episodes" if stem == "fault_detection" else None
        for gi, (grid, glabel) in enumerate(pl.GRID_ROWS):
            cells = []
            for test_set in pl.TEST_SETS:
                # every defined view of a function shares one mask -> one count per (grid, family)
                ns = {counts.get((f"{stem}_{v}", grid, test_set)) for v in views}
                ns.discard(None)
                if not ns:
                    cells.append(r"\pending")
                elif len(ns) > 1:
                    raise AssertionError(
                        f"view-dependent valid-sample count for {stem}/{grid}/{test_set}: {ns}"
                    )
                else:
                    cells.append(f"{ns.pop():,}")
            head = f"\\multirow{{{len(pl.GRID_ROWS)}}}{{*}}{{{label}}}" if gi == 0 else ""
            body.append(f"{head:<34} & {glabel:<24} & " + " & ".join(cells) + r" \\")
        if fi < len(pl.PAPER_FUNCTIONS) - 1:
            body.append(r"\midrule")
        del scope
    return (
        pl.provenance_header(
            "hpc/paper_artifacts/build_supp_tab_primary_valid_counts.py",
            [
                pl._held_out_glob(
                    "{fault_detection,fault_classification,fault_location}_{local,line,global}",
                    "{test,benchmark}",
                )
                + "  (n_test)",
                pl._held_out_glob(
                    "{event_detection,event_classification}_global", "{test,benchmark}"
                )
                + "  (n_test)",
            ],
        )
        + "\\begin{table}[!htbp]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.05}\n"
        "\\caption{Evaluated sample counts for the five primary functions: the number of "
        "\\emph{windows} (not episodes) admitted by each function's frozen valid-sample mask, per "
        "grid and evaluation family. This is the $n$ behind every mean and confidence interval in "
        "Tables~\\ref{tab:res_fd}--\\ref{tab:res_ec}, read from the committed result artefacts. The "
        "mask is a function of the window label frame only, so it is identical across the three "
        "observability views of a function and across baselines --- one count per (function, grid, "
        "family). Counts are far below episodes\\,$\\times\\,23$ (the per-episode window count "
        "under the frozen contract) wherever the mask restricts "
        "scoring: \\ac{fd} scores windows of fault episodes; \\ac{fc} and \\ac{fl} score "
        "fault-active windows (\\ac{fl} additionally only line-fault episodes); \\ac{ed} and "
        "\\ac{ec} score every window with an in-vocabulary event type.}\n"
        "\\label{tab:supp_primary_valid_counts}\n"
        "\\begin{tabular}{ll rr}\n\\toprule\n"
        "\\textbf{Function} & \\textbf{Grid} & \\textbf{\\texttt{adapt\\_grid} test} & "
        "\\textbf{\\texttt{eval\\_grid}} \\\\\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


# ==================================================================================================
# Baseline configuration  ->  supp_tab_baseline_cfg.tex
# ==================================================================================================
def _defaults_from_source(module: str, cls: str) -> dict:
    """Constructor keyword defaults of a baseline class, read from its SOURCE rather than imported.

    Parsed with :mod:`ast` on purpose: the deep baselines ``import torch``, and this reporting
    layer must stay importable where torch is absent — CI's mirror runner has no torch, and the
    pre-commit freshness gate rebuilds this table there. The values are still exactly the shipped
    ones; this reads the same ``def __init__`` the runtime uses, it just does not execute it.
    """
    import ast

    path = _REPO / "src" / "evemtbench" / "baselines" / f"{module}.py"
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == cls):
            continue
        for item in node.body:
            if not (isinstance(item, ast.FunctionDef) and item.name == "__init__"):
                continue
            a = item.args
            out: dict = {}
            # keyword-only defaults (every baseline uses `def __init__(self, *, ...)`)
            for name, default in zip(a.kwonlyargs, a.kw_defaults, strict=True):
                if default is not None:
                    out[name.arg] = ast.literal_eval(default)
            # ...and any trailing positional-or-keyword defaults, for robustness
            positional = a.args[1:]  # drop self
            for name, default in zip(
                positional[len(positional) - len(a.defaults) :], a.defaults, strict=True
            ):
                out[name.arg] = ast.literal_eval(default)
            return out
    raise LookupError(f"no {cls}.__init__ found in {path}")


def baseline_cfg_rows() -> dict:
    """Every reference baseline's frozen configuration, read from the shipped classes so a
    hyperparameter cannot drift between the code and the SI table."""
    return {
        "mlp": _defaults_from_source("mlp", "MLPBaseline"),
        "gru": _defaults_from_source("gru", "GRUBaseline"),
        "cnn": _defaults_from_source("cnn", "CNNBaseline"),
        "resnet": _defaults_from_source("resnet", "ResNetBaseline"),
        "tst": _defaults_from_source("tst", "TSTBaseline"),
        "random_forest": _defaults_from_source("random_forest", "RandomForestBaseline"),
        "threshold": _defaults_from_source("threshold", "ThresholdBaseline"),
        "impedance": _defaults_from_source("impedance", "ImpedanceBaseline"),
    }


def protocol_seeds() -> list[int]:
    """The seed set actually executed, read from the committed held_out result artefacts.

    Taken from the artefacts rather than from a constant so the SI can never claim a protocol the
    results do not have. Raises if the cells disagree — a five-seed claim must be true of every
    stochastic cell."""
    seen: set[tuple[int, ...]] = set()
    for grid, _ in pl.GRID_ROWS:
        for (_task, baseline), by_set in load_rows(grid).items():
            if baseline not in ("mlp", "gru", "cnn", "resnet", "tst", "random_forest"):
                continue
            for doc in by_set.values():
                seeds = tuple(doc.get("seeds") or ())
                if seeds:
                    seen.add(seeds)
    if len(seen) != 1:
        raise AssertionError(f"stochastic cells do not share one seed set: {sorted(seen)}")
    return list(next(iter(seen)))


def baseline_cfg_table() -> str:
    """SI table: the exact configuration of every reference baseline, with the shared deep-baseline
    training regime factored out. Hyperparameter values are introspected from the shipped baseline
    classes; only the prose descriptions are authored here."""
    cfg = baseline_cfg_rows()
    deep = cfg["mlp"]  # the training regime is shared by all five deep baselines (asserted below)
    regime = ("epochs", "batch_size", "lr", "weight_decay", "patience")
    for name in ("gru", "cnn", "resnet", "tst"):
        # dropout is architecture-level for tst (its source prescribes 0.1); the row states it
        for k in regime if name == "tst" else (*regime, "dropout"):
            if cfg[name][k] != deep[k]:
                raise AssertionError(
                    f"deep baselines no longer share the training regime: {name}.{k}="
                    f"{cfg[name][k]} vs mlp.{k}={deep[k]} — split the table before shipping"
                )
    rf, thr, imp = cfg["random_forest"], cfg["threshold"], cfg["impedance"]
    if not isinstance(imp["impedance_scale"], (int, float)):
        raise AssertionError(
            "the impedance locator default is no longer a fixed scale — reword its row"
        )
    seeds = protocol_seeds()
    hidden = "$-$".join(str(h) for h in deep["hidden_sizes"])
    rows = [
        (
            r"\multicolumn{2}{l}{\textit{Shared deep-baseline training regime "
            r"(\ac{mlp}/\ac{gru}/\ac{cnn}/\ac{resnet}/\ac{tst})}} \\"
        ),
        f"Optimizer & AdamW, learning rate ${deep['lr']:g}$, weight decay "
        f"${deep['weight_decay']:g}$ \\\\",
        f"Batch / epochs & batch size {deep['batch_size']}; up to {deep['epochs']} epochs \\\\",
        f"Learning-rate schedule & cosine annealing, $T_{{\\max}}={deep['epochs']}$ "
        "(stepped per epoch) \\\\",
        f"Early stopping & on validation loss, patience {deep['patience']}, best-checkpoint "
        "restore \\\\",
        r"Loss & cross-entropy with inverse-frequency class weights (classification); "
        r"mean-squared error (MSE) on train-standardized targets, de-standardized at predict "
        r"(regression, \ac{fl}) \\",
        r"Input & per-channel standardization fitted on train only \\",
        f"Seeds & {len(seeds)} per cell (${min(seeds)}$--${max(seeds)}$): model init, "
        "batch-shuffle order, dropout (not the split) \\\\",
        r"\midrule",
        f"\\ac{{mlp}} & flattened window $\\to {hidden}$; each layer Linear\\,+\\,BatchNorm\\,+\\,"
        f"ReLU\\,+\\,Dropout(${deep['dropout']}$) \\\\",
        f"\\ac{{gru}} & {cfg['gru']['num_layers']}-layer GRU, hidden size {cfg['gru']['hidden']}, "
        f"read at the final step; Dropout(${cfg['gru']['dropout']}$)\\,+\\,Linear head \\\\",
        f"\\ac{{cnn}} & three dilated Conv1d blocks (Conv\\,+\\,BatchNorm\\,+\\,ReLU), width "
        f"{cfg['cnn']['width']}, kernels $7/5/3$ at dilations $1/2/4$, global average pooling, "
        f"Dropout(${cfg['cnn']['dropout']}$) \\\\",
        f"\\ac{{resnet}} & three residual blocks (filters ${cfg['resnet']['width']}/"
        f"{2 * cfg['resnet']['width']}/{2 * cfg['resnet']['width']}$), each three Conv1d with "
        f"kernels $8/5/3$ and a $1{{\\times}}1$ shortcut; global average pooling, "
        f"Dropout(${cfg['resnet']['dropout']}$) \\\\",
        f"\\ac{{tst}} & Time Series Transformer encoder: patch embedding Conv1d (kernel "
        f"{cfg['tst']['patch_len']}, stride {cfg['tst']['stride']}; 59 tokens for a 480-sample "
        f"window) scaled by $\\sqrt{{d}}$ and layer-normalised per token, learnable positional encoding, "
        f"{cfg['tst']['n_layers']} post-norm encoder blocks (width $d={cfg['tst']['d_model']}$, "
        f"{cfg['tst']['n_heads']} heads, feed-forward {cfg['tst']['dim_ff']}, GELU, BatchNorm in "
        f"place of LayerNorm), Dropout(${cfg['tst']['dropout']}$), Linear head on the "
        "concatenated token outputs. Deviations from the source: the strided patch embedding "
        f"(one token per time step there), {cfg['tst']['n_heads']} heads (16 there) and the token "
        "LayerNorm (none there) \\\\",
        r"\midrule",
        r"\emph{majority} & most-frequent train class (classification) / train-mean target "
        r"(regression) \\",
        f"\\emph{{threshold}} & phase-maximum \\ac{{rms}}-current statistic; threshold maximizes "
        f"balanced accuracy over a {thr['n_candidates']}-point train quantile grid; binary "
        r"\ac{fd}, \emph{local} only \\",
        f"\\emph{{random forest}} & {rf['n_estimators']} trees, "
        f"{'unbounded' if rf['max_depth'] is None else rf['max_depth']} depth, balanced class "
        "weights (classifier); seven per-channel time-axis features: mean, std, min, max, "
        "peak-to-peak, \\ac{rms}, peak $|\\cdot|$ \\\\",
        r"\emph{impedance locator} & fundamental-frequency phasors by a full-cycle DFT over the last "
        r"complete cycle of the window; \emph{local}: one-terminal reactance method on the fault "
        r"loop chosen by a current-based phase selector, zero-sequence compensated for ground loops; "
        r"\emph{line}: two-terminal method on positive-sequence phasors (one-terminal fallback where "
        r"the far terminal is not measured); series impedance and length of the faulted line from "
        f"the grid topology, impedance scale ${imp['impedance_scale']:g}$, no fitted parameter; "
        r"\ac{fl}, \emph{local} and \emph{line} only \\",
    ]
    return (
        pl.provenance_header(
            "hpc/paper_artifacts/build_supp_tab_baseline_cfg.py",
            [
                "src/evemtbench/baselines/{mlp,gru,cnn,resnet,tst,random_forest,threshold,impedance}.py"
                "  (frozen constructor defaults, introspected)",
                pl._held_out_glob("*", "{val,test,benchmark}") + "  (executed seed set)",
            ],
        )
        + "\\begin{table}[!htbp]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.05}\n"
        f"\\caption{{Reference-baseline configurations (benchmark \\texttt{{v{BENCHMARK_VERSION}}}). "
        "The five deep baselines share one training regime (top) and differ only in architecture "
        "(and, for the \\ac{tst}, the dropout rate its source prescribes); "
        "the classical baselines are fully specified by their listed rule. Every numeric "
        "hyperparameter is read from the shipped baseline classes when this table is generated, so "
        "it cannot drift from the code.}\n"
        "\\label{tab:supp_baseline_cfg}\n"
        "\\begin{tabular}{L{3.0cm} L{12.5cm}}\n\\toprule\n"
        "\\textbf{Component} & \\textbf{Configuration} \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


# ==================================================================================================
# Dataset-backed diagnostics: EC support, ED negative composition, raw current scale.
# ==================================================================================================
#: Evaluation families of the held_out protocol as (split name, column label).
_EVAL_FAMILIES = (("test", r"\texttt{adapt\_grid} test"), ("benchmark", r"\texttt{eval\_grid}"))


def _task_view(task_id: str, grid: str, split: str, root=None):
    """One evaluation view of a scored task, loaded through the benchmark's own task loader.

    Using ``load_task`` (rather than reading the window frames directly) means the episode
    selection, family resolution, window contract, valid-sample mask and label derivation are
    byte-for-byte the ones every scored run used — the diagnostic cannot drift from the benchmark
    it describes. ``view.labels`` is the frozen label vector; ``view.metadata`` the per-window
    key/label frame. Features are streamed only where a diagnostic needs them."""
    from evemtbench.tasks.task import load_task

    return getattr(load_task(task_id, "held_out", grid=grid, root=root), split)


def ec_support_counts(root=None) -> tuple[dict, list[str]]:
    """``{(grid, split): Counter(class name -> window count)}`` for the 23-class EC vocabulary.

    Read off the frozen ``event_classification_global`` label vector of each evaluation view, so
    the counts are exactly the support the reported EC metrics are computed over (out-of-vocabulary
    active windows are already removed by the task's valid-sample mask). Support is a property of
    the evaluation set, not of any model."""
    from evemtbench.tasks.labels.event_type import EVENT_TYPE_CLASSES

    out = {}
    for grid, _ in pl.GRID_ROWS:
        for split, _lab in _EVAL_FAMILIES:
            labels = _task_view("event_classification_global", grid, split, root=root).labels
            c = Counter()
            for code in labels:
                c[EVENT_TYPE_CLASSES[int(code)]] += 1
            out[(grid, split)] = c
    return out, list(EVENT_TYPE_CLASSES)


def ec_support_table(root=None) -> str:
    """SI table: per-class support of the frozen 23-class EC vocabulary, per grid and family."""
    counts, vocab = ec_support_counts(root=root)
    heads = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{\\textbf{{{label}}}}}" for _g, label in pl.GRID_ROWS
    )
    cmids = "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(pl.GRID_ROWS)))
    sub = " & ".join([r"\textbf{test} & \textbf{eval}"] * len(pl.GRID_ROWS))
    body = []
    for idx, cls in enumerate(vocab):
        cells = [
            f"{counts[(g, s)].get(cls, 0):,}" for g, _ in pl.GRID_ROWS for s, _l in _EVAL_FAMILIES
        ]
        body.append(
            f"{idx} & \\texttt{{{cls.replace('_', chr(92) + '_')}}} & " + " & ".join(cells) + r" \\"
        )
    totals = [
        f"{sum(counts[(g, s)].values()):,}" for g, _ in pl.GRID_ROWS for s, _l in _EVAL_FAMILIES
    ]
    body.append(r"\midrule")
    # NB no math-italic in the header: the offline TeX bundle carries no 7 pt math
    # italic (lmmi7.pfb), which a scaled box would need as a physical font.
    body.append(
        r"\multicolumn{2}{l}{\textbf{total (evaluated EC windows)}} & "
        + " & ".join(totals)
        + r" \\"
    )
    return (
        pl.provenance_header(
            "hpc/paper_artifacts/build_supp_tab_ec_support.py",
            [
                "src/evemtbench/splits/v1.0/held_out/<grid>/{test,benchmark}.txt",
                "$EVEMT_ROOT/{adapt_grid,benchmark}/windows/<grid>/y_*.parquet  (released dataset)",
                "src/evemtbench/tasks/labels/event_type.py  (frozen 23-class vocabulary+deriver)",
            ],
        )
        + "\\begin{table}[!htbp]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\renewcommand{\\arraystretch}{1.05}\n"
        "\\caption{Per-class support of the frozen 23-class \\ac{ec} vocabulary: number of "
        "\\emph{windows} carrying each class in the in-distribution test split and the "
        "never-trained \\texttt{eval\\_grid} family of each grid. Class indices are the frozen "
        "label ids. A count of $0$ is genuine zero support --- the class does not occur in that "
        "evaluation set --- and its per-class $\\text{F}_1$ is reported as \\undefm{} (undefined), "
        "never as zero; the grid-restricted classes (the \\ac{mv}-only \\ac{hif} and cable/motor "
        "events, and the \\ac{ehv}-only generator trip) are the systematic zeros. Column totals "
        "are the evaluated \\ac{ec} sample count $n$ and match "
        "Table~\\ref{tab:supp_primary_valid_counts}. Derived from the committed partitions and the "
        "released window labels with the benchmark's frozen deriver.}\n"
        "\\label{tab:supp_ec_support}\n"
        + pl._SHRINK_TO_FIT
        + f"\\begin{{tabular}}{{r l {'rr ' * len(pl.GRID_ROWS)}}}\n\\toprule\n"
        f" & & {heads} \\\\\n{cmids}\n"
        f"\\textbf{{idx}} & \\textbf{{\\ac{{ec}} class}} & {sub} \\\\\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


#: ED negative categories, in the order the table reports them.
_ED_NEG_ORDER = ("pre-event", "post-event", "quiescent episode")


def ed_negative_composition(root=None) -> dict:
    """``{(grid, split): {category -> window count}}`` for the ED evaluation sets.

    ED admits every window; a window is positive iff it overlaps an active fault or switching event
    (``tasks.labels.event_detection``). Positives are read off the frozen ED label vector. The
    negatives are then split using only metadata the benchmark already carries — no new heuristic:

    * **quiescent episode** — the episode's ``event_family`` is neither ``fault`` nor ``switching``,
      so the episode contains no event at all and every one of its windows is negative;
    * **pre-event** / **post-event** — a negative window of an *event* episode, classified by
      whether its ``window_idx`` precedes that episode's first positive window or follows it.
    """
    import numpy as np

    out = {}
    for grid, _ in pl.GRID_ROWS:
        for split, _lab in _EVAL_FAMILIES:
            view = _task_view("event_detection_global", grid, split, root=root)
            md = view.metadata.reset_index(drop=True)
            pos = np.asarray(view.labels).astype(bool)
            event_episode = np.isin(
                md["event_family"].astype(str).to_numpy(), ("fault", "switching")
            )
            sid = md["sample_id"].to_numpy()
            widx = md["window_idx"].to_numpy()
            # first positive window index per episode (episodes with no positive -> absent)
            first_active: dict[int, int] = {}
            for s_, w_, p_ in zip(sid[pos], widx[pos], pos[pos], strict=True):
                k = int(s_)
                w = int(w_)
                if k not in first_active or w < first_active[k]:
                    first_active[k] = w
                del p_
            neg = ~pos
            quiescent = int((neg & ~event_episode).sum())
            pre = post = 0
            for s_, w_ in zip(sid[neg & event_episode], widx[neg & event_episode], strict=True):
                fa = first_active.get(int(s_))
                if fa is None or int(w_) < fa:
                    pre += 1
                else:
                    post += 1
            fam = md["event_family"].astype(str).to_numpy()
            out[(grid, split)] = {
                "positive": int(pos.sum()),
                "pre-event": pre,
                "post-event": post,
                "quiescent episode": quiescent,
                "neg. in fault episodes": int((neg & (fam == "fault")).sum()),
                "neg. in switching episodes": int((neg & (fam == "switching")).sum()),
                "negative": int(neg.sum()),
                "total": int(len(md)),
            }
    return out


def ed_negative_support_table(root=None) -> str:
    """SI table: the composition of the \\ac{ed} evaluation sets, positives and negatives by
    origin, for each grid and evaluation family."""
    comp = ed_negative_composition(root=root)
    shares = {
        split: [
            100.0 * comp[(g, split)]["neg. in switching episodes"] / comp[(g, split)]["negative"]
            for g, _ in pl.GRID_ROWS
        ]
        for split, _l in _EVAL_FAMILIES
    }
    # rendered as "$a$--$b$": an en-dash between two math groups, never "--" inside one (which
    # LaTeX would typeset as two minus signs).
    sw_id = f"${min(shares['test']):.0f}$--${max(shares['test']):.0f}"
    sw_ev = f"${min(shares['benchmark']):.0f}$--${max(shares['benchmark']):.0f}"
    body_a: list[str] = []
    body_b: list[str] = []
    for grid, glabel in pl.GRID_ROWS:
        for split, slabel in _EVAL_FAMILIES:
            c = comp[(grid, split)]
            share = f"{100.0 * c['negative'] / c['total']:.1f}" if c["total"] else "0.0"
            ident = f"{glabel} & {slabel} & "
            cells_a = [f"{c['positive']:,}"] + [f"{c[k]:,}" for k in _ED_NEG_ORDER]
            cells_a += [f"{c['negative']:,}"]
            cells_b = [f"{c['neg. in fault episodes']:,}", f"{c['neg. in switching episodes']:,}"]
            cells_b += [f"{c['total']:,}", share]
            body_a.append(ident + " & ".join(cells_a) + r" \\")
            body_b.append(ident + " & ".join(cells_b) + r" \\")
        body_a.append(r"\midrule")
        body_b.append(r"\midrule")
    body_a, body_b = body_a[:-1], body_b[:-1]
    return (
        pl.provenance_header(
            "hpc/paper_artifacts/build_supp_tab_ed_negative_support.py",
            [
                "src/evemtbench/splits/v1.0/held_out/<grid>/{test,benchmark}.txt",
                "$EVEMT_ROOT/{adapt_grid,benchmark}/windows/<grid>/y_*.parquet  (released dataset)",
                "src/evemtbench/tasks/labels/event_detection.py  (frozen ED label deriver)",
            ],
        )
        + "\\begin{table}[!htbp]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\renewcommand{\\arraystretch}{1.05}\n"
        "\\caption{Composition of the \\ac{ed} evaluation sets, in \\emph{windows}. \\ac{ed} scores "
        "every window: a window is positive iff it overlaps an active fault or switching event, and "
        "every other window is negative. Negatives are decomposed twice --- by position relative to "
        "the episode's active block (\\emph{pre-event} / \\emph{post-event}, plus every window of a "
        "\\emph{quiescent} episode that carries no event at all), and by the event family of the "
        "episode they come from. This quantifies the change in negative-class composition between "
        "evaluation families discussed in Methods. At \\texttt{v1.1.0} no reference-grid family "
        "contains dedicated quiescent episodes, so that column is structurally zero on every row; "
        f"the composition shift is instead a shift in \\emph{{episode mix}} --- {sw_id}\\%$ of the "
        "in-distribution negatives come from benign switching episodes, against only "
        f"{sw_ev}\\%$ in the \\texttt{{eval\\_grid}} family, which is almost entirely fault "
        "episodes --- and in the pre-/post-event balance. By construction "
        "pre\\,+\\,post\\,+\\,quiescent\\,=\\,fault-episode\\,+\\,switching-episode\\,+\\,"
        "quiescent\\,=\\,all negatives, and positives\\,+\\,negatives\\,=\\,total; the artefact "
        "checker asserts both. Derived from the committed partitions and the released window labels "
        "with the benchmark's frozen \\ac{ed} label deriver.}\n"
        "\\label{tab:supp_ed_negative_support}\n"
        + pl.si_panels(
            [
                "\\begin{tabular}{ll r rrr r}\n\\toprule\n"
                " & & & \\multicolumn{3}{c}{\\textbf{Negatives by position}} & \\\\\n"
                "\\cmidrule(lr){4-6}\n"
                "\\textbf{Grid} & \\textbf{Family} & \\textbf{positive} & \\textbf{pre} & "
                "\\textbf{post} & \\textbf{quiesc.} & \\textbf{all neg.} \\\\\n\\midrule\n"
                + "\n".join(body_a)
                + "\n\\bottomrule\n\\end{tabular}",
                "\\begin{tabular}{ll rr r r}\n\\toprule\n"
                " & & \\multicolumn{2}{c}{\\textbf{Negatives by episode family}} & & \\\\\n"
                "\\cmidrule(lr){3-4}\n"
                "\\textbf{Grid} & \\textbf{Family} & \\textbf{fault ep.} & "
                "\\textbf{switch.\\ ep.} & \\textbf{total} & "
                "\\textbf{neg./total (\\%)} \\\\\n\\midrule\n"
                + "\n".join(body_b)
                + "\n\\bottomrule\n\\end{tabular}",
            ]
        )
        + "\\end{table}\n"
    )


# --------------------------------------------------------------------------------------------------
# Raw current-scale diagnostic (S: supp_tab_current_shift.tex)
# --------------------------------------------------------------------------------------------------
#: The diagnostic's statistic and window subset are PRE-SPECIFIED by committed benchmark code, not
#: chosen to reproduce a target number:
#:   * statistic  = ``baselines.threshold._current_statistic`` — the phase-maximum RMS current of a
#:                  window, i.e. the decision variable of the shipped overcurrent baseline and the
#:                  same primitive as the random forest's per-channel ``rms`` feature;
#:   * view       = the LOCAL observability view (the six channels [Ia,Ib,Ic,Va,Vb,Vc] of the
#:                  faulted line's near-end cubicle), which is what those two baselines consume;
#:   * subset     = the NON-ACTIVE windows of the fault-detection valid-sample set, i.e. the
#:                  pre-/post-fault (steady-state) windows of fault episodes — the operating-point
#:                  windows the original observation refers to and the exact windows on which the
#:                  collapsed detectors raise their false alarms;
#:   * aggregate  = median over windows (robust central tendency), plus the interquartile range.
#: Values are raw physical amperes as simulated, before any standardisation (the deep baselines
#: standardise on train; the RF/threshold consume these raw magnitudes directly).
_CURRENT_TASK = "fault_detection_local"


def current_scale_stats(root=None) -> dict:
    """``{(grid, split): {median, q1, q3, n}}`` of the per-window phase-maximum RMS current.

    ``split`` is ``train`` (the fitting distribution) or ``benchmark`` (the never-trained
    ``eval_grid`` family). Streams the local view from the memmap; features are never materialised
    in full."""
    import numpy as np

    from evemtbench.baselines.threshold import _current_statistic

    out = {}
    for grid, _ in pl.GRID_ROWS:
        for split in ("train", "benchmark"):
            view = _task_view(_CURRENT_TASK, grid, split, root=root)
            # the FD negatives: windows of a fault episode that do NOT overlap the fault, i.e. the
            # steady-state pre-/post-fault windows -- read off the frozen FD label vector itself.
            quiet = ~np.asarray(view.labels).astype(bool)
            stats, offset = [], 0
            for x, _y in view.iter_batches(8192):
                s = _current_statistic(np.asarray(x))
                stats.append(s[quiet[offset : offset + len(s)]])
                offset += len(s)
            vals = np.concatenate(stats) if stats else np.empty(0, np.float64)
            out[(grid, split)] = {
                "median": float(np.median(vals)),
                "q1": float(np.percentile(vals, 25)),
                "q3": float(np.percentile(vals, 75)),
                "n": int(vals.size),
            }
    return out


def current_shift_table(root=None) -> str:
    """SI table: raw steady-state current scale of the fitting distribution vs. the never-trained
    evaluation family, per grid, with the ratio."""
    st = current_scale_stats(root=root)
    body, ratios = [], []
    for grid, glabel in pl.GRID_ROWS:
        a, b = st[(grid, "train")], st[(grid, "benchmark")]
        ratio = b["median"] / a["median"]
        ratios.append(ratio)
        body.append(
            f"{glabel} & {a['median']:,.1f} & [{a['q1']:,.1f}, {a['q3']:,.1f}] & {a['n']:,} & "
            f"{b['median']:,.1f} & [{b['q1']:,.1f}, {b['q3']:,.1f}] & {b['n']:,} & "
            f"{ratio:,.1f}$\\times$ \\\\"
        )
    return (
        pl.provenance_header(
            "hpc/paper_artifacts/build_supp_tab_current_shift.py",
            [
                "src/evemtbench/splits/v1.0/held_out/<grid>/{train,benchmark}.txt",
                "$EVEMT_ROOT/{adapt_grid,benchmark}/windows/<grid>/X_*.raw  (released dataset,"
                " raw physical units)",
                "src/evemtbench/baselines/threshold.py::_current_statistic  (the statistic)",
                "src/evemtbench/tasks/valid_samples.py  (fault-detection valid-sample mask)",
            ],
        )
        + "\\begin{table}[!htbp]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.05}\n"
        "\\caption{Raw current scale of the fitting distribution versus the never-trained "
        "\\texttt{eval\\_grid} family. The per-window statistic is the \\emph{phase-maximum "
        "\\ac{rms} current}: the root-mean-square of each phase current over the $480$ samples "
        "of a $50\\,$ms window, maximised over the three phases --- the decision variable of the "
        "shipped overcurrent "
        "baseline and the same primitive as the random forest's per-channel \\ac{rms} feature. It "
        "is evaluated on the six channels of the \\emph{local} observability view (the faulted "
        "line's near-end cubicle), in \\textbf{raw simulated amperes, before any standardisation}. "
        "Windows are the \\emph{non-active} (steady-state) windows of the fault-detection "
        "valid-sample set, i.e.\\ the pre-/post-fault windows of fault episodes --- the operating "
        "point of the set, and exactly the windows on which a collapsed detector raises its false "
        "alarms. Aggregation is the median over windows, with the interquartile range in brackets; "
        "\\emph{ratio} is median(\\texttt{eval\\_grid})\\,/\\,median(train). $n$ is the number of "
        "windows entering each statistic. The statistic, view, window subset and aggregation are "
        "fixed by committed benchmark code and were specified before the values were computed.}\n"
        "\\label{tab:supp_current_shift}\n"
        + pl._SHRINK_TO_FIT
        + "\\begin{tabular}{l rlr rlr r}\n\\toprule\n"
        " & \\multicolumn{3}{c}{\\textbf{\\texttt{adapt\\_grid} train}} & "
        "\\multicolumn{3}{c}{\\textbf{\\texttt{eval\\_grid}}} & \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
        "\\textbf{Grid} & \\textbf{median (A)} & \\textbf{[IQR]} & \\textbf{$n$} & "
        "\\textbf{median (A)} & \\textbf{[IQR]} & \\textbf{$n$} & \\textbf{ratio} \\\\\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


_FL_VIEWS = ("line", "local")
_FL_ONE_ENDED_GRID = "cigre_mv"


def fl_locator_breakdown(root=None) -> list[dict]:
    """Fault-location MAE of the impedance locator and of the validation-selected learned reference,
    per grid, view and evaluation family, split by window position and, on CIGRE MV, by line.

    ``full`` windows lie entirely inside the fault interval (``overlap_ratio == 1``); ``partial``
    windows also contain samples before inception or after clearing (``0 < overlap_ratio < 1``). The one-ended lines come from
    the locator's own topology rule (:func:`evemtbench.baselines.impedance.lines_measured_at_far_end`).
    Errors are recomputed per seed from the committed prediction files joined to the task view, and
    each seed's overall MAE must equal the stored ``per_seed`` value (the join is exact)."""
    import numpy as np
    import pandas as pd

    from evemtbench.baselines.impedance import lines_measured_at_far_end
    from evemtbench.evaluation.predictions_io import align_to_view
    from evemtbench.tasks.task import load_task
    from evemtbench.version import results_version

    out = []
    for view in _FL_VIEWS:
        task_id = f"fault_location_{view}"
        tree = _REPO / "outputs" / results_version("held_out", task_id)
        for grid, _ in pl.GRID_ROWS:
            rows = load_rows(grid)
            task = load_task(task_id, "held_out", grid=grid, root=root)
            for split, _lab in _EVAL_FAMILIES:
                v = task.view(split)
                meta = v.metadata.reset_index(drop=True)
                y = np.asarray(v.labels, dtype=float)
                ov = np.concatenate(
                    [s.ws.y["overlap_ratio"].to_numpy()[s.rows] for s in v._sources]
                )
                keys = np.concatenate(
                    [s.ws.y[["sample_id", "window_idx"]].to_numpy()[s.rows] for s in v._sources]
                )
                if not (keys == meta[["sample_id", "window_idx"]].to_numpy()).all():
                    raise RuntimeError(f"{task_id}/{grid}/{split}: overlap_ratio not aligned")
                full = ov >= 1.0 - 1e-9
                groups = {"full": full, "partial": (ov > 0) & ~full}
                one_ended, _mirrored = lines_measured_at_far_end(v)
                if grid == _FL_ONE_ENDED_GRID and view == "line":
                    lines = meta["y_fault_line"].astype(str).to_numpy()
                    is_one = np.isin(lines, one_ended)
                    groups["one_ended"] = is_one
                    groups["two_ended"] = ~is_one
                choice = pl.best_cell_choice(rows, task_id, split, "mae", grid=grid)
                for role, baseline in (("locator", "impedance"), ("learned", choice["baseline"])):
                    d = tree / task_id / "held_out" / grid / baseline / split
                    res = json.loads((d / "result.json").read_text())
                    stored = {int(p["seed"]): p["metrics"]["mae"] for p in res["per_seed"]}
                    per = {g: [] for g in groups}
                    for pp in sorted(d.glob("predictions_seed*.parquet")):
                        seed = int(pp.stem.removeprefix("predictions_seed"))
                        err = np.abs(align_to_view(meta, pd.read_parquet(pp)).astype(float) - y)
                        if abs(err.mean() - stored[seed]) > 1e-9:
                            raise RuntimeError(f"{d}: seed {seed} MAE does not match result.json")
                        for g, m in groups.items():
                            per[g].append(float(err[m].mean()))
                    if len(per["full"]) != len(stored):
                        raise RuntimeError(f"{d}: prediction files do not cover every seed")
                    for g, m in groups.items():
                        out.append(
                            {
                                "view": view,
                                "grid": grid,
                                "set": split,
                                "role": role,
                                "baseline": baseline,
                                "group": g,
                                "n": int(m.sum()),
                                "mae": float(np.mean(per[g])),
                            }
                        )
    return out


def fl_locator_breakdown_table(root=None) -> str:
    """SI table: :func:`fl_locator_breakdown` — locator vs selected learned reference by window
    position, plus the one-ended CIGRE MV line."""
    recs = fl_locator_breakdown(root=root)
    at = {(r["view"], r["grid"], r["set"], r["role"], r["group"]): r for r in recs}
    lab = dict(pl.GRID_ROWS)

    def mae(view, grid, split, role, group):
        return f"{at[(view, grid, split, role, group)]['mae']:.1f}"

    def acs(view, grid, split):
        b = at[(view, grid, split, "learned", "full")]["baseline"]
        return pl._BASELINE_ACS.get(b, b)

    body = []
    for view in _FL_VIEWS:
        body.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{view} view}}}} \\\\")
        for grid, _ in pl.GRID_ROWS:
            for i, (split, slab) in enumerate(_EVAL_FAMILIES):
                n_full = at[(view, grid, split, "locator", "full")]["n"]
                n_part = at[(view, grid, split, "locator", "partial")]["n"]
                body.append(
                    (lab[grid] if i == 0 else "")
                    + f" & {slab} & {100 * n_part / (n_full + n_part):.0f} & "
                    + " & ".join(
                        mae(view, grid, split, role, g)
                        for g in ("full", "partial")
                        for role in ("locator", "learned")
                    )
                    + f" & {acs(view, grid, split)} \\\\"
                )
        body.append(r"\midrule")
    body.pop()
    one = []
    g = _FL_ONE_ENDED_GRID
    for split, slab in _EVAL_FAMILIES:
        n_one = at[("line", g, split, "locator", "one_ended")]["n"]
        n_two = at[("line", g, split, "locator", "two_ended")]["n"]
        one.append(
            f"{slab} & {100 * n_one / (n_one + n_two):.0f} & "
            + " & ".join(
                mae("line", g, split, role, grp)
                for grp in ("one_ended", "two_ended")
                for role in ("locator", "learned")
            )
            + f" & {acs('line', g, split)} \\\\"
        )
    return (
        pl.provenance_header(
            "hpc/paper_artifacts/build_supp_tab_fl_locator_breakdown.py",
            [
                "outputs/<fl version>/fault_location_{line,local}/held_out/<grid>/"
                "<impedance|selected>/{test,benchmark}/{result.json,predictions_seed*.parquet}",
                "outputs/<fl version>/fault_location_{line,local}/held_out/<grid>/<baseline>/val/"
                "result.json  (selection)",
                "$EVEMT_ROOT/{adapt_grid,benchmark}/windows/<grid>/y_*.parquet  (released dataset:"
                " overlap_ratio, y_fault_line)",
            ],
        )
        + "\\begin{table}[!htbp]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.05}\n"
        "\\caption{Fault-location error of the impedance-based locator and of the "
        "validation-selected learned reference by window position. \\ac{mae} in \\% of line "
        "length, five-seed mean for the learned reference (the locator is deterministic). "
        "\\emph{Full} windows lie entirely inside the fault interval; \\emph{partial} windows "
        "also contain samples before fault inception or after clearing, which can enter the "
        "one-cycle phasor estimate. "
        "\\emph{Partial \\%} is the share of partial windows in the evaluated set. The lower "
        "block splits the CIGR\\'E \\ac{mv} line view into line 8--14, whose line view repeats its "
        "single measured terminal so that the locator falls back to the one-terminal method, and "
        "all other lines. Computed from the stored per-seed predictions of the reported cells; the "
        "per-seed totals reproduce the reported values.}\n"
        "\\label{tab:supp_fl_locator_breakdown}\n"
        + pl._SHRINK_TO_FIT
        + "\\begin{tabular}{ll r rr rr l}\n\\toprule\n"
        " & & & \\multicolumn{2}{c}{\\textbf{full}} & \\multicolumn{2}{c}{\\textbf{partial}} & \\\\\n"
        "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
        "\\textbf{Grid} & \\textbf{Set} & \\textbf{partial \\%} & \\textbf{locator} & "
        "\\textbf{learned} & \\textbf{locator} & \\textbf{learned} & \\textbf{selected} \\\\\n"
        "\\midrule\n" + "\n".join(body) + "\n\\midrule\n"
        " & & & \\multicolumn{2}{c}{\\textbf{line 8--14}} & "
        "\\multicolumn{2}{c}{\\textbf{other lines}} & \\\\\n"
        "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
        "\\textbf{CIGR\\'E \\ac{mv}, line view} & \\textbf{Set} & \\textbf{8--14 \\%} & "
        "\\textbf{locator} & \\textbf{learned} & \\textbf{locator} & \\textbf{learned} & \\\\\n"
        "\\midrule\n"
        + "\n".join(" & " + r for r in one)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


# ==================================================================================================
# Registry — the SI artifacts owned by this module (one entry = one file = one build_*.py).
# ==================================================================================================
SUPP_ARTIFACTS: dict[str, dict] = {
    "supp_tab_partition_audit.tex": {
        "builder": partition_audit_table,
        "script": "hpc/paper_artifacts/build_supp_tab_partition_audit.py",
        "dataset": False,
    },
    "supp_tab_transfer_scratch.tex": {
        "builder": transfer_scratch_table,
        "script": "hpc/paper_artifacts/build_supp_tab_transfer_scratch.py",
        "dataset": False,
    },
    "supp_tab_primary_valid_counts.tex": {
        "builder": primary_valid_counts_table,
        "script": "hpc/paper_artifacts/build_supp_tab_primary_valid_counts.py",
        "dataset": False,
    },
    "supp_tab_baseline_cfg.tex": {
        "builder": baseline_cfg_table,
        "script": "hpc/paper_artifacts/build_supp_tab_baseline_cfg.py",
        "dataset": False,
    },
    "supp_tab_ec_support.tex": {
        "builder": ec_support_table,
        "script": "hpc/paper_artifacts/build_supp_tab_ec_support.py",
        "dataset": True,
    },
    "supp_tab_ed_negative_support.tex": {
        "builder": ed_negative_support_table,
        "script": "hpc/paper_artifacts/build_supp_tab_ed_negative_support.py",
        "dataset": True,
    },
    "supp_tab_current_shift.tex": {
        "builder": current_shift_table,
        "script": "hpc/paper_artifacts/build_supp_tab_current_shift.py",
        "dataset": True,
    },
    "supp_tab_fl_locator_breakdown.tex": {
        "builder": fl_locator_breakdown_table,
        "script": "hpc/paper_artifacts/build_supp_tab_fl_locator_breakdown.py",
        "dataset": True,
    },
}


def dataset_free_artifacts() -> dict[str, str]:
    """``{filename: latex}`` for the SI artifacts that need only committed repo/outputs artefacts —
    the set the pre-commit freshness gate can rebuild anywhere."""
    return {
        n: pl.manuscript_caption(n, spec["builder"]())
        for n, spec in SUPP_ARTIFACTS.items()
        if not spec["dataset"]
    }


__all__ = [
    "DATASET_BACKED",
    "SUPP_ARTIFACTS",
    "baseline_cfg_table",
    "current_scale_stats",
    "current_shift_table",
    "dataset_free_artifacts",
    "ec_support_counts",
    "ec_support_table",
    "ed_negative_composition",
    "ed_negative_support_table",
    "fl_locator_breakdown",
    "fl_locator_breakdown_table",
    "partition_audit_rows",
    "protocol_seeds",
    "partition_audit_table",
    "primary_valid_counts_table",
    "transfer_scratch_rows",
    "transfer_scratch_table",
    "valid_counts",
]
