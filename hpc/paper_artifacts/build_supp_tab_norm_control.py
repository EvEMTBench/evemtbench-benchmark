#!/usr/bin/env python3
"""Standalone builder → paper/manuscript/generated/supp_tab_norm_control.tex
(the input-normalization control for the cross-grid transfer comparison).

    PYTHONPATH=src python hpc/paper_artifacts/build_supp_tab_norm_control.py --abl <tree>
    PYTHONPATH=src python hpc/paper_artifacts/build_supp_tab_norm_control.py --abl <tree> --check

WHY THIS BUILDER IS NOT IN THE COMMIT-TIME REGISTRY
    ``check_results_fresh._paper_fragment_checks`` rebuilds every fragment named by
    ``build_paper_tables`` / ``build_supplement_tables`` / ``build_transfer_tables`` /
    ``build_spec_tables`` / ``dataset_free_artifacts`` and byte-compares it. This fragment is
    deliberately absent from all of those, for the same reason the three ``DATASET_BACKED`` SI
    fragments are: it needs an input the repository does not contain. Here that input is the
    CONTROL TREE — an isolated ``--out`` root holding ``held_out`` cells re-run with
    ``--normalize per_window``. It is an analysis-only ablation by decision: it is not in
    ``outputs/``, not in coverage, not in the leaderboard, and does not move any version
    identifier. A machine without the control tree therefore cannot rebuild this fragment, so
    gating it at commit time would fail everywhere except one host. Regenerate it explicitly
    (``make check-norm-control ABL=...``) on a host that has the tree.

    It therefore also does NOT live in ``paper/manuscript/generated/``. That directory is defined as
    fragments which are a pure function of ``outputs/``, and it is consumed by a bare
    ``GEN_DIR.glob("*.tex")`` in ``make_results_bundle`` (:219) that copies every fragment into the
    PUBLIC results bundle ``results/<version>/tables/`` — which the freshness gate then byte-checks.
    Dropping an ablation fragment in there put an unrebuildable file into the published bundle and
    turned the gate red, which is how this was caught. It lives in
    ``paper/manuscript/ablation/`` instead; the manuscript takes it with
    ``\\input{ablation/supp_tab_norm_control.tex}``.

WHAT IT REPORTS
    Three arms per (function, target grid, view), all fitted on the TARGET grid's ``adapt_grid``
    training partition, selected on that grid's validation split out of the four deep baselines,
    and reported on that grid's never-trained ``eval_grid`` family:

        A  held_out, normalize=none        (the published from-scratch reference)
        C  held_out, normalize=per_window  (this control)
        B  transfer_finetune               (pre-trained init, normalize=per_window)

    ``C-A`` is the cost of the transfer protocols' normalization at random initialisation; ``B-C``
    is the effect of the pre-trained initialisation under that normalization. Deliberately NOT
    presented as a 2x2 decomposition: ``(B-A) = (C-A) + (B-C)`` is an algebraic identity, and the
    interaction term is not identifiable without the (pretrained, none) cell — which is
    inadmissible, since per-window normalization is what makes one model portable across
    20--345 kV at all.

    B is the best (per the metric direction) over the three pre-training sources, and the source
    that achieves it is named, because the control is source-independent while B is not.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paper_lib as pl

from evemtbench.coverage import DEEP_BASELINES, PRETRAIN_GRIDS
from make_results import load_rows

NAME = "supp_tab_norm_control.tex"
#: NOT paper_lib.GEN_DIR — see the module docstring: generated/ is glob-copied into the public
#: results bundle and byte-gated, and this ablation fragment cannot be rebuilt without the
#: control tree.
OUT_DIR = Path(__file__).resolve().parents[2] / "paper" / "manuscript" / "ablation"
VIEWS = ("local", "line")
FUNCS = [
    ("fd", r"\ac{fd}", "fault_detection", "balanced_accuracy", r"bal.\ acc.\ $\uparrow$"),
    ("fc", r"\ac{fc}", "fault_classification", "balanced_accuracy", r"bal.\ acc.\ $\uparrow$"),
    ("fl", r"\ac{fl}", "fault_location", "mae", r"\acs{mae}\,\% $\downarrow$"),
]
_SRC_SHORT = {"template_20kv": "20", "template_110kv": "110", "template_345kv": "345"}


def _fmt(v, metric):
    if v is None:
        return r"\pending"
    return f"{v:.3f}" if metric == "balanced_accuracy" else f"{v:.2f}"


def _signed(v, metric):
    if v is None:
        return r"\pending"
    return f"{v:+.3f}" if metric == "balanced_accuracy" else f"{v:+.2f}"


def rows(abl: Path) -> list[dict]:
    """One record per (function, grid, view) with the three arms and the two contrasts."""
    repo = Path.cwd()
    per_grid_a = {g: load_rows(g) for g, _ in pl.GRID_ROWS}
    idx, val_idx = pl._transfer_index("outputs"), pl._transfer_val_index("outputs")
    os.chdir(abl)  # rescore/load_rows resolve outputs/<version> relative to cwd
    try:
        per_grid_c = {g: load_rows(g) for g, _ in pl.GRID_ROWS}
    finally:
        os.chdir(repo)

    out = []
    for key, flabel, stem, metric, arrow in FUNCS:
        for grid, glabel in pl.GRID_ROWS:
            for view in VIEWS:
                tid = f"{stem}_{view}"
                a = pl.best_cell_choice(per_grid_a[grid], tid, "benchmark", metric, DEEP_BASELINES)
                c = pl.best_cell_choice(per_grid_c[grid], tid, "benchmark", metric, DEEP_BASELINES)
                best_b, best_src = None, None
                for src in PRETRAIN_GRIDS:
                    r = pl.best_transfer_row(
                        idx, tid, grid, "transfer_finetune", metric, src, val_idx
                    )
                    if r is None:
                        continue
                    v = r.mean(metric)
                    if best_b is None or pl.METRIC_DIRECTION.get(metric, 1) * (v - best_b) > 0:
                        best_b, best_src = v, src
                out.append(
                    {
                        "key": key,
                        "flabel": flabel,
                        "arrow": arrow,
                        "grid": grid,
                        "grid_label": glabel,
                        "view": view,
                        "metric": metric,
                        "A": None if a is None else a["mean"],
                        "C": None if c is None else c["mean"],
                        "B": best_b,
                        "B_src": best_src,
                    }
                )
    return out


def table(abl: Path) -> str:
    recs = {(r["key"], r["grid"], r["view"]): r for r in rows(abl)}
    body: list[str] = []
    for fi, (key, flabel, _stem, metric, arrow) in enumerate(FUNCS):
        for gi, (grid, glabel) in enumerate(pl.GRID_ROWS):
            cells: list[str] = []
            for view in VIEWS:
                r = recs[(key, grid, view)]
                ca = None if (r["C"] is None or r["A"] is None) else r["C"] - r["A"]
                bc = None if (r["B"] is None or r["C"] is None) else r["B"] - r["C"]
                src = "" if r["B_src"] is None else f"\\,({_SRC_SHORT[r['B_src']]})"
                cells += [
                    _fmt(r["A"], metric),
                    _fmt(r["C"], metric),
                    _fmt(r["B"], metric) + src,
                    _signed(ca, metric),
                    _signed(bc, metric),
                ]
            head = (
                f"\\multirow{{{len(pl.GRID_ROWS)}}}{{*}}{{{flabel}\\,({arrow})}}" if gi == 0 else ""
            )
            body.append(f"{head:<40} & {glabel:<24} & " + " & ".join(cells) + r" \\")
        if fi < len(FUNCS) - 1:
            body.append(r"\midrule")
    return (
        pl.provenance_header(
            "hpc/paper_artifacts/build_supp_tab_norm_control.py",
            [
                "outputs/1.0.0/{fault_detection,fault_classification,fault_location}_{local,line}"
                "/held_out/<grid>/<baseline>/{val,benchmark}/result.json  (arm A)",
                "<control tree>/1.0.0/.../held_out/<grid>/<baseline>/{val,benchmark}/result.json"
                "  (arm C: --normalize per_window; ANALYSIS-ONLY, not in outputs/)",
                "outputs/1.1.0/.../transfer_finetune/<grid>/<baseline>/[pretrain-<src>/]"
                "{val,benchmark}/result.json  (arm B)",
            ],
        )
        + "\\begin{table*}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
        "\\caption{Input-normalization control for the cross-grid transfer comparison. "
        "\\textbf{A} is the from-scratch reference of Table~\\ref{tab:supp_transfer_scratch} "
        "(\\texttt{held\\_out}, \\texttt{normalize=none}); \\textbf{C} repeats it with the "
        "per-window, per-channel normalization the transfer protocols require for voltage "
        "invariance, changing nothing else; \\textbf{B} is target-grid fine-tuning, reported for "
        "the best of the three pre-training sources with that source in parentheses. All three are "
        "fitted on the target grid's \\texttt{adapt\\_grid} training partition, select their "
        "architecture on that grid's validation split out of the four deep baselines, and are "
        "scored on its never-trained \\texttt{eval\\_grid} family. $C-A$ is therefore the cost of "
        "the normalization at random initialisation and $B-C$ the effect of the pre-trained "
        "initialisation under that normalization; the (pre-trained, \\texttt{none}) condition is "
        "not evaluated because that normalization is not voltage-invariant and so is not an "
        "admissible cross-grid pre-training configuration. This control is an ablation: it adds no "
        "protocol, leaves the committed \\texttt{held\\_out} and transfer artifacts unchanged, and "
        "moves no version identifier."
        + pl._pending_legend("\n".join(body), " \\pending{}~= pending.")
        + "}\n"
        "\\label{tab:supp_norm_control}\n"
        "\\begin{tabular}{ll rrrrr rrrrr}\n\\toprule\n"
        " & & \\multicolumn{5}{c}{\\textbf{local}} & \\multicolumn{5}{c}{\\textbf{line}} \\\\\n"
        "\\cmidrule(lr){3-7}\\cmidrule(lr){8-12}\n"
        "\\textbf{Function} & \\textbf{Target grid} "
        "& \\textbf{A} & \\textbf{C} & \\textbf{B} & $C-A$ & $B-C$ "
        "& \\textbf{A} & \\textbf{C} & \\textbf{B} & $C-A$ & $B-C$ \\\\\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--abl", required=True, help="control tree (the dir CONTAINING outputs/)")
    ap.add_argument("--check", action="store_true", help="verify only; write nothing")
    args = ap.parse_args()
    content = table(Path(args.abl).resolve())
    dest = OUT_DIR / NAME
    if args.check:
        cur = dest.read_text() if dest.is_file() else ""
        if cur == content:
            print(f"OK {dest}")
        else:
            print(f"STALE {dest} — regenerate (drop --check)")
            sys.exit(1)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
        print(f"wrote {dest}")
