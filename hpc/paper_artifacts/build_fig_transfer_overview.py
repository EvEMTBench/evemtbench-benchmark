#!/usr/bin/env python3
"""Standalone builder → paper/manuscript/figures/fig_transfer_overview.pdf.

A source × function panel grid of cross-grid transfer on the never-trained benchmark family: for
each pre-training source grid (row, voltage order) and fault-triad function (column), the best
learned baseline's headline metric, zero-shot vs fine-tune, across the target grids (voltage order),
at the differential ``line`` view. The cross-grid transfer protocols apply only to the fixed-width
``local``/``line`` views (a model with a fixed input head cannot ingest the grid-dependent
``global`` channel count — that is what the planned cross-topology + GNN track is for), so the
figure covers the three fault-triad functions that carry transferable weights (FD/FC/FL); the
event-scope functions (ED/EC) are global-only and held-out-only and thus have no transfer panel.
Reuses the same selection as the transfer tables (``paper_lib.best_transfer_row``), so the figure
and ``tab_transfer_*.tex`` agree cell-for-cell.

Requires ``matplotlib`` (NOT in the sweep's ``py_dl`` env — provision it in a separate env so the
running sweep is untouched). This is compute — run via ``srun``, never the login node. Example:

    srun --partition=a100mig --gres=gpu:a100small:1 --time=10:00 bash -c \
      'CUDA_VISIBLE_DEVICES= PYTHONPATH=src <python-with-matplotlib> \
       hpc/paper_artifacts/build_fig_transfer_overview.py'

The PDF is committed ONLY from a frozen tag (like the populated tables); pre-freeze ``main.tex`` keeps
its ``\\figplaceholder`` box.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("MPLBACKEND", "Agg")  # headless PDF rendering (no display / no X server)

import matplotlib.pyplot as plt  # noqa: E402

# Embed TrueType fonts (Type 42), not Type 3 bitmap fonts, so the PDF meets the journal's
# scalable-font figure requirement (see paper/manuscript/document_check.md).
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

from paper_lib import (  # noqa: E402
    FIG_DIR,
    GRID_ROWS,
    PAPER_FUNCTIONS,
    PRETRAIN_GRIDS,
    _transfer_index,
    _transfer_val_index,
    best_transfer_row,
)

NAME = "fig_transfer_overview.pdf"
OBS_VIEW = "line"  # differential two-terminal view; transfer applies to fixed-width local/line only
# Only the fault-triad functions carry cross-grid transfer (deep baselines, fixed 6/12-ch input);
# ED/EC are global-only + held_out-only, so they have no transfer cell to plot.
TRANSFER_FUNCTIONS = ("fd", "fc", "fl")
_PROTOS = [("transfer_zeroshot", "zero-shot"), ("transfer_finetune", "fine-tune")]
_GRID_SHORT = {
    "cigre_mv": "CIGRÉ MV\n20 kV",
    "double_line": "Double Line\n110 kV",
    "testgrid_110kv": "Prot. Ref.\n110 kV",
    "ieee39": "IEEE-39\n345 kV",
}
_FUNC_TITLE = {
    "fd": "Fault detection",
    "fc": "Fault classification",
    "fl": "Fault localisation",
    "ed": "Event detection",
    "ec": "Event classification",
}
_METRIC_AXIS = {"balanced_accuracy": "balanced accuracy ↑", "mae": "MAE % ↓"}
# One row of panels per pre-training source (voltage order); label the left column with the source.
_SOURCE_LABEL = {
    "template_20kv": "from 20 kV",
    "template_110kv": "from 110 kV",
    "template_345kv": "from 345 kV",
}


def build_figure():
    """Build the matplotlib Figure (does not write it) — returns the Figure.

    A ``source × function`` panel grid: one row per pre-training source grid (20/110/345 kV, the
    ``PRETRAIN_GRIDS`` order) and one column per fault-triad function. ``sharey="col"`` keeps each
    function's metric axis comparable across sources, so the matched-voltage diagonal (a source
    transferring best to the same-voltage target) reads off directly. Selection matches the transfer
    tables cell-for-cell (``best_transfer_row`` per source)."""
    idx = _transfer_index("outputs")
    val_idx = _transfer_val_index("outputs")
    funcs = [pf for pf in PAPER_FUNCTIONS if pf[0] in TRANSFER_FUNCTIONS]
    grids = [g for g, _ in GRID_ROWS]
    sources = list(PRETRAIN_GRIDS)
    x = list(range(len(grids)))
    width = 0.38
    fig, axes = plt.subplots(
        len(sources),
        len(funcs),
        figsize=(4.0 * len(funcs), 2.7 * len(sources)),
        squeeze=False,
        sharey="col",
    )
    for si, source in enumerate(sources):
        for fi, (key, _name, stem, metric, _label) in enumerate(funcs):
            ax = axes[si][fi]
            task_id = f"{stem}_{OBS_VIEW}"
            any_val = False
            for j, (proto, plabel) in enumerate(_PROTOS):
                offs = [xi + (j - 0.5) * width for xi in x]
                rows = [
                    best_transfer_row(idx, task_id, g, proto, metric, source, val_idx)
                    for g in grids
                ]
                pairs = [
                    (o, r.mean(metric)) for o, r in zip(offs, rows, strict=True) if r is not None
                ]
                if pairs:
                    any_val = True
                    ax.bar([o for o, _ in pairs], [v for _, v in pairs], width=width, label=plabel)
            if si == 0:  # the metric differs per column (accuracy vs MAE %) → name it in the title
                ax.set_title(
                    f"{_FUNC_TITLE.get(key, key)}\n{_METRIC_AXIS.get(metric, metric)}", fontsize=9
                )
            ax.set_xticks(x)
            if si == len(sources) - 1:
                ax.set_xticklabels([_GRID_SHORT.get(g, g) for g in grids], fontsize=7)
            else:
                ax.set_xticklabels([])
            if fi == 0:  # left column: a rotated row label naming the pre-training source
                ax.text(
                    -0.42,
                    0.5,
                    _SOURCE_LABEL.get(source, source),
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                )
            ax.grid(axis="y", alpha=0.3)
            if not any_val:
                ax.text(
                    0.5,
                    0.5,
                    "no result yet",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9,
                    color="0.5",
                )
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(rect=(0.02, 0, 1, 0.94))
    if handles:  # legend in the reserved top strip (title text lives in the LaTeX caption)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=2,
            frameon=False,
            fontsize=10,
        )
    return fig


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure = build_figure()
    out = FIG_DIR / NAME
    figure.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
