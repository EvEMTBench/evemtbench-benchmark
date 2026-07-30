#!/usr/bin/env python3
"""Standalone builder → paper/manuscript/figures/fig_synthesis.pdf.

The main-text result synthesis: a single at-a-glance view of the ``held_out`` headline results for
the five primary functions, so the reader sees the whole benchmark surface without paging through
five near-identical tables. Rows are ``(function, observability view)``; columns are the four
reference grids, grouped into the in-distribution ``test`` block and the never-trained
``benchmark`` block so the in-distribution-vs-shifted comparison is visual. Two panels keep the two
metric families on their own native colour scales (never one colourbar for two metrics):

  * top  — balanced accuracy ↑ (FD, FC, ED, EC), sequential ``cividis`` (colour-blind-safe);
  * bottom — fault localisation MAE in % of line length ↓ (FL), reversed sequential.

Every cell is the best learned baseline (RF/MLP/GRU/CNN/ResNet), read through the same
``paper_lib.best_cell_choice`` the audit paper-numbers page and ``tab_res_*.tex`` use, so the figure
agrees with the tables cell-for-cell. Pending cells (no result yet) are left blank and marked
``P``; the raw value is annotated in every populated cell.

Requires ``matplotlib`` (NOT in the sweep's ``py_dl`` env — provision it separately). This is
compute — run via ``srun``, never the login node. Example:

    srun --partition=a100mig --gres=gpu:a100small:1 --time=10:00 bash -c \
      'CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
       "$PAPERFIGS_PY" hpc/paper_artifacts/build_fig_synthesis.py'

(``PAPERFIGS_PY`` = a python with matplotlib, e.g. from the untracked hpc/clusters/local.env.)

The PDF is committed alongside the populated tables (a live gated view of the current sweep;
provisional until a frozen tag).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("MPLBACKEND", "Agg")  # headless PDF rendering (no display / no X server)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

# Embed TrueType fonts (Type 42), not Type 3 bitmap fonts, so the PDF meets the journal's
# scalable-font figure requirement (see paper/manuscript/document_check.md).
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# paper_lib puts hpc/ on sys.path and re-exports load_rows from make_results, so import it from
# there (importing make_results directly here would run before hpc/ is on the path).
from paper_lib import GRID_ROWS, best_cell_choice, load_rows  # noqa: E402

NAME = "fig_synthesis.pdf"
TEST_SETS = ("test", "benchmark")

# (row label, task-id stem, observability view) for the balanced-accuracy panel …
_BACC_ROWS = [
    ("FD · local", "fault_detection", "local"),
    ("FD · line", "fault_detection", "line"),
    ("FD · global", "fault_detection", "global"),
    ("FC · local", "fault_classification", "local"),
    ("FC · line", "fault_classification", "line"),
    ("FC · global", "fault_classification", "global"),
    ("ED · global", "event_detection", "global"),
    ("EC · global", "event_classification", "global"),
]
# … and the fault-localisation panel (its own metric + scale).
_FL_ROWS = [
    ("FL · local", "fault_location", "local"),
    ("FL · line", "fault_location", "line"),
    ("FL · global", "fault_location", "global"),
]
_GRID_SHORT = {
    "cigre_mv": "CIGRÉ\n20 kV",
    "double_line": "Double\n110 kV",
    "testgrid_110kv": "Prot.Ref\n110 kV",
    "ieee39": "IEEE-39\n345 kV",
}


def _matrix(rows, metric):
    """(values, present) arrays of shape (len(rows), 2*len(grids)); columns = test-block then
    benchmark-block, each in GRID_ROWS order. ``present`` is False for pending cells."""
    grids = [g for g, _ in GRID_ROWS]
    per_grid = {g: load_rows(g) for g in grids}
    n_col = len(TEST_SETS) * len(grids)
    values = np.full((len(rows), n_col), np.nan)
    for i, (_label, stem, view) in enumerate(rows):
        task_id = f"{stem}_{view}"
        for ts_i, ts in enumerate(TEST_SETS):
            for g_i, g in enumerate(grids):
                choice = best_cell_choice(per_grid[g], task_id, ts, metric)
                if choice is not None:
                    values[i, ts_i * len(grids) + g_i] = choice["mean"]
    return values, ~np.isnan(values)


def _draw_panel(ax, rows, values, cmap, vmin, vmax, fmt):
    n_row, n_col = values.shape
    masked = np.ma.masked_invalid(values)
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(color="0.92")  # pending / no result yet
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    for i in range(n_row):
        for j in range(n_col):
            v = values[i, j]
            if np.isnan(v):
                ax.text(j, i, "P", ha="center", va="center", fontsize=6.5, color="0.55")
                continue
            # white ink on dark cells, dark ink on light — by the cell colour's luminance
            r, gg, b, _ = cmap(norm(v))
            lum = 0.299 * r + 0.587 * gg + 0.114 * b
            ax.text(
                j,
                i,
                fmt.format(v),
                ha="center",
                va="center",
                fontsize=6.8,
                color="white" if lum < 0.5 else "0.12",
            )
    ax.set_yticks(range(n_row))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_xticks([])
    # in-distribution | never-trained divider
    ax.axvline(n_col / 2 - 0.5, color="white", lw=2.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def build_figure():
    """Build the matplotlib Figure (does not write it) — returns the Figure."""
    grids = [g for g, _ in GRID_ROWS]
    n_grid = len(grids)
    bacc_vals, _ = _matrix(_BACC_ROWS, "balanced_accuracy")
    fl_vals, _ = _matrix(_FL_ROWS, "mae")

    fig = plt.figure(figsize=(9.2, 6.2))
    gs = GridSpec(
        2,
        2,
        height_ratios=[len(_BACC_ROWS), len(_FL_ROWS)],
        width_ratios=[40, 1],
        hspace=0.16,
        wspace=0.03,
        figure=fig,
    )
    ax_b = fig.add_subplot(gs[0, 0])
    ax_f = fig.add_subplot(gs[1, 0])
    cax_b = fig.add_subplot(gs[0, 1])
    cax_f = fig.add_subplot(gs[1, 1])

    im_b = _draw_panel(ax_b, _BACC_ROWS, bacc_vals, "cividis", 0.40, 1.0, "{:.2f}")
    im_f = _draw_panel(ax_f, _FL_ROWS, fl_vals, "cividis_r", 0.0, 32.0, "{:.0f}")

    fig.colorbar(im_b, cax=cax_b).set_label("balanced accuracy ↑", fontsize=7.5)
    fig.colorbar(im_f, cax=cax_f).set_label("MAE % ↓", fontsize=7.5)
    cax_b.tick_params(labelsize=6.5)
    cax_f.tick_params(labelsize=6.5)

    # column headers: the two blocks, then the per-grid labels under each
    for ax in (ax_b, ax_f):
        ax.set_xlim(-0.5, 2 * n_grid - 0.5)
    ax_b.set_xticks(range(2 * n_grid))
    ax_b.set_xticklabels([_GRID_SHORT[g] for g in grids] * 2, fontsize=6.5)
    ax_b.xaxis.set_ticks_position("top")
    ax_b.tick_params(axis="x", length=0)
    ax_f.set_xticks(range(2 * n_grid))
    ax_f.set_xticklabels([_GRID_SHORT[g] for g in grids] * 2, fontsize=6.5)
    ax_f.tick_params(axis="x", length=0)
    # block captions above the grid labels
    ax_b.text(
        (n_grid - 1) / 2,
        -1.05,
        "In-distribution (test)",
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        transform=ax_b.transData,
    )
    ax_b.text(
        n_grid + (n_grid - 1) / 2,
        -1.05,
        "Never-trained (benchmark)",
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        transform=ax_b.transData,
    )
    fig.suptitle(
        "Benchmark v1.0.0 — held-out reference baselines (best of five, validation-selected). "
        "P = pending.",
        y=0.99,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


if __name__ == "__main__":
    from paper_lib import FIG_DIR

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure = build_figure()
    out = FIG_DIR / NAME
    figure.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
