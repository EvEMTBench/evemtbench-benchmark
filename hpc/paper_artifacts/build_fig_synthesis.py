#!/usr/bin/env python3
"""Standalone builder → paper/manuscript/figures/fig_synthesis.{pdf,svg,png}.

The main-text result synthesis: a single at-a-glance view of the ``held_out`` headline results for
the five primary functions, so the reader sees the whole benchmark surface without paging through
five near-identical tables. Rows are ``(function, observability view)``; columns are the four
reference grids, grouped into the in-distribution ``test`` block and the never-trained
``eval_grid`` block so the in-distribution-vs-shifted comparison is visual. Two panels keep the two
metric families on their own native colour scales (never one colourbar for two metrics):

  * top  — balanced accuracy ↑ (FD, FC, ED, EC), sequential single-hue ``Blues``;
  * bottom — fault localization MAE in % of line length ↓ (FL), the same ramp reversed.

Because the two metrics point in opposite directions, the ramp is reversed for MAE so that in BOTH
panels a **darker cell is a better result** — and, since ``Blues`` is monotonic in lightness, that
reading survives greyscale printing. The value is annotated in every cell regardless, in black or
white by measured contrast against the cell fill, so nothing is carried by colour alone.

Every cell is the best learned baseline (RF/MLP/GRU/CNN/ResNet), read through the same
``paper_lib.best_cell_choice`` the audit paper-numbers page and ``tab_res_*.tex`` use, so the figure
agrees with the tables cell-for-cell. There is no in-figure title: the LaTeX caption
(``main.tex``, ``fig:synthesis``) supplies the version, the protocol and the selection rule, as it
already does for ``fig_transfer_overview``.

Authored at exactly ``paperfig_style.FIG_WIDTH`` (= ``\\textwidth``), so LaTeX places it at scale
1.0 and the point sizes set here are the point sizes on the page.

Requires ``matplotlib`` (NOT in the sweep's ``py_dl`` env — provision it separately). This is
compute — run via ``srun``, never the login node. Example:

    srun --partition=a100mig --gres=gpu:a100small:1 --time=10:00 bash -c \
      'CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
       "$PAPERFIGS_PY" hpc/paper_artifacts/build_fig_synthesis.py'

(``PAPERFIGS_PY`` = a python with matplotlib, e.g. from the untracked hpc/clusters/local.env.)

The artifacts are committed alongside the populated tables (a live gated view of the current sweep;
provisional until a frozen tag).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("MPLBACKEND", "Agg")  # headless rendering (no display / no X server)

import matplotlib.pyplot as plt
import numpy as np

# paper_lib is the data seam AND the sys.path shim: it puts hpc/ and src/ on the path and
# re-exports the make_results helpers (warn_once), so nothing here imports across hpc/ directly.
import paperfig_style as style
from matplotlib.gridspec import GridSpec
from paper_lib import (
    GRID_ROWS,
    SYNTHESIS_BACC_ROWS,
    SYNTHESIS_FL_ROWS,
    held_out_rows,
    synthesis_matrix,
    warn_once,
)

STEM = "fig_synthesis"
TEST_SETS = ("test", "benchmark")
# DISPLAY labels for the two held_out families. The manuscript calls the never-trained family
# `eval_grid` to keep it distinct from "the benchmark" (the evaluation framework as a whole); the
# on-disk/result-path key stays `benchmark`, exactly as in paper_lib._TS_LABEL.
_BLOCK_LABEL = {"test": "In-distribution (test)", "benchmark": "Evaluation-only (eval_grid)"}

# The two panels' rows live in paper_lib so the figure-value test checks the same coverage.
_BACC_ROWS = SYNTHESIS_BACC_ROWS
_FL_ROWS = SYNTHESIS_FL_ROWS

# Scale limits, unchanged from the pre-refinement figure so no cell's reading changes.
_BACC_LIM = (0.40, 1.00)
_MAE_LIM = (0.0, 32.0)
_BACC_TICKS = [0.4, 0.6, 0.8, 1.0]
_MAE_TICKS = [0, 8, 16, 24, 32]

# Extra whitespace between the two evaluation families, in cell widths. The families are the top of
# this figure's hierarchy, so the boundary has to be visible before any individual cell is read; a
# real gap does that without a box, a shaded band, or a heavy rule.
_FAMILY_GAP = 0.15

# Type ladder. The family headers are bold but set SMALLER than the task labels: with a two-weight
# font (see paperfig_style._SANS_STACK) weight is the only prominence axis available, so size is
# what keeps a bold header from overpowering the row it labels.
_CELL_FONTSIZE = 7.0
_ROW_FONTSIZE = 8.0  # task labels (FD · local): the size ceiling for the family headers
_COL_FONTSIZE = 7.4  # grid names, two lines (CIGRÉ MV / 20 kV)
_BLOCK_FONTSIZE = 7.9  # family headers, bold: +0.5 over the grid names, <= the task labels
_CBAR_LABEL_FONTSIZE = 7.8
_CBAR_TICK_FONTSIZE = 6.8  # deliberately left behind — the scale is the last thing read
#: White separator between cells. At the journal's 1 pt stroke floor this is the widest gap in the
#: figure, so the cells are drawn edge-to-edge and the separator is the only thing between them —
#: it must stay white, not grey, or the block reads as a ruled table instead of a heatmap.
_CELL_GAP_LINEWIDTH = style.MIN_LINEWIDTH


def _values(matrix) -> np.ndarray:
    """``(len(rows), n_col)`` float array of cell means; NaN where a cell has no result."""
    return np.array(
        [[np.nan if cell is None else cell["mean"] for cell in row] for row in matrix], dtype=float
    )


def _col_x(j: int, n_col: int) -> float:
    """Data-space centre of matrix column ``j``, with the second family block pushed right.

    Single source of truth for the shifted geometry: the cell fills, the value annotations, the
    major ticks and the inter-cell hairlines all derive their x from here, so they cannot drift
    apart if :data:`_FAMILY_GAP` changes."""
    return j + (_FAMILY_GAP if j >= n_col // 2 else 0.0)


def _draw_panel(ax, rows, matrix, cmap, vmin, vmax, fmt):
    values = _values(matrix)
    n_row, n_col = values.shape
    half = n_col // 2
    cmap = cmap.copy()
    cmap.set_bad(color=style.NO_RESULT_FILL)
    masked = np.ma.masked_invalid(values)
    # Two imshow calls on ONE axes, the right block offset by _FAMILY_GAP, rather than one image or
    # two axes: a single image cannot carry an interior gap, and splitting the axes would put the
    # upper and lower panels' column alignment at the mercy of two independent layout solves.
    # Same cmap/vmin/vmax for both halves, so no cell's colour changes.
    im = None
    for lo, hi in ((0, half), (half, n_col)):
        left = _col_x(lo, n_col) - 0.5
        im = ax.imshow(
            masked[:, lo:hi],
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",  # never let a cell edge blur
            # (left, right, bottom, top) — bottom > top keeps row 0 at the top, as origin="upper"
            extent=(left, left + (hi - lo), n_row - 0.5, -0.5),
        )
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    for i in range(n_row):
        for j in range(n_col):
            v = values[i, j]
            if np.isnan(v):
                # Unreachable while the sweep is complete. Leave the cell blank rather than print a
                # marker the caption would then have to explain, and say so in the build log.
                warn_once(f"{STEM}: no result for {rows[i][0]}, column {j} — cell left blank")
                continue
            ax.text(
                _col_x(j, n_col),
                i,
                fmt.format(v),
                ha="center",
                va="center",
                fontsize=_CELL_FONTSIZE,
                color=style.annot_color(cmap(norm(v))),
            )
    ax.set_yticks(range(n_row))
    ax.set_yticklabels([r[0] for r in rows], fontsize=_ROW_FONTSIZE)
    ax.set_xticks([_col_x(j, n_col) for j in range(n_col)])
    ax.set_ylim(n_row - 0.5, -0.5)
    ax.set_xlim(-0.5, _col_x(n_col - 1, n_col) + 0.5)
    # hairline surface gaps between cells, rather than a border drawn around each one — per block,
    # so no line is drawn across the family gap itself
    edges = [_col_x(j, n_col) + 0.5 for j in range(n_col - 1) if j != half - 1]
    ax.set_xticks(edges, minor=True)
    ax.set_yticks(np.arange(-0.5, n_row, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=_CELL_GAP_LINEWIDTH)
    ax.tick_params(which="both", length=0)
    # The gap already separates the families; this rule only marks where. Subordinate to the
    # headers by construction, and the one cue that survives a greyscale print.
    ax.axvline(
        half - 0.5 + _FAMILY_GAP / 2,
        color=style.AXIS_LINE,
        linewidth=style.MIN_LINEWIDTH,
        zorder=4,
    )
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def _add_colorbar(fig, im, cax, metric, ticks):
    cbar = fig.colorbar(im, cax=cax, ticks=ticks)
    cbar.set_label(style.metric_label(metric, cap=True), fontsize=_CBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=_CBAR_TICK_FONTSIZE, length=2.5, width=style.MIN_LINEWIDTH)
    cbar.outline.set_linewidth(style.MIN_LINEWIDTH)
    cbar.outline.set_edgecolor(style.AXIS_LINE)
    return cbar


def build_figure():
    """Build the matplotlib Figure (does not write it) — returns the Figure."""
    style.apply_style()
    grids = [g for g, _ in GRID_ROWS]
    per_grid = held_out_rows()  # one scan of outputs/, shared by both matrices
    bacc = synthesis_matrix(_BACC_ROWS, "balanced_accuracy", per_grid, TEST_SETS)
    fl = synthesis_matrix(_FL_ROWS, "mae", per_grid, TEST_SETS)

    fig = plt.figure(figsize=(style.FIG_WIDTH, 4.3), layout="constrained")
    # Identical width_ratios on both rows are what make the two metric blocks — and therefore the
    # row labels and the grid columns — line up exactly between the upper and lower panel.
    gs = GridSpec(
        2,
        2,
        height_ratios=[len(_BACC_ROWS), len(_FL_ROWS)],
        width_ratios=[1.0, 0.018],
        figure=fig,
    )
    ax_b = fig.add_subplot(gs[0, 0])
    ax_f = fig.add_subplot(gs[1, 0])
    cax_b = fig.add_subplot(gs[0, 1])
    cax_f = fig.add_subplot(gs[1, 1])

    # One perceptually ordered single-hue family, reversed for the lower-is-better metric, so
    # "darker = better" holds in both panels (and in greyscale). Trimmed at both ends: the light
    # end stays distinguishable from the page, the dark end stays a readable navy.
    ramp = style.truncated_cmap("Blues", 0.10, 0.85)
    im_b = _draw_panel(ax_b, _BACC_ROWS, bacc, ramp, *_BACC_LIM, "{:.2f}")
    im_f = _draw_panel(ax_f, _FL_ROWS, fl, ramp.reversed(), *_MAE_LIM, "{:.0f}")

    _add_colorbar(fig, im_b, cax_b, "balanced_accuracy", _BACC_TICKS)
    _add_colorbar(fig, im_f, cax_f, "mae", _MAE_TICKS)

    # Grid labels on the outer edges only (top of the upper panel, bottom of the lower one), so the
    # two blocks meet without a repeated axis between them.
    col_labels = [style.grid_label(g) for g in grids] * len(TEST_SETS)
    ax_b.set_xticklabels(col_labels, fontsize=_COL_FONTSIZE)
    ax_b.xaxis.set_ticks_position("top")
    ax_f.set_xticklabels(col_labels, fontsize=_COL_FONTSIZE)
    # Family headers above the grid labels. Anchored in DATA x (via get_xaxis_transform) at the
    # centre of each four-grid block, not at a fraction of the axes: once the family gap shifts the
    # right-hand block, the two halves are no longer symmetric about the axes midpoint, so an
    # axes-fraction anchor would sit off-centre over its own columns.
    n_col = len(grids) * len(TEST_SETS)
    for i in range(len(TEST_SETS)):
        first, last = i * len(grids), (i + 1) * len(grids) - 1
        centre = (_col_x(first, n_col) + _col_x(last, n_col)) / 2
        ax_b.annotate(
            _BLOCK_LABEL[TEST_SETS[i]],
            xy=(centre, 1.0),
            xycoords=ax_b.get_xaxis_transform(),
            # clear of the two-line grid labels below (2 x ~8.9 pt of leading plus the tick pad)
            xytext=(0, 27),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=_BLOCK_FONTSIZE,
            fontweight="bold",  # "semibold" is unavailable in this font stack — see paperfig_style
            annotation_clip=False,
        )
    fig.get_layout_engine().set(h_pad=0.01, w_pad=0.01, hspace=0.04, wspace=0.01)
    return fig


if __name__ == "__main__":
    from paper_lib import FIG_DIR

    figure = build_figure()
    print(f"sans face: {style.resolved_sans()}")
    for path in style.save_figure(figure, STEM, FIG_DIR):
        print(f"wrote {path}")
