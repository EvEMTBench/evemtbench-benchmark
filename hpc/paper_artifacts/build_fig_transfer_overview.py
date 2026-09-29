#!/usr/bin/env python3
"""Standalone builder → paper/manuscript/figures/fig_transfer_overview.{pdf,svg,png}.

A source × function panel grid of cross-grid transfer on the never-trained ``eval_grid`` family: for
each pre-training source grid (row, voltage order) and fault-triad function (column), the best
learned baseline's headline metric, zero-shot vs fine-tune, across the target grids (voltage order),
at the differential ``line`` view. The cross-grid transfer protocols apply only to the fixed-width
``local``/``line`` views (a model with a fixed input head cannot ingest the grid-dependent
``global`` channel count — that is what the planned cross-topology + GNN track is for), so the
figure covers the three fault-triad functions that carry transferable weights (FD/FC/FL); the
event-scope functions (ED/EC) are global-only and held-out-only and thus have no transfer panel.
Reuses the same selection as the transfer tables (``paper_lib.best_transfer_row`` via
``paper_lib.transfer_series``), so the figure and ``tab_transfer_*.tex`` agree cell-for-cell.

**Paired points, not grouped bars.** The quantity of interest is the *adaptation gain* — the
zero-shot → fine-tune step for one target grid — so the two protocols are drawn as two marks on a
short connector per target, and never joined across targets. Two things follow that grouped bars
could not give: the axis is free to span only the data (FD lives in 0.87–0.99, which a bar anchored
at zero compresses into the top few percent of the panel), and each mark carries its 95\\% CI. Marks
differ in shape as well as colour (zero-shot = circle, fine-tune = square), so the pairing survives
greyscale and colour-vision deficiency. Y limits are shared within a task column — the comparison
the figure exists to support is across source rows — and deliberately not across tasks, whose
metrics differ.

Authored at exactly ``paperfig_style.FIG_WIDTH`` (= ``\\textwidth``), so LaTeX places it at scale
1.0 and the point sizes set here are the point sizes on the page. No in-figure title: that text
lives in the LaTeX caption (``main.tex``, ``fig:transfer_overview``).

Requires ``matplotlib`` (NOT in the sweep's ``py_dl`` env — provision it in a separate env so the
running sweep is untouched). This is compute — run via ``srun``, never the login node. Example:

    srun --partition=a100mig --gres=gpu:a100small:1 --time=10:00 bash -c \
      'CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
       "$PAPERFIGS_PY" hpc/paper_artifacts/build_fig_transfer_overview.py'
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("MPLBACKEND", "Agg")  # headless rendering (no display / no X server)

import matplotlib.pyplot as plt
import numpy as np
import paperfig_style as style
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from paper_lib import (
    GRID_ROWS,
    PAPER_FUNCTIONS,
    PRETRAIN_GRIDS,
    TRANSFER_FIG_FUNCTIONS,
    TRANSFER_FIG_VIEW,
    _is_bounded01,
    _transfer_index,
    _transfer_val_index,
    transfer_series,
)

STEM = "fig_transfer_overview"
# Coverage lives in paper_lib so the figure-value test checks the same view and functions.
OBS_VIEW = TRANSFER_FIG_VIEW
TRANSFER_FUNCTIONS = TRANSFER_FIG_FUNCTIONS

# Horizontal offset of each protocol's mark from its target-grid tick, in data units.
_DX = {"transfer_zeroshot": -0.14, "transfer_finetune": 0.14}
# The connector is a PAIRING AID, not a trajectory: at full strength a grey line between two points
# reads as "the value moved along here", which is not what a zero-shot/fine-tune pair means. It
# groups the pair and then gets out of the way, leaving the markers and their intervals as the
# strongest ink. Colour is the shared muted ink rather than a fourth private grey. The journal's
# 1 pt stroke floor takes away width as the way to hold it back, so the alpha absorbs that: it drops
# in step with the width so the connector keeps roughly the ink it had at 0.7 pt.
_CONNECTOR_LINEWIDTH = style.MIN_LINEWIDTH
_CONNECTOR_ALPHA = 0.35
# Error bars: at the stroke floor, small-capped, and just under full strength, so uncertainty is
# legible without out-weighing the point estimate it belongs to. The cap has to be visibly longer
# than the bar is thick or it stops reading as a cap, hence the capsize moving with the width.
_ERRORBAR_LINEWIDTH = style.MIN_LINEWIDTH
_ERRORBAR_CAPSIZE = 2.2
_ERRORBAR_ALPHA = 0.85
_PANEL_TITLE_FONTSIZE = 9.0
_PANEL_METRIC_FONTSIZE = 7.5
_ROW_LABEL_FONTSIZE = 8.0  # quieter than the panel titles: the source is context, not the subject
# The target grids are compared repeatedly across all nine panels, so their names are the labels
# most worth reading; size is bounded by four two-line names sitting side by side in a ~2 in panel
# without touching. "CIGRÉ MV" against "Double Line" is always the binding pair.
#
# RE-MEASURED 2026-08-12 for the Helvetica-metric face (paperfig_style._SANS_STACK). The previous
# 7.5 pt was calibrated against the old serif stack, where it cleared by 1.80 pt; Helvetica is wider
# and the same size now OVERLAPS by 0.34 pt. Clearance on the live renderer, at the final figure
# width: 7.5 -> -0.34 (overlap); 7.2 -> +1.00; 7.0 -> +2.87; 6.8 -> +3.35. Widening the panels does
# almost nothing (wspace 0.015 -> 0.008 buys 0.2 pt), because the pitch is set by the figure width.
# 7.0 restores the slack the design had before the face changed. THIS NUMBER IS FACE-SPECIFIC: any
# change to the font stack invalidates it, so re-run the measurement rather than trusting the value.
_XTICK_FONTSIZE = 7.0
# Short numeric ticks, held one step below the grid names so the hierarchy survives the x labels
# coming down to 7.0 — the target names stay the louder of the two.
_YTICK_FONTSIZE = 6.8
_Y_PAD_FRAC = 0.07  # headroom around the data+CI extent, as a fraction of its span


def _panel_limits(cells, metric):
    """Shared y limits for one task column from every drawn cell's CI extent, so no whisker is
    clipped. Bounded metrics are held just inside their own support so the axis cannot suggest that
    a balanced accuracy above 1 is possible."""
    lows = [c["lo"] for c in cells]
    highs = [c["hi"] for c in cells]
    if not lows:
        return None
    lo, hi = min(lows), max(highs)
    pad = max((hi - lo) * _Y_PAD_FRAC, 1e-6)
    lo, hi = lo - pad, hi + pad
    if _is_bounded01(metric):
        return max(lo, 0.0), min(hi, 1.02)
    return max(lo, 0.0), hi


def _draw_panel(ax, series, x):
    """One source × function panel: a connector plus two CI-bearing marks per target grid."""
    protos = list(style.SERIES)
    zs, ft = (series.get(p) or [None] * len(x) for p in protos)
    # connector first, and only within a target grid — never across target grids
    for i, (a, b) in enumerate(zip(zs, ft, strict=True)):
        if a is None or b is None:
            continue
        ax.plot(
            [i + _DX[protos[0]], i + _DX[protos[1]]],
            [a["mean"], b["mean"]],
            color=style.INK_MUTED,
            alpha=_CONNECTOR_ALPHA,
            linewidth=_CONNECTOR_LINEWIDTH,
            solid_capstyle="round",
            zorder=1,
        )
    drawn = []
    for proto in protos:
        spec = style.SERIES[proto]
        cells = [(i, c) for i, c in enumerate(series.get(proto) or []) if c is not None]
        if not cells:
            continue
        drawn.extend(c for _, c in cells)
        xs = [i + _DX[proto] for i, _ in cells]
        ys = [c["mean"] for _, c in cells]
        # asymmetric where ci_bounds clipped the interval to the metric's support, so the drawn
        # whisker is exactly the interval tab_transfer_*.tex prints
        yerr = np.array(
            [[c["mean"] - c["lo"] for _, c in cells], [c["hi"] - c["mean"] for _, c in cells]]
        )
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            fmt="none",
            ecolor=spec["color"],
            elinewidth=_ERRORBAR_LINEWIDTH,
            capsize=_ERRORBAR_CAPSIZE,
            capthick=_ERRORBAR_LINEWIDTH,
            alpha=_ERRORBAR_ALPHA,
            zorder=2,
        )
        ax.plot(
            xs,
            ys,
            linestyle="none",
            marker=spec["marker"],
            markersize=spec["markersize"],  # per-series: equal AREA, not equal size (see SERIES)
            color=spec["color"],
            markeredgecolor="white",
            markeredgewidth=style.MARKER_EDGEWIDTH,
            zorder=3,
        )
    ax.grid(axis="y")
    # 5 bins rather than 4: a column's shared range can be wide (FC spans 0.27-0.86), and 4 leaves
    # the low end of such a column without a gridline to read the weakest zero-shot point against.
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    # Both tick label sizes are set per-axes, here, and NOT via set_xticklabels(fontsize=...) in
    # build_figure. Under sharex/sharey the shared Formatter propagates the label *text* to every
    # panel, but each axes keeps its own label artists — and matplotlib hides the inner rows' — so a
    # fontsize passed to set_xticklabels on the top-left panel lands only on invisible artists while
    # the visible bottom row silently keeps the rcParams default.
    ax.tick_params(axis="x", labelsize=_XTICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=_YTICK_FONTSIZE)
    return drawn


def build_figure():
    """Build the matplotlib Figure (does not write it) — returns the Figure.

    A ``source × function`` panel grid: one row per pre-training source grid (20/110/345 kV, the
    ``PRETRAIN_GRIDS`` order) and one column per fault-triad function. ``sharey="col"`` plus limits
    computed from the whole column keeps each function's metric axis comparable across sources, so
    the matched-voltage diagonal (a source transferring best to the same-voltage target) reads off
    directly. Selection matches the transfer tables cell-for-cell."""
    style.apply_style()
    idx = _transfer_index("outputs")
    val_idx = _transfer_val_index("outputs")
    funcs = [pf for pf in PAPER_FUNCTIONS if pf[0] in TRANSFER_FUNCTIONS]
    grids = [g for g, _ in GRID_ROWS]
    sources = list(PRETRAIN_GRIDS)
    x = np.arange(len(grids))

    fig, axes = plt.subplots(
        len(sources),
        len(funcs),
        figsize=(style.FIG_WIDTH, 5.35),
        squeeze=False,
        sharey="col",
        sharex=True,
        layout="constrained",
    )
    # Select every cell once up front: the per-column y limits need the whole column before any
    # panel can be scaled, and re-selecting per panel would repeat the validation-selection walk.
    series_by = {
        (si, fi): transfer_series(idx, f"{stem}_{OBS_VIEW}", source, metric, val_idx, grids)
        for si, source in enumerate(sources)
        for fi, (_key, _name, stem, metric, _label) in enumerate(funcs)
    }

    for si, source in enumerate(sources):
        for fi, (key, _name, _stem, metric, _label) in enumerate(funcs):
            ax = axes[si][fi]
            drawn = _draw_panel(ax, series_by[(si, fi)], x)
            if not drawn:
                ax.text(
                    0.5,
                    0.5,
                    "no result yet",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                    color=style.INK_MUTED,
                )
            if si == 0:  # the metric differs per column (accuracy vs MAE %) → name it in the title
                # Bold task name over a smaller regular metric line: the task is the top of this
                # figure's hierarchy. ("semibold" is unavailable in this font stack — see
                # paperfig_style; bold is the only heavier face, so size carries the rest.)
                ax.set_title(
                    style.FUNC_TITLE.get(key, key),
                    fontsize=_PANEL_TITLE_FONTSIZE,
                    fontweight="bold",
                    pad=13,  # unchanged, so the title block's vertical footprint does not grow
                )
                ax.annotate(
                    style.metric_label(metric),
                    xy=(0.5, 1.0),
                    xycoords="axes fraction",
                    xytext=(0, 3.5),
                    textcoords="offset points",
                    ha="center",
                    va="baseline",
                    fontsize=_PANEL_METRIC_FONTSIZE,
                    color=style.INK_MUTED,
                    annotation_clip=False,
                )
            if fi == 0:  # left column: name the pre-training source (the only rotated text here)
                ax.set_ylabel(
                    style.SOURCE_LABEL.get(source, source),
                    fontsize=_ROW_LABEL_FONTSIZE,
                    fontweight="bold",
                )

    # Shared limits per task column, from every cell drawn in that column (CI extent included).
    for fi, (_key, _name, _stem, metric, _label) in enumerate(funcs):
        cells = [
            c
            for si in range(len(sources))
            for proto_cells in series_by[(si, fi)].values()
            for c in proto_cells
            if c is not None
        ]
        limits = _panel_limits(cells, metric)
        if limits is not None:
            axes[0][fi].set_ylim(*limits)  # sharey="col" propagates to the rows below

    axes[0][0].set_xticks(x)
    # Text only — the size is applied per-axes in _draw_panel (see the note there).
    axes[0][0].set_xticklabels([style.grid_label(g) for g in grids])
    # Tight x padding buys tick pitch: four two-line target names have to sit side by side inside a
    # ~2 in panel without touching, and the marks only reach +/-0.14 from their tick.
    axes[0][0].set_xlim(-0.35, len(grids) - 0.65)

    # One shared legend in a reserved strip at the very top (``rect`` below keeps the axes out of
    # it), so it cannot crowd the top row's panel titles. Marker only — no connector segment in the
    # key, because the connector is a pairing aid rather than a third series.
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=spec["marker"],
            markersize=spec["markersize"],  # same per-series sizes as the panels
            color=spec["color"],
            markeredgecolor="white",
            markeredgewidth=style.MARKER_EDGEWIDTH,
            label=spec["label"],
        )
        for spec in style.SERIES.values()
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles))
    fig.get_layout_engine().set(
        h_pad=0.01, w_pad=0.01, hspace=0.05, wspace=0.015, rect=(0, 0, 1, 0.955)
    )
    return fig


if __name__ == "__main__":
    from paper_lib import FIG_DIR

    figure = build_figure()
    print(f"sans face: {style.resolved_sans()}")
    for path in style.save_figure(figure, STEM, FIG_DIR):
        print(f"wrote {path}")
