#!/usr/bin/env python3
"""Shared drawing code for the Supplementary Information figures. Generates NO artifact itself.

The artifact rule (``paper/manuscript/generated/README.md``) is one scientific figure -> one
standalone ``build_fig_<name>.py`` -> one ``figures/<name>.pdf``. This module is the artifact-free
helper those builders share, exactly as ``paper_lib`` / ``supplement_lib`` are for the tables: it
never writes, and it is never the owner of a figure.

**Why these figures exist.** The S2 per-baseline tables carried 13 columns of ``mean ± CI`` at
``\\scriptsize`` inside a scale-to-fit box. The box has no floor, so on the page the type rendered
at **3.76--4.98 pt** across six SI pages -- below the ~5 pt practical legibility floor, and against
the journal's "same typeface in the same font size" rule, since every table ended up at a different
effective size. A dot plot carries the same comparison at 7 pt or larger.

**What is lost, and where it went.** A figure cannot be read to three decimals. The exact values
stay machine-readable in ``results/<version>/aggregated_results.csv`` (one row per
task x protocol x grid x baseline x test_set x metric), which the SI already cites, and per-cell in
the provenance index. Nothing is only in the figure.

Every number is read through :func:`paper_lib.per_baseline_matrix`, i.e. the same
``load_rows``/``mean_val``/``ci_val`` objects the tables were rendered from, so a figure and the
published CSV cannot disagree.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("MPLBACKEND", "Agg")  # headless rendering (no display / no X server)

import matplotlib.pyplot as plt  # noqa: E402
import paperfig_style as style  # noqa: E402
from paper_lib import (  # noqa: E402
    _BINARY_RATE_KEYS,
    _PF_BY_KEY,
    BASELINES,
    GRID_ROWS,
    OBS,
    TEST_SETS,
    _transfer_index,
    _transfer_val_index,
    best_cell_choice,
    ci_val,
    held_out_rows,
    mean_val,
    per_baseline_matrix,
    transfer_series,
)
from supplement_lib import TRANSFER_VIEWS  # noqa: E402

from evemtbench.coverage import PRETRAIN_GRIDS  # noqa: E402

#: Functions whose only defined view is ``global`` — one panel row instead of three.
_GLOBAL_ONLY = {"ed", "ec"}

#: Plain-text baseline labels. ``paper_lib._BASELINE_LABEL`` carries LaTeX ``\ac{}`` macros, which
#: a figure cannot expand; these are the same names the acronym list expands them to.
BASELINE_LABEL = {
    "majority": "majority",
    "threshold": "threshold",
    "random_forest": "RF",
    "mlp": "MLP",
    "gru": "GRU",
    "cnn": "CNN",
    "resnet": "ResNet",
}

#: Per-test-set marker style. Filled = in-distribution, hollow = never-trained, so the pair reads
#: without colour; the two hues match the transfer figure's zero-shot/fine-tune pair.
_TEST_SET_STYLE = {
    "test": {
        "label": "in-distribution (test)",
        "color": "#2a78d6",
        "marker": "o",
        "markersize": 4.6,
        "fill": True,
    },
    "benchmark": {
        "label": "never-trained (eval_grid)",
        "color": "#eb6834",
        "marker": "s",
        "markersize": 4.2,
        "fill": False,
    },
}

#: Vertical offset of the two markers about their shared baseline row, in row units.
_PAIR_OFFSET = 0.17

# Type sizes. The floor across every SI figure is 7 pt, matching the tables' enforced floor and
# sitting inside the 6.8-9.0 pt band of the two main-text figures.
_TICK_FONTSIZE = 7.0
_BASELINE_FONTSIZE = 7.2
_PANEL_TITLE_FONTSIZE = 7.8
_ROW_LABEL_FONTSIZE = 8.0
_LEGEND_FONTSIZE = 7.4
_MIN_FONTSIZE = 7.0


def _panel_rows(key: str) -> tuple[str, ...]:
    return ("global",) if key in _GLOBAL_ONLY else OBS


def build_per_baseline_figure(key: str, out_dir: Path | None = None):
    """Render and write the per-baseline S2 figure for one primary function.

    Rows are observability views, columns the four reference grids; inside a panel the seven
    baselines run top to bottom, each carrying its in-distribution and never-trained value with the
    95% interval the corresponding table cell prints.
    """
    _, _name, stem, metric, _ = _PF_BY_KEY[key]
    views = _panel_rows(key)
    per_grid = held_out_rows()
    cells = per_baseline_matrix(stem, metric, per_grid=per_grid, views=views)

    # Drop a baseline the function never defines anywhere in this figure. `threshold` is the case
    # that matters: the overcurrent relay is binary-FD-local only, so for FC/FL/ED/EC its row is
    # empty in every panel — dead vertical space that reads as a missing result rather than an
    # undefined one. A baseline defined in *some* panel keeps its row, so a genuine gap stays visible.
    drawn = [
        b
        for b in BASELINES
        if any(
            cells.get((grid, view, b, ts)) is not None
            for grid, _ in GRID_ROWS
            for view in views
            for ts in TEST_SETS
        )
    ]
    dropped = [b for b in BASELINES if b not in drawn]
    if dropped:
        print(f"omitted (undefined for {key} at every view): {', '.join(dropped)}")

    style.apply_style()
    n_row, n_col = len(views), len(GRID_ROWS)
    # 0.15 in per drawn baseline row, plus the column-title and legend strips. Scaling with
    # len(drawn) keeps the row pitch constant across figures, so a dot sits the same distance
    # from its neighbour whether the function defines seven baselines or five.
    height = 0.15 * len(drawn) * n_row + 1.05
    fig, axes = plt.subplots(
        n_row,
        n_col,
        figsize=(style.SI_FIG_WIDTH, height),
        sharex="row",
        sharey=True,
        squeeze=False,
        layout="constrained",
    )

    y_of = {b: len(drawn) - 1 - i for i, b in enumerate(drawn)}
    for r, view in enumerate(views):
        for c, (grid, _label) in enumerate(GRID_ROWS):
            ax = axes[r][c]
            ax.set_axisbelow(True)
            ax.grid(axis="x", color=style.GRID_LINE, linewidth=style.MIN_LINEWIDTH)
            for baseline in drawn:
                y = y_of[baseline]
                for test_set in TEST_SETS:
                    cell = cells.get((grid, view, baseline, test_set))
                    if cell is None:
                        continue
                    spec = _TEST_SET_STYLE[test_set]
                    dy = -_PAIR_OFFSET if test_set == "test" else _PAIR_OFFSET
                    if cell["lo"] is not None:
                        ax.plot(
                            [cell["lo"], cell["hi"]],
                            [y + dy, y + dy],
                            color=spec["color"],
                            linewidth=style.MIN_LINEWIDTH,
                            solid_capstyle="butt",
                            zorder=2,
                        )
                    ax.plot(
                        [cell["mean"]],
                        [y + dy],
                        marker=spec["marker"],
                        markersize=spec["markersize"],
                        color=spec["color"],
                        markerfacecolor=spec["color"] if spec["fill"] else "white",
                        markeredgecolor=spec["color"],
                        markeredgewidth=style.MARKER_EDGEWIDTH,
                        linestyle="none",
                        zorder=3,
                    )
            ax.set_ylim(-0.6, len(drawn) - 0.4)
            ax.set_yticks(list(y_of.values()))
            if c == 0:
                ax.set_yticklabels([BASELINE_LABEL[b] for b in drawn], fontsize=_BASELINE_FONTSIZE)
            ax.tick_params(axis="x", labelsize=_TICK_FONTSIZE)
            if r == 0:
                ax.set_title(
                    style.grid_label(grid, sep="\n"),
                    fontsize=_PANEL_TITLE_FONTSIZE,
                    pad=4,
                )
            if c == n_col - 1 and len(views) > 1:
                ax.annotate(
                    view,
                    xy=(1.0, 0.5),
                    xycoords="axes fraction",
                    xytext=(14, 0),
                    textcoords="offset points",
                    rotation=270,
                    ha="left",
                    va="center",
                    fontsize=_ROW_LABEL_FONTSIZE,
                    fontweight="bold",
                    annotation_clip=False,
                )
    # One metric label for the whole figure, not one under every column: the four panels of a row
    # share an axis, and repeating the label four times crowded it into the legend strip.
    fig.supxlabel(style.metric_label(metric), fontsize=_ROW_LABEL_FONTSIZE)

    handles = [
        plt.Line2D(
            [],
            [],
            marker=spec["marker"],
            markersize=spec["markersize"],
            color=spec["color"],
            markerfacecolor=spec["color"] if spec["fill"] else "white",
            markeredgecolor=spec["color"],
            markeredgewidth=style.MARKER_EDGEWIDTH,
            linestyle="none",
            label=spec["label"],
        )
        for spec in _TEST_SET_STYLE.values()
    ]
    # Above the panels, not below: the bottom strip already carries supxlabel, and both an
    # "outside lower center" legend and a bbox-anchored one land on top of it.
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=2,
        frameon=False,
        fontsize=_LEGEND_FONTSIZE,
    )

    stem_name = f"fig_supp_{key}"
    if out_dir is None:
        from paper_lib import FIG_DIR

        out_dir = FIG_DIR
    print(f"sans face: {style.resolved_sans()}")
    _assert_type_floor(fig)
    for path in style.save_figure(fig, stem_name, out_dir):
        print(f"wrote {path}")
    return fig


def _assert_type_floor(fig, floor: float = _MIN_FONTSIZE) -> None:
    """Fail the build if any text in the figure is set below the SI legibility floor.

    The S2 tables reached 3.76 pt precisely because nothing checked. This makes the floor a
    build-time property of every figure this module draws rather than a convention.
    """
    small = sorted(
        {
            round(t.get_fontsize(), 2)
            for ax in fig.get_axes()
            for t in [*ax.get_xticklabels(), *ax.get_yticklabels(), ax.title]
            if t.get_text() and t.get_fontsize() < floor - 1e-9
        }
    )
    if small:
        raise AssertionError(f"figure text below the {floor} pt SI floor: {small}")


#: Pre-training source labels — the voltage template, named by its level.
SOURCE_LABEL = {"template_20kv": "20 kV", "template_110kv": "110 kV", "template_345kv": "345 kV"}

#: Per-protocol marker style for the transfer figures. Deliberately the same filled/hollow pair
#: and the same two hues as the S2 figures and the main-text transfer figure, so a reader who has
#: learned the encoding once carries it through the whole document.
_PROTO_STYLE = {
    "transfer_zeroshot": {
        "label": "zero-shot",
        "color": "#2a78d6",
        "marker": "o",
        "markersize": 4.6,
        "fill": True,
    },
    "transfer_finetune": {
        "label": "fine-tune",
        "color": "#eb6834",
        "marker": "s",
        "markersize": 4.2,
        "fill": False,
    },
}


def build_transfer_figure(key: str, out_dir: Path | None = None):
    """Render and write the SI cross-grid transfer figure for one fault-triad function.

    Rows are the two fixed-width observability views, columns the four target grids; inside a panel
    the three pre-training sources run top to bottom, each carrying its zero-shot and fine-tuned
    value with the 95% interval the corresponding table cell prints. Replaces a 8-column table that
    the scale-to-fit box rendered below the legibility floor.

    Selection is :func:`paper_lib.transfer_series`, i.e. the same ``best_transfer_row`` the
    ``tab_transfer_*.tex`` fragments and the main-text figure use — zero-shot selected on the
    source corpus's validation split, fine-tune on the target grid's, and never on the reported
    ``eval_grid`` family.
    """
    from supplement_lib import TRANSFER_FUNCTIONS

    spec = {row[0]: row for row in TRANSFER_FUNCTIONS}[key]
    stem, metric = spec[2], spec[3]
    idx, val_idx = _transfer_index("outputs"), _transfer_val_index("outputs")

    # {(view, source): {protocol: [cell per target grid]}}
    data = {
        (view, source): transfer_series(idx, f"{stem}_{view}", source, metric, val_idx)
        for view in TRANSFER_VIEWS
        for source in PRETRAIN_GRIDS
    }

    style.apply_style()
    n_row, n_col = len(TRANSFER_VIEWS), len(GRID_ROWS)
    height = 0.15 * len(PRETRAIN_GRIDS) * n_row + 1.15
    fig, axes = plt.subplots(
        n_row,
        n_col,
        figsize=(style.SI_FIG_WIDTH, height),
        sharex="row",
        sharey=True,
        squeeze=False,
        layout="constrained",
    )

    y_of = {s: len(PRETRAIN_GRIDS) - 1 - i for i, s in enumerate(PRETRAIN_GRIDS)}
    for r, view in enumerate(TRANSFER_VIEWS):
        for c, (grid, _label) in enumerate(GRID_ROWS):
            ax = axes[r][c]
            ax.set_axisbelow(True)
            ax.grid(axis="x", color=style.GRID_LINE, linewidth=style.MIN_LINEWIDTH)
            for source in PRETRAIN_GRIDS:
                y = y_of[source]
                for protocol, pspec in _PROTO_STYLE.items():
                    cells = data[(view, source)].get(protocol) or []
                    cell = cells[c] if c < len(cells) else None
                    if cell is None:
                        continue
                    dy = -_PAIR_OFFSET if protocol == "transfer_zeroshot" else _PAIR_OFFSET
                    if cell["lo"] is not None and cell["lo"] != cell["hi"]:
                        ax.plot(
                            [cell["lo"], cell["hi"]],
                            [y + dy, y + dy],
                            color=pspec["color"],
                            linewidth=style.MIN_LINEWIDTH,
                            solid_capstyle="butt",
                            zorder=2,
                        )
                    ax.plot(
                        [cell["mean"]],
                        [y + dy],
                        marker=pspec["marker"],
                        markersize=pspec["markersize"],
                        color=pspec["color"],
                        markerfacecolor=pspec["color"] if pspec["fill"] else "white",
                        markeredgecolor=pspec["color"],
                        markeredgewidth=style.MARKER_EDGEWIDTH,
                        linestyle="none",
                        zorder=3,
                    )
            ax.set_ylim(-0.6, len(PRETRAIN_GRIDS) - 0.4)
            ax.set_yticks(list(y_of.values()))
            if c == 0:
                ax.set_yticklabels(
                    [SOURCE_LABEL[s] for s in PRETRAIN_GRIDS], fontsize=_BASELINE_FONTSIZE
                )
            ax.tick_params(axis="x", labelsize=_TICK_FONTSIZE)
            if r == 0:
                ax.set_title(
                    style.grid_label(grid, sep="\n"), fontsize=_PANEL_TITLE_FONTSIZE, pad=4
                )
            if c == n_col - 1:
                ax.annotate(
                    view,
                    xy=(1.0, 0.5),
                    xycoords="axes fraction",
                    xytext=(14, 0),
                    textcoords="offset points",
                    rotation=270,
                    ha="left",
                    va="center",
                    fontsize=_ROW_LABEL_FONTSIZE,
                    fontweight="bold",
                    annotation_clip=False,
                )
    fig.supxlabel(style.metric_label(metric), fontsize=_ROW_LABEL_FONTSIZE)
    fig.supylabel("pre-training source", fontsize=_ROW_LABEL_FONTSIZE)

    handles = [
        plt.Line2D(
            [],
            [],
            marker=s["marker"],
            markersize=s["markersize"],
            color=s["color"],
            markerfacecolor=s["color"] if s["fill"] else "white",
            markeredgecolor=s["color"],
            markeredgewidth=style.MARKER_EDGEWIDTH,
            linestyle="none",
            label=s["label"],
        )
        for s in _PROTO_STYLE.values()
    ]
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=2,
        frameon=False,
        fontsize=_LEGEND_FONTSIZE,
    )

    if out_dir is None:
        from paper_lib import FIG_DIR

        out_dir = FIG_DIR
    print(f"sans face: {style.resolved_sans()}")
    _assert_type_floor(fig)
    for path in style.save_figure(fig, f"fig_supp_transfer_{key}", out_dir):
        print(f"wrote {path}")
    return fig


#: The binary protection rows: (function stem, view, display label). FD at all three views plus
#: global ED — the four functions whose dependability/security pair the SI reports.
_PROTECTION_ROWS = [
    ("fault_detection", "local", "FD / local"),
    ("fault_detection", "line", "FD / line"),
    ("fault_detection", "global", "FD / global"),
    ("event_detection", "global", "ED / global"),
]

#: The three quantities per row, top to bottom, all on a shared [0, 1] axis.
_PROTECTION_METRICS = [("balanced_accuracy", "bal. acc."), ("mdr", "MDR"), ("far", "FAR")]


def build_protection_figure(out_dir: Path | None = None):
    """Render and write the protection-grade dependability/security figure.

    One panel per (function/view, grid). Inside a panel, balanced accuracy, MDR and FAR for the
    validation-selected reference, in-distribution against never-trained, on a shared [0, 1] axis.

    The point of the layout is that an always-positive collapse has a *shape*: balanced accuracy
    falling to ~0.5 while MDR goes to ~0 and FAR to ~1. In the 10-column table that signature was
    three numbers on one 6.97 pt line; here it is a visual pattern that cannot be misread as random
    prediction. No statistical test against a random predictor is performed or implied.
    """
    per_grid = held_out_rows()
    style.apply_style()
    n_row, n_col = len(_PROTECTION_ROWS), len(GRID_ROWS)
    height = 0.15 * len(_PROTECTION_METRICS) * n_row + 1.15
    fig, axes = plt.subplots(
        n_row,
        n_col,
        figsize=(style.SI_FIG_WIDTH, height),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )

    y_of = {m: len(_PROTECTION_METRICS) - 1 - i for i, (m, _) in enumerate(_PROTECTION_METRICS)}
    for r, (stem, view, row_label) in enumerate(_PROTECTION_ROWS):
        mk, fk = _BINARY_RATE_KEYS[stem]
        key_of = {"balanced_accuracy": "balanced_accuracy", "mdr": mk, "far": fk}
        for c, (grid, _label) in enumerate(GRID_ROWS):
            ax = axes[r][c]
            ax.set_axisbelow(True)
            ax.grid(axis="x", color=style.GRID_LINE, linewidth=style.MIN_LINEWIDTH)
            rows = per_grid[grid]
            tid = f"{stem}_{view}"
            for test_set in TEST_SETS:
                choice = best_cell_choice(rows, tid, test_set, "balanced_accuracy")
                if choice is None:
                    continue
                doc = rows.get((tid, choice["baseline"]), {}).get(test_set)
                spec = _TEST_SET_STYLE[test_set]
                dy = -_PAIR_OFFSET if test_set == "test" else _PAIR_OFFSET
                for name, _lab in _PROTECTION_METRICS:
                    v = mean_val(doc, key_of[name])
                    if v is None:
                        continue
                    y = y_of[name] + dy
                    ci = ci_val(doc, key_of[name])
                    if ci:
                        ax.plot(
                            [max(0.0, v - ci), min(1.0, v + ci)],
                            [y, y],
                            color=spec["color"],
                            linewidth=style.MIN_LINEWIDTH,
                            solid_capstyle="butt",
                            zorder=2,
                        )
                    ax.plot(
                        [v],
                        [y],
                        marker=spec["marker"],
                        markersize=spec["markersize"],
                        color=spec["color"],
                        markerfacecolor=spec["color"] if spec["fill"] else "white",
                        markeredgecolor=spec["color"],
                        markeredgewidth=style.MARKER_EDGEWIDTH,
                        linestyle="none",
                        zorder=3,
                    )
            ax.set_xlim(-0.03, 1.03)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_ylim(-0.6, len(_PROTECTION_METRICS) - 0.4)
            ax.set_yticks(list(y_of.values()))
            if c == 0:
                ax.set_yticklabels(
                    [lab for _, lab in _PROTECTION_METRICS], fontsize=_BASELINE_FONTSIZE
                )
            ax.tick_params(axis="x", labelsize=_TICK_FONTSIZE)
            if r == 0:
                ax.set_title(
                    style.grid_label(grid, sep="\n"), fontsize=_PANEL_TITLE_FONTSIZE, pad=4
                )
            if c == n_col - 1:
                ax.annotate(
                    row_label,
                    xy=(1.0, 0.5),
                    xycoords="axes fraction",
                    xytext=(14, 0),
                    textcoords="offset points",
                    rotation=270,
                    ha="left",
                    va="center",
                    fontsize=_ROW_LABEL_FONTSIZE,
                    fontweight="bold",
                    annotation_clip=False,
                )
    fig.supxlabel("rate / balanced accuracy", fontsize=_ROW_LABEL_FONTSIZE)

    handles = [
        plt.Line2D(
            [],
            [],
            marker=s["marker"],
            markersize=s["markersize"],
            color=s["color"],
            markerfacecolor=s["color"] if s["fill"] else "white",
            markeredgecolor=s["color"],
            markeredgewidth=style.MARKER_EDGEWIDTH,
            linestyle="none",
            label=s["label"],
        )
        for s in _TEST_SET_STYLE.values()
    ]
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=2,
        frameon=False,
        fontsize=_LEGEND_FONTSIZE,
    )

    if out_dir is None:
        from paper_lib import FIG_DIR

        out_dir = FIG_DIR
    print(f"sans face: {style.resolved_sans()}")
    _assert_type_floor(fig)
    for path in style.save_figure(fig, "fig_supp_protection", out_dir):
        print(f"wrote {path}")
    return fig
