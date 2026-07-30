#!/usr/bin/env python3
"""Standalone builder → paper/manuscript/figures/fig_overview.pdf.

The conceptual benchmark-overview figure (hand-authored, not data-derived): a shared EMT dataset
enters a FROZEN evaluation contract into which an OPEN, replaceable method plugs, producing
conditional, comparable, reproducible outputs. Rendered with matplotlib rather than TikZ so the
offline SLURM build (Tectonic on a network-less compute node) needs no extra LaTeX packages.

Requires ``matplotlib`` (the paperfigs-venv). Run via ``srun``, never the login node.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402

# Embed TrueType fonts (Type 42), not Type 3 bitmap fonts, so the PDF meets the journal's
# scalable-font figure requirement (see paper/manuscript/document_check.md).
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from paper_lib import FIG_DIR  # noqa: E402

NAME = "fig_overview.pdf"
_INK = "#1a1a1a"
_MUTED = "#555555"


def _box(ax, x, y, w, h, title, body, *, emph=False, dashed=False):
    """Rounded box with a bold title line and a body block; returns (cx, edges) anchors."""
    style = dict(
        boxstyle="round,pad=0.4,rounding_size=6",
        linewidth=1.9 if emph else 1.1,
        edgecolor=_INK if emph else _MUTED,
        facecolor="#f4f4f2" if emph else "#fbfbfa",
    )
    if dashed:
        style["linestyle"] = (0, (4, 3))
    ax.add_patch(FancyBboxPatch((x, y), w, h, mutation_scale=1, **style))
    ax.text(
        x + w / 2,
        y + h - 5.5,
        title,
        ha="center",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color=_INK,
    )
    ax.text(
        x + w / 2,
        y + h - 13,
        body,
        ha="center",
        va="top",
        fontsize=7.4,
        color=_INK,
        linespacing=1.35,
    )
    return x + w / 2


def _arrow(ax, xy0, xy1, *, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            xy0,
            xy1,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.5,
            color=_INK if not dashed else _MUTED,
            linestyle=(0, (4, 3)) if dashed else "solid",
            shrinkA=2,
            shrinkB=2,
        )
    )


def _tag(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="center", fontsize=7.2, style="italic", color=_MUTED)


def build_figure():
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    _box(
        ax,
        1,
        49,
        22,
        34,
        "Shared EMT dataset",
        "point-on-wave $V,\\,I$\n4 grids · 20–345 kV\n\nFamilies:\nTrain · Fine-tune ·\n"
        "Benchmark (eval-only)",
    )
    cx_f = _box(
        ax,
        31,
        29,
        36,
        58,
        "FROZEN BENCHMARK CONTRACT",
        "• 24 scored tasks, 3 tiers\n   (5 primary: ED EC FD FC FL)\n"
        "• Views: local ⊂ line ⊂ global\n• Protocols: held-out\n"
        "   (in-dist. + never-trained)\n   · cross-grid transfer\n"
        "• Metrics + episode-level\n   leakage controls",
        emph=True,
    )
    _box(
        ax,
        75,
        49,
        24,
        34,
        "CONDITIONAL EVIDENCE",
        "score per\n(task × view × grid ×\nprotocol × set × metric)\n\n"
        "mean ± 95% CI ·\nleaderboard · provenance",
    )
    _box(
        ax,
        29,
        2,
        40,
        24,
        "ANY METHOD PLUGS IN",
        "shipped: majority, overcurrent, RF,\nMLP, GRU, CNN, ResNet   ·   future: GNN,\n"
        "physics-informed, foundation, uncertainty",
        dashed=True,
    )

    _arrow(ax, (23, 66), (31, 66))  # dataset -> contract
    _arrow(ax, (67, 66), (75, 66))  # contract -> outputs
    _arrow(ax, (cx_f, 26), (cx_f, 29), dashed=True)  # method -> contract

    _tag(ax, cx_f, 90, "fixed within a benchmark version")
    _tag(ax, 87, 86, "comparable & reproducible")
    _tag(ax, cx_f, 27.6, "open / replaceable")

    fig.tight_layout(pad=0.3)
    return fig


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure = build_figure()
    out = FIG_DIR / NAME
    figure.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
