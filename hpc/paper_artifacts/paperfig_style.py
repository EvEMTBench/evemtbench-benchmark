#!/usr/bin/env python3
"""Shared matplotlib style for the manuscript figure builders (``build_fig_*.py``).

One place for everything the figures must agree on, because it used to be copied per builder and
drifted: the rcParams block, the print-width constant, the grid/source/function display labels, the
metric-arrow strings, the two-series palette, and the multi-format writer.

Two rules this module exists to enforce:

* **Author at final size.** ``wlscirep.cls`` is single-column with ``left=2cm, right=2cm,
  letterpaper``, so ``\\textwidth`` is exactly :data:`FIG_WIDTH` inches and both main-text figures
  sit in a ``figure*`` at ``width=\\linewidth``. A figure authored at :data:`FIG_WIDTH` therefore
  reaches the page at essentially scale 1.0 — ``bbox_inches="tight"`` trims to the drawn content and
  adds :data:`_PAD_INCHES` per side, which in practice lands within a few tenths of a percent of
  ``\\textwidth`` — so the point sizes set here are the point sizes the reader sees. (Before this
  module the two figures were authored at 8.3 in and 11.8 in and downscaled x0.83 / x0.59, which put
  Fig. 3's tick labels on the page at ~4 pt.)
* **One visual language.** The grid abbreviations, the metric labels and the series colours are
  defined once here and consumed by both figures, so Fig. 2 and Fig. 3 cannot disagree about what
  the Protection Reference grid is called or which way "better" points.

Fonts: *Scientific Reports* asks for "a clear, sans-serif typeface (for example, Helvetica) for
figure lettering", and for "the same typeface in the same font size for all figures" — so the
figures deliberately do NOT match the Times body text, and they follow Fig. 1 (drawn outside this
repo, in Arial) rather than the journal's example face. The stack therefore asks for Arial, then
Arial-metric clones, and only then the Helvetica family; see the note on :data:`_SANS_STACK`. On
Alex the pick is Liberation Sans, provisioned into the figure venv's ``mpl-data/fonts/ttf``;
:func:`resolved_sans` reports what was actually chosen so a build log records it rather than the
figure silently claiming a face it is not set in. Text stays text in the SVG
(``svg.fonttype="none"``) and embeds as TrueType in the PDF (``pdf.fonttype=42``), per the
journal's scalable-font requirement.

Stroke weights: the same guidelines require that "the thinnest lines in the final figure should be
no smaller than one point wide", so :data:`MIN_LINEWIDTH` is the floor for every stroke either
figure draws — axes, ticks, grid, cell separators, connectors, error bars and marker edges. Nothing
here may go below it; the recessive elements stay recessive through colour and length, not through
hairline widths.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------------------------------------------
# Print geometry
# --------------------------------------------------------------------------------------------------
#: ``\textwidth`` of the manuscript class, in inches: letterpaper 8.5 in less 2 x 2 cm margins
#: (``wlscirep.cls`` ``\RequirePackage[left=2cm, right=2cm, ...]{geometry}``). The class is
#: single-column, so inside ``figure*`` ``\linewidth == \textwidth``.
FIG_WIDTH = 6.925

#: ``\textwidth`` of the Supplementary Information class, in inches: A4 210 mm less 2 x 15 mm
#: margins (``supplementary.tex`` ``\usepackage[margin=15mm]{geometry}``). SI figures are authored
#: at this width so LaTeX places them at scale 1.0 and the point sizes set in the builder are the
#: point sizes on the page — the same contract the main-text figures have with FIG_WIDTH.
SI_FIG_WIDTH = 7.087

#: The journal's floor for every stroke in a figure, in points ("the thinnest lines in the final
#: figure should be no smaller than one point wide"). Because the figures are authored at final
#: size, a point here is a point on the page — no scaling shrinks it below the floor.
MIN_LINEWIDTH = 1.0

# --------------------------------------------------------------------------------------------------
# Display labels — the single source of truth for both figures
# --------------------------------------------------------------------------------------------------
#: Grid key -> (short name, voltage). Matches the ``GRID_ROWS`` LaTeX labels in ``paper_lib`` and
#: the manuscript prose; used for the column headers of Fig. 2 and the x axis of Fig. 3.
GRID_SHORT: dict[str, tuple[str, str]] = {
    "cigre_mv": ("CIGRÉ MV", "20 kV"),
    "double_line": ("Double Line", "110 kV"),
    "testgrid_110kv": ("Prot. Ref.", "110 kV"),
    "ieee39": ("IEEE-39", "345 kV"),
}

#: Pre-training source grid -> Fig. 3 row label.
SOURCE_LABEL: dict[str, str] = {
    "template_20kv": "Source: 20 kV",
    "template_110kv": "Source: 110 kV",
    "template_345kv": "Source: 345 kV",
}

#: Paper-function key -> title case name (American English, matching ``main.tex``).
FUNC_TITLE: dict[str, str] = {
    "fd": "Fault detection",
    "fc": "Fault classification",
    "fl": "Fault localization",
    "ed": "Event detection",
    "ec": "Event classification",
}

#: Metric key -> axis/colourbar label with its direction arrow.
METRIC_LABEL: dict[str, str] = {
    "balanced_accuracy": "balanced accuracy ↑",
    "mae": "MAE (% line length) ↓",
}

#: Transfer protocol -> marks. Colour AND marker shape, so the two series stay separable in
#: greyscale and under colour-vision deficiency without relying on hue. Measured on this pair
#: (Machado-Oliveira-Fernandes 2009 severity 1.0, Euclidean OKLab x100, white surface): CVD
#: dE 24.7 (protan) / 32.7 (tritan) and unsimulated dE 33.6 -- clear of the 8.0 / 15.0 gates.
#:
#: ``markersize`` is per-series because equal ``markersize`` is NOT equal ink: matplotlib sizes a
#: square by its side and a circle by its diameter, so a square of side m covers m^2 while a circle
#: of diameter m covers pi*m^2/4. Matching area needs side = diameter * sqrt(pi)/2 ~= 0.886, hence
#: 5.2 -> 4.6. Without this the fine-tune squares read ~27% heavier than the zero-shot circles and
#: the eye mistakes a shape difference for a magnitude one. Consumed by the panels AND the legend,
#: so the two cannot disagree.
#:
#: The diameters are set against :data:`MARKER_EDGEWIDTH`, not chosen freely: matplotlib centres the
#: edge stroke on the marker path, so a marker of size d with a white edge of width w shows a
#: coloured core of d - w. At the journal's 1 pt stroke floor, holding the previous ~4 pt core needs
#: d = 5.2 rather than the 4.6 that a 0.6 pt edge allowed.
SERIES: dict[str, dict] = {
    "transfer_zeroshot": {
        "label": "zero-shot",
        "color": "#2a78d6",
        "marker": "o",
        "markersize": 5.2,
    },
    "transfer_finetune": {
        "label": "fine-tune",
        "color": "#eb6834",
        "marker": "s",
        "markersize": 4.6,
    },
}

#: White separating edge around every plotted marker, so overlapping marks and the error bar behind
#: them stay separable. At :data:`MIN_LINEWIDTH` per the journal's stroke floor.
MARKER_EDGEWIDTH = MIN_LINEWIDTH

# Ink. Near-black rather than pure black for body text, one recessive grey for chrome.
INK = "#1a1a1a"
INK_MUTED = "#5a5a5a"
#: Lighter than it looks it should be: the grid is drawn at :data:`MIN_LINEWIDTH`, roughly double
#: the hairline it used to be, so the tone is pulled back to hold the same perceived weight against
#: the page. Width carries the compliance, colour carries the hierarchy.
GRID_LINE = "#e4e4e4"
AXIS_LINE = "#555555"
#: Fill for a cell with no result. Unreachable while the sweep is complete; a build that hits it
#: warns rather than printing a legend key for a state the figure does not otherwise document.
NO_RESULT_FILL = "#ebebeb"

# NOTE ON WEIGHTS. Every face in this stack that is actually present here ships exactly two weights,
# 400 and 700 — TeX Gyre Heros has texgyreheros-regular.otf and texgyreheros-bold.otf and nothing
# between. So ``fontweight="semibold"`` (and "demibold", "medium") cannot be honoured: matplotlib
# logs "Failed to find font weight semibold, now using 700" and loads the same file as "bold". The
# figures therefore say ``"bold"`` deliberately and get the *subordinate* half of a
# semibold-like hierarchy from SIZE instead (e.g. Fig. 2's family headers are bold but set smaller
# than the task labels). Do not "improve" these to "semibold": it changes nothing and adds a warning.
#
# ARIAL, not Helvetica, and that is a decision rather than a default: Fig. 1 is drawn outside this
# repo and is set in Arial, and the journal requires "the same typeface ... for all figures". The
# two are metric-compatible but visually distinct (look at R, G, a, t), so matching the face the
# hand-drawn figure already uses is what makes the set consistent. Liberation Sans is Arial-metric
# and near-identical in shape; it is provisioned into the figure venv's mpl-data alongside
# TeX Gyre Heros. If Fig. 1 is ever redrawn in Helvetica, move the Helvetica clones back to the top.
_SANS_STACK = [
    "Arial",  # the face Fig. 1 is set in
    "Liberation Sans",  # Arial-metric clones
    "Arimo",
    "Helvetica",  # Helvetica-metric fallbacks: metric-compatible, visually distinct
    "Nimbus Sans",
    "TeX Gyre Heros",
    "DejaVu Sans",  # matplotlib's bundled last resort
]


def apply_style() -> None:
    """Install the shared rcParams. Call once, before creating a Figure."""
    mpl.rcParams.update(
        {
            # --- typography: Helvetica family, mathtext to match -----------------------------
            "font.family": "sans-serif",
            "font.sans-serif": _SANS_STACK,
            "mathtext.fontset": "stixsans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 8.0,
            "figure.titlesize": 9.0,
            # --- export: scalable fonts in the PDF, live text in the SVG --------------------
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            # --- chrome: no stroke below MIN_LINEWIDTH, no top/right spines, recessive grid --
            "axes.linewidth": MIN_LINEWIDTH,
            "axes.edgecolor": AXIS_LINE,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.grid": False,
            "grid.color": GRID_LINE,
            "grid.linestyle": "-",
            "grid.linewidth": MIN_LINEWIDTH,
            "grid.alpha": 1.0,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.width": MIN_LINEWIDTH,
            "ytick.major.width": MIN_LINEWIDTH,
            # Ticks grow with their stroke: a 2 pt tick drawn 1 pt wide reads as a smudge, not a
            # tick, so the length is raised in step with the width floor.
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.width": MIN_LINEWIDTH,
            "ytick.minor.width": MIN_LINEWIDTH,
            # --- marks ----------------------------------------------------------------------
            "lines.linewidth": MIN_LINEWIDTH,
            "lines.markersize": 4.0,
            "legend.frameon": False,
            "legend.borderpad": 0.2,
            "legend.handletextpad": 0.4,
            "legend.columnspacing": 1.6,
        }
    )


def resolved_sans() -> str:
    """The sans-serif face matplotlib actually resolves :data:`_SANS_STACK` to on this host.

    Worth printing in a build log: the PDF embeds whatever this returns, so a figure rendered on a
    host without a Helvetica clone does not silently claim to be set in one.
    """
    from matplotlib.font_manager import FontProperties, findfont, get_font

    # The family MUST be passed as a list. Given a bare string and no other keyword, FontProperties
    # treats it as a fontconfig pattern, and "sans-serif" fails that parse on its hyphen
    # (``ParseException: Expected end of text, found '-'``) — which "serif" never did.
    return get_font(findfont(FontProperties(family=["sans-serif"]))).family_name


def grid_label(key: str, sep: str = "\n") -> str:
    """``cigre_mv`` -> ``"CIGRÉ MV\\n20 kV"``; unknown keys fall back to the raw key."""
    if key not in GRID_SHORT:
        return key
    name, kv = GRID_SHORT[key]
    return f"{name}{sep}{kv}"


def metric_label(metric: str, cap: bool = False) -> str:
    """The metric's arrow label; ``cap`` capitalises the first letter (colourbars vs axis titles)."""
    label = METRIC_LABEL.get(metric, metric)
    return label[:1].upper() + label[1:] if cap else label


# --------------------------------------------------------------------------------------------------
# Colour
# --------------------------------------------------------------------------------------------------
def truncated_cmap(name: str, lo: float = 0.0, hi: float = 1.0, n: int = 256):
    """A sequential colormap resampled from ``name[lo:hi]``.

    Trimming both ends of ``Blues`` keeps the light end distinguishable from the page and the dark
    end a readable navy instead of near-black, so the in-cell values stay legible at either extreme
    while the ramp remains monotonic in lightness (and therefore in greyscale).
    """
    base = mpl.colormaps[name]
    return LinearSegmentedColormap.from_list(
        f"{name}[{lo:.2f},{hi:.2f}]", base(np.linspace(lo, hi, n)), N=n
    )


def _relative_luminance(rgb) -> float:
    """WCAG relative luminance of an (r, g, b) triple of sRGB values in [0, 1]."""
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def annot_color(rgba, dark: str = "#000000", light: str = "#ffffff") -> str:
    """Black or white in-cell text, whichever contrasts more with the cell fill.

    Picking by measured contrast rather than a fixed luminance cut means the annotation stays
    readable if the colormap or its truncation ever changes.

    The dark ink is pure black here, not the :data:`INK` used for labels, and that is load-bearing:
    the worst case is a cell whose luminance sits exactly where the two inks tie, so the tie value
    *is* the guaranteed floor for every cell. Against pure black and white that floor is 4.58:1,
    which clears WCAG AA for body text; against ``#111111`` it is only 4.35:1 and against
    ``#1a1a1a`` 4.16:1 — i.e. a slightly softer ink would put the mid-tone cells under AA.
    """
    lum = _relative_luminance(rgba)
    contrast_dark = (lum + 0.05) / (_relative_luminance(mpl.colors.to_rgb(dark)) + 0.05)
    contrast_light = (_relative_luminance(mpl.colors.to_rgb(light)) + 0.05) / (lum + 0.05)
    return dark if contrast_dark >= contrast_light else light


# --------------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------------
FORMATS = ("pdf", "svg", "png")
#: Small but non-zero, so a descender or an error-bar cap at the very edge is not clipped by the
#: tight bounding box. It is also the only reason the delivered width differs from FIG_WIDTH at all.
_PAD_INCHES = 0.02


def save_figure(
    fig,
    stem: str,
    out_dir: Path,
    formats: tuple[str, ...] = FORMATS,
    pad_inches: float = _PAD_INCHES,
) -> list[Path]:
    """Write ``<stem>.{pdf,svg,png}`` into ``out_dir`` and return the paths.

    PDF is the manuscript version, SVG the editable vector one, PNG the >=300 dpi preview. A
    missing glyph is escalated to an error instead of shipping a tofu box: the arrows, the middle
    dot and the acute E are only in the figure because the resolved sans face covers them.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for ext in formats:
            path = out_dir / f"{stem}.{ext}"
            fig.savefig(path, bbox_inches="tight", pad_inches=pad_inches)
            written.append(path)
    missing = [str(w.message) for w in caught if "missing from font" in str(w.message)]
    if missing:
        raise RuntimeError(
            f"{stem}: the resolved sans face ({resolved_sans()}) lacks glyphs used by the figure "
            f"({len(missing)} warning(s)); first: {missing[0]}"
        )
    for w in caught:  # re-emit anything else so it is not swallowed by the recorder
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    return written
