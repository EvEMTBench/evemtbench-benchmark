"""Every value the two main-text figures plot equals the published table cell it came from.

``check_results_fresh.py`` byte-gates ``paper/manuscript/generated/*.tex`` against a fresh rebuild
from ``outputs/``, but it says nothing about the figure PDFs — so before this file nothing checked
that Fig. 2 and Fig. 3 agreed with the tables a reviewer reads. Both figures already select through
the same ``paper_lib`` helpers as the tables, so this is a regression test on that coupling: it
re-derives each figure's cells via the builders' own data seam (``synthesis_matrix`` /
``transfer_series``, both matplotlib-free, which is why this runs in a torch- and matplotlib-free
CI) and compares them against the committed ``.tex`` fragments, parsed — never hand-typed.

Agreement is asserted at the tables' own formatting precision by rendering the figure's
``(mean, ci)`` back through ``paper_lib._fmt_mean_ci`` and requiring the exact cell string, which
covers the three forms the tables use: ``mean $\\pm$ ci``, a bare ``mean`` (half-width rounds to
zero), and the ``mean [lo, hi]`` truncation for a [0,1]-bounded interval that overflows its bound.

Skips until ``outputs/`` is populated, like ``test_paper_numbers_covered.py``.
"""

import re
from pathlib import Path

import paper_lib
import pytest
from paper_lib import (
    _PF_BY_KEY,
    _TRANSFER_PROTOS,
    GRID_ROWS,
    OBS,
    PRETRAIN_GRIDS,
    SYNTHESIS_BACC_ROWS,
    SYNTHESIS_FL_ROWS,
    TEST_SETS,
    TRANSFER_FIG_FUNCTIONS,
    TRANSFER_FIG_VIEW,
    ci_bounds,
    held_out_rows,
    synthesis_matrix,
    transfer_series,
)

REPO = Path(__file__).resolve().parents[2]

MANUSCRIPT = REPO / "paper" / "manuscript"
GEN = MANUSCRIPT / "generated"
pytestmark = pytest.mark.skipif(
    not GEN.is_dir(), reason="private manuscript not present (public release clone)"
)

# A five-state marker cell, i.e. no number to compare.
_MARKERS = {r"\pending", r"\notapp", r"\oos", r"\undefm", r"\res", ""}
# ``0.893$\,{\scriptscriptstyle[0.781,\,1.000]}$`` -> the truncated-interval display form.
_CLIPPED = re.compile(
    r"^(?P<mean>[\d.]+)\$\\,\{\\scriptscriptstyle\[(?P<lo>[\d.]+),\\,(?P<hi>[\d.]+)\]\}\$$"
)


def _body_rows(name: str) -> list[list[str]]:
    """The data rows of a generated table as lists of stripped cell strings."""
    text = (GEN / name).read_text()
    body = text.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    rows = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("%") or line == r"\midrule":
            continue
        rows.append([c.strip() for c in line.rstrip("\\").split("&")])
    return rows


def _outputs_populated() -> bool:
    per_grid = held_out_rows()
    return any(per_grid.values())


def _assert_agrees(cell: dict, table_cell: str, metric: str, where: str) -> None:
    """The figure's cell renders to exactly the string the table prints."""
    rendered = paper_lib._fmt_mean_ci(cell["mean"], cell["ci"], metric)
    assert rendered == table_cell, (
        f"{where}: figure would plot mean={cell['mean']!r} ci={cell['ci']!r} -> {rendered!r}, "
        f"but {metric} table cell is {table_cell!r}"
    )


# --------------------------------------------------------------------------------------------------
# Fig. 2 — held-out synthesis heat map vs tab_res_*.tex
# --------------------------------------------------------------------------------------------------
def test_synthesis_figure_cells_match_the_result_tables():
    """Each Fig. 2 cell equals its ``tab_res_<fn>.tex`` cell, both selected by best_cell_choice.

    ``result_table`` lays a function's row out as grid x (test_set x observability), so the column
    index of a (test_set, view) pair is fixed by ``TEST_SETS``/``OBS`` order."""
    if not _outputs_populated():
        pytest.skip("outputs/ not populated on this host")
    per_grid = held_out_rows()
    grids = [g for g, _ in GRID_ROWS]
    specs = [(SYNTHESIS_BACC_ROWS, "balanced_accuracy"), (SYNTHESIS_FL_ROWS, "mae")]
    compared = 0
    for rows, metric in specs:
        matrix = synthesis_matrix(rows, metric, per_grid, TEST_SETS)
        for (label, stem, view), fig_row in zip(rows, matrix, strict=True):
            key = next(k for k, pf in _PF_BY_KEY.items() if pf[2] == stem)
            table = _body_rows(f"tab_res_{key}.tex")
            for col, cell in enumerate(fig_row):
                test_set = TEST_SETS[col // len(grids)]
                grid = grids[col % len(grids)]
                table_cell = table[grids.index(grid)][
                    1 + TEST_SETS.index(test_set) * len(OBS) + OBS.index(view)
                ]
                if cell is None:
                    assert table_cell in _MARKERS, (
                        f"{label}/{grid}/{test_set}: figure has no value but the table prints "
                        f"{table_cell!r} — the figure would silently omit a published number"
                    )
                    continue
                _assert_agrees(cell, table_cell, metric, f"fig_synthesis {label}/{grid}/{test_set}")
                compared += 1
    assert compared == len(SYNTHESIS_BACC_ROWS + SYNTHESIS_FL_ROWS) * len(TEST_SETS) * len(
        GRID_ROWS
    )


def test_synthesis_display_rounding_is_faithful():
    """The heat map prints balanced accuracy at 2 d.p. and MAE as an integer; check that neither
    display rounding misstates the underlying value by more than half its last place."""
    if not _outputs_populated():
        pytest.skip("outputs/ not populated on this host")
    per_grid = held_out_rows()
    for rows, metric, fmt, tol in (
        (SYNTHESIS_BACC_ROWS, "balanced_accuracy", "{:.2f}", 0.005),
        (SYNTHESIS_FL_ROWS, "mae", "{:.0f}", 0.5),
    ):
        for (label, _stem, _view), fig_row in zip(
            rows, synthesis_matrix(rows, metric, per_grid, TEST_SETS), strict=True
        ):
            for cell in fig_row:
                if cell is None:
                    continue
                shown = float(fmt.format(cell["mean"]))
                assert abs(shown - cell["mean"]) <= tol, (
                    f"{label}: printed {shown} for {cell['mean']} ({metric})"
                )


# --------------------------------------------------------------------------------------------------
# Fig. 3 — cross-grid transfer dumbbells vs tab_transfer_*.tex
# --------------------------------------------------------------------------------------------------
def test_transfer_figure_points_and_cis_match_the_transfer_tables():
    """Each plotted point AND its 95% CI equals the corresponding ``tab_transfer_<fn>.tex`` cell.

    ``transfer_table`` lays each row out as grid x source, with columns
    (protocol x observability) — so the ``line`` column of a protocol is at a fixed offset."""
    idx = paper_lib._transfer_index("outputs")
    if not idx:
        pytest.skip("outputs/ has no transfer results on this host")
    val_idx = paper_lib._transfer_val_index("outputs")
    grids = [g for g, _ in GRID_ROWS]
    protos = [p for p, _ in _TRANSFER_PROTOS]
    compared = 0
    for key in TRANSFER_FIG_FUNCTIONS:
        _k, _name, stem, metric, _label = _PF_BY_KEY[key]
        table = _body_rows(f"tab_transfer_{key}.tex")
        task_id = f"{stem}_{TRANSFER_FIG_VIEW}"
        for si, source in enumerate(PRETRAIN_GRIDS):
            series = transfer_series(idx, task_id, source, metric, val_idx, grids)
            for gi, grid in enumerate(grids):
                # rows run grid-major, source-minor; the grid label only appears on si == 0
                row = table[gi * len(PRETRAIN_GRIDS) + si]
                for proto in protos:
                    cell = series[proto][gi]
                    col = 2 + protos.index(proto) * len(OBS) + OBS.index(TRANSFER_FIG_VIEW)
                    table_cell = row[col]
                    where = f"fig_transfer {key}/{source}->{grid}/{proto}"
                    if cell is None:
                        assert table_cell in _MARKERS, (
                            f"{where}: figure plots nothing but the table prints {table_cell!r}"
                        )
                        continue
                    _assert_agrees(cell, table_cell, metric, where)
                    # the drawn whisker must be the interval the table displays
                    clipped = _CLIPPED.match(table_cell)
                    if clipped:
                        assert (
                            f"{cell['lo']:.3f}" == clipped["lo"]
                            and f"{cell['hi']:.3f}" == clipped["hi"]
                        ), (
                            f"{where}: error bar [{cell['lo']:.3f}, {cell['hi']:.3f}] != printed "
                            f"[{clipped['lo']}, {clipped['hi']}]"
                        )
                    elif cell["ci"] is not None:
                        assert (cell["lo"], cell["hi"]) == (
                            cell["mean"] - cell["ci"],
                            cell["mean"] + cell["ci"],
                        ), f"{where}: unclipped error bar is not mean +/- ci"
                    else:
                        assert cell["lo"] == cell["hi"] == cell["mean"], (
                            f"{where}: no CI, so the mark must carry no whisker"
                        )
                    compared += 1
    expected = len(TRANSFER_FIG_FUNCTIONS) * len(PRETRAIN_GRIDS) * len(GRID_ROWS) * len(protos)
    assert compared == expected, f"compared {compared} transfer cells, expected {expected}"


def test_ci_bounds_is_the_tables_own_truncation_rule():
    """Data-free: the helper the figure draws from is the one the tables print with.

    Balanced accuracy is clipped to [0, 1] (F10); MAE in % of line length is not clipped; a missing
    half-width yields no interval at all."""
    # approx throughout: these are binary floats, and 0.98 - 0.05 is 0.9299999999999999
    assert ci_bounds(0.98, 0.05, "balanced_accuracy") == pytest.approx((0.93, 1.0))
    assert ci_bounds(0.02, 0.05, "balanced_accuracy") == pytest.approx((0.0, 0.07))
    assert ci_bounds(31.0, 2.0, "mae") == pytest.approx((29.0, 33.0))
    assert ci_bounds(0.5, None, "balanced_accuracy") is None
    # and the printed form of a clipped interval really is the bracket notation
    assert _CLIPPED.match(paper_lib._fmt_mean_ci(0.893, 0.112, "balanced_accuracy"))
