"""Bidirectional paper<->dashboard number coverage (Part B / B3).

``test_audit_manifest.py`` checks one direction (each ledger ``paper_anchor`` still occurs in the
manuscript). This is the reverse and the numeric side: *every result number the paper reports is
on the audit dashboard, traced to its artifacts* — the user's traceability requirement made
mechanical.

The design that makes this checkable: the paper's result numbers live ONLY in the auto-generated
tables (``paper/manuscript/generated/tab_res_*.tex`` / ``supp_tab_*.tex``), each cell selected by
``make_paper_tables.best_cell_choice`` from the same ``result.json`` the dashboard reads. So the
checks are: (1) no result number is hand-typed outside ``generated/`` (else it would be
untraceable); (2) every paper-table coordinate maps to a real, dashboard-covered benchmark cell;
(3) the dashboard's paper-numbers page names every paper table and offers the five trace links;
(4) when the sweep has run, every number in the paper appears verbatim on the dashboard page.

Data-free except (4), which skips until ``outputs/`` is populated. Torch-free."""

import os
import re
from collections import Counter
from pathlib import Path

import pytest

import make_audit
from evemtbench.coverage import REFERENCE_GRIDS
from evemtbench.tasks.registry import get_task, list_tasks
from make_paper_tables import GRID_ROWS, OBS, PAPER_FUNCTIONS

REPO = Path(__file__).resolve().parents[2]

MANUSCRIPT = REPO / "paper" / "manuscript"
# Every test here is a paper<->dashboard correspondence check — meaningless without the private
# manuscript tree (pruned from the public release export, where only the dashboard ships).
pytestmark = pytest.mark.skipif(
    not MANUSCRIPT.is_dir(), reason="private manuscript not present (public release clone)"
)
# A result cell is unambiguously "mean ± ci" with three-decimal numbers joined by \pm; layout
# constants (4.5pt) and descriptive digits (9600 Hz, 70/15/15, grant ids) never take this shape.
# Tolerant of the delimiter style so a number hand-typed as "0.9 \pm 0.0" (no $…$) is caught too.
_RESULT_CELL = re.compile(r"\d\.\d{3}\s*\$?\s*\\pm\s*\$?\s*\d\.\d{3}")
# Any three-decimal token — a paper "number" once populated (means and CIs).
_NUM = re.compile(r"\d\.\d{3}")
# `$\pm{<}0.001$` states the DISPLAY PRECISION of a confidence interval, not a measured value, so
# its literal must not be counted as a paper number the dashboard has to reproduce. Stripped before
# extraction rather than added to an ignore set, so a genuine 0.001 measurement elsewhere is still
# checked.
_BOUND = re.compile(r"\$\\pm\{<\}[0-9.]+\$")


def _nums(text: str) -> list[str]:
    return _NUM.findall(_BOUND.sub("", text))


def _handwritten_tex() -> list[Path]:
    """The hand-authored manuscript bodies (NOT the generated fragments)."""
    return [p for p in (MANUSCRIPT / "main.tex", MANUSCRIPT / "supplementary.tex") if p.is_file()]


def test_result_numbers_live_only_in_generated_fragments():
    """Reverse guarantee: a result number never appears hand-typed in the manuscript body — every
    reported number is machine-generated from result.json (hence traceable)."""
    for tex in _handwritten_tex():
        hits = _RESULT_CELL.findall(tex.read_text())
        assert not hits, (
            f"{tex.relative_to(REPO)} contains hand-typed result numbers {hits} — result numbers "
            "must live ONLY in paper/manuscript/generated/*.tex (auto-generated from result.json), "
            "or they are untraceable. Move the value into the generated table."
        )


def test_paper_table_coordinates_are_valid_dashboard_cells():
    """Forward: every paper-table coordinate that CAN carry a number (its task id exists) is a real
    benchmark cell on a reference grid under held_out — i.e. dashboard-coverable."""
    paper_grids = [g for g, _ in GRID_ROWS]
    for g in paper_grids:
        assert g in REFERENCE_GRIDS, (
            f"paper table row grid {g!r} is not a reference grid {REFERENCE_GRIDS} — the dashboard "
            "coverage matrix would not cover it"
        )
    tasks = set(list_tasks())
    for _key, _name, stem, _metric, _label in PAPER_FUNCTIONS:
        real = [f"{stem}_{obs}" for obs in OBS if f"{stem}_{obs}" in tasks]
        assert real, f"paper function stem {stem!r} maps to no real task id (checked {OBS})"
        for task_id in real:
            protos = get_task(task_id).protocols
            assert "held_out" in protos, (
                f"paper reports {task_id} but its task defines protocols {protos} without "
                "held_out — the paper tables are held_out only"
            )


def test_dashboard_paper_numbers_page_names_every_table_and_offers_traces():
    """The dashboard's paper-numbers page references every paper result table + its generated
    fragment, and advertises the five trace links a reviewer needs from each number."""
    page = make_audit.build_paper_numbers([], None)
    for key, _name, _stem, _metric, _label in PAPER_FUNCTIONS:
        assert f"tab:res_{key}" in page, f"paper-numbers page omits paper table tab:res_{key}"
        assert f"tab_res_{key}.tex" in page, f"paper-numbers page omits fragment tab_res_{key}.tex"
    for g, _ in GRID_ROWS:
        assert f"g_{g}.html" in page, f"paper-numbers page omits grid drill-down link for {g}"
    for token in ("output file", "training log", "training code", "config", "W&amp;B"):
        assert token in page, f"paper-numbers page never mentions the trace target {token!r}"


def test_dashboard_paper_numbers_page_builds_with_provenance():
    """The page builds against a provenance map without error and reflects the version."""
    prov = {}
    page = make_audit.build_paper_numbers([], prov, version=make_audit.BENCHMARK_VERSION)
    assert f"provenance/{make_audit.BENCHMARK_VERSION}/index.json" in page


def test_every_paper_number_appears_on_the_dashboard():
    """When the sweep has run: every three-decimal number in the generated paper tables appears
    verbatim on the dashboard's paper-numbers page (same selection, same result.json). Skips until
    outputs/ is populated (pre-sweep the tables are all \\res)."""
    from evemtbench.leaderboard import scan_results
    from make_paper_tables import build_paper_tables
    from make_supplement_tables import build_supplement_tables

    os.chdir(REPO)  # load_rows / scan_results glob 'outputs/...' relative to cwd
    # Multiset, not set: a dropped/misplaced cell must not be masked when its value string happens
    # to coincide with another number elsewhere on the page — require the page to contain each
    # value at least as many times as the paper does.
    paper_counts = Counter(_nums("".join(build_paper_tables().values())))
    if not paper_counts:
        pytest.skip("no paper numbers wired yet (pre-sweep: generated tables are all \\res)")
    rows = scan_results("outputs")
    prov = make_audit.provenance_cells(rows, make_audit.collect_run_records("outputs"))
    page_counts = Counter(_nums(make_audit.build_paper_numbers(rows, prov)))
    deficient = {n: (c, page_counts[n]) for n, c in paper_counts.items() if page_counts[n] < c}
    assert not deficient, (
        f"main-text paper numbers under-represented on the dashboard paper-numbers page "
        f"{{value: (paper_count, page_count)}} = {deficient} — the paper and the dashboard have "
        "diverged (they must share make_paper_tables.best_cell_choice)"
    )
    # the per-baseline supplement draws from the same cells; its numbers must be traceable too via
    # the per-baseline drill-down pages (built from the same rows). The validation-selection audit
    # table (supp_tab_val_selection) is the exception: it reports the *validation-split* selection
    # scores, not held_out results, so those trace to the validation re-score artifacts
    # (hpc/rescore_validation.py) that drive best_cell_choice, not to the held_out drill-down pages.
    _skip = "supp_tab_val_selection.tex"
    _supp = {k: v for k, v in build_supplement_tables().items() if k != _skip}
    supp_counts = Counter(_nums("".join(_supp.values())))
    baseline_pages = "".join(
        make_audit.build_baseline_page(b, rows, prov) for b in make_audit.ALL_BASELINES
    )
    bp_counts = Counter(_nums(baseline_pages))
    # Membership, not multiplicity: an SI table may legitimately restate a value the headline
    # tables already carry (the protection table repeats the selected reference's balanced accuracy
    # beside its MDR/FAR so a collapse cell is readable in one row). What must hold is that every
    # value the supplement prints is traceable to a drill-down page, not that it is printed there
    # exactly as often.
    supp_missing = sorted(n for n in supp_counts if bp_counts[n] == 0)
    assert not supp_missing, (
        f"supplement numbers absent from the baseline drill-down pages: {supp_missing}"
    )
