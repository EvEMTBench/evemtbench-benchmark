"""Invariants for the supplementary per-baseline tables builder (hpc/make_supplement_tables.py):
deterministic, empty-outputs → all-`\\res`, a real result populates exactly its cell, and the
committed fragments stay well-formed (they now populate live from the sweep — no freeze)."""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

N_BODY_CELLS = 7 * 12  # 7 baselines × (4 grids × 3 observability views)

# the 10 standard per-baseline tables (each with N_BODY_CELLS cells); Package C adds further SI
# tables (protection rates, per-class F1, FC support, split counts, diagnostics) of other shapes.
STD_PER_BASELINE = {
    f"supp_tab_{k}_{ts}.tex" for k in ("fd", "fc", "fl", "ed", "ec") for ts in ("test", "benchmark")
}


def _load_builder():
    sys.path.insert(0, str(REPO / "hpc"))  # hpc/ isn't a package
    import make_supplement_tables as mst

    return mst


def test_build_is_deterministic_and_all_res_without_outputs(tmp_path, monkeypatch):
    mst = _load_builder()
    monkeypatch.chdir(tmp_path)  # no outputs/ -> every cell must be \res
    a = mst.build_supplement_tables()
    b = mst.build_supplement_tables()
    assert a == b
    assert STD_PER_BASELINE <= set(a)  # the 10 standard tables, plus the Package C SI tables
    for name, latex in a.items():
        assert "\\begin{table" in latex and "AUTO-GENERATED" in latex, name
    for name in STD_PER_BASELINE:
        # B2: pre-outputs placeholder is the five-state marker set (\pending/\notapp), not \res
        body = a[name].split("\\midrule")[1]
        assert body.count("\\pending") + body.count("\\notapp") == N_BODY_CELLS, name
        assert "$\\pm$" not in body, name


def test_real_result_populates_only_its_cell(tmp_path, monkeypatch):
    mst = _load_builder()
    from evemtbench.version import BENCHMARK_VERSION

    d = (
        tmp_path
        / "outputs"
        / BENCHMARK_VERSION
        / "fault_detection_local"
        / "held_out"
        / "double_line"
        / "threshold"
        / "test"
    )
    d.mkdir(parents=True)
    (d / "result.json").write_text(
        json.dumps(
            {
                "task": "fault_detection_local",
                "baseline": "threshold",
                "test_set": "test",
                "aggregate": {"balanced_accuracy": {"mean": 0.777, "std": 0.01, "ci95": 0.011}},
                "per_seed": [{"seed": 0, "metrics": {"balanced_accuracy": 0.777}}],
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    tables = mst.build_supplement_tables()
    fd_test = tables["supp_tab_fd_test.tex"]
    # the threshold row shows in the supplement, in the compact per-baseline CI form (subscript)
    assert "0.777$\\,{\\scriptscriptstyle\\pm0.011}$" in fd_test
    body = fd_test.split("\\midrule")[1]
    # exactly one cell populated; every other cell stays a five-state marker
    assert body.count("\\pending") + body.count("\\notapp") == N_BODY_CELLS - 1
    # the benchmark-set twin and other functions stay placeholder-only (fully markered)
    for other in ("supp_tab_fd_benchmark.tex", "supp_tab_ec_test.tex"):
        ob = tables[other].split("\\midrule")[1]
        assert ob.count("\\pending") + ob.count("\\notapp") == N_BODY_CELLS


def test_committed_supplement_fragments_present():
    """The 10 committed supplement fragments exist and are self-contained table envs. They now
    populate live from the sweep (a gated view, like RESULTS.md) — there is no pre-freeze \\res
    requirement; freshness is enforced by the byte-compare gate where outputs/ lives."""
    import pytest

    gen = REPO / "paper" / "manuscript" / "generated"
    if not gen.is_dir():
        pytest.skip("generated/ not present (public release clone)")
    frags = sorted(gen.glob("supp_tab_*.tex"))
    # 10 standard per-baseline tables + 5 Package C SI tables (protection, fc_perclass, fc_support,
    # splits, diagnostics) + 1 validation-selection audit table (val_selection)
    assert len(frags) == 16, f"expected 16 committed supplement fragments, found {len(frags)}"
    assert STD_PER_BASELINE <= {f.name for f in frags}
    for f in frags:
        txt = f.read_text()
        assert "\\begin{table" in txt and "\\end{table" in txt, f.name
        assert "AUTO-GENERATED" in txt, f.name
