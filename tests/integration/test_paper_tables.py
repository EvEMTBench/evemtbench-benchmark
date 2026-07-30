"""Invariants for the paper's results→LaTeX builder (hpc/make_paper_tables.py):
deterministic, empty-outputs → all-`\\res` (so the manuscript builds with no data), and a real
result populates its cell as `mean ± ci`."""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load_builder():
    sys.path.insert(0, str(REPO / "hpc"))  # hpc/ isn't a package
    import make_paper_tables as mpt

    return mpt


def test_build_is_deterministic_and_all_res_without_outputs(tmp_path, monkeypatch):
    mpt = _load_builder()
    monkeypatch.chdir(tmp_path)  # no outputs/ here -> every cell must be \res
    a = mpt.build_paper_tables()
    b = mpt.build_paper_tables()
    assert a == b  # pure/deterministic
    assert set(a) == {f"tab_res_{k}.tex" for k in ("fd", "fc", "fl", "ed", "ec")}
    for name, latex in a.items():
        assert "\\begin{table*}" in latex and f"tab:res_{name[8:-4]}" in latex
        # 4 grids × 6 columns = 24 body cells, all \res (no numeric ± cells)
        body = latex.split("\\midrule")[1]
        # B2: pre-outputs placeholder is the five-state marker set (\pending/\notapp), not \res
        assert body.count("\\pending") + body.count("\\notapp") == 24, name
        assert "$\\pm$" not in body, name


def test_real_result_populates_its_cell(tmp_path, monkeypatch):
    mpt = _load_builder()
    from evemtbench.version import BENCHMARK_VERSION

    # a done fault_detection_local cell on double_line/test for a learned baseline (mlp)
    d = (
        tmp_path
        / "outputs"
        / BENCHMARK_VERSION
        / "fault_detection_local"
        / "held_out"
        / "double_line"
        / "mlp"
        / "test"
    )
    d.mkdir(parents=True)
    (d / "result.json").write_text(
        json.dumps(
            {
                "task": "fault_detection_local",
                "baseline": "mlp",
                "test_set": "test",
                "aggregate": {"balanced_accuracy": {"mean": 0.912, "std": 0.01, "ci95": 0.023}},
                "per_seed": [{"seed": s, "metrics": {"balanced_accuracy": 0.9}} for s in range(5)],
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    fd = mpt.build_paper_tables()["tab_res_fd.tex"]
    assert (
        "0.912 $\\pm$ 0.023" in fd
    )  # the cell is populated as mean ± ci (main tables, non-compact)
    # a different task's table stays fully placeholder (five-state markers, not the old \res)
    ec_body = mpt.build_paper_tables()["tab_res_ec.tex"].split("\\midrule")[1]
    assert ec_body.count("\\pending") + ec_body.count("\\notapp") == 24


def test_committed_fragments_are_well_formed():
    """The committed generated/*.tex are valid, self-contained table envs (survives real
    wiring — every body cell is either \\res or a `mean ± ci` number, never a stray)."""
    import pytest

    gen = REPO / "paper" / "manuscript" / "generated"
    if not gen.is_dir():
        pytest.skip("generated/ not present")
    for key in ("fd", "fc", "fl", "ed", "ec"):
        latex = (gen / f"tab_res_{key}.tex").read_text()
        assert "\\begin{table*}" in latex and "\\end{table*}" in latex
        assert f"\\label{{tab:res_{key}}}" in latex
        body = latex.split("\\midrule")[1].split("\\bottomrule")[0]
        for row in (ln for ln in body.splitlines() if "&" in ln):
            cells = [c.strip() for c in row.split("&")[1:]]  # drop the grid label
            assert len(cells) == 6, f"{key}: row has {len(cells)} cells, expected 6"
            for c in cells:
                c = c.rstrip("\\").strip()
                # a cell is a five-state marker, a `mean ± ci` value, an F10-clipped interval
                # (mean + a bracketed [lo,hi], no ±), or a bare number
                assert (
                    c in ("\\res", "\\pending", "\\notapp", "\\oos", "\\undefm")
                    or "$\\pm$" in c
                    or "scriptscriptstyle" in c
                    or _is_number(c)
                ), f"{key}: bad cell {c!r}"


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
