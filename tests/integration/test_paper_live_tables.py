"""The manuscript's generated fragments are a LIVE gated view (paper/AGENTS.md): they populate
from the current sweep and must byte-match a fresh rebuild from outputs/ (``--fix`` regenerates
them), exactly like RESULTS.md. There is no freeze gate — a committed populated fragment is allowed,
not a policy violation."""

import json
from pathlib import Path

import check_results_fresh as gate
from evemtbench.version import results_version

REPO = Path(__file__).resolve().parents[2]

POPULATED_FRAGMENT = "% AUTO-GENERATED\n\\begin{tabular}{l}\n0.883 $\\pm$ 0.030\n\\end{tabular}\n"


def _gen_dir(root: Path) -> Path:
    gen = root / "paper" / "manuscript" / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    return gen


def test_populated_fragment_is_allowed(tmp_path, monkeypatch):
    """Live-tables policy: a committed populated fragment is fine (the old \\res-until-freeze block
    is gone). With no outputs/ the gate skips cleanly rather than flagging a 'violation'."""
    monkeypatch.setenv("EVEMTBENCH_REPO", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    gen = _gen_dir(tmp_path)
    (gen / "tab_res_fd.tex").write_text(POPULATED_FRAGMENT)
    assert gate.main([]) == 0  # no outputs -> skip; populated is NOT a violation anymore
    assert gate.main(["--fix"]) == 0


def test_paper_fragments_are_a_live_gated_view(tmp_path, monkeypatch):
    """With outputs present and NO FROZEN_TAG, the paper fragments join the byte-compare gate:
    missing/stale -> FAIL; --fix regenerates them (populated from the data); tamper -> FAIL."""
    monkeypatch.setenv("EVEMTBENCH_REPO", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    # seed one held_out result so outputs/ is present (builders glob outputs/<version>/…)
    d = (
        tmp_path
        / "outputs"
        / results_version("held_out")
        / "fault_classification_local"
        / "held_out"
        / "cigre_mv"
        / "mlp"
        / "test"
    )
    d.mkdir(parents=True)
    (d / "result.json").write_text(
        json.dumps(
            {
                "benchmark_version": results_version("held_out"),
                "task": "fault_classification_local",
                "protocol": "held_out",
                "grid": "cigre_mv",
                "baseline": "mlp",
                "test_set": "test",
                "n_test": 10,
                "seeds": [0],
                "aggregate": {"balanced_accuracy": {"mean": 0.5, "std": 0.0}},
            }
        )
    )
    gen = _gen_dir(tmp_path)  # NO FROZEN_TAG — fragments are gated regardless

    # committed fragments missing -> the gate flags them stale; --fix regenerates all views
    assert gate.main([]) == 1
    assert gate.main(["--fix"]) == 0
    frag = gen / "tab_res_fc.tex"
    assert frag.exists() and "AUTO-GENERATED" in frag.read_text()
    assert "0.500" in frag.read_text()  # populated live from the seeded result (no freeze needed)
    assert gate.main([]) == 0  # now fresh -> passes

    # tamper a fragment -> stale again
    frag.write_text(frag.read_text() + "% drift\n")
    assert gate.main([]) == 1
