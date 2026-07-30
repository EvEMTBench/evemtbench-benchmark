"""Invariants the results-freshness gate (hpc/check_results_fresh.py) relies on:
deterministic RESULTS builder, and the leaderboard "updated <ts>" line being the ONLY volatile
content (so the gate can normalize it away and still detect real drift)."""

import json
import re
import sys
from pathlib import Path

from evemtbench.leaderboard import render_html, scan_results

REPO = Path(__file__).resolve().parents[2]
_VOLATILE = re.compile(r"<div class=v>updated [^<]*</div>")


def _build_results():
    sys.path.insert(0, str(REPO / "hpc"))  # hpc/ isn't a package; load make_results directly
    from make_results import build_results

    return build_results()


def test_build_results_deterministic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no outputs/ here -> narrative only, but pure + repeatable
    a = _build_results()
    b = _build_results()
    assert isinstance(a, str) and a.startswith("# EvEMTBench")
    assert a == b
    assert "Tier 1" in a  # tier legend is in the narrative


def test_make_results_labels_every_function():
    """Every registry Function has a human label + tier, so no task renders as a raw enum and the
    tier grouping is total (a new task without a label/tier fails here, not silently in RESULTS)."""
    sys.path.insert(0, str(REPO / "hpc"))
    import make_results as mr
    from evemtbench.tasks.registry import get_task, list_tasks
    from evemtbench.tasks.taxonomy import Function

    for fn in Function:
        assert fn in mr.FUNCTION_LABEL, f"make_results.FUNCTION_LABEL missing {fn}"
    for tid in list_tasks():
        assert get_task(tid).tier in mr.TIER_TITLE


def test_leaderboard_timestamp_is_the_only_volatile_line(tmp_path):
    d = tmp_path.joinpath(
        "0.1.0-dev", "fault_detection_local", "held_out", "double_line", "mlp", "test"
    )
    d.mkdir(parents=True)
    (d / "result.json").write_text(
        json.dumps(
            {
                "task": "fault_detection_local",
                "protocol": "held_out",
                "grid": "double_line",
                "baseline": "mlp",
                "test_set": "test",
                "n_test": 10,
                "seeds": [0],
                "aggregate": {"balanced_accuracy": {"mean": 0.8, "std": 0.0}},
            }
        )
    )
    rows = scan_results(tmp_path)
    a = render_html(rows, version="0.1.0-dev", generated_at="2026-01-01 00:00 UTC")
    b = render_html(rows, version="0.1.0-dev", generated_at="2099-12-31 23:59 UTC")
    assert a != b  # the timestamps differ...
    assert _VOLATILE.sub("", a) == _VOLATILE.sub("", b)  # ...and that is the ONLY difference
