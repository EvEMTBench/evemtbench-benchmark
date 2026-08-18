"""Golden fixture-metrics ratchet (the UnderstandingVCS divergence-count-ratchet idiom).

Pins the EXACT aggregate metrics of the analytically deterministic baselines (majority,
threshold) on the synthetic fixtures against a committed golden file. Any change to label
derivation, alignment, casting or metric wiring moves these values and fails HERE by name —
silent scoring drift becomes impossible. The UnderstandingVCS same-commit rule applies: a
legitimate change must refresh the golden file **in the same commit**
(``GOLDEN_UPDATE=1 pytest tests/integration/test_golden_fixture_metrics.py``).

random_forest and the deep baselines are deliberately NOT pinned: their exact values depend on
sklearn/torch versions, so pinning them would ratchet the library lockfile, not our code.
"""

import json
import os
from pathlib import Path

import pytest

from evemtbench import results_version
from evemtbench.baselines import runner

HO_VER = results_version("held_out")

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "fixture_metrics.json"

# (baseline, task) — analytic/deterministic pairs only, covering binary + regression paths.
PINNED = [
    ("majority", "fault_detection_local"),
    ("threshold", "fault_detection_local"),
    ("majority", "fault_location_local"),
]


def _aggregates(baseline, task, fixture_root, held_out_splits, out) -> dict:
    runner.main(
        [
            "--task", task,
            "--protocol", "held_out",
            "--grid", "double_line",
            "--baseline", baseline,
            "--seeds", "0",
            "--root", str(fixture_root),
            "--splits-root", str(held_out_splits),
            "--out", str(out),
        ]
    )  # fmt: skip
    base = out / HO_VER / task / "held_out" / "double_line" / baseline
    return {
        ts: json.loads((base / ts / "result.json").read_text())["aggregate"]
        for ts in ("test", "benchmark")
    }


@pytest.mark.golden
def test_fixture_metrics_match_golden(fixture_root, held_out_splits, tmp_path):
    actual = {
        f"{baseline}/{task}": _aggregates(
            baseline, task, fixture_root, held_out_splits, tmp_path / "out"
        )
        for baseline, task in PINNED
    }
    if os.environ.get("GOLDEN_UPDATE") == "1":
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"golden file rewritten: {GOLDEN}")
    assert GOLDEN.is_file(), (
        "no committed golden file — generate with GOLDEN_UPDATE=1 and commit it"
    )
    expected = json.loads(GOLDEN.read_text())
    assert actual == expected, (
        "fixture metrics moved vs the committed golden values — if the change is intended "
        "(new label derivation / metric fix), refresh with GOLDEN_UPDATE=1 and commit the "
        "golden file IN THE SAME COMMIT; otherwise this is silent scoring drift"
    )
