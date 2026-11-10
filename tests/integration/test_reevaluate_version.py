"""reevaluate resolves each protocol's own results tree by default (held_out lives in 1.0.0, not in
BENCHMARK_VERSION's tree), so its shipped sbatch can find the held_out cells at all."""

import json

from evemtbench.coverage import cell_dir, iter_cells
from evemtbench.evaluation.reevaluate import Report, reevaluate_unit
from evemtbench.version import BENCHMARK_VERSION, results_version

TASK = "fault_detection_local"


def _cells():
    return list(iter_cells(tasks=[TASK], grids=["double_line"], protocols=("held_out",)))


def test_default_version_reads_the_held_out_tree(fixture_root, held_out_splits, tmp_path):
    assert results_version("held_out") != BENCHMARK_VERSION  # the case this guards
    cells = _cells()
    assert cells
    # a run record exists ONLY in the held_out tree (as on the cluster)
    for c in cells:
        d = cell_dir(tmp_path, c)
        assert d.parts[-5] == results_version("held_out")
        d.mkdir(parents=True)
        (d / "run_record.json").write_text(json.dumps({"versions": {}}))

    report = Report()
    reevaluate_unit(
        TASK,
        "held_out",
        "double_line",
        cells,
        out=tmp_path,
        version=None,
        report=report,
        check_only=True,
        root=fixture_root,
        splits_root=held_out_splits,
    )
    reasons = {s["reason"] for s in report.skipped}
    # the record was found (it then fails the environment control, which is fine here)
    assert "run_record.json missing/unreadable" not in reasons
    assert "environment differs" in reasons

    # an explicit version still wins: BENCHMARK_VERSION's tree holds no held_out records
    report = Report()
    reevaluate_unit(
        TASK,
        "held_out",
        "double_line",
        cells,
        out=tmp_path,
        version=BENCHMARK_VERSION,
        report=report,
        check_only=True,
        root=fixture_root,
        splits_root=held_out_splits,
    )
    assert {s["reason"] for s in report.skipped} == {"run_record.json missing/unreadable"}
