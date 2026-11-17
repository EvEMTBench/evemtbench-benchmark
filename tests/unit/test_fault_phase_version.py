"""The per-cell results version of the v1.3.0 scoped revision (label patch LP-1, fault_phase).

Exactly the fault_phase cells that read a repaired label move to 1.3.0 — 151 cells — and every
other cell keeps its tree. A wrong entry would either leave a cell scored without the
high-impedance-with-arc class reported as current, or churn an unaffected published cell."""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from evemtbench.coverage import (
    PRETRAIN_GRID,
    SNAPSHOT_PROTOCOLS,
    Cell,
    cell_dir,
    cell_from_key,
    cell_results_version,
    iter_cells,
)
from evemtbench.version import (
    DEFAULT_PRETRAIN_GRID,
    is_fault_phase_task,
    label_patches_enabled,
    results_version,
)

FP_TASKS = ("fault_phase_local", "fault_phase_line", "fault_phase_global")


def _fp_cells():
    return [c for c in iter_cells(protocols=SNAPSHOT_PROTOCOLS) if is_fault_phase_task(c.task)]


def test_default_source_agrees_with_coverage():
    assert DEFAULT_PRETRAIN_GRID == PRETRAIN_GRID


@pytest.mark.parametrize("task", FP_TASKS)
@pytest.mark.parametrize("protocol", SNAPSHOT_PROTOCOLS)
def test_fault_phase_without_grid_raises(task, protocol):
    with pytest.raises(ValueError, match="needs grid="):
        results_version(protocol, task)


def test_task_none_and_other_tasks_need_no_grid():
    assert results_version("held_out") == "1.0.0"
    assert results_version("transfer_finetune", "fault_detection_local") == "1.1.0"
    assert results_version("held_out", "fault_location_line") == "1.2.0"


@pytest.mark.parametrize(
    "protocol, task, grid, source, expected",
    [
        ("held_out", "fault_phase_global", "cigre_mv", None, "1.3.0"),
        ("held_out", "fault_phase_line", "double_line", None, "1.0.0"),
        ("held_out", "fault_phase_local", "ieee39", None, "1.0.0"),
        ("multi_grid_pretrain", "fault_phase_line", "template_20kv", None, "1.3.0"),
        ("multi_grid_pretrain", "fault_phase_line", "template_110kv", None, "1.1.0"),
        ("multi_grid_pretrain", "fault_phase_local", "template_345kv", None, "1.1.0"),
        ("transfer_zeroshot", "fault_phase_local", "cigre_mv", "template_345kv", "1.3.0"),
        ("transfer_finetune", "fault_phase_line", "cigre_mv", "template_110kv", "1.3.0"),
        ("transfer_finetune", "fault_phase_line", "double_line", "template_20kv", "1.3.0"),
        ("transfer_zeroshot", "fault_phase_line", "ieee39", None, "1.3.0"),  # default source
        ("transfer_finetune", "fault_phase_local", "testgrid_110kv", "template_110kv", "1.1.0"),
        ("transfer_zeroshot", "fault_phase_line", "double_line", "template_345kv", "1.1.0"),
    ],
)
def test_version_per_cell(protocol, task, grid, source, expected):
    assert results_version(protocol, task, grid=grid, source=source) == expected


def test_exactly_151_cells_move_to_1_3_0():
    cells = _fp_cells()
    assert len(cells) == 354  # every fault_phase cell of the matrix
    moved = [c for c in cells if cell_results_version(c) == "1.3.0"]
    assert len(moved) == 151
    by_group = Counter(
        (
            c.protocol,
            "cigre_mv" if c.grid == "cigre_mv" else "other",
            c.source == "template_20kv" if c.source else None,
        )
        for c in moved
    )
    assert by_group == {
        ("held_out", "cigre_mv", None): 21,  # 3 tasks x 7 applicable baselines
        ("multi_grid_pretrain", "other", None): 10,  # template_20kv, 2 tasks x 5 deep
        ("transfer_zeroshot", "cigre_mv", True): 10,
        ("transfer_zeroshot", "cigre_mv", False): 20,
        ("transfer_zeroshot", "other", True): 30,
        ("transfer_finetune", "cigre_mv", True): 10,
        ("transfer_finetune", "cigre_mv", False): 20,
        ("transfer_finetune", "other", True): 30,
    }
    assert all(c.grid == "template_20kv" for c in moved if c.protocol == "multi_grid_pretrain")
    # every unaffected fault_phase cell keeps its pre-1.3.0 tree
    for c in cells:
        if c not in moved:
            want = "1.0.0" if c.protocol == "held_out" else "1.1.0"
            assert cell_results_version(c) == want, c


def test_cell_dir_and_runner_paths_follow_the_cell_version(tmp_path):
    from evemtbench.baselines.runner import (
        _cell_dir_from_args,
        _pretrained_bundle_path,
        cell_identity,
    )

    c = Cell("fault_phase_line", "transfer_finetune", "double_line", "gru", source="template_110kv")
    assert cell_dir(Path("o"), c).parts[1] == "1.1.0"
    c20 = Cell("fault_phase_line", "transfer_finetune", "double_line", "gru", source=PRETRAIN_GRID)
    assert cell_dir(Path("o"), c20) == Path(
        "o/1.3.0/fault_phase_line/transfer_finetune/double_line/gru"
    )

    def args(protocol, grid, src):
        return SimpleNamespace(
            out=str(tmp_path),
            task="fault_phase_line",
            protocol=protocol,
            grid=grid,
            baseline="gru",
            zero_seq=False,
            pretrain_grid=src,
            pretrained_from=None,
        )

    a = args("transfer_finetune", "cigre_mv", "template_345kv")
    assert _cell_dir_from_args(a).relative_to(tmp_path).parts[0] == "1.3.0"
    # its source model is the 345 kV pretrain cell, which the patch does not touch
    assert _pretrained_bundle_path(a, 0, "gru").relative_to(tmp_path).parts[0] == "1.1.0"
    b = args("transfer_zeroshot", "ieee39", "template_20kv")
    assert _pretrained_bundle_path(b, 0, "gru").relative_to(tmp_path).parts[0] == "1.3.0"
    _, key = cell_identity(
        "fault_phase_line", "transfer_finetune", "ieee39", "gru", "template_110kv"
    )
    assert key.startswith("1.1.0/")
    _, key = cell_identity("fault_phase_line", "held_out", "cigre_mv", "gru", PRETRAIN_GRID)
    assert key.startswith("1.3.0/")


def test_cell_from_key_round_trips():
    for c in _fp_cells():
        d = cell_dir(Path("o"), c)
        back = cell_from_key(Path(*d.parts[2:]))
        assert back == c, (c, back)
        assert cell_results_version(back) == d.parts[1]
    assert cell_from_key("a/b") is None


def test_label_patches_switch():
    assert label_patches_enabled("fault_phase_line", "1.3.0")
    assert not label_patches_enabled("fault_phase_line", "1.0.0")
    assert not label_patches_enabled("fault_phase_line", "1.1.0")
    assert not label_patches_enabled("fault_detection_line", "1.3.0")


def test_evaluate_cli_needs_the_source_of_a_fault_phase_transfer_cell(tmp_path):
    from evemtbench.evaluation import cli

    with pytest.raises(SystemExit):  # argparse error before any data is read
        cli.main(
            [
                "--task",
                "fault_phase_line",
                "--protocol",
                "transfer_finetune",
                "--grid",
                "ieee39",
                "--predictions",
                str(tmp_path / "p.parquet"),
            ]
        )
