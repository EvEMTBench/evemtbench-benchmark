"""The per-cell results-tree mapping of the v1.2.0 and v1.3.0 scoped revisions.

``results_version(protocol, task)`` decides which ``outputs/<version>/`` tree a cell is written to
and read from, so a wrong entry silently reports a superseded number as current (or trains into
the wrong tree). Every protocol x {fault-location, other} combination is pinned explicitly."""

from pathlib import Path

import pytest

from evemtbench.coverage import SNAPSHOT_PROTOCOLS, Cell, cell_dir, iter_cells
from evemtbench.splits.protocols import PROTOCOLS
from evemtbench.tasks.registry import get_task, list_tasks
from evemtbench.tasks.taxonomy import Function
from evemtbench.version import (
    BENCHMARK_VERSION,
    CURRENT_RESULTS_VERSIONS,
    DATA_CACHE_VERSION,
    FL_RESULTS_VERSION,
    FP_RESULTS_VERSION,
    HELD_OUT_RESULTS_VERSION,
    TRANSFER_RESULTS_VERSION,
    is_fault_location_task,
    is_fault_phase_task,
    results_version,
)

FL_TASKS = ("fault_location_local", "fault_location_line", "fault_location_global")
OTHER_TASKS = ("fault_detection_local", "fault_classification_line", "event_state_global")

# (protocol, is fault location) -> tree. Literal on purpose: the mapping IS the contract.
EXPECTED = {
    ("held_out", True): "1.2.0",
    ("transfer_zeroshot", True): "1.2.0",
    ("transfer_finetune", True): "1.2.0",
    ("multi_grid_pretrain", True): "1.1.0",
    ("held_out", False): "1.0.0",
    ("transfer_zeroshot", False): "1.1.0",
    ("transfer_finetune", False): "1.1.0",
    ("multi_grid_pretrain", False): "1.1.0",
}


def test_version_constants():
    assert BENCHMARK_VERSION == "1.3.0"
    assert HELD_OUT_RESULTS_VERSION == "1.0.0"
    assert TRANSFER_RESULTS_VERSION == "1.1.0"
    assert FL_RESULTS_VERSION == "1.2.0"
    assert FP_RESULTS_VERSION == "1.3.0"
    assert CURRENT_RESULTS_VERSIONS == ("1.0.0", "1.1.0", "1.2.0", "1.3.0")


def test_expected_table_covers_every_protocol():
    assert {p for p, _ in EXPECTED} == set(PROTOCOLS) == set(SNAPSHOT_PROTOCOLS)


@pytest.mark.parametrize("protocol", sorted(PROTOCOLS))
@pytest.mark.parametrize("task", FL_TASKS)
def test_fault_location_tasks(protocol, task):
    assert results_version(protocol, task) == EXPECTED[(protocol, True)]


@pytest.mark.parametrize("protocol", sorted(PROTOCOLS))
@pytest.mark.parametrize("task", OTHER_TASKS)
def test_other_tasks(protocol, task):
    assert results_version(protocol, task) == EXPECTED[(protocol, False)]


@pytest.mark.parametrize("protocol", sorted(PROTOCOLS))
def test_without_a_task_is_the_non_fault_location_tree(protocol):
    """``task=None`` is the label/heading form: it must equal every non-FL task's tree."""
    assert results_version(protocol) == EXPECTED[(protocol, False)]
    assert results_version(protocol, None) == EXPECTED[(protocol, False)]


def test_task_id_prefix_agrees_with_the_registry_for_every_task():
    """FL-ness is decided from the task-id prefix (no registry import in ``version``); it must be
    exactly the registry's ``Function.FAULT_LOCATION`` tasks, for every registered task and
    protocol."""
    tasks = list_tasks()
    assert set(FL_TASKS) <= set(tasks)
    for task in tasks:
        is_fl = get_task(task).function is Function.FAULT_LOCATION
        assert is_fault_location_task(task) is is_fl, task
        assert is_fault_phase_task(task) is (get_task(task).function is Function.FAULT_PHASE)
        if is_fault_phase_task(task):
            continue  # scoped per cell: tests/unit/test_fault_phase_version.py
        for protocol in PROTOCOLS:
            assert results_version(protocol, task) == EXPECTED[(protocol, is_fl)], (task, protocol)
    assert {t for t in tasks if is_fault_location_task(t)} == set(FL_TASKS)
    assert not is_fault_location_task(None)


def test_every_matrix_cell_resolves_into_a_current_tree_by_task():
    """coverage.cell_dir (where the runner writes and every reader looks) follows the mapping for
    every cell of the matrix, and an explicit version still wins."""
    cells = iter_cells(protocols=SNAPSHOT_PROTOCOLS)
    assert any(is_fault_location_task(c.task) for c in cells)
    seen = set()
    for c in cells:
        tree = cell_dir(Path("o"), c).parts[1]
        seen.add(tree)
        if is_fault_phase_task(c.task):
            continue  # scoped per cell: tests/unit/test_fault_phase_version.py
        assert tree == EXPECTED[(c.protocol, is_fault_location_task(c.task))], c
    assert seen == set(CURRENT_RESULTS_VERSIONS)
    fl = Cell("fault_location_line", "held_out", "double_line", "cnn")
    assert cell_dir(Path("o"), fl) == Path("o/1.2.0/fault_location_line/held_out/double_line/cnn")
    assert cell_dir(Path("o"), fl, version="1.0.0").parts[1] == "1.0.0"


def test_runner_paths_follow_the_mapping(tmp_path):
    """The runner's own path helpers (cell dir known before the task loads; the pre-trained
    bundle a transfer run initialises from) use the same mapping: a fault-location fine-tune run
    writes into 1.2.0 but reads its source model from the 1.1.0 pretrain tree."""
    from types import SimpleNamespace

    from evemtbench.baselines.runner import _cell_dir_from_args, _pretrained_bundle_path
    from evemtbench.coverage import PRETRAIN_GRID

    def args(task, protocol):
        return SimpleNamespace(
            out=str(tmp_path),
            task=task,
            protocol=protocol,
            grid="double_line",
            baseline="cnn",
            zero_seq=False,
            pretrain_grid=PRETRAIN_GRID,
            pretrained_from=None,
        )

    fl = args("fault_location_line", "transfer_finetune")
    assert _cell_dir_from_args(fl).relative_to(tmp_path).parts[0] == "1.2.0"
    assert _pretrained_bundle_path(fl, 0, "cnn").relative_to(tmp_path).parts[:3] == (
        "1.1.0",
        "fault_location_line",
        "multi_grid_pretrain",
    )
    fd = args("fault_detection_line", "transfer_finetune")
    assert _cell_dir_from_args(fd).relative_to(tmp_path).parts[0] == "1.1.0"
    assert _pretrained_bundle_path(fd, 0, "cnn").relative_to(tmp_path).parts[0] == "1.1.0"
    assert (
        _cell_dir_from_args(args("fault_location_global", "held_out"))
        .relative_to(tmp_path)
        .parts[0]
        == "1.2.0"
    )
    assert (
        _cell_dir_from_args(args("fault_detection_local", "held_out"))
        .relative_to(tmp_path)
        .parts[0]
        == "1.0.0"
    )


def test_cache_version_does_not_follow_the_benchmark_version(tmp_path, monkeypatch):
    """The projection cache root is keyed by DATA_CACHE_VERSION, which stays at 1.1.0: the v1.2.0
    revision changes no input, so it must neither relocate nor mark as stale the existing cache."""
    from evemtbench.data import staging

    assert DATA_CACHE_VERSION == "1.1.0"
    assert DATA_CACHE_VERSION != BENCHMARK_VERSION
    monkeypatch.setenv("EVEMTBENCH_PROJECTION_CACHE_ROOT", str(tmp_path))
    assert staging.persistent_cache_root() == tmp_path / "1.1.0"
    # the existing 1.1.0 cache is the current one: the GC must not list it as stale
    cur = tmp_path / "1.1.0"
    cur.mkdir()
    (cur / "proj_x.f32").write_bytes(b"0")
    assert staging.prune_stale_versions(dry_run=True) == []
