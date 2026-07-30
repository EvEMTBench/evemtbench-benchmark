"""Coverage enumeration + applicability for the pre-training/transfer protocols, and the
protocol-driven normalization default. Pure (registry + protocol logic; no data)."""

from evemtbench.coverage import (
    DEEP_BASELINES,
    PRETRAIN_GRID,
    PRETRAIN_GRIDS,
    REFERENCE_GRIDS,
    expected_test_sets,
    iter_cells,
    protocol_grid_applies,
)
from evemtbench.splits.protocols import default_input_normalize
from evemtbench.tasks.registry import get_task, list_tasks

_TRANSFER = ("transfer_zeroshot", "transfer_finetune")


def test_only_local_line_fault_tasks_support_transfer():
    supported = {t for t in list_tasks() if "transfer_zeroshot" in get_task(t).protocols}
    # exactly the 6 fault-family objectives x {local, line}
    assert len(supported) == 12
    assert all(t.endswith(("_local", "_line")) for t in supported)
    assert not any(t.endswith("_global") for t in supported)
    # event/switch-scope tasks never transfer
    assert "event_state_global" not in supported
    assert "held_out" in get_task("fault_detection_global").protocols  # global keeps held_out only
    assert "transfer_zeroshot" not in get_task("fault_detection_global").protocols


def test_protocol_grid_applicability():
    for g in PRETRAIN_GRIDS:
        assert protocol_grid_applies("multi_grid_pretrain", g)
    assert not protocol_grid_applies("multi_grid_pretrain", "double_line")
    for g in REFERENCE_GRIDS:
        assert protocol_grid_applies("transfer_zeroshot", g)
        assert protocol_grid_applies("held_out", g)
    assert not protocol_grid_applies("transfer_zeroshot", PRETRAIN_GRID)
    assert not protocol_grid_applies("held_out", PRETRAIN_GRID)  # no adapt_grid/benchmark there


def test_iter_cells_shapes():
    pre = iter_cells(protocols=("multi_grid_pretrain",))
    assert {c.grid for c in pre} == set(PRETRAIN_GRIDS)
    assert {c.baseline for c in pre} == set(DEEP_BASELINES)  # deep only
    assert len(pre) == 12 * len(DEEP_BASELINES) * len(PRETRAIN_GRIDS)
    for proto in _TRANSFER:
        cells = iter_cells(protocols=(proto,))
        assert {c.grid for c in cells} == set(REFERENCE_GRIDS)  # grid = transfer TARGET
        assert {c.baseline for c in cells} == set(DEEP_BASELINES)
        # transfer fans out over the pre-training SOURCE grids (20/110/345 kV)
        assert {c.source for c in cells} == set(PRETRAIN_GRIDS)
        assert len(cells) == 12 * len(DEEP_BASELINES) * len(REFERENCE_GRIDS) * len(PRETRAIN_GRIDS)


def test_expected_test_sets_pretrain_is_in_corpus_only():
    # multi_grid_pretrain has no separate benchmark family -> test only, regardless of grid splits
    assert expected_test_sets(PRETRAIN_GRID, "multi_grid_pretrain") == ("test",)


def test_default_normalize_per_protocol():
    assert default_input_normalize("held_out") == "none"
    for proto in ("multi_grid_pretrain", *_TRANSFER):
        assert default_input_normalize(proto) == "per_window"
