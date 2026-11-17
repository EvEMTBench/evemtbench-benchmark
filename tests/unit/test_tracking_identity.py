"""W&B run identity must be one-to-one with a result directory.

``run_id_for`` is a pure hash of ``cell_key`` and ``start_run`` uses ``resume="allow"``, so two
cells sharing a key do not merely look alike in the UI — the second run resumes into the first,
appending to one history and overwriting its config. The transfer protocols run the same
(task, protocol, grid, baseline) from three pre-training sources into three distinct output
directories, so the source must be part of the key.

These tests call the runner's own ``cell_identity`` rather than restating its formula, so the
assertions cannot drift away from the code that builds the real ids.
"""

from __future__ import annotations

import pytest

from evemtbench.baselines import _tracking
from evemtbench.baselines.runner import cell_identity
from evemtbench.coverage import (
    PRETRAIN_GRID,
    PRETRAIN_GRIDS,
    TRANSFER_PROTOCOLS,
    Cell,
    cell_dir,
)
from evemtbench.version import TRANSFER_RESULTS_VERSION, results_version

TASK, GRID, BASELINE = "fault_category_line", "cigre_mv", "cnn"
PROTOCOL = TRANSFER_PROTOCOLS[0]


def _key(source):
    return cell_identity(TASK, PROTOCOL, GRID, BASELINE, source)[1]


def test_run_ids_are_distinct_across_pretrain_sources():
    ids = {src: _tracking.run_id_for(_key(src)) for src in PRETRAIN_GRIDS}
    assert len(set(ids.values())) == len(PRETRAIN_GRIDS), (
        f"colliding run ids across pre-training sources: {ids}"
    )


def test_default_source_key_is_unchanged():
    """Default source keeps the legacy source-less key, so already-logged runs keep their ids."""
    name, key = cell_identity(TASK, PROTOCOL, GRID, BASELINE, PRETRAIN_GRID)
    assert name == f"{TASK}/{PROTOCOL}/{GRID}/{BASELINE}"
    # the key lives in the cell's CURRENT results tree; for this (non fault-location) transfer cell
    # that is still the 1.1.0 tree after the v1.2.0 revision, which is what keeps the logged ids
    assert results_version(PROTOCOL, TASK) == TRANSFER_RESULTS_VERSION == "1.1.0"
    assert key == f"1.1.0/{name}"


def test_fault_location_transfer_cell_gets_a_new_key():
    """A fault-location transfer cell is re-run under the v1.2.0 gate into a new tree, so its
    tracking key (and run id) must differ from the superseded 1.1.0 run of the same cell."""
    name, key = cell_identity("fault_location_line", PROTOCOL, GRID, BASELINE, PRETRAIN_GRID)
    assert key == f"1.2.0/{name}"
    assert _tracking.run_id_for(key) != _tracking.run_id_for(f"1.1.0/{name}")


@pytest.mark.parametrize("source", [s for s in PRETRAIN_GRIDS if s != PRETRAIN_GRID])
def test_identity_matches_the_output_directory(source, tmp_path):
    """The online key and the offline directory must be qualified the same way — that 1:1 mapping is
    the whole point of recording run_id in each cell's run_record.json."""
    name, _ = cell_identity(TASK, PROTOCOL, GRID, BASELINE, source)
    d = cell_dir(tmp_path, Cell(TASK, PROTOCOL, GRID, BASELINE, source=source))
    assert name.endswith(f"/pretrain-{source}")
    assert d.name == f"pretrain-{source}"


def test_non_transfer_protocols_ignore_pretrain_grid():
    """A held_out cell has no source dimension; passing one must not change its identity."""
    a = cell_identity(TASK, "held_out", GRID, BASELINE, None)
    b = cell_identity(TASK, "held_out", GRID, BASELINE, "template_345kv")
    assert a == b


def test_normalization_control_gets_its_own_run_id():
    """The normalization control re-runs a held_out cell with per-window normalization into a
    separate output root. Sharing the main cell's key made that run resume into the main cell's
    W&B run and overwrite its summary, so a non-default normalization must qualify the key."""
    held_out = "held_out"
    _, main_key = cell_identity("fault_location_line", held_out, GRID, BASELINE, None)
    _, default_key = cell_identity(
        "fault_location_line", held_out, GRID, BASELINE, None, normalize="none"
    )
    name, control_key = cell_identity(
        "fault_location_line", held_out, GRID, BASELINE, None, normalize="per_window"
    )
    assert default_key == main_key
    assert name.endswith("/norm-per_window")
    assert _tracking.run_id_for(control_key) != _tracking.run_id_for(main_key)


def test_transfer_default_normalization_keeps_the_key():
    """per_window is the transfer default, so passing it must not change already-logged ids."""
    assert cell_identity(TASK, PROTOCOL, GRID, BASELINE, PRETRAIN_GRID, normalize="per_window")[
        1
    ] == _key(PRETRAIN_GRID)
