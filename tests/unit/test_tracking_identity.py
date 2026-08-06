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
from evemtbench.version import BENCHMARK_VERSION

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
    assert key == f"{BENCHMARK_VERSION}/{name}"


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
