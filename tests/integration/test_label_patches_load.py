"""``load_task`` applies a label patch end to end (fixture), and only when switched on.

A copy of the synthetic fixture has the phase bits of one single-phase fault episode zeroed —
the shape of the 1.x ``flt_1phg_hif_w_arc`` defect — and a patch keyed to that partition's
manifest restores them. Checks: patched load == the original labels (and records its provenance),
unpatched load drops the episode, the default resolution leaves a 1.0.0 cell unpatched, and the
same patch on the UNmodified fixture (same manifest, bits present) fails loudly."""

from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from evemtbench.data import label_patches as lp
from evemtbench.data.windows import load_windows
from evemtbench.tasks.labels.event_present import derive_fault_active_series
from evemtbench.tasks.task import load_task

TASK = "fault_phase_line"
COLS = ["y_phase_A", "y_phase_B", "y_phase_C"]


@pytest.fixture
def patched_setup(fixture_root, tmp_path, monkeypatch):
    root = tmp_path / "data"
    shutil.copytree(fixture_root, root)
    ydir = root / "adapt_grid" / "windows" / "DoubleLine"
    ypath = next(ydir.glob("y_*.parquet"))
    y = pd.read_parquet(ypath)
    fa = derive_fault_active_series(y).astype(bool)
    single = (
        fa
        & (y[COLS].fillna(0).astype(int).sum(axis=1) == 1)
        & y.event_type.str.startswith("flt_1phg")
    )
    sid = int(y.loc[single, "sample_id"].iloc[0])
    et = str(y.loc[single & (y.sample_id == sid), "event_type"].iloc[0])
    target = fa & (y.sample_id == sid)
    bits = y.loc[target, COLS].iloc[0].astype(int).tolist()
    y.loc[target, COLS] = 0
    y.to_parquet(ypath)

    ws = load_windows("double_line", "adapt_grid", root=root)
    manifest = ws.provenance["manifest_sha256"]
    pdir = tmp_path / "patches"
    pdir.mkdir()
    sel = "abc"[bits.index(1)]
    table = pd.DataFrame(
        [["double_line", "adapt_grid", sid, et, "fixture", f"general/sim_idx={sid}", sel, *bits]],
        columns=list(lp.CSV_COLUMNS),
    )
    (pdir / "t.csv").write_bytes(table.to_csv(index=False, lineterminator="\n").encode())
    index = {
        "schema_version": 1,
        "patches": [
            {
                "id": "LP-FIX",
                "file": "t.csv",
                "sha256": hashlib.sha256((pdir / "t.csv").read_bytes()).hexdigest(),
                "event_types": [et],
                "affects_columns": COLS,
                "affects_functions": ["fault_phase"],
                "partitions": [
                    {
                        "grid": "double_line",
                        "family": "adapt_grid",
                        "manifest_sha256": manifest,
                        "episodes": 1,
                        "windows_patched": int(target.sum()),
                    }
                ],
            }
        ],
    }
    (pdir / "index.json").write_text(json.dumps(index))
    monkeypatch.setattr(lp, "INDEX_PATH", pdir / "index.json")
    monkeypatch.setattr(lp, "PATCH_DIR", pdir)
    lp.load_index.cache_clear()
    lp.load_patch_table.cache_clear()
    yield root, sid, int(target.sum())
    lp.load_index.cache_clear()
    lp.load_patch_table.cache_clear()


def _all(task, names=("train", "val", "test", "benchmark")):
    views = [task.view(n) for n in names]
    labels = np.concatenate([v.labels for v in views])
    sids = np.concatenate([v.sample_ids for v in views])
    return labels, sids


def test_patch_restores_the_original_labels(patched_setup, fixture_root, held_out_splits):
    root, sid, n = patched_setup
    kw = dict(grid="double_line", splits_root=held_out_splits)
    orig = load_task(TASK, "held_out", root=fixture_root, label_patches=False, **kw)
    on = load_task(TASK, "held_out", root=root, label_patches=True, **kw)
    off = load_task(TASK, "held_out", root=root, label_patches=False, **kw)
    default = load_task(TASK, "held_out", root=root, **kw)  # held_out double_line = 1.0.0: off

    lo, so = _all(orig)
    lp_on, s_on = _all(on)
    np.testing.assert_array_equal(lo, lp_on)
    np.testing.assert_array_equal(so, s_on)
    (rec,) = on.provenance["label_patches"]
    assert rec["id"] == "LP-FIX" and rec["rows_patched"] == n and rec["family"] == "adapt_grid"

    # the patched episode is an adapt_grid one (train/val/test); sample ids repeat in benchmark
    adapt = ("train", "val", "test")
    l_off, s_off = _all(off, adapt)
    _, s_on_adapt = _all(on, adapt)
    assert sid in set(s_on_adapt.tolist()) and sid not in set(s_off.tolist())
    assert len(l_off) == len(s_on_adapt) - int((s_on_adapt == sid).sum())
    assert "label_patches" not in off.provenance
    assert "label_patches" not in default.provenance
    assert len(_all(default, adapt)[0]) == len(l_off)
    assert default.results_version == "1.0.0"


def test_other_tasks_are_never_patched(patched_setup, held_out_splits):
    root, _sid, _n = patched_setup
    t = load_task(
        "fault_grounded_line",
        "held_out",
        grid="double_line",
        root=root,
        splits_root=held_out_splits,
        label_patches=True,
    )
    assert "label_patches" not in t.provenance


def test_patch_on_an_already_labelled_partition_fails(patched_setup, fixture_root, held_out_splits):
    """Same manifest bytes, but the targets already carry their bits: refuse, never double-apply."""
    with pytest.raises(lp.LabelPatchError, match="already labelled"):
        load_task(
            TASK,
            "held_out",
            grid="double_line",
            root=fixture_root,
            splits_root=held_out_splits,
            label_patches=True,
        )


def test_task_results_version_uses_grid_and_source(fixture_root, held_out_splits):
    t = load_task(
        "fault_detection_line",
        "held_out",
        grid="double_line",
        root=fixture_root,
        splits_root=held_out_splits,
    )
    assert t.results_version == "1.0.0" and t.pretrain_grid is None
