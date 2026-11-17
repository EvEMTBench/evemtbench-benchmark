"""Committed label patches (``evemtbench.data.label_patches``): pin, application, loud failures.

The patch must apply only to the exact partition it was built for (by ``manifest_sha256``), only
to the functions it names, and must refuse — loudly — anything that looks like a changed dataset:
a target that already carries a phase bit, a missing episode, a different event type, a count that
differs from the pin, an edited CSV. The committed LP-1 itself is pinned against its index and the
input digests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evemtbench.data import label_patches as lp
from evemtbench.data.input_digests import load_table as load_digests

MAN = "a" * 64
COLS = ["y_phase_A", "y_phase_B", "y_phase_C"]
ET = "flt_1phg_hif_w_arc"


def _y() -> pd.DataFrame:
    """Two patch episodes (10, 11) with 3 fault-active windows each and one pre-fault window; one
    labelled 1phg episode (12); one no-event episode (13)."""
    rows = []
    for sid, et, bits in (
        (10, ET, (0, 0, 0)),
        (11, ET, (0, 0, 0)),
        (12, "flt_1phg_shc", (0, 1, 0)),
    ):
        rows.append(dict(sample_id=sid, event_type=et, fa=False, A=0, B=0, C=0))  # pre-fault
        for _ in range(3):
            rows.append(
                dict(sample_id=sid, event_type=et, fa=True, A=bits[0], B=bits[1], C=bits[2])
            )
    rows.append(dict(sample_id=13, event_type="no_event", fa=False, A=0, B=0, C=0))
    df = pd.DataFrame(rows).rename(columns={"A": "y_phase_A", "B": "y_phase_B", "C": "y_phase_C"})
    return df


def _write_patch(d: Path, *, rows=None, windows=6, manifest=MAN, functions=("fault_phase",)):
    rows = rows or [(10, "a", 1, 0, 0), (11, "c", 0, 0, 1)]
    table = pd.DataFrame(
        [
            dict(
                grid="g",
                family="adapt_grid",
                sample_id=s,
                event_type=ET,
                source_file="settings_clean.csv",
                source_key=f"general/sim_idx={s}",
                event_phase_select_1ph=sel,
                y_phase_A=a,
                y_phase_B=b,
                y_phase_C=c,
            )
            for s, sel, a, b, c in rows
        ],
        columns=list(lp.CSV_COLUMNS),
    )
    csv = d / "p.csv"
    csv.write_bytes(table.to_csv(index=False, lineterminator="\n").encode())
    index = {
        "schema_version": 1,
        "patches": [
            {
                "id": "LP-T",
                "file": "p.csv",
                "sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
                "event_types": [ET],
                "affects_columns": COLS,
                "affects_functions": list(functions),
                "partitions": [
                    {
                        "grid": "g",
                        "family": "adapt_grid",
                        "manifest_sha256": manifest,
                        "episodes": len(rows),
                        "windows_patched": windows,
                    }
                ],
            }
        ],
    }
    (d / "index.json").write_text(json.dumps(index))
    return str(d / "index.json")


def _apply(y, idx, *, manifest=MAN, function="fault_phase"):
    return lp.apply_label_patches(
        y,
        grid="g",
        family="adapt_grid",
        manifest_sha256=manifest,
        function=function,
        fault_active=y["fa"].to_numpy(),
        index_path=idx,
    )


def test_apply_sets_the_bits_on_fault_active_windows_only(tmp_path):
    y = _y()
    before = y.copy()
    out, prov = _apply(y, _write_patch(tmp_path))
    pd.testing.assert_frame_equal(y, before)  # the input frame is never mutated
    fa10 = (out.sample_id == 10) & out.fa
    fa11 = (out.sample_id == 11) & out.fa
    assert (out.loc[fa10, COLS].to_numpy() == [1, 0, 0]).all()
    assert (out.loc[fa11, COLS].to_numpy() == [0, 0, 1]).all()
    assert (out.loc[~out.fa, COLS].to_numpy() == 0).all()  # pre-fault windows untouched
    pd.testing.assert_frame_equal(out[out.sample_id >= 12], before[before.sample_id >= 12])
    assert prov == [
        {
            "id": "LP-T",
            "sha256": json.loads(Path(tmp_path / "index.json").read_text())["patches"][0]["sha256"],
            "grid": "g",
            "family": "adapt_grid",
            "manifest_sha256": MAN,
            "episodes": 2,
            "rows_patched": 6,
        }
    ]
    assert (
        lp.expected_provenance("g", "adapt_grid", MAN, "fault_phase", str(tmp_path / "index.json"))
        == prov
    )


def test_inert_on_another_manifest_or_function(tmp_path):
    y = _y()
    idx = _write_patch(tmp_path)
    out, prov = _apply(y, idx, manifest="b" * 64)
    assert prov == [] and out is y
    out, prov = _apply(y, idx, function="fault_grounded")
    assert prov == [] and out is y
    assert lp.expected_provenance("g", "adapt_grid", "b" * 64, "fault_phase", idx) == []


def test_already_labelled_target_fails(tmp_path):
    y = _y()
    y.loc[(y.sample_id == 10) & y.fa, "y_phase_B"] = 1
    with pytest.raises(lp.LabelPatchError, match="already labelled"):
        _apply(y, _write_patch(tmp_path))


def test_count_mismatch_fails(tmp_path):
    with pytest.raises(lp.LabelPatchError, match="index pins 7"):
        _apply(_y(), _write_patch(tmp_path, windows=7))


def test_missing_episode_fails(tmp_path):
    y = _y()
    y = y[y.sample_id != 11].reset_index(drop=True)
    with pytest.raises(lp.LabelPatchError, match="no fault-active window"):
        _apply(y, _write_patch(tmp_path))


def test_other_event_type_fails(tmp_path):
    y = _y()
    y.loc[y.sample_id == 10, "event_type"] = "flt_1phg_hif"
    with pytest.raises(lp.LabelPatchError, match="event types"):
        _apply(y, _write_patch(tmp_path))


def test_incomplete_patch_fails(tmp_path):
    """A fault-active window of the patched type left without a bit is refused (the patch was
    built for a partition with more episodes of the type than it lists)."""
    y = _y()
    extra = y[y.sample_id == 11].assign(sample_id=99)
    y = pd.concat([y, extra], ignore_index=True)
    with pytest.raises(lp.LabelPatchError, match="still carry no phase bit"):
        _apply(y, _write_patch(tmp_path))


def test_edited_csv_fails_its_sha_pin(tmp_path):
    idx = _write_patch(tmp_path)
    csv = tmp_path / "p.csv"
    csv.write_text(csv.read_text().replace(",a,1,0,0", ",b,0,1,0"))
    with pytest.raises(lp.LabelPatchError, match="sha256"):
        _apply(_y(), idx)


# ------------------------------------------------------------------- the committed patch (LP-1)


def test_committed_lp1_matches_its_pins():
    index = lp.load_index()
    (entry,) = [e for e in index["patches"] if e["id"] == "LP-1"]
    table = lp.load_patch_table("LP-1")  # raises on a sha mismatch
    assert entry["affects_functions"] == ["fault_phase"]
    assert entry["affects_columns"] == COLS
    assert entry["event_types"] == [ET]
    assert entry["from_results_version"] == "1.3.0"
    parts = {(p["grid"], p["family"]): p for p in entry["partitions"]}
    assert {k: (v["episodes"], v["windows_patched"]) for k, v in parts.items()} == {
        ("cigre_mv", "adapt_grid"): (456, 7752),
        ("cigre_mv", "benchmark"): (87, 1392),
        ("template_20kv", "multi_grid"): (254, 4318),
    }
    digests = load_digests()["entries"]
    for (grid, family), p in parts.items():
        assert p["manifest_sha256"] == digests[f"{grid}/{family}"]["manifest_sha256"]
        assert (table[(table.grid == grid) & (table.family == family)].shape[0]) == p["episodes"]
    assert len(table) == 797
    assert (table.event_type == ET).all()
    assert (table[COLS].sum(axis=1) == 1).all()  # exactly one phase per single-phase fault
    sel_bits = table.event_phase_select_1ph.map({"a": "100", "b": "010", "c": "001"})
    assert (sel_bits == table[COLS].astype(str).agg("".join, axis=1)).all()
    key = ["grid", "family", "sample_id"]
    assert table[key].equals(table[key].sort_values(key).reset_index(drop=True))
    raw = (lp.PATCH_DIR / entry["file"]).read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")


def test_only_fault_phase_and_the_1_3_0_cells_switch_patches_on():
    from evemtbench.version import label_patches_enabled, results_version

    v = results_version("held_out", "fault_phase_line", grid="cigre_mv")
    assert label_patches_enabled("fault_phase_line", v)
    v = results_version("held_out", "fault_phase_line", grid="double_line")
    assert not label_patches_enabled("fault_phase_line", v)
    assert not label_patches_enabled("fault_grounded_line", "1.3.0")


# ------------------------------------------------------------------- provenance consumers


def _record(version, task_parts, patches):
    return {
        "benchmark_version": version,
        "data": {"partitions": task_parts, "label_patches": patches},
    }


def test_integrity_requires_exactly_the_expected_patches():
    from evemtbench.coverage import Cell
    from evemtbench.integrity import _check_label_patches

    entry = lp.get_patch("LP-1")
    part = next(p for p in entry["partitions"] if p["family"] == "benchmark")
    parts = [
        {"grid": "cigre_mv", "family": "benchmark", "manifest_sha256": part["manifest_sha256"]}
    ]
    want = lp.expected_provenance("cigre_mv", "benchmark", part["manifest_sha256"], "fault_phase")
    assert len(want) == 1
    cell = Cell("fault_phase_line", "held_out", "cigre_mv", "cnn")
    assert _check_label_patches(cell, _record("1.3.0", parts, want), "p") == []
    assert _check_label_patches(cell, _record("1.3.0", parts, []), "p")  # missing on a 1.3.0 run
    assert _check_label_patches(cell, _record("1.0.0", parts, want), "p")  # present on a 1.0.0 run
    assert _check_label_patches(cell, _record("1.0.0", parts, None), "p") == []
    tampered = [{**want[0], "sha256": "0" * 64}]
    assert _check_label_patches(cell, _record("1.3.0", parts, tampered), "p")
    other = Cell("fault_grounded_line", "held_out", "cigre_mv", "cnn")
    assert _check_label_patches(other, _record("1.0.0", parts, None), "p") == []


def test_reevaluate_compares_label_patch_provenance():
    from evemtbench.evaluation.reevaluate import _provenance_matches

    p = {"id": "LP-1", "sha256": "s", "grid": "g", "family": "f", "manifest_sha256": "m",
         "rows_patched": 3}  # fmt: skip
    base = {"splits": {"files": {"a": "1"}}, "partitions": []}
    assert _provenance_matches(base, base) is None  # pre-1.3.0 record, patches off
    assert (
        _provenance_matches({**base, "label_patches": [p]}, {**base, "label_patches": [p]}) is None
    )
    assert "label patches differ" in _provenance_matches({**base, "label_patches": [p]}, base)
    assert "label patches differ" in _provenance_matches(
        base, {**base, "label_patches": [{**p, "rows_patched": 4}]}
    )


def test_fault_active_mask_is_boolean_safe(tmp_path):
    y = _y()
    out, prov = lp.apply_label_patches(
        y,
        grid="g",
        family="adapt_grid",
        manifest_sha256=MAN,
        function="fault_phase",
        fault_active=y["fa"].astype(np.int64).to_numpy(),
        index_path=_write_patch(tmp_path),
    )
    assert prov[0]["rows_patched"] == 6
