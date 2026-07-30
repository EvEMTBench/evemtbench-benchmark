"""The input-digest pins (evemtbench.data.input_digests): round-trip on the synthetic fixtures,
mismatch detection, and the integrity cross-check that catches a dataset swapped under a fixed
version (same-instance/shape-gated, so fixture runs never false-positive against real pins)."""

import json
from pathlib import Path

from evemtbench import integrity
from evemtbench.coverage import Cell
from evemtbench.data import input_digests

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic"


def test_build_and_verify_roundtrip_on_fixtures(monkeypatch):
    table = input_digests.build_table(FIXTURE_ROOT)
    assert "double_line/adapt_grid" in table["entries"]
    entry = table["entries"]["double_line/adapt_grid"]
    assert len(entry["manifest_sha256"]) == 64 and entry["shape"]

    monkeypatch.setattr(input_digests, "load_table", lambda: table)
    assert input_digests.verify(FIXTURE_ROOT) == []

    # corrupt one committed digest -> named mismatch
    entry["manifest_sha256"] = "0" * 64
    problems = input_digests.verify(FIXTURE_ROOT)
    assert any("manifest_sha256 mismatch" in p for p in problems)


def test_integrity_flags_swapped_input_same_shape(tmp_path):
    """A run record whose loaded partition matches a pin's (grid, family, shape) but not its
    digests must be flagged; a different shape (another dataset instance, e.g. fixtures) is
    skipped."""
    cell = Cell("fault_detection_local", "held_out", "double_line", "majority")
    record = {
        "git_sha": "a" * 40,
        "lockfile_sha256": "b" * 64,
        "seeds": [0],
        "data": {
            "partitions": [
                {
                    "grid": "double_line",
                    "family": "adapt_grid",
                    "shape": {"n": 100, "l": 480, "f": 66},
                    "manifest_sha256": "a" * 64,
                    "feature_hash": "b" * 64,
                }
            ]
        },
        "checkpoints": {},
    }
    (tmp_path / "run_record.json").write_text(json.dumps(record))
    pin_same_shape = {
        "entries": {
            "double_line/adapt_grid": {
                "shape": {"n": 100, "l": 480, "f": 66},
                "manifest_sha256": "c" * 64,  # different bytes, same instance shape
                "feature_hash": "b" * 64,
            }
        }
    }
    problems = integrity.check_cell(cell, tmp_path, expected_seeds=None, digests=pin_same_shape)
    assert any("does not match the committed digest pin" in p for p in problems)

    pin_other_instance = {
        "entries": {
            "double_line/adapt_grid": {
                "shape": {"n": 25000, "l": 480, "f": 66},  # the real corpus, not this instance
                "manifest_sha256": "c" * 64,
                "feature_hash": "d" * 64,
            }
        }
    }
    problems = integrity.check_cell(cell, tmp_path, expected_seeds=None, digests=pin_other_instance)
    assert not any("digest pin" in p for p in problems)
