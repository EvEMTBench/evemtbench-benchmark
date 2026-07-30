"""Consumer-side contract conformance (Deliverable C).

Asserts the benchmark stays aligned with the vendored psp_helper windowed-dataset contract:
the vendored copy is internally consistent; the columns the benchmark requires are a subset of
the contract's required set; the fault-class vocabulary matches the contract; the synthetic
fixture's sidecars validate against the vendored JSON Schemas; and the observability oracle's
descriptor path agrees with the legacy name-regex path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evemtbench import contract as C
from evemtbench.data.schema import WINDOWS_Y_REQUIRED
from evemtbench.tasks import line_view
from evemtbench.tasks.labels.fault_class import FAULT_CLASS_CLASSES


def test_vendored_contract_pin_is_consistent():
    assert C.verify_pin() == []


def test_windows_required_is_subset_of_contract():
    contract_required = set(C.required_y_columns())
    extra = set(WINDOWS_Y_REQUIRED) - contract_required
    assert not extra, f"WINDOWS_Y_REQUIRED requires columns the contract does not: {extra}"


def test_fault_class_vocab_matches_contract():
    assert set(FAULT_CLASS_CLASSES) == C.fault_event_types()


def test_fixture_sidecars_validate_against_vendored_schemas(fixture_root):
    pytest.importorskip("jsonschema")
    from jsonschema import Draft202012Validator

    base = Path(C.__file__).resolve().parent
    man_schema = json.loads((base / "manifest.schema.json").read_text())
    meta_schema = json.loads((base / "meta.schema.json").read_text())
    seen = 0
    for fam in ("adapt_grid", "benchmark"):
        d = fixture_root / fam / "windows" / "DoubleLine"
        for x in d.glob("X_*.raw"):
            man = json.loads((d / (x.name + ".manifest.json")).read_text())
            meta = json.loads((d / (x.name + ".meta.json")).read_text())
            assert list(Draft202012Validator(man_schema).iter_errors(man)) == [], x.name
            assert list(Draft202012Validator(meta_schema).iter_errors(meta)) == [], x.name
            seen += 1
    assert seen > 0, "no fixture window sets found"


def test_oracle_regex_and_descriptor_paths_agree(fixture_root):
    from evemtbench.data.windows import load_windows

    ws = load_windows("double_line", "adapt_grid", root=fixture_root)
    assert ws.channels is not None, "fixture meta.json must carry the channels descriptor"

    line_view._CACHE.clear()
    o_regex = line_view.build_oracle("double_line", ws.feature_names)
    line_view._CACHE.clear()
    o_desc = line_view.build_oracle("double_line", ws.feature_names, ws.channels)

    assert o_regex.line_channels == o_desc.line_channels
    assert o_regex.n_line_cubicles == o_desc.n_line_cubicles
    assert o_regex._pairs == o_desc._pairs
