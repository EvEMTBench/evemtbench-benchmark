"""Vendored copy of the psp_helper windowed-dataset contract (the producer owns the canonical
version in ``power-grid-and-ai-helper-functions/contract``).

The benchmark takes **no import dependency** on ``psp_helper``; it vendors the machine-readable
artifacts (``contract.json`` + JSON Schemas) and pins the version + checksum. A unit test
asserts the vendored copy is internally consistent (offline), and a soft sibling-diff check
(``hpc/check_contract_sync.py``) catches drift from the upstream producer when reachable.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent

#: The contract version this benchmark build is written against. A **major** bump here forces a
#: BENCHMARK_VERSION major bump (see ``evemtbench.version``).
EXPECTED_CONTRACT_VERSION = (_HERE / "CONTRACT_VERSION").read_text(encoding="utf-8").strip()

#: sha256 of the vendored ``contract.json`` — pins the exact vendored bytes so an accidental or
#: unreviewed edit fails the consistency test. Regenerate with::
#:     sha256sum src/evemtbench/contract/contract.json
CONTRACT_SHA256 = "76a8f6eb814ca3db0bcb40faf6e2216228645ca04228b21aa545394dab891d6a"


@lru_cache(maxsize=1)
def load_contract() -> dict[str, Any]:
    """Parsed vendored ``contract.json``."""
    return json.loads((_HERE / "contract.json").read_text(encoding="utf-8"))


def contract_json_sha256() -> str:
    """Actual sha256 of the vendored ``contract.json`` on disk."""
    return hashlib.sha256((_HERE / "contract.json").read_bytes()).hexdigest()


def fault_event_types() -> set[str]:
    """The Core event vocabulary's fault ``event_type`` values (``flt_*``)."""
    return set(load_contract()["event_vocab"]["event_type_fault"])


def required_y_columns() -> list[str]:
    """Required ``y.parquet`` label-primitive column names from the contract."""
    return [c["name"] for c in load_contract()["y_columns"]["required"]]


def verify_pin() -> list[str]:
    """Offline internal-consistency check of the vendored contract. Returns problems (empty=OK)."""
    problems: list[str] = []
    contract = load_contract()
    if contract.get("contract_version") != EXPECTED_CONTRACT_VERSION:
        problems.append(
            f"contract.json version {contract.get('contract_version')!r} != "
            f"CONTRACT_VERSION {EXPECTED_CONTRACT_VERSION!r}"
        )
    actual = contract_json_sha256()
    if actual != CONTRACT_SHA256:
        problems.append(
            f"vendored contract.json sha256 {actual} != pinned {CONTRACT_SHA256} "
            "(update CONTRACT_SHA256 after an intentional re-vendor)"
        )
    return problems
