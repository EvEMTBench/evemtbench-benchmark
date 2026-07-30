"""Soft cross-repo contract drift check (offline-safe).

Compares the vendored ``src/evemtbench/contract`` against the sibling producer
(``power-grid-and-ai-helper-functions/contract``, the canonical owner). It **skips cleanly** when
the sibling is not present (FAU GitLab has no Pages; CI may be offline) — never a hard network or
path dependency — mirroring ``hpc/check_results_fresh.py``'s skip-if-absent idiom. Advisory: wire
into CI/pre-commit to catch a stale vendored copy, but it must not block offline work.

    python hpc/check_contract_sync.py     # 0 = in sync or skipped; 1 = drift detected
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_FILES = ("contract.json", "manifest.schema.json", "meta.schema.json", "CONTRACT_VERSION")


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def _vendored() -> Path:
    return _repo() / "src" / "evemtbench" / "contract"


def _sibling() -> Path | None:
    cand = _repo().parent / "power-grid-and-ai-helper-functions" / "contract"
    return cand if (cand / "contract.json").is_file() else None


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    sib = _sibling()
    if sib is None:
        print("check_contract_sync: sibling producer repo not found — skipping (offline OK).")
        return 0

    vend = _vendored()
    problems: list[str] = []
    for name in _FILES:
        upstream = sib / name
        if not upstream.is_file():
            continue
        local = vend / name
        if not local.is_file():
            problems.append(f"vendored {name} is missing")
        elif _sha(local) != _sha(upstream):
            problems.append(f"{name} differs from the upstream producer copy")

    if problems:
        print("check_contract_sync: DRIFT vs upstream producer contract:")
        for p in problems:
            print(f"  - {p}")
        print(
            "  fix: re-vendor with\n"
            "       cp ../power-grid-and-ai-helper-functions/contract/"
            "{contract.json,manifest.schema.json,meta.schema.json,CONTRACT_VERSION} "
            "src/evemtbench/contract/\n"
            "     then update CONTRACT_SHA256 in src/evemtbench/contract/__init__.py."
        )
        return 1

    print("check_contract_sync: vendored contract matches the upstream producer. OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
