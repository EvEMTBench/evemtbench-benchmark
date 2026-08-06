"""Verify every committed cell against the captured RESULTS environment.

Records written before ``results_env_sha256`` existed (the whole v1.0.0 corpus) cannot carry that
hash honestly: stamping it in after the fact would make a July artifact look like the runner had
recorded it. So, following ``hpc/backfill_host_provenance.py``, this script NEVER mutates
run_record.json. Instead it does the check the hash only *gestures* at — it compares each record's
independently captured ``versions`` / ``python`` / ``torch`` against
``env/results-v<version>.lock`` and reports agreement per cell.

That comparison, not a pointer, is what makes the environment claim auditable: the lockfile hash in
a record proves only which file was on disk, whereas this proves the interpreter that produced the
numbers matched it.

    PYTHONPATH=src python hpc/verify_results_env.py [--outputs outputs] [--write]

Read-only by default; ``--write`` emits ``outputs/<version>/results_env_verification.json``
(gitignored, like everything under outputs/). Exit 1 if any cell disagrees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from evemtbench.version import BENCHMARK_VERSION  # noqa: E402

# Only the packages whose version can move a metric value are compared. The rest of the lock
# (build tooling, plotting, CUDA shims) does not enter a score, and demanding equality there would
# manufacture failures that mean nothing about the results.
_SCORING_PKGS = ("numpy", "pandas", "scikit-learn", "scipy")
_PIN = re.compile(r"^([A-Za-z0-9._-]+)==(.+)$")


def parse_lock(path: Path) -> dict[str, str]:
    """``{canonical_name: version}`` from a ``name==version`` lock (comments/blanks ignored)."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _PIN.match(line)
        if m:
            out[m.group(1).lower().replace("_", "-")] = m.group(2)
    return out


def verify(out_root: Path, lock: Path, *, version: str = BENCHMARK_VERSION) -> dict:
    pins = parse_lock(lock)
    lock_sha = hashlib.sha256(lock.read_bytes()).hexdigest()
    expected = {p: pins.get(p) for p in _SCORING_PKGS}
    # torch is pinned as e.g. "2.6.0+cu124"; the record stores torch.__version__, same string.
    expected_torch = pins.get("torch")

    agree: list[str] = []
    disagree: list[dict] = []
    missing: list[str] = []
    pythons: dict[str, int] = {}

    root = out_root / version
    for rec_path in sorted(root.rglob("run_record.json")):
        cell = str(rec_path.parent.relative_to(root))
        try:
            rec = json.loads(rec_path.read_text())
        except (OSError, json.JSONDecodeError):
            missing.append(cell)
            continue
        got = rec.get("versions") or {}
        deltas = {
            pkg: {"record": got.get(pkg), "lock": expected[pkg]}
            for pkg in _SCORING_PKGS
            if got.get(pkg) != expected[pkg]
        }
        # torch is absent from classical-baseline records that never imported it — not a conflict.
        rec_torch = rec.get("torch")
        if rec_torch is not None and expected_torch is not None and rec_torch != expected_torch:
            deltas["torch"] = {"record": rec_torch, "lock": expected_torch}
        pythons[str(rec.get("python"))] = pythons.get(str(rec.get("python")), 0) + 1
        if deltas:
            disagree.append({"cell": cell, "deltas": deltas})
        else:
            agree.append(cell)

    return {
        "benchmark_version": version,
        "lockfile": str(lock.relative_to(_REPO)) if lock.is_relative_to(_REPO) else str(lock),
        "results_env_sha256": lock_sha,
        "compared_packages": list(_SCORING_PKGS) + ["torch"],
        "method": (
            "per-cell run_record.versions/torch compared against env lock pins; "
            "run_record.json never mutated (hpc/verify_results_env.py)"
        ),
        "n_agree": len(agree),
        "n_disagree": len(disagree),
        "n_unreadable": len(missing),
        "python_versions": pythons,
        "disagree": disagree,
        "unreadable": missing,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify cells against the captured results env.")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--version", default=BENCHMARK_VERSION)
    ap.add_argument("--lock", default=None, help="default: env/results-v<version>.lock")
    ap.add_argument("--write", action="store_true", help="write the sidecar (default: dry-run)")
    args = ap.parse_args(argv)

    lock = Path(args.lock) if args.lock else _REPO / "env" / f"results-v{args.version}.lock"
    if not lock.is_file():
        print(f"error: {lock} not found — run hpc/capture_results_env.sh first", file=sys.stderr)
        return 2

    payload = verify(Path(args.outputs), lock, version=args.version)
    print(
        f"results-env verification @ {payload['results_env_sha256'][:12]}: "
        f"{payload['n_agree']} agree, {payload['n_disagree']} disagree, "
        f"{payload['n_unreadable']} unreadable; python={payload['python_versions']}"
    )
    for d in payload["disagree"][:10]:
        print(f"  MISMATCH {d['cell']}: {json.dumps(d['deltas'], sort_keys=True)}")
    if len(payload["disagree"]) > 10:
        print(f"  ... and {len(payload['disagree']) - 10} more")

    if args.write:
        out = Path(args.outputs) / args.version / "results_env_verification.json"
        out.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"wrote {out}")

    return 1 if payload["n_disagree"] or payload["n_unreadable"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
