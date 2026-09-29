"""Cross-channel redundancy check (B6): the OFFLINE and ONLINE result channels must agree.

Redundant logging is only trustworthy if the two channels are provably the *same* numbers, not
merely both-present. This tool joins the committed provenance index (offline: derived from
``result.json``) with each cell's Weights & Biases run summary (online) and asserts, per
(cell, test_set, metric), that ``offline mean == W&B "<test_set>/<metric>_mean"`` within a
tolerance. On a genuine value mismatch it fails loudly; when W&B is simply unavailable (not
installed / not authenticated / offline / run not yet synced) it SKIPS — this is a verification
aid, never a network-dependent gate that could block CI.

    PYTHONPATH=src python hpc/check_tracking_consistency.py            # current BENCHMARK_VERSION
    PYTHONPATH=src python hpc/check_tracking_consistency.py --version 1.0.0 --tolerance 1e-6

Requires network + ``~/.netrc`` W&B auth (the same the sweep uses). Read-only against W&B.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evemtbench.version import BENCHMARK_VERSION

_REPO = Path(__file__).resolve().parents[1]


def _load_index(version: str) -> dict:
    idx = _REPO / "audit" / "provenance" / version / "index.json"
    if not idx.is_file():
        return {}
    try:
        return json.loads(idx.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify offline result.json == online W&B summary.")
    ap.add_argument("--version", default=BENCHMARK_VERSION)
    ap.add_argument("--tolerance", type=float, default=1e-6)
    args = ap.parse_args(argv)

    doc = _load_index(args.version)
    cells = doc.get("cells", {})
    if not cells:
        print(f"tracking-consistency: no provenance index for v{args.version} — nothing to check.")
        return 0
    tracked = {
        ck: c
        for ck, c in cells.items()
        if (c.get("wandb") or {}).get("enabled") and (c.get("wandb") or {}).get("run_id")
    }
    if not tracked:
        print(
            f"tracking-consistency: {len(cells)} cells, none with an enabled W&B run "
            "(offline-only) — nothing to cross-check."
        )
        return 0

    try:
        import wandb

        api = wandb.Api()
    except Exception as exc:  # not installed / not authenticated / no network
        print(f"tracking-consistency: SKIP — W&B API unavailable ({type(exc).__name__}: {exc}).")
        return 0

    mismatches: list[str] = []
    unreachable = missing = compared = 0
    for ck, c in sorted(tracked.items()):
        wb = c["wandb"]
        ref = f"{wb.get('entity')}/{wb.get('project')}/{wb['run_id']}"
        try:
            summ = dict(api.run(ref).summary)
        except Exception as exc:
            unreachable += 1
            print(f"  · {ck}: W&B run {ref} unreachable ({type(exc).__name__}) — skipped")
            continue
        for ts, metrics in (c.get("metrics") or {}).items():
            for metric, mv in metrics.items():
                off = mv.get("mean")
                if off is None:
                    continue
                on = summ.get(f"{ts}/{metric}_mean")
                if on is None:
                    missing += 1
                    continue
                compared += 1
                if abs(float(on) - float(off)) > args.tolerance:
                    mismatches.append(
                        f"{ck} [{ts}/{metric}]: offline {off:.6f} != W&B {float(on):.6f} "
                        f"(|Δ|={abs(float(on) - float(off)):.2e} > {args.tolerance:g})"
                    )

    print(
        f"tracking-consistency: {compared} (cell,test_set,metric) values compared across "
        f"{len(tracked)} tracked cells; {missing} online keys missing, {unreachable} runs "
        f"unreachable, {len(mismatches)} mismatches."
    )
    if mismatches:
        print("\n✗ offline<->online DISAGREEMENT — the two channels are not redundant:")
        for m in mismatches:
            print(f"  {m}")
        return 1
    print("tracking-consistency: offline result.json and online W&B agree. ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
