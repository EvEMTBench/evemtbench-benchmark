"""Best-effort hardware backfill for cells that ran before host provenance existed.

Runs recorded before run_record.json carried a ``host`` block have their hardware identity only
in the SLURM ``.out`` logs (``sweep job <id> on <host> | start <date> ...``, the nvidia-smi
header, and one ``===== HH:MM:SS  --task … =====`` marker per cell). This script parses those
logs into a **sidecar** — it NEVER mutates run_record.json — mapping each cell to (job id,
hostname, GPU model, log file), disambiguating retries by proximity of the cell marker to the
run record's own timestamp. Every entry is labelled ``"source": "best-effort log parse"`` so the
audit can display it honestly, distinct from artifact-backed provenance.

    PYTHONPATH=src python hpc/backfill_host_provenance.py \\
        [--logs hpc/hpc_logs] [--outputs outputs] [--write]

Read-only by default (prints a summary); ``--write`` emits
``outputs/<version>/provenance_backfill.json`` (gitignored, like everything under outputs/).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from evemtbench.version import BENCHMARK_VERSION  # noqa: E402

_SWEEP = re.compile(r"sweep job (\d+) on (\S+) \| start (.+?) \| GPU")
_PROLOGUE = re.compile(r"Starting TaskPrologue of job (\d+) on (\S+) at (.+)$")
_GPU = re.compile(r"\|\s+\d+\s+(NVIDIA \S+(?: \S+)?)\s+(?:On|Off)")
_MARKER = re.compile(
    r"^===== (\d{2}:\d{2}:\d{2})\s+--task (\S+) --protocol (\S+) --grid (\S+) --baseline (\S+)"
)


def _parse_start(s: str) -> float | None:
    """Epoch of a log-header date (local cluster time); None if unparseable."""
    for fmt in ("%a %b %d %I:%M:%S %p %Z %Y", "%a %b %d %H:%M:%S %Z %Y", "%a %b %d %H:%M:%S %Y"):
        try:
            return time.mktime(time.strptime(" ".join(s.split()), fmt))
        except ValueError:
            continue
    return None


@dataclass
class Candidate:
    job_id: str
    hostname: str
    gpu: str | None
    log_file: str
    marker_epoch: float | None  # absolute time of the cell's ===== marker, if derivable


def _scan_log(path: Path) -> tuple[dict, list[tuple[str, str]]]:
    """(header {job_id, hostname, gpu, start_epoch}, [(cell_key, marker HH:MM:SS)])."""
    header: dict = {}
    markers: list[tuple[str, str]] = []
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return header, markers
    m = _SWEEP.search(text) or _PROLOGUE.search(text)
    if m:
        header = {
            "job_id": m.group(1),
            "hostname": m.group(2),
            "start_epoch": _parse_start(m.group(3)),
        }
    g = _GPU.search(text)
    header["gpu"] = g.group(1) if g else None
    for line in text.splitlines():
        mm = _MARKER.match(line)
        if mm:
            hms, task, protocol, grid, baseline = mm.groups()
            markers.append((f"{task}/{protocol}/{grid}/{baseline}", hms))
    return header, markers


def _marker_epoch(start_epoch: float | None, start_hms: str | None, hms: str) -> float | None:
    """Absolute epoch of a marker: job-start date + marker clock, rolling over midnight."""
    if start_epoch is None:
        return None
    day = start_epoch - (start_epoch % 86400)
    h, m, s = (int(x) for x in hms.split(":"))
    t = day + h * 3600 + m * 60 + s
    while t < start_epoch - 60:  # marker clock wrapped past midnight relative to job start
        t += 86400
    return t


def build_backfill(logs_dir: Path, out_root: Path, *, version: str = BENCHMARK_VERSION) -> dict:
    candidates: dict[str, list[Candidate]] = {}
    for log in sorted(logs_dir.glob("*.out")):
        header, markers = _scan_log(log)
        if not header.get("job_id"):
            continue
        for cell_key, hms in markers:
            candidates.setdefault(cell_key, []).append(
                Candidate(
                    job_id=header["job_id"],
                    hostname=header["hostname"],
                    gpu=header.get("gpu"),
                    log_file=log.name,
                    marker_epoch=_marker_epoch(header.get("start_epoch"), None, hms),
                )
            )

    entries: dict[str, dict] = {}
    unmatched: list[str] = []
    for cell_key, cands in sorted(candidates.items()):
        record_ts = None
        rec_path = out_root / version / Path(cell_key) / "run_record.json"
        try:
            record_ts = json.loads(rec_path.read_text()).get("timestamp")
        except (OSError, json.JSONDecodeError):
            pass
        if record_ts is None:
            unmatched.append(cell_key)  # cell seen in logs but no record on disk (or unreadable)
            continue
        # retries/chain links: pick the run whose marker precedes the record's own write time
        # and lies closest to it; fall back to the last candidate when times are unusable.
        timed = [c for c in cands if c.marker_epoch is not None and c.marker_epoch <= record_ts]
        best = max(timed, key=lambda c: c.marker_epoch) if timed else cands[-1]
        entries[cell_key] = {
            "slurm_job_id": best.job_id,
            "hostname": best.hostname,
            "gpu": best.gpu,
            "log_file": best.log_file,
            "n_log_occurrences": len(cands),
            "source": "best-effort log parse",
        }
    return {
        "benchmark_version": version,
        "source": "best-effort log parse (hpc/backfill_host_provenance.py); never authoritative",
        "n_cells": len(entries),
        "unmatched": unmatched,
        "cells": entries,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Backfill host provenance from SLURM logs.")
    ap.add_argument("--logs", default="hpc/hpc_logs")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--version", default=BENCHMARK_VERSION)
    ap.add_argument("--write", action="store_true", help="write the sidecar (default: dry-run)")
    args = ap.parse_args(argv)
    payload = build_backfill(Path(args.logs), Path(args.outputs), version=args.version)
    hosts = sorted({c["hostname"] for c in payload["cells"].values()})
    print(
        f"backfill: {payload['n_cells']} cells mapped from logs "
        f"({len(payload['unmatched'])} log-only/unmatched) across {len(hosts)} hosts"
    )
    if args.write:
        out = Path(args.outputs) / args.version / "provenance_backfill.json"
        out.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
