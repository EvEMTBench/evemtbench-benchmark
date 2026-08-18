"""Post-hoc artifact integrity of the outputs/ corpus — the audit's sixth gate.

CI tests validate the code; this module validates the *artifacts* the sweep left behind, which
only exist on the loop host. For every done cell it checks that the seed set is complete and
consistent across result.json / run_record.json, that the provenance record exists, is
schema-valid and carries the non-null pins (git sha, lockfile hash), and that saved checkpoints
match their recorded SHA-256 paths. ``--deep`` additionally opens every predictions parquet
(metadata only) and asserts its row count equals ``n_test`` — catching truncated NFS writes that
no code test can see. Pending/failed cells are skipped, never failed: coverage owns those.

    python -m evemtbench.integrity                     # fast, JSON-only (seconds)
    python -m evemtbench.integrity --deep              # + parquet metadata (freeze-time)
    python -m evemtbench.integrity --expect-seeds 0    # fixture/test corpora
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .coverage import (
    DEEP_BASELINES,
    SNAPSHOT_PROTOCOLS,
    Cell,
    cell_dir,
    cell_status,
    expected_test_sets,
    iter_cells,
)
from .data.input_digests import load_table as _load_digest_table

# per-protocol version resolution lives in cell_dir/cell_status (evemtbench.version.results_version)

# Deep baselines persist best-val weights under these protocols (zero-shot only loads them).
_CHECKPOINTED_PROTOCOLS = ("held_out", "multi_grid_pretrain", "transfer_finetune")

_SCHEMA_PATH = Path(__file__).resolve().parent / "baselines" / "run_record.schema.json"


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _validator() -> Any:  # jsonschema.Draft202012Validator | None (typed loosely: optional dep)
    """A jsonschema validator for run_record.json, or None when jsonschema isn't installed
    (it lives in the dev extras; the checker degrades to structural checks without it)."""
    try:
        import jsonschema
    except ImportError:
        return None
    schema = json.loads(_SCHEMA_PATH.read_text())
    return jsonschema.Draft202012Validator(schema)


def check_cell(
    cell: Cell,
    cdir: Path,
    *,
    expected_seeds: list[int] | None,
    validator: Any = None,
    deep: bool = False,
    digests: dict | None = None,
) -> list[str]:
    """Problems found in one done cell's artifacts (empty list = clean)."""
    prefix = f"{cell.task}/{cell.protocol}/{cell.grid}/{cell.baseline}"
    problems: list[str] = []

    results: dict[str, dict] = {}
    for ts in expected_test_sets(cell.grid, cell.protocol):
        doc = _load(cdir / ts / "result.json")
        if doc is None:  # cell_status said done, so this indicates a racing write
            problems.append(f"{prefix}/{ts}: result.json unreadable")
            continue
        results[ts] = doc
        seeds = doc.get("seeds") or []
        if expected_seeds is not None and seeds != expected_seeds:
            problems.append(f"{prefix}/{ts}: seeds {seeds} != expected {expected_seeds}")
        for metric, agg in (doc.get("aggregate") or {}).items():
            n = agg.get("n_seeds") if isinstance(agg, dict) else None
            if n is not None and n != len(seeds):
                problems.append(f"{prefix}/{ts}: {metric}.n_seeds={n} != len(seeds)={len(seeds)}")

    record = _load(cdir / "run_record.json")
    if record is None:
        problems.append(f"{prefix}: run_record.json missing/unreadable")
        return problems
    for key in ("git_sha", "lockfile_sha256"):
        if not record.get(key):
            problems.append(f"{prefix}: run_record.{key} is null/absent")
    for ts, doc in results.items():
        if record.get("seeds") != doc.get("seeds"):
            problems.append(
                f"{prefix}: run_record.seeds {record.get('seeds')} != result[{ts}].seeds"
            )
    if validator is not None:
        for err in validator.iter_errors(record):
            problems.append(f"{prefix}: run_record schema: {err.message} (at {list(err.path)})")

    # input-digest cross-check (deep-audit W4): a partition this run actually loaded must match
    # the committed digest pin — SAME dataset instance only (shape equality gates the compare, so
    # legitimately different builds like the test fixtures are skipped, silent same-shape content
    # swaps are caught via manifest_sha256 + feature_hash).
    digest_entries = (digests or {}).get("entries", {})
    for part in (record.get("data") or {}).get("partitions") or []:
        pin = digest_entries.get(f"{part.get('grid')}/{part.get('family')}")
        if not pin or pin.get("shape") != part.get("shape"):
            continue
        for field in ("manifest_sha256", "feature_hash"):
            if part.get(field) and pin.get(field) and part[field] != pin[field]:
                problems.append(
                    f"{prefix}: input {field} for {part['grid']}/{part['family']} does not "
                    f"match the committed digest pin (dataset swapped under the version?)"
                )

    checkpoints = record.get("checkpoints") or {}
    if (
        cell.baseline in DEEP_BASELINES
        and cell.protocol in _CHECKPOINTED_PROTOCOLS
        and not checkpoints
    ):
        problems.append(f"{prefix}: deep baseline has no recorded checkpoints")
    for seed, entry in sorted(checkpoints.items()):
        sha = (entry or {}).get("sha256") or ""
        if len(sha) != 64:
            problems.append(f"{prefix}: checkpoint seed {seed} sha256 malformed")
        path = (entry or {}).get("path") or ""
        if path and not (cdir / path).is_file():
            problems.append(f"{prefix}: checkpoint file missing: {path}")

    if deep:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            problems.append(f"{prefix}: --deep requested but pyarrow is not installed")
            return problems
        for ts, doc in results.items():
            for seed in doc.get("seeds") or []:
                f = cdir / ts / f"predictions_seed{seed}.parquet"
                if not f.is_file():
                    problems.append(f"{prefix}/{ts}: predictions_seed{seed}.parquet missing")
                    continue
                try:
                    n_rows = pq.ParquetFile(f).metadata.num_rows
                except Exception as e:  # truncated/corrupt parquet
                    problems.append(
                        f"{prefix}/{ts}: predictions_seed{seed}.parquet unreadable: {e}"
                    )
                    continue
                if n_rows != doc.get("n_test"):
                    problems.append(
                        f"{prefix}/{ts}: predictions_seed{seed}.parquet has {n_rows} rows, "
                        f"result.n_test={doc.get('n_test')}"
                    )
    return problems


def check_outputs(
    out: str | Path = "outputs",
    *,
    version: str | None = None,  # None -> per-protocol current-results tree
    protocols: tuple[str, ...] = SNAPSHOT_PROTOCOLS,
    expected_seeds: list[int] | None = [0, 1, 2, 3, 4],  # noqa: B006 — the production seed set
    deep: bool = False,
) -> dict:
    """Integrity report over every done cell: {ok, n_done, n_skipped, problems, provenance_pairs}.

    ``provenance_pairs`` reports (not fails) the distinct (git_sha, lockfile_sha256) pairs across
    the corpus — the input for the paper's "attributable to a frozen configuration" sentence."""
    out = Path(out)
    validator = _validator()
    digests = _load_digest_table()
    problems: list[str] = []
    pairs: set[tuple[str, str]] = set()
    n_done = n_skipped = 0
    for cell in iter_cells(protocols=tuple(protocols)):
        if cell_status(out, cell, version=version) != "done":
            n_skipped += 1
            continue
        n_done += 1
        cdir = cell_dir(out, cell, version=version)
        problems += check_cell(
            cell,
            cdir,
            expected_seeds=expected_seeds,
            validator=validator,
            deep=deep,
            digests=digests,
        )
        record = _load(cdir / "run_record.json") or {}
        pairs.add((str(record.get("git_sha")), str(record.get("lockfile_sha256"))))
    return {
        "ok": not problems,
        "n_done": n_done,
        "n_skipped": n_skipped,
        "schema_validated": validator is not None,
        "problems": problems,
        "provenance_pairs": sorted(pairs),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Artifact integrity over the outputs/ corpus.")
    p.add_argument("--out", default="outputs")
    p.add_argument("--version", default=None, help="explicit tree (default: per-protocol)")
    p.add_argument("--protocols", nargs="*", default=list(SNAPSHOT_PROTOCOLS))
    p.add_argument(
        "--expect-seeds",
        type=int,
        nargs="*",
        default=[0, 1, 2, 3, 4],
        help="the seed set every done cell must carry (empty = don't check seeds)",
    )
    p.add_argument(
        "--deep",
        action="store_true",
        help="also open every predictions parquet (metadata) and check row counts",
    )
    args = p.parse_args(argv)
    report = check_outputs(
        args.out,
        version=args.version,
        protocols=tuple(args.protocols),
        expected_seeds=args.expect_seeds or None,
        deep=args.deep,
    )
    for problem in report["problems"]:
        print(f"PROBLEM: {problem}")
    status = "OK" if report["ok"] else "FAIL"
    print(
        f"integrity: {status} — {report['n_done']} done cells checked "
        f"({report['n_skipped']} pending/failed skipped, schema "
        f"{'validated' if report['schema_validated'] else 'NOT validated: jsonschema missing'}), "
        f"{len(report['problems'])} problems, "
        f"{len(report['provenance_pairs'])} distinct (git_sha, lockfile) pairs"
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
