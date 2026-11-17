"""Outside submissions: layout, metadata contract and validator (roadmap W4).

A submission is one directory per model, added to the repository by a pull/merge request::

    submissions/<model>/
        metadata.json
        results/<task>/<protocol>/<grid>/[pretrain-<source>/]<test_set>/result.json
        results/<task>/<protocol>/<grid>/[pretrain-<source>/]run_record.json   (optional)

Each ``result.json`` has exactly the schema the reference runner writes (``docs/submission.md``
§3), with ``baseline`` set to the submission's ``model`` name. ``metadata.json`` describes the
model and declares its training data (:data:`METADATA_FIELDS`).

:func:`validate_submission` checks a submission without the dataset. It checks:

* the metadata (required fields, allowed values, and no declared training or tuning on
  ``test`` / ``benchmark``);
* every cell: it exists in the registry, the protocol applies to the grid, the test set is one
  the grid reports, and the result has ≥ 3 seeds;
* the official metric set is present, and ``aggregate`` matches a recomputation from
  ``per_seed`` (a hand-edited aggregate fails);
* value ranges;
* ``n_test`` matches the reference results bundle when one is available;
* track completeness;
* the run record's no-tuning attestation, when a run record is included.

It cannot verify how results were produced; that is what the *reproduced* tier is for. A
maintainer re-runs the submitted code and sets ``reproduced`` in the metadata. The roadmap
decision (2026-11-10) is that self-reported results are listed with a badge once they pass this
validator.

Uses only the standard library plus the evaluation layer; it never imports
:mod:`evemtbench.baselines`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .coverage import (
    ALL_BASELINES,
    PRETRAIN_GRID,
    PRETRAIN_GRIDS,
    expected_test_sets,
    iter_cells,
    protocol_grid_applies,
)
from .evaluation.aggregate import aggregate_seed_metrics
from .splits.protocols import TRANSFER_PROTOCOLS
from .tasks.registry import get_task, list_tasks
from .version import results_version

SCHEMA_VERSION = 1
MIN_SEEDS = 3
MODEL_TYPES = ("classical", "deep", "pretrained", "fine-tuned", "zero-shot")
TRACKS = ("full", "partial")  # "lite" joins once the lite track is defined (roadmap W2)
#: tokens allowed in ``trained_on``. ``external:<description>`` covers outside pretraining data.
TRAINED_ON_TOKENS = ("adapt_grid:train", "multi_grid:train")
_MODEL_NAME = re.compile(r"^[a-z0-9][a-z0-9._+-]{1,63}$")
_URL = re.compile(r"^https?://\S+$")
_FORBIDDEN_SPLIT_WORDS = re.compile(r"\b(test|benchmark)\b", re.I)
_TOL = 1e-6

#: metadata.json fields → (required, description). The single source for the validator and the
#: documentation table in docs/submission.md.
METADATA_FIELDS: dict[str, tuple[bool, str]] = {
    "schema_version": (True, f"integer, currently {SCHEMA_VERSION}"),
    "model": (True, "model id; also the `baseline` field of every result.json and the directory "
              "name; lowercase `[a-z0-9._+-]`, not an in-house baseline name"),
    "display_name": (False, "human-readable name for the leaderboard"),
    "org": (True, "organisation or team"),
    "model_type": (True, f"one of {', '.join(MODEL_TYPES)}"),
    "code_link": (True, "public URL of the code that produced the results"),
    "paper_link": (False, "URL of a paper or report, or null"),
    "benchmark_version": (True, "the EvEMTBench contract version evaluated against"),
    "track": (True, f"one of {', '.join(TRACKS)}; `full` = every held_out cell"),
    "trained_on": (True, "list of training data: `adapt_grid:train`, `multi_grid:train`, or "
                   "`external:<description>` for outside (pre)training data"),
    "tuned_on": (True, "must be `val` — model selection and hyperparameters on validation only"),
    "test_data_leakage": (True, "must be false — no test/benchmark data, and no simulation of "
                          "them, in any training or pretraining data"),
    "reproduced": (True, "false on submission; set to true only by a maintainer after re-running "
                   "`code_link`"),
    "reproduced_by": (False, "maintainer who re-ran it (set with `reproduced`)"),
    "reproduced_commit": (False, "commit of `code_link` that was re-run (set with `reproduced`)"),
    "notes": (False, "free text"),
}  # fmt: skip

_RANGE_01 = ("balanced_accuracy", "accuracy", "macro_f1")
_RANGE_NONNEG = ("mae", "rmse", "median_abs_error")


@dataclass
class Report:
    """Validation outcome for one submission directory."""

    path: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_results: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


# ------------------------------------------------------------------------------------------------
# metadata
# ------------------------------------------------------------------------------------------------
def check_metadata(meta: dict, rep: Report, *, dirname: str | None = None) -> None:
    for key, (required, _) in METADATA_FIELDS.items():
        if required and key not in meta:
            rep.error(f"metadata: missing required field '{key}'")
    for key in meta:
        if key not in METADATA_FIELDS:
            rep.warn(f"metadata: unknown field '{key}' (ignored)")
    if meta.get("schema_version") != SCHEMA_VERSION:
        rep.error(f"metadata: schema_version must be {SCHEMA_VERSION}")
    model = meta.get("model")
    if not isinstance(model, str) or not _MODEL_NAME.match(model):
        rep.error("metadata: 'model' must match [a-z0-9][a-z0-9._+-]{1,63}")
    elif model in ALL_BASELINES:
        rep.error(f"metadata: 'model' {model!r} collides with an in-house baseline name")
    elif dirname is not None and model != dirname:
        rep.error(f"metadata: 'model' {model!r} differs from the directory name {dirname!r}")
    if meta.get("model_type") not in MODEL_TYPES:
        rep.error(f"metadata: 'model_type' must be one of {MODEL_TYPES}")
    if meta.get("track") not in TRACKS:
        rep.error(f"metadata: 'track' must be one of {TRACKS}")
    for key in ("code_link", "paper_link"):
        v = meta.get(key)
        if v is not None and (not isinstance(v, str) or not _URL.match(v)):
            rep.error(f"metadata: '{key}' must be an http(s) URL")
    if meta.get("tuned_on") != "val":
        rep.error("metadata: 'tuned_on' must be 'val' (never tune on test/benchmark)")
    if meta.get("test_data_leakage") is not False:
        rep.error("metadata: 'test_data_leakage' must be false")
    trained = meta.get("trained_on")
    if not isinstance(trained, list) or not trained:
        rep.error("metadata: 'trained_on' must be a non-empty list")
    else:
        for t in trained:
            if not isinstance(t, str):
                rep.error(f"metadata: 'trained_on' entry {t!r} is not a string")
            elif t.startswith("external:"):
                if len(t) <= len("external:"):
                    rep.error("metadata: 'external:' needs a description of the data")
            elif t not in TRAINED_ON_TOKENS:
                rep.error(
                    f"metadata: 'trained_on' entry {t!r} not allowed "
                    f"(use {', '.join(TRAINED_ON_TOKENS)} or external:<description>)"
                )
            if isinstance(t, str) and not t.startswith("external:"):
                if _FORBIDDEN_SPLIT_WORDS.search(t):
                    rep.error(f"metadata: 'trained_on' entry {t!r} names a held-out split")
    if not isinstance(meta.get("reproduced"), bool):
        rep.error("metadata: 'reproduced' must be true or false")
    elif meta["reproduced"] and not (meta.get("reproduced_by") and meta.get("reproduced_commit")):
        rep.error("metadata: 'reproduced: true' needs 'reproduced_by' and 'reproduced_commit'")


# ------------------------------------------------------------------------------------------------
# reference n_test
# ------------------------------------------------------------------------------------------------
def load_reference_n_test(path: str | Path | None) -> dict[tuple, int]:
    """``{(task, protocol, grid, source, test_set): n_test}`` from a results bundle's
    ``aggregated_results.csv`` (``results/<version>/``). ``source`` is ``""`` outside transfer.
    Empty when the file is absent."""
    if path is None or not Path(path).is_file():
        return {}
    ref: dict[tuple, int] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                n = int(float(row["n_test"]))
            except (KeyError, TypeError, ValueError):
                continue
            key = (row["task"], row["protocol"], row["grid"], row.get("source") or "",
                   row["test_set"])  # fmt: skip
            ref.setdefault(key, n)
    return ref


def default_reference(repo: Path | None = None) -> Path | None:
    """The committed bundle of the current version, when running inside the repository."""
    from .version import BENCHMARK_VERSION

    repo = repo or Path(__file__).resolve().parents[2]
    p = repo / "results" / BENCHMARK_VERSION / "aggregated_results.csv"
    return p if p.is_file() else None


# ------------------------------------------------------------------------------------------------
# one result.json
# ------------------------------------------------------------------------------------------------
def _close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=_TOL, abs_tol=_TOL)


def check_result(doc: dict, rel: str, model: str, rep: Report, ref: dict[tuple, int]) -> tuple:
    """Validate one result document; returns its cell key (or ``()`` when unusable)."""
    where = f"{rel}:"
    req = ("task", "protocol", "grid", "baseline", "test_set", "benchmark_version", "n_test",
           "seeds", "per_seed", "aggregate")  # fmt: skip
    missing = [k for k in req if k not in doc]
    if missing:
        rep.error(f"{where} missing fields {missing}")
        return ()
    task, protocol, grid, ts = doc["task"], doc["protocol"], doc["grid"], doc["test_set"]
    source = doc.get("pretrain_grid") if protocol in TRANSFER_PROTOCOLS else None
    if doc["baseline"] != model:
        rep.error(f"{where} baseline {doc['baseline']!r} must equal the model name {model!r}")
    if task not in list_tasks():
        rep.error(f"{where} unknown task {task!r}")
        return ()
    td = get_task(task)
    if protocol not in td.protocols:
        rep.error(f"{where} protocol {protocol!r} is not defined for {task}")
        return ()
    if not protocol_grid_applies(protocol, grid):
        rep.error(f"{where} protocol {protocol!r} does not apply to grid {grid!r}")
        return ()
    if ts not in expected_test_sets(grid, protocol):
        rep.error(f"{where} test_set {ts!r} is not scored for {grid}/{protocol}")
        return ()
    if protocol in TRANSFER_PROTOCOLS and source not in PRETRAIN_GRIDS:
        rep.error(f"{where} transfer results need pretrain_grid in {PRETRAIN_GRIDS}")
        return ()
    expected_ver = results_version(protocol, task, grid=grid, source=source)
    if doc["benchmark_version"] != expected_ver:
        rep.error(f"{where} benchmark_version {doc['benchmark_version']!r} != {expected_ver!r}")
    # path must match the fields (results/<task>/<protocol>/<grid>/[pretrain-<src>/]<ts>/)
    parts = Path(rel).parts
    want = ("results", task, protocol, grid, *([f"pretrain-{source}"] if source else []), ts,
            "result.json")  # fmt: skip
    if parts != want:
        rep.error(f"{where} path does not match its fields; expected {'/'.join(want)}")

    seeds = doc["seeds"]
    if not isinstance(seeds, list) or len(set(seeds)) != len(seeds):
        rep.error(f"{where} 'seeds' must be a list of distinct seeds")
    elif len(seeds) < MIN_SEEDS:
        rep.error(f"{where} {len(seeds)} seed(s); at least {MIN_SEEDS} are required")
    per_seed = doc["per_seed"]
    if not isinstance(per_seed, list) or [p.get("seed") for p in per_seed] != list(seeds):
        rep.error(f"{where} 'per_seed' must hold one entry per seed, in 'seeds' order")
        return (task, protocol, grid, source or "", ts)

    official = list(td.metrics)
    agg = doc["aggregate"]
    absent = [m for m in official if m not in agg]
    if absent:
        rep.error(f"{where} official metrics missing from aggregate: {absent}")
    try:
        per_seed_metrics = [
            {k: (math.nan if v is None else float(v)) for k, v in p["metrics"].items()}
            for p in per_seed
        ]
        recomputed = aggregate_seed_metrics(per_seed_metrics)
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        rep.error(f"{where} per_seed metrics unreadable ({type(e).__name__})")
        recomputed = {}
    for m in official:
        if m not in agg or m not in recomputed:
            continue
        for stat in ("mean", "std", "ci95"):
            if not _close(agg[m].get(stat), recomputed[m].get(stat)):
                rep.error(
                    f"{where} aggregate.{m}.{stat}={agg[m].get(stat)} does not match the value "
                    f"recomputed from per_seed ({recomputed[m].get(stat)})"
                )
        mean = agg[m].get("mean")
        if mean is not None and (
            (m in _RANGE_01 and not 0.0 <= mean <= 1.0)
            or (m in _RANGE_NONNEG and mean < 0.0)
            or (m == "r2" and mean > 1.0 + 1e-9)
        ):
            rep.error(f"{where} {m} mean {mean} is out of range")

    key = (task, protocol, grid, source or "", ts)
    if ref:
        n_ref = ref.get(key)
        if n_ref is None and source == PRETRAIN_GRID:  # default source: legacy source-less rows
            n_ref = ref.get((task, protocol, grid, "", ts))
        if n_ref is None:
            rep.warn(f"{where} no reference n_test for this cell (not checked)")
        elif doc["n_test"] != n_ref:
            rep.error(f"{where} n_test {doc['n_test']} != reference {n_ref} (wrong eval set?)")
    return key


def check_run_record(rec: dict, rel: str, rep: Report) -> None:
    att = rec.get("attestation") or {}
    untouched = att.get("held_out_untouched") or []
    if not {"test", "benchmark"} <= set(untouched):
        rep.error(f"{rel}: attestation.held_out_untouched must include test and benchmark")
    sel = att.get("model_selection_on")
    if sel not in ("val", "none"):
        rep.error(f"{rel}: attestation.model_selection_on must be 'val' (or 'none'), got {sel!r}")


# ------------------------------------------------------------------------------------------------
# a whole submission
# ------------------------------------------------------------------------------------------------
def full_track_cells() -> set[tuple]:
    """Every held_out cell key ``(task, "held_out", grid, "", test_set)`` — the `full` track."""
    keys = set()
    for c in iter_cells(protocols=("held_out",), baselines=("mlp",)):  # mlp applies everywhere
        for ts in expected_test_sets(c.grid, c.protocol):
            keys.add((c.task, c.protocol, c.grid, "", ts))
    return keys


def validate_submission(path: str | Path, *, reference: str | Path | None = None) -> Report:
    path = Path(path)
    rep = Report(path)
    meta_path = path / "metadata.json"
    if not meta_path.is_file():
        rep.error("metadata.json is missing")
        return rep
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        rep.error(f"metadata.json is not valid JSON ({e})")
        return rep
    if not isinstance(meta, dict):
        rep.error("metadata.json must be a JSON object")
        return rep
    check_metadata(meta, rep, dirname=path.name)
    model = meta.get("model", path.name)
    ref = load_reference_n_test(reference)
    if not ref:
        rep.warn("no reference results bundle: n_test was not checked")

    seen: dict[tuple, str] = {}
    for f in sorted((path / "results").rglob("result.json")) if (path / "results").is_dir() else []:
        rel = f.relative_to(path).as_posix()
        try:
            doc = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError) as e:
            rep.error(f"{rel}: not valid JSON ({e})")
            continue
        rep.n_results += 1
        key = check_result(doc, rel, model, rep, ref)
        if key:
            if key in seen:
                rep.error(f"{rel}: duplicate cell (also {seen[key]})")
            seen[key] = rel
    for f in sorted(path.rglob("run_record.json")):
        rel = f.relative_to(path).as_posix()
        try:
            check_run_record(json.loads(f.read_text()), rel, rep)
        except (OSError, json.JSONDecodeError) as e:
            rep.error(f"{rel}: not valid JSON ({e})")
    if rep.n_results == 0:
        rep.error("no results/**/result.json found")
    if meta.get("track") == "full":
        missing = sorted(full_track_cells() - set(seen))
        if missing:
            shown = ", ".join("/".join(k for k in m if k) for m in missing[:5])
            rep.error(f"track 'full' is missing {len(missing)} held_out cell(s), e.g. {shown}")
    return rep


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Validate EvEMTBench submissions (metadata + result.json files)."
    )
    p.add_argument("paths", nargs="+", help="submission directories, or a parent with --all")
    p.add_argument("--all", action="store_true", help="validate every subdirectory of each path")
    p.add_argument(
        "--reference",
        default=None,
        help="results bundle aggregated_results.csv for the n_test check "
        "(default: the committed results/<version>/ bundle when run inside the repository)",
    )
    args = p.parse_args(argv)
    reference = args.reference or default_reference()
    dirs: list[Path] = []
    for raw in args.paths:
        d = Path(raw)
        dirs += sorted(x for x in d.iterdir() if x.is_dir()) if args.all else [d]
    failed = 0
    for d in dirs:
        rep = validate_submission(d, reference=reference)
        status = "OK" if rep.ok else "FAIL"
        print(f"{status}  {d}  ({rep.n_results} result file(s))")
        for e in rep.errors:
            print(f"  error: {e}")
        for w in rep.warnings:
            print(f"  warning: {w}")
        failed += not rep.ok
    print(f"{len(dirs) - failed} of {len(dirs)} submission(s) valid")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
