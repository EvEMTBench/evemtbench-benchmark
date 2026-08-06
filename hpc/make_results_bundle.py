"""Build the public ``results/<version>/`` bundle — the committed, self-contained results tier.

The intermediate layer between the git-ignored raw ``outputs/`` (per-cell result.json, ~21 GB with
checkpoints) and the PRIVATE ``paper/manuscript/`` (never published). It carries everything a public
reader/reviewer needs to inspect and verify the numbers WITHOUT the manuscript:

    results/<version>/
      aggregated_results.csv    flat: one row per (task, protocol, grid, baseline, test_set, metric)
      aggregated_results.json   same records, exact-typed (null for absent-class metrics)
      tables/                   the LaTeX result tables (5 main + 10 supplement), copy-paste ready
        tab_res_*.tex / supp_tab_*_*.tex / _preamble.tex
      README.md                 schema + provenance pointer

Like ``make_results``/``make_audit``, the builders here are pure (no writes, no clock) and
deterministic, so the freshness gate (``hpc/check_results_fresh.py``) can byte-compare the committed
bundle against a fresh rebuild. The LaTeX comes from the SAME builders the private paper uses
(``make_paper_tables`` / ``make_supplement_tables``) — one table-logic source, two write
targets — so ``results/tables/*.tex`` is provably identical to the manuscript's fragments. The
aggregated CSV/JSON is derived from the same ``leaderboard.scan_results`` rows the leaderboard/audit
read, version-scoped to ``BENCHMARK_VERSION``.

    PYTHONPATH=src python hpc/make_results_bundle.py            # outputs/ -> results/<version>/
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO / "src", _REPO / "hpc", _REPO / "hpc" / "paper_artifacts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from paper_lib import build_transfer_tables  # noqa: E402

from evemtbench.leaderboard import Row, scan_results  # noqa: E402
from evemtbench.version import BENCHMARK_VERSION  # noqa: E402
from make_paper_tables import build_paper_tables  # noqa: E402
from make_supplement_tables import build_supplement_tables  # noqa: E402

# Flat schema: one record per (task, protocol, grid, baseline, test_set, metric).
COLUMNS = (
    "version",
    "task",
    "protocol",
    "grid",
    "baseline",
    "source",
    "test_set",
    "metric",
    "mean",
    "std",
    "ci95",
    "n_seeds",
    "n_test",
    "n_params",
    "fit_seconds",
)


def aggregated_records(rows: list[Row], *, version: str = BENCHMARK_VERSION) -> list[dict]:
    """Flatten scanned Rows to one record per (cell × test_set × metric), version-scoped.

    Exact values (no rounding): ``scan_results`` carries ``metrics[m] = {mean, std, ci95}`` and
    ``seeds``. Absent-class metrics keep ``None`` mean/std/ci95 (null in JSON / empty in CSV), so
    the file honestly states the class had no support rather than hiding it as a 0."""
    recs: list[dict] = []
    for r in rows:
        if r.version != version:
            continue
        for metric, m in r.metrics.items():
            recs.append(
                {
                    "version": r.version,
                    "task": r.task,
                    "protocol": r.protocol,
                    "grid": r.grid,
                    "baseline": r.baseline,
                    # transfer pre-training source (20/110/345 kV); None for non-transfer cells
                    "source": (
                        (r.source or "template_20kv")
                        if r.protocol in ("transfer_zeroshot", "transfer_finetune")
                        else None
                    ),
                    "test_set": r.test_set,
                    "metric": metric,
                    "mean": m.get("mean"),
                    "std": m.get("std"),
                    "ci95": m.get("ci95"),
                    "n_seeds": r.seeds,
                    "n_test": r.n_test,
                    "n_params": r.n_params,
                    "fit_seconds": r.fit_seconds,
                }
            )
    recs.sort(
        key=lambda d: (
            d["task"],
            d["protocol"],
            d["grid"],
            d["baseline"],
            d["source"] or "",
            d["test_set"],
            d["metric"],
        )
    )
    return recs


def _csv_cell(v: object) -> str:
    """Deterministic, human-readable CSV cell: '' for None, 6 significant figures for floats."""
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _to_csv(records: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(COLUMNS)
    for rec in records:
        w.writerow([_csv_cell(rec[c]) for c in COLUMNS])
    return buf.getvalue()


def _to_json(records: list[dict]) -> str:
    # Exact float values (round-trips result.json); records already sorted deterministically.
    return json.dumps(records, indent=2, allow_nan=False) + "\n"


_PREAMBLE = r"""% Minimal standalone preamble for the results/<version>/tables/*.tex fragments.
% The (private) manuscript uses the full acronym package; these \providecommand stubs let the
% fragments compile on their own — \input any of them after \input{_preamble.tex} — so a public
% reader can typeset the tables without the manuscript. AUTO-GENERATED; do not edit by hand.
\providecommand{\res}{--}
\providecommand{\ac}[1]{#1}
\providecommand{\acs}[1]{#1}
"""


def _readme(version: str, n_records: int, n_cells: int) -> str:
    cols = "".join(f"- `{c}`\n" for c in COLUMNS)
    return (
        f"# EvEMTBench results — v{version}\n\n"
        "The public, self-contained results tier: the aggregated numbers and the LaTeX result\n"
        "tables for this benchmark version. Derived from the raw per-cell artifacts under\n"
        "`outputs/` (git-ignored) by `hpc/make_results_bundle.py`; **do not edit by hand** —\n"
        "regenerate with:\n\n"
        "```\nPYTHONPATH=src python hpc/make_results_bundle.py\n```\n\n"
        "Every file here is byte-checked against a fresh rebuild by the freshness gate\n"
        "(`hpc/check_results_fresh.py`), so the committed bundle provably equals the data.\n\n"
        "## `aggregated_results.csv` / `aggregated_results.json`\n\n"
        f"One record per (task × protocol × grid × baseline × test_set × metric) — {n_records}\n"
        f"records across {n_cells} cells for this version. Columns / keys:\n\n"
        f"{cols}\n"
        "`mean`/`std`/`ci95` are aggregated over the `n_seeds` random seeds; a null (JSON) /\n"
        "empty (CSV) value marks a metric with no support on that eval set (e.g. an absent\n"
        "per-class F1), reported honestly rather than as a misleading 0. The CSV shows 6\n"
        "significant figures for readability; the JSON preserves the exact values.\n\n"
        "One convention split, stated because the column is shared: rows with `test_set=val`\n"
        "carry a **sample** std (`ddof=1`, written by `hpc/rescore_validation.py`), while\n"
        "`test`/`benchmark` rows carry a **population** std (`ddof=0`, written by the runner).\n"
        "`ci95` is t-based (`ddof=1`) in both. Validation rows are published so that baseline\n"
        "selection is auditable — the reported results are always the `test`/`benchmark` rows.\n\n"
        "## `tables/`\n\n"
        "The LaTeX result tables, copy-paste ready: `tab_res_<fn>.tex` (5 main, best learned\n"
        "baseline per grid × observability view × held_out test set) and\n"
        "`supp_tab_<fn>_<test_set>.tex` (10 full per-baseline supplement tables). Each cell is\n"
        "`mean $\\pm$ 95% CI`; `\\res` = no result yet. Compile standalone by prepending\n"
        "`\\input{_preamble.tex}` (it stubs `\\res`/`\\ac`/`\\acs`).\n\n"
        "## Full provenance\n\n"
        "For per-number traceability — output file, evaluation code, training log, training code,\n"
        "config, W&B run and the exact reproduce command behind each number — see the audit\n"
        f"site: `audit/paper_numbers.html` and `audit/provenance/{version}/index.json`.\n"
    )


def build_results_bundle(
    outputs: str | Path = "outputs", *, version: str = BENCHMARK_VERSION
) -> dict[str, str]:
    """{rel-path-under-repo: content} for the whole ``results/<version>/`` bundle. Pure +
    deterministic (no timestamps, sorted) so the freshness gate byte-compares it like the rest."""
    out = Path(outputs)
    rows = scan_results(out) if out.is_dir() else []
    records = aggregated_records(rows, version=version)
    n_cells = len(
        {
            (r["task"], r["protocol"], r["grid"], r["baseline"], r["source"], r["test_set"])
            for r in records
        }
    )
    base = f"results/{version}"
    bundle = {
        f"{base}/aggregated_results.csv": _to_csv(records),
        f"{base}/aggregated_results.json": _to_json(records),
        f"{base}/tables/_preamble.tex": _PREAMBLE,
        f"{base}/README.md": _readme(version, len(records), n_cells),
    }
    for fname, latex in {
        **build_paper_tables(),
        **build_supplement_tables(),
        **build_transfer_tables(outputs),
    }.items():
        bundle[f"{base}/tables/{fname}"] = latex
    return bundle


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the public results/<version>/ bundle.")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--out", default=".", help="repo root under which results/ is written")
    args = ap.parse_args(argv)
    bundle = build_results_bundle(args.outputs)
    root = Path(args.out)
    for rel, content in sorted(bundle.items()):
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
    print(f"wrote {len(bundle)} files to {root}/results/{BENCHMARK_VERSION}/")


if __name__ == "__main__":
    main()
