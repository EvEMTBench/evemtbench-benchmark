"""The public results bundle (hpc/make_results_bundle.py): the builder is pure + deterministic, the
aggregated CSV/JSON is a single-source-of-truth flattening of the same result.json the paper tables
read, it is strictly version-scoped, and its LaTeX tables are byte-identical to the paper builders'
output. Torch-free; drives everything against a tmp outputs/ tree."""

import csv
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "hpc"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import make_results_bundle as MRB  # noqa: E402
from evemtbench.version import BENCHMARK_VERSION  # noqa: E402

VER = BENCHMARK_VERSION
BASE = f"results/{VER}"


def _write_cell(root: Path, task, grid, baseline, mean, *, version=VER, ci95=0.02):
    """Minimal valid done cell: one result.json per test set with a balanced_accuracy aggregate."""
    for ts in ("test", "benchmark"):
        d = root / "outputs" / version / task / "held_out" / grid / baseline / ts
        d.mkdir(parents=True, exist_ok=True)
        (d / "result.json").write_text(
            json.dumps(
                {
                    "benchmark_version": version,
                    "task": task,
                    "protocol": "held_out",
                    "grid": grid,
                    "baseline": baseline,
                    "test_set": ts,
                    "n_test": 100,
                    "seeds": [0, 1, 2, 3, 4],
                    "n_params": 12345,
                    "compute": {"fit_seconds_mean": 4.2},
                    "aggregate": {"balanced_accuracy": {"mean": mean, "std": 0.01, "ci95": ci95}},
                }
            )
        )


def test_builder_deterministic_and_wellformed_without_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no outputs/ -> empty aggregate, tables all \res
    a = MRB.build_results_bundle("outputs")
    b = MRB.build_results_bundle("outputs")
    assert a == b, "build_results_bundle is not deterministic"
    # expected members
    assert f"{BASE}/aggregated_results.csv" in a
    assert f"{BASE}/aggregated_results.json" in a
    assert f"{BASE}/tables/_preamble.tex" in a
    assert f"{BASE}/README.md" in a
    assert f"{BASE}/tables/tab_res_fd.tex" in a
    assert f"{BASE}/tables/supp_tab_fd_test.tex" in a
    # empty: header-only CSV, empty JSON array, tables all placeholder markers (B2 five-state)
    csv_txt = a[f"{BASE}/aggregated_results.csv"]
    assert csv_txt.splitlines()[0] == ",".join(MRB.COLUMNS)
    assert len(csv_txt.splitlines()) == 1, "no data rows expected without outputs"
    assert json.loads(a[f"{BASE}/aggregated_results.json"]) == []
    assert r"\pending" in a[f"{BASE}/tables/tab_res_fd.tex"]


def test_aggregated_single_source_of_truth(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_cell(tmp_path, "fault_detection_local", "double_line", "resnet", 0.951, ci95=0.006)
    bundle = MRB.build_results_bundle("outputs")
    records = json.loads(bundle[f"{BASE}/aggregated_results.json"])
    rec = next(
        r
        for r in records
        if r["baseline"] == "resnet"
        and r["test_set"] == "test"
        and r["metric"] == "balanced_accuracy"
    )
    # JSON preserves the exact source value + provenance-free aggregate fields
    assert rec["mean"] == 0.951 and rec["ci95"] == 0.006 and rec["std"] == 0.01
    assert rec["n_seeds"] == 5 and rec["n_test"] == 100 and rec["version"] == VER
    # CSV row carries the same value (6g formatting) for the same key
    rows = list(csv.DictReader(io.StringIO(bundle[f"{BASE}/aggregated_results.csv"])))
    crow = next(
        r
        for r in rows
        if r["baseline"] == "resnet"
        and r["test_set"] == "test"
        and r["metric"] == "balanced_accuracy"
    )
    assert crow["mean"] == "0.951" and crow["ci95"] == "0.006" and crow["n_seeds"] == "5"


def test_strictly_version_scoped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_cell(tmp_path, "fault_detection_local", "double_line", "mlp", 0.80, version=VER)
    _write_cell(tmp_path, "fault_detection_local", "double_line", "mlp", 0.10, version="9.9.9")
    records = json.loads(MRB.build_results_bundle("outputs")[f"{BASE}/aggregated_results.json"])
    versions = {r["version"] for r in records}
    assert versions == {VER}, f"foreign-version rows leaked into the bundle: {versions}"
    assert all(r["mean"] != 0.10 for r in records), "the 9.9.9 value must not appear"


def test_tables_match_the_paper_builders(tmp_path, monkeypatch):
    """results/<v>/tables/*.tex must be byte-identical to the paper builders' output (one source of
    table logic) — so the public bundle and the private manuscript can never disagree."""
    monkeypatch.chdir(tmp_path)
    from make_paper_tables import build_paper_tables
    from make_supplement_tables import build_supplement_tables

    bundle = MRB.build_results_bundle("outputs")
    for fname, latex in {**build_paper_tables(), **build_supplement_tables()}.items():
        assert bundle[f"{BASE}/tables/{fname}"] == latex, f"{fname} diverges from the paper builder"


def test_csv_and_json_record_counts_agree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_cell(tmp_path, "fault_detection_local", "double_line", "resnet", 0.9)
    _write_cell(tmp_path, "fault_detection_local", "cigre_mv", "mlp", 0.8)
    bundle = MRB.build_results_bundle("outputs")
    n_csv = len(bundle[f"{BASE}/aggregated_results.csv"].splitlines()) - 1  # minus header
    n_json = len(json.loads(bundle[f"{BASE}/aggregated_results.json"]))
    assert n_csv == n_json > 0
