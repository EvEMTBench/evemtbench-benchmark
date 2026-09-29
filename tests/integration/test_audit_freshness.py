"""Invariants the freshness gate relies on for the audit site (hpc/make_audit.py): the builder
is pure and deterministic, the "updated <ts> · git <sha>" line is the ONLY volatile content,
pages are self-contained, internal links resolve, and status="auto" claims can never render
"measured" while any selected cell is still pending. Torch-free; drives everything against a tmp
outputs/ tree."""

import json
import re
from pathlib import Path

import make_audit
from evemtbench.coverage import expected_test_sets, iter_cells
from evemtbench.version import results_version

REPO = Path(__file__).resolve().parents[2]

_VOLATILE = re.compile(r"<div class=v>updated [^<]*</div>")


def _write_cell(root: Path, task: str, protocol: str, grid: str, baseline: str) -> None:
    """A minimal-but-valid done cell: one result.json per expected test set + a run record."""
    # per-protocol current-results tree (held_out resolves to 1.0.0 under the scoped revision)
    cell = root / "outputs" / results_version(protocol) / task / protocol / grid / baseline
    for ts in expected_test_sets(grid, protocol):
        d = cell / ts
        d.mkdir(parents=True, exist_ok=True)
        (d / "result.json").write_text(
            json.dumps(
                {
                    "benchmark_version": results_version(protocol),
                    "task": task,
                    "protocol": protocol,
                    "grid": grid,
                    "baseline": baseline,
                    "test_set": ts,
                    "n_test": 10,
                    "seeds": [0, 1, 2, 3, 4],
                    "aggregate": {"balanced_accuracy": {"mean": 0.9, "std": 0.01, "ci95": 0.02}},
                    "compute": {"fit_seconds_mean": 60.0},
                }
            )
        )
    (cell / "run_record.json").write_text(
        json.dumps(
            {
                "git_sha": "a" * 40,
                "lockfile_sha256": "b" * 64,
                "python": "3.12.0",
                "attestation": {"held_out_untouched": ["test", "benchmark"]},
            }
        )
    )


def test_build_deterministic_and_wellformed_without_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no outputs/ -> manifest-only page set, still complete
    a = make_audit.fresh_pages()
    b = make_audit.fresh_pages()
    assert a == b, "build_audit_pages is not deterministic"
    assert set(a) >= {"index.html", "ledger.html", "coverage.html", "gates.html"}
    for name, page in a.items():
        assert page.startswith("<!doctype html"), name
    assert "class=pillar" in a["index.html"]
    assert "class=claimrow" in a["ledger.html"]
    # with nothing measured, no auto claim may render as measured
    assert "in progress" in a["ledger.html"]


def test_timestamp_and_sha_are_the_only_volatile_line(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_cell(tmp_path, "fault_detection_local", "held_out", "double_line", "resnet")
    a = make_audit.fresh_pages("outputs", generated_at="2026-01-01 00:00 UTC · git aaaa")
    b = make_audit.fresh_pages("outputs", generated_at="2099-12-31 23:59 UTC · git bbbb")
    for name in a:
        assert a[name] != b[name], f"{name}: volatile line missing entirely"
        assert _VOLATILE.sub("", a[name]) == _VOLATILE.sub("", b[name]), (
            f"{name}: content beyond the 'updated' line differs between two stamped builds"
        )


def test_pages_are_self_contained_and_links_resolve(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pages = make_audit.fresh_pages()
    for name, page in pages.items():
        assert "<style>" in page and 'src="http' not in page, f"{name}: external asset"
        assert "<link rel=stylesheet" not in page, f"{name}: external stylesheet"
    hrefs = set()
    for page in pages.values():
        hrefs |= {
            h
            for h in re.findall(r'href="([^"]+)"', page)
            if not h.startswith(("http", "../", "#", "data:"))
        }
    # progress.html is built by make_burnup.py from git history and deliberately ungated
    known = set(pages) | {"progress.html"}
    dangling = {h for h in hrefs if h.split("#")[0] not in known}
    assert not dangling, f"internal links point at missing pages: {dangling}"


def test_auto_status_downgrades_until_every_selected_cell_is_done(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    task = "fault_detection_local"
    cells = [c for c in iter_cells(protocols=("held_out",)) if c.task == task]
    assert cells, "selector matches no cells — registry changed?"

    # seed all but one cell of the FD selector -> the FD headline row must stay "in progress"
    for c in cells[:-1]:
        _write_cell(tmp_path, c.task, c.protocol, c.grid, c.baseline)
    ledger = make_audit.fresh_pages("outputs")["ledger.html"]
    assert re.search(r"Fault detection[^<]*</span>.*?badge inprogress", ledger, re.S), (
        "FD claim rendered without an in-progress badge while a cell is pending"
    )
    assert "1 pending" in ledger

    # complete the last cell -> the FD row flips to measured
    last = cells[-1]
    _write_cell(tmp_path, last.task, last.protocol, last.grid, last.baseline)
    ledger = make_audit.fresh_pages("outputs")["ledger.html"]
    m = re.search(r"Fault detection.*?badge (\w+)", ledger, re.S)
    assert m and m.group(1) == "measured", "FD claim did not flip to measured when all cells done"


def test_committed_assets_match_manifest_and_pages(tmp_path, monkeypatch):
    """Figures are inputs, not derived views: every figure listed in the committed assets
    manifest must exist on disk, every asset a page references must be listed, and pages must
    render the honest pending note (never a broken img) for anything unrendered."""
    assets_dir = REPO / "audit" / "assets"
    if not (assets_dir / "manifest.json").is_file():
        import pytest

        pytest.skip("no committed assets yet")
    manifest = json.loads((assets_dir / "manifest.json").read_text())
    listed = {e["figure"] for sec in ("waveforms", "topologies") for e in manifest[sec].values()}
    on_disk = {p.name for p in assets_dir.glob("*.svg")}
    assert listed <= on_disk, f"manifest lists missing files: {listed - on_disk}"
    assert on_disk <= listed, f"unlisted stray assets: {on_disk - listed}"

    monkeypatch.chdir(tmp_path)  # build without outputs; asset refs come from the manifest
    pages = make_audit.fresh_pages()
    referenced = set()
    for page in pages.values():
        referenced |= set(re.findall(r'src="assets/([^"]+)"', page))
    assert referenced <= listed, f"pages reference unlisted assets: {referenced - listed}"
