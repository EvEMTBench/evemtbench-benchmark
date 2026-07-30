"""The freshness gate (hpc/check_results_fresh.py) main(): passes when the committed views match a
fresh rebuild, fails on drift, and skips when outputs/ is absent. Driven against a temp 'repo' via
the EVEMTBENCH_REPO override so it never touches the real committed artifacts. Torch-free."""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "hpc"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import check_results_fresh as gate  # noqa: E402  (needs hpc/ on the path)
from evemtbench.version import BENCHMARK_VERSION  # noqa: E402


def _seed_repo(root: Path) -> None:
    d = (
        root
        / "outputs"
        / BENCHMARK_VERSION
        / "fault_classification_local"
        / "held_out"
        / "cigre_mv"
        / "mlp"
        / "test"
    )
    d.mkdir(parents=True)
    (d / "result.json").write_text(
        json.dumps(
            {
                "benchmark_version": BENCHMARK_VERSION,
                "task": "fault_classification_local",
                "protocol": "held_out",
                "grid": "cigre_mv",
                "baseline": "mlp",
                "test_set": "test",
                "n_test": 10,
                "seeds": [0, 1],
                "aggregate": {
                    "balanced_accuracy": {"mean": 0.5, "std": 0.0},
                    "macro_f1": {"mean": 0.3, "std": 0.0},
                },
            }
        )
    )


def test_gate_pass_fail_skip(tmp_path, monkeypatch):
    monkeypatch.setenv("EVEMTBENCH_REPO", str(tmp_path))
    monkeypatch.chdir(tmp_path)  # also restores cwd at teardown (gate chdir's internally)
    _seed_repo(tmp_path)

    # --fix seeds the committed views to match a fresh build...
    assert gate.main(["--fix"]) == 0
    assert (tmp_path / "RESULTS.md").exists() and (tmp_path / "leaderboard.html").exists()
    assert (tmp_path / "audit" / "index.html").exists(), "--fix must build the audit site too"
    assert (tmp_path / "coverage_snapshot.json").exists(), "--fix must write the snapshot"
    # ...so a verify now PASSES (the volatile 'updated …' lines are normalized, not false fails).
    assert gate.main([]) == 0

    # tamper the committed RESULTS.md -> the gate must FAIL
    (tmp_path / "RESULTS.md").write_text((tmp_path / "RESULTS.md").read_text() + "\nDRIFT\n")
    assert gate.main([]) == 1
    assert gate.main(["--fix"]) == 0

    # tamper the committed snapshot -> FAIL, --fix restores
    snap = tmp_path / "coverage_snapshot.json"
    snap.write_text(snap.read_text().replace('"done"', '"pending"', 1))
    assert gate.main([]) == 1
    assert gate.main(["--fix"]) == 0

    # tamper an audit page -> FAIL; a stray page a rebuild doesn't produce -> also FAIL
    ledger = tmp_path / "audit" / "ledger.html"
    ledger.write_text(ledger.read_text().replace("</body>", "<p>DRIFT</p></body>"))
    assert gate.main([]) == 1
    assert gate.main(["--fix"]) == 0
    (tmp_path / "audit" / "zombie.html").write_text("<!doctype html>stray")
    assert gate.main([]) == 1
    assert gate.main(["--fix"]) == 0  # --fix removes the stray page
    assert not (tmp_path / "audit" / "zombie.html").exists()
    assert gate.main([]) == 0

    # a physically impossible aggregate must trip the snapshot's embedded self-check
    bad = next((tmp_path / "outputs").rglob("result.json"))
    orig = bad.read_text()
    doc = json.loads(orig)
    doc["aggregate"]["balanced_accuracy"]["mean"] = 1.7
    bad.write_text(json.dumps(doc))
    assert gate.main([]) == 1
    bad.write_text(orig)
    assert gate.main(["--fix"]) == 0

    # an UNGATED page (progress.html, git-history-derived burnup) must never trip the gate:
    # neither its presence with arbitrary content, nor its absence
    (tmp_path / "audit" / "progress.html").write_text("<!doctype html>anything at all")
    assert gate.main([]) == 0
    (tmp_path / "audit" / "progress.html").unlink()
    assert gate.main([]) == 0

    # remove outputs/ -> the gate SKIPS (exit 0) where the data can't be rebuilt
    for f in (tmp_path / "outputs").rglob("result.json"):
        f.unlink()
    assert gate.main([]) == 0
