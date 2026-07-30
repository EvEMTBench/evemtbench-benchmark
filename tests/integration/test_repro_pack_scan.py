"""The repro-pack leak-scan gate (hpc/make_repro_pack.sh --scan-only): planted forbidden
strings refuse the pack, and the pack's stricter policy holds — unlike the public-release scan it
also forbids cluster hostnames and ANY private-forge reference (the pack ships no schema files,
so there is no $id allowlist). Drives the REAL script's scan (via --scan-only) so the tests
cannot drift from the shipped patterns.

Planted forbidden strings are assembled from adjacent string literals so this file itself passes
the public-release scan (it ships in the export and is scanned — no exclusions)."""

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "hpc" / "make_repro_pack.sh"


def _scan(target: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), "--scan-only", str(target)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_planted_home_path_refuses(tmp_path):
    (tmp_path / "leak.md").write_text("data lives at " + "/home/" + "hpc/somewhere/secret" + "\n")
    res = _scan(tmp_path)
    assert res.returncode == 1
    assert "LEAK SCAN FAILED" in res.stderr
    assert "leak.md" in res.stdout  # the hit is named


def test_planted_token_and_hostname_refuse(tmp_path):
    (tmp_path / "a.txt").write_text("token " + "glpat-" + "abcdefghijklmnop\n")
    (tmp_path / "b.txt").write_text("host " + "alex" + ".nhr.fau.de\n")
    assert _scan(tmp_path).returncode == 1


def test_private_forge_url_refuses(tmp_path):
    """Pack policy: no private-forge reference at all — the pack ships to reviewers/Zenodo and
    every link must resolve publicly."""
    (tmp_path / "README.md").write_text(
        "mirror: https://" + "git5.cs" + ".fau.de/evemt/evemtbench-benchmark\n"
    )
    assert _scan(tmp_path).returncode == 1


def test_scan_errors_fail_closed(tmp_path):
    res = _scan(tmp_path / "does-not-exist")
    assert res.returncode == 1
    assert "LEAK SCAN FAILED" in res.stderr


def test_clean_dir_passes(tmp_path):
    (tmp_path / "ok.md").write_text("nothing to see: results are in outputs/1.0.0\n")
    res = _scan(tmp_path)
    assert res.returncode == 0
    assert "leak scan clean" in res.stdout
