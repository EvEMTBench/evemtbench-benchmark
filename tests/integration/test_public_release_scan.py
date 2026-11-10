"""The public-release gates (hpc/make_public_release.sh): the leak scan refuses forbidden
strings and FAILS CLOSED on scan errors (via --scan-only, so the tests drive the REAL shipped
patterns), the documented policy differences from the repro-pack scan hold (hostnames allowed,
private-forge URLs forbidden with the producer-schema $id exception applied per OCCURRENCE), and
every path in the prune set exists in the private repo — a rename would otherwise silently stop
being pruned.

Planted forbidden strings are assembled from adjacent string literals so this file itself passes
the scan (the scan covers the whole export, this file included — no exclusions)."""

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "hpc" / "make_public_release.sh"


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


def test_planted_account_and_cluster_path_refuse(tmp_path):
    (tmp_path / "a.txt").write_text("account " + "b304" + "dc submitted this\n")
    (tmp_path / "b.txt").write_text("legacy root " + "/clu" + "ster/someone/data\n")
    assert _scan(tmp_path).returncode == 1


def test_absolute_cluster_path_refuses_but_relative_segment_passes(tmp_path):
    # The cluster-path pattern targets absolute roots; tests/cluster/ is the repo's own layout
    # (the data_integrity suite) and is cited in the public docs.
    for absolute in (
        'cd "/clu' + 'ster/x/y"',
        "root=/clu" + "ster/x",
        "/clu" + "ster/x at line start",
    ):
        (tmp_path / "a.md").write_text(absolute + "\n")
        assert _scan(tmp_path).returncode == 1, absolute
    (tmp_path / "a.md").write_text(
        "run `pytest -m data_integrity tests/" + "cluster/test_x.py` (see ./" + "cluster/)\n"
    )
    assert _scan(tmp_path).returncode == 0


def test_planted_preview_token_refuses(tmp_path):
    (tmp_path / "a.md").write_text(
        "https://zenodo.org/records/1?preview=1&" + "token=ey" + "JhbGciOiJIUzUxMiJ9abcdef\n"
    )
    assert _scan(tmp_path).returncode == 1


def test_private_forge_url_refuses_but_schema_id_passes(tmp_path):
    (tmp_path / "bad.md").write_text(
        "see https://" + "git5.cs" + ".fau.de/juoelhaf/evemtbench-benchmark\n"
    )
    assert _scan(tmp_path).returncode == 1
    (tmp_path / "bad.md").unlink()
    (tmp_path / "schema.json").write_text(
        '{"$id": "https://git5.cs.fau.de/juoelhaf/power-grid-and-ai-helper-functions'
        '/contract/manifest.schema.json"}\n'
    )
    assert _scan(tmp_path).returncode == 0


def test_allowlisted_url_cannot_mask_a_same_line_secret(tmp_path):
    """The allowlist is applied per matched occurrence, never by dropping whole lines — a
    forbidden string co-located with the allowlisted schema $id (e.g. minified JSON) still
    refuses."""
    (tmp_path / "minified.json").write_text(
        '{"$id": "https://git5.cs.fau.de/juoelhaf/power-grid-and-ai-helper-functions'
        '/contract/manifest.schema.json", "root": "' + "/home/" + 'atuin/somewhere/secret"}\n'
    )
    res = _scan(tmp_path)
    assert res.returncode == 1
    assert "minified.json" in res.stdout


def test_scan_errors_fail_closed(tmp_path):
    """A grep hard error (missing dir, unreadable file) must refuse, never report clean."""
    import os

    res = _scan(tmp_path / "does-not-exist")
    assert res.returncode == 1
    assert "LEAK SCAN FAILED" in res.stderr
    if os.geteuid() == 0:  # root reads anything — the unreadable-file case can't be staged
        return
    unreadable = tmp_path / "unreadable.txt"
    unreadable.write_text("nothing forbidden\n")
    unreadable.chmod(0o000)
    try:
        assert _scan(tmp_path).returncode == 1
    finally:
        unreadable.chmod(0o644)


def test_hostnames_are_allowed(tmp_path):
    """Policy difference vs the repro-pack scan: NHR node/proxy hostnames are public
    infrastructure and appear in provenance and docs — they must pass."""
    (tmp_path / "prov.json").write_text(
        '{"hostname": "a0605", "proxy": "http://proxy.nhr.fau.de:80"}\n'
    )
    res = _scan(tmp_path)
    assert res.returncode == 0
    assert "leak scan clean" in res.stdout


def test_prune_set_paths_exist_and_cover_the_private_trees():
    import pytest

    if not (REPO / "paper").is_dir():
        pytest.skip("public release clone — the prune set is absent by construction")
    res = subprocess.run(
        ["bash", str(SCRIPT), "--print-prune"], capture_output=True, text=True, check=True
    )
    prune = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
    assert prune, "--print-prune returned nothing"
    for p in prune:
        assert (REPO / p).exists(), f"prune set names a missing path (renamed?): {p}"
    must_prune = {"paper", "docs/internal", "CLAUDE.md", ".claude", "ONBOARDING.md"}
    assert must_prune <= set(prune), f"missing critical prune entries: {must_prune - set(prune)}"
