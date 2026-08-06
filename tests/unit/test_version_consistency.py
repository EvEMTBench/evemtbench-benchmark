"""Version identifiers must agree everywhere they appear — drift becomes a failing test.

The same contract version is declared in pyproject.toml ([project].version, PEP 440:
``0.2.0.dev0``), src/evemtbench/version.py (``BENCHMARK_VERSION``, semver-ish: ``0.2.0-dev``),
and CITATION.cff (``version:`` + ``date-released:`` once out of dev). CHANGELOG.md must carry a
section for the current version. docs/release.md step 4 relies on this test to force the three
files to move together."""

import re
from datetime import date
from pathlib import Path

try:
    import tomllib  # stdlib from Python 3.11
except ModuleNotFoundError:  # Python 3.10 (the CI floor): the API-compatible backport
    import tomli as tomllib  # type: ignore[no-redef]

from evemtbench.version import BENCHMARK_VERSION

REPO = Path(__file__).resolve().parents[2]


def _pyproject_version() -> str:
    return tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"]


def _base(version: str) -> tuple[str, bool]:
    """(X.Y.Z, is_dev) for either the PEP 440 (0.2.0.dev0) or semver-ish (0.2.0-dev) spelling."""
    m = re.fullmatch(r"(\d+\.\d+\.\d+)(?:[.-]dev\d*)?", version)
    assert m, f"unparseable version: {version!r}"
    return m.group(1), version != m.group(1)


def test_pyproject_matches_benchmark_version():
    assert _base(_pyproject_version()) == _base(BENCHMARK_VERSION), (
        f"pyproject {_pyproject_version()!r} and BENCHMARK_VERSION {BENCHMARK_VERSION!r} "
        "disagree — bump them together (docs/release.md step 4)"
    )


def test_citation_cff_matches():
    cff = (REPO / "CITATION.cff").read_text()
    m = re.search(r"^version:\s*\"?([0-9][\w.+-]*)\"?\s*$", cff, re.M)
    assert m, "CITATION.cff has no version: field"
    assert _base(m.group(1)) == _base(BENCHMARK_VERSION), (
        f"CITATION.cff version {m.group(1)!r} != BENCHMARK_VERSION {BENCHMARK_VERSION!r} — "
        "update CITATION.cff (docs/release.md step 4)"
    )


def test_citation_cff_has_a_release_date_on_release_versions():
    """docs/release.md step 4 moves CITATION.cff ``version`` AND ``date-released`` together, but
    only the version half was enforced here — so 1.0.0 shipped with no release date at all, which
    Zenodo and every other CFF consumer reads. Dev versions are exempt: they are not released."""
    _, is_dev = _base(BENCHMARK_VERSION)
    if is_dev:
        return
    cff = (REPO / "CITATION.cff").read_text()
    m = re.search(r"^date-released:\s*\"?(\d{4}-\d{2}-\d{2})\"?\s*$", cff, re.M)
    assert m, (
        f"CITATION.cff has no 'date-released: YYYY-MM-DD' but BENCHMARK_VERSION "
        f"{BENCHMARK_VERSION!r} is a release version (docs/release.md step 4)"
    )
    date.fromisoformat(m.group(1))  # raises on 2026-13-45 and friends


def test_changelog_has_a_section_for_the_current_version():
    base, _ = _base(BENCHMARK_VERSION)
    changelog = (REPO / "CHANGELOG.md").read_text()
    assert re.search(rf"^## {re.escape(base)}", changelog, re.M), (
        f"CHANGELOG.md has no '## {base}…' section for the current version"
    )


def test_frozen_tag_absent_on_dev_versions():
    """The paper freeze marker may only exist on a release version (the freshness gate enforces
    this too; the test makes it visible in CI, where the gate skips without outputs/)."""
    _, is_dev = _base(BENCHMARK_VERSION)
    marker = REPO / "paper" / "manuscript" / "generated" / "FROZEN_TAG"
    if is_dev:
        assert not marker.exists(), "FROZEN_TAG exists but BENCHMARK_VERSION is a dev version"
