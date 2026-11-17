"""Every non-Python file the package reads at runtime ships in the wheel: each file under
src/evemtbench/ is matched by a [tool.setuptools.package-data] glob (roadmap W3). A new data file
that nobody adds to package-data would otherwise work in a checkout and break after
`pip install evemtbench`."""

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "src" / "evemtbench"


def test_every_package_file_is_covered_by_package_data():
    cfg = tomllib.loads((REPO / "pyproject.toml").read_text())
    globs = cfg["tool"]["setuptools"]["package-data"]["evemtbench"]
    shipped = {p for g in globs for p in PKG.glob(g) if p.is_file()}
    data_files = {
        p
        for p in PKG.rglob("*")
        if p.is_file() and p.suffix not in (".py", ".pyc") and "__pycache__" not in p.parts
    }
    missing = sorted(str(p.relative_to(PKG)) for p in data_files - shipped)
    assert not missing, f"not in package-data (would be missing from the wheel): {missing}"
