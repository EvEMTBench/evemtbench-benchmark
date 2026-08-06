"""Tarball discovery for the aux overlay (``evemtbench.data.extract_aux._find_tar``).

Regression: the v1.0 corpus ships ``<family>-<RawFolder>.tar.gz`` (and ``multigrid-…`` for the
template families), while the original glob matched ``<RawFolder>*`` only — a PREFIX. After the
dataset re-upload that silently found nothing, so every aux extraction raised FileNotFoundError.
The producer's naming has changed twice now, so pin all three conventions.
"""

import pytest

from evemtbench.data.extract_aux import _find_tar


def _root(tmp_path, files: dict[str, list[str]]):
    for family, names in files.items():
        d = tmp_path / family
        d.mkdir(parents=True, exist_ok=True)
        for n in names:
            (d / n).touch()
    return tmp_path


def test_current_family_prefixed_naming(tmp_path):
    """v1.0 layout: `<family>-<RawFolder>.tar.gz`, and `multigrid-` for template families."""
    root = _root(
        tmp_path,
        {
            "adapt_grid": ["adapt_grid-CigreMVGrid.tar.gz"],
            "benchmark": ["benchmark-IEEE39BusSystem.tar.gz"],
            "multi_grid": ["multigrid-Template20kV.tar.gz"],
        },
    )
    assert _find_tar("cigre_mv", "adapt_grid", root=root).name == "adapt_grid-CigreMVGrid.tar.gz"
    assert _find_tar("ieee39", "benchmark", root=root).name == "benchmark-IEEE39BusSystem.tar.gz"
    # the on-disk family is multi_grid but the producer writes "multigrid" in the tar name
    got = _find_tar("template_20kv", "multi_grid", root=root).name
    assert got == "multigrid-Template20kV.tar.gz"


def test_legacy_suffix_and_bare_naming(tmp_path):
    """Earlier layouts: `<RawFolder>_<family>.tar.gz`, or a bare `<RawFolder>.tar.gz`."""
    root = _root(
        tmp_path,
        {
            "benchmark": [
                "DoubleLine_benchmark.tar.gz",
                "DoubleLine_adapt_grid.tar.gz",
                "IEEE39BusSystem.tar.gz",
            ]
        },
    )
    # family named in the basename wins over a same-grid candidate for another family
    assert _find_tar("double_line", "benchmark", root=root).name == "DoubleLine_benchmark.tar.gz"
    # bare name still resolves (no family token to prefer)
    assert _find_tar("ieee39", "benchmark", root=root).name == "IEEE39BusSystem.tar.gz"


def test_missing_tarball_fails_loudly(tmp_path):
    root = _root(tmp_path, {"benchmark": []})
    with pytest.raises(FileNotFoundError, match="no tarball"):
        _find_tar("cigre_mv", "benchmark", root=root)
