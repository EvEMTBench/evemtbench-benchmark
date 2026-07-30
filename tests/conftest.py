"""Shared pytest fixtures: the synthetic DoubleLine fixture root + a held_out split for it."""

from __future__ import annotations

from pathlib import Path

import pytest

from evemtbench.version import SPLITS_VERSION

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "synthetic"


@pytest.fixture(scope="session")
def fixture_root() -> Path:
    if not FIXTURE_ROOT.is_dir():
        pytest.skip("synthetic fixture missing; run `python tests/fixtures/generate_fixture.py`")
    return FIXTURE_ROOT


@pytest.fixture(scope="session")
def held_out_splits(fixture_root, tmp_path_factory) -> Path:
    """Generate the held_out split for the fixture double_line into a tmp splits root."""
    from evemtbench.splits.generator import generate_held_out, write_held_out_artifact

    root = tmp_path_factory.mktemp("splits")
    art = generate_held_out("double_line", root=fixture_root)
    write_held_out_artifact(art, root, version=SPLITS_VERSION)
    return root
