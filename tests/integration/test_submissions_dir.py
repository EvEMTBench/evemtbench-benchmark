"""Every committed submission under submissions/ passes the validator (the CI gate for
submission pull requests), and the docs list every metadata field the validator knows."""

from pathlib import Path

import pytest

from evemtbench import submission as S

REPO = Path(__file__).resolve().parents[2]
SUBMISSIONS = sorted(p for p in (REPO / "submissions").glob("*") if p.is_dir())


@pytest.mark.parametrize("path", SUBMISSIONS, ids=[p.name for p in SUBMISSIONS])
def test_committed_submission_is_valid(path):
    rep = S.validate_submission(path, reference=S.default_reference(REPO))
    assert rep.ok, "\n".join(rep.errors)


def test_docs_list_every_metadata_field():
    doc = (REPO / "docs" / "submission.md").read_text()
    missing = [k for k in S.METADATA_FIELDS if f"`{k}`" not in doc]
    assert not missing, f"docs/submission.md does not document metadata fields {missing}"
