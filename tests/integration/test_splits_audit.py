"""Stratified split generator + audit verifier (pure — synthetic episodes, no dataset)."""

from __future__ import annotations

import pytest

from evemtbench.splits.audit import audit_partition
from evemtbench.splits.generator import split_episodes
from evemtbench.splits.protocols import HELD_OUT_FRACTIONS


def _synthetic(n_per_class: int, classes: list[str]):
    ids: list[int] = []
    strata: dict[int, str] = {}
    i = 0
    for c in classes:
        for _ in range(n_per_class):
            strata[i] = c
            ids.append(i)
            i += 1
    return ids, strata


def test_stratified_split_deterministic_and_covers_every_class():
    ids, strata = _synthetic(100, ["A", "B", "C"])  # 300 episodes, 3 classes
    a = split_episodes(ids, HELD_OUT_FRACTIONS, seed=7, strata=strata)
    b = split_episodes(ids, HELD_OUT_FRACTIONS, seed=7, strata=strata)
    assert a == b  # byte-deterministic

    rep = audit_partition(a, strata, min_per_class=7)
    assert rep["ok"], rep["issues"]
    assert rep["disjoint"] and rep["coverage_ok"] and rep["fractions_ok"]
    for split in ("val", "test"):  # every class present in val AND test
        assert {strata[i] for i in a[split]} == {"A", "B", "C"}


@pytest.mark.parametrize(
    ("m", "expected"),
    [
        (1, (1, 0, 0)),
        (2, (1, 0, 1)),
        (3, (2, 0, 1)),  # no val
        (4, (3, 1, 0)),  # no test
        (5, (4, 1, 0)),  # no test
        (6, (4, 1, 1)),  # first m present in all three
    ],
)
def test_stratified_split_small_classes(m, expected):
    """Pin the LIMIT of "rare classes appear in every split".

    Per-class sizes are rounded independently, so a class needs m >= 6 episodes to land in all three
    partitions. The docstrings used to promise presence in every split unconditionally, which is
    false below 6 — and would fail silently, since a class missing from a partition just reads n/a
    rather than wrong. No committed v1.0 split is affected (>= 31 episodes per class per partition);
    this test is the tripwire for a future grid with a genuinely rare class.

    The arithmetic is deliberately NOT changed: it would alter the committed splits.
    """
    ids, strata = _synthetic(m, ["R"])
    out = split_episodes(ids, HELD_OUT_FRACTIONS, seed=11, strata=strata)
    assert (len(out["train"]), len(out["val"]), len(out["test"])) == expected
    assert sum(len(out[s]) for s in ("train", "val", "test")) == m  # nothing dropped
    present_everywhere = all(out[s] for s in ("train", "val", "test"))
    assert present_everywhere == (m >= 6)


def test_uniform_split_is_the_default_and_deterministic():
    ids = list(range(50))
    a = split_episodes(ids, HELD_OUT_FRACTIONS, seed=3)
    b = split_episodes(ids, HELD_OUT_FRACTIONS, seed=3, strata=None)
    assert a == b  # strata=None is the default (committed splits unchanged)
    assert sum(len(v) for v in a.values()) == 50
    assert not (set(a["train"]) & set(a["test"]))  # disjoint


def test_stratified_split_respects_fractions():
    ids, strata = _synthetic(40, ["X", "Y"])  # 80 episodes
    rep = audit_partition(
        split_episodes(ids, HELD_OUT_FRACTIONS, seed=1, strata=strata), strata, min_per_class=7
    )
    assert rep["fractions_ok"] and rep["ok"]


def test_audit_flags_episode_leakage():
    strata = {i: ("A" if i < 10 else "B") for i in range(20)}
    parts = {  # episode 0 appears in both train and val -> leakage
        "train": list(range(0, 14)),
        "val": [0, *range(14, 17)],
        "test": list(range(17, 20)),
    }
    rep = audit_partition(parts, strata, min_per_class=3)
    assert not rep["disjoint"] and not rep["ok"]


def test_audit_flags_missing_class_in_heldout():
    # class B never reaches val/test although it has enough episodes -> coverage failure
    strata = {i: ("A" if i < 20 else "B") for i in range(40)}
    parts = {
        "train": list(range(20, 40)) + list(range(0, 14)),  # all of B in train
        "val": list(range(14, 17)),  # only A
        "test": list(range(17, 20)),  # only A
    }
    rep = audit_partition(parts, strata, min_per_class=7)
    assert not rep["coverage_ok"] and not rep["ok"]
    assert any("'B'" in issue for issue in rep["issues"])
