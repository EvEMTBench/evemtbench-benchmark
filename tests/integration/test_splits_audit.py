"""Stratified split generator + audit verifier (pure — synthetic episodes, no dataset)."""

from __future__ import annotations

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
