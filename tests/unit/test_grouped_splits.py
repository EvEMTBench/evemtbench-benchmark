"""Grouped source-corpus splits: group integrity, determinism, stratification, and the grouped
audit — the leakage guarantees behind the corrected ``multi_grid_pretrain`` v1.1 splits.

Pure unit tests (synthetic dicts, no dataset / no fixture I/O). The end-to-end path (generate on
the synthetic corpus -> write -> reload -> regenerate identically) lives in
``tests/integration/test_leakage.py``; the committed-artifact gates live there too.
"""

from __future__ import annotations

from collections import Counter

import pytest

from evemtbench.splits.audit import audit_grouped_partition
from evemtbench.splits.generator import split_grouped_episodes
from evemtbench.splits.leakage import LeakageError, assert_group_disjoint

FRACTIONS = (0.70, 0.15, 0.15)


def _synthetic(n_groups: int = 60, rare_every: int = 6):
    """~20-episode groups; a rare class confined to every ``rare_every``-th group."""
    eps, groups, strata = [], {}, {}
    eid = 0
    for g in range(n_groups):
        for k in range(20 + (g % 5)):
            eps.append(eid)
            groups[eid] = f"topo{g:03d}"
            strata[eid] = (
                "rare" if (g % rare_every == 0 and k < 2) else ("faultA" if k < 5 else "no_event")
            )
            eid += 1
    return eps, groups, strata


def test_groups_never_straddle_partitions():
    eps, groups, strata = _synthetic()
    parts = split_grouped_episodes(eps, groups, FRACTIONS, seed=7, strata=strata)
    owner: dict[str, str] = {}
    for name, ids in parts.items():
        for i in ids:
            assert owner.setdefault(groups[i], name) == name, "group split across partitions"
    # complete + episode-disjoint
    assert sorted(i for ids in parts.values() for i in ids) == sorted(eps)


def test_grouped_split_is_deterministic_and_seed_sensitive():
    eps, groups, strata = _synthetic()
    a = split_grouped_episodes(eps, groups, FRACTIONS, seed=7, strata=strata)
    b = split_grouped_episodes(eps, groups, FRACTIONS, seed=7, strata=strata)
    c = split_grouped_episodes(eps, groups, FRACTIONS, seed=8, strata=strata)
    assert a == b
    assert a != c


def test_grouped_fractions_near_target():
    eps, groups, strata = _synthetic()
    parts = split_grouped_episodes(eps, groups, FRACTIONS, seed=7, strata=strata)
    n = len(eps)
    for name, target in zip(("train", "val", "test"), FRACTIONS, strict=True):
        assert abs(len(parts[name]) / n - target) < 0.05


def test_grouped_stratification_spreads_rare_class():
    """A rare class carried by >= 3 groups must reach every partition (that is what the
    rarest-class-first assignment buys over plain size balancing)."""
    eps, groups, strata = _synthetic()
    parts = split_grouped_episodes(eps, groups, FRACTIONS, seed=7, strata=strata)
    for name in ("train", "val", "test"):
        hist = Counter(strata[i] for i in parts[name])
        assert hist["rare"] > 0, f"rare class absent from {name}"


def test_grouped_split_unstratified_still_groups():
    eps, groups, _ = _synthetic()
    parts = split_grouped_episodes(eps, groups, FRACTIONS, seed=7, strata=None)
    owner: dict[str, str] = {}
    for name, ids in parts.items():
        for i in ids:
            assert owner.setdefault(groups[i], name) == name


def test_grouped_split_rejects_missing_group_metadata():
    eps, groups, strata = _synthetic()
    del groups[eps[3]]
    with pytest.raises(ValueError, match="no group key"):
        split_grouped_episodes(eps, groups, FRACTIONS, seed=7, strata=strata)


def test_single_group_class_lands_wholesale_without_failing():
    """A class confined to ONE group cannot be spread — it must land in exactly one partition
    (documented data property, not an assignment bug)."""
    eps, groups, strata = _synthetic(rare_every=1000)  # only group 0 carries "rare"
    parts = split_grouped_episodes(eps, groups, FRACTIONS, seed=7, strata=strata)
    partitions_with_rare = [name for name in parts if any(strata[i] == "rare" for i in parts[name])]
    assert len(partitions_with_rare) == 1


# --- leakage validator ----------------------------------------------------------------------


def _triples(parts, groups):
    return {name: [("multi_grid", "template_20kv", i) for i in ids] for name, ids in parts.items()}


def test_assert_group_disjoint_passes_on_grouped_split():
    eps, groups, strata = _synthetic()
    parts = split_grouped_episodes(eps, groups, FRACTIONS, seed=7, strata=strata)
    assert_group_disjoint(_triples(parts, groups), groups, level="topology")  # must not raise


def test_assert_group_disjoint_catches_straddling_group():
    groups = {1: "gA", 2: "gA"}
    splits = {
        "train": [("multi_grid", "template_20kv", 1)],
        "val": [("multi_grid", "template_20kv", 2)],
        "test": [],
    }
    with pytest.raises(LeakageError, match="appears in both"):
        assert_group_disjoint(splits, groups, level="topology")


def test_assert_group_disjoint_catches_missing_metadata():
    splits = {"train": [("multi_grid", "template_20kv", 1)], "val": [], "test": []}
    with pytest.raises(LeakageError, match="metadata missing"):
        assert_group_disjoint(splits, {}, level="simulation")


# --- grouped audit ---------------------------------------------------------------------------


def _audit_inputs():
    eps, topos, strata = _synthetic()
    # two simulations per topology (nested consistently: sim -> exactly one topo)
    sims = {i: f"{topos[i]}_s{i % 2}" for i in eps}
    return eps, topos, sims, strata


def test_audit_grouped_partition_ok_and_zero_overlaps():
    eps, topos, sims, strata = _audit_inputs()
    parts = split_grouped_episodes(eps, topos, FRACTIONS, seed=7, strata=strata)
    rep = audit_grouped_partition(
        parts, strata=strata, simulations=sims, topologies=topos, grouping="topology"
    )
    assert rep["ok"], rep["issues"]
    for level in ("episode", "simulation", "topology"):
        assert all(v == 0 for v in rep["overlaps"][level].values()), (level, rep["overlaps"])
    for name in ("train", "val", "test"):
        c = rep["counts"][name]
        assert c["episodes"] > 0 and c["simulations"] > 0 and c["topologies"] > 0
        assert c["fault_episodes"] == sum(
            v for k, v in rep["class_support"][name].items() if k != "no_event"
        )


def test_audit_grouped_partition_flags_topology_overlap():
    eps, topos, sims, strata = _audit_inputs()
    # simulation-grouped assignment CAN split a topology; audit at topology grouping must fail
    parts = split_grouped_episodes(eps, sims, FRACTIONS, seed=7, strata=strata)
    rep = audit_grouped_partition(
        parts, strata=strata, simulations=sims, topologies=topos, grouping="topology"
    )
    assert not rep["ok"]
    assert any("topology overlap" in s for s in rep["issues"])
    # ... but passes when audited at its own (simulation) grouping level
    rep_sim = audit_grouped_partition(
        parts, strata=strata, simulations=sims, topologies=topos, grouping="simulation"
    )
    assert rep_sim["ok"], rep_sim["issues"]
    assert all(v == 0 for v in rep_sim["overlaps"]["simulation"].values())


def test_audit_grouped_partition_flags_missing_metadata():
    eps, topos, sims, strata = _audit_inputs()
    parts = split_grouped_episodes(eps, topos, FRACTIONS, seed=7, strata=strata)
    del sims[eps[0]]
    rep = audit_grouped_partition(
        parts, strata=strata, simulations=sims, topologies=topos, grouping="topology"
    )
    assert not rep["ok"]
    assert any("missing simulation/topology metadata" in s for s in rep["issues"])
