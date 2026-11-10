"""Split-quality audit — the verification gate for stratified, leakage-free splits.

Given a train/val/test partition of episode ids and per-episode strata (e.g. ``event_type``):

* **episode-disjoint** train/val/test (no window/episode leakage across partitions);
* **fractions** within tolerance of the target 70/15/15;
* **class coverage** — every class with enough episodes appears in *both* val and test (so per-class
  metrics like HIF F1 are defined on the held-out sets, not silently absent);
* per-split **class histograms** + proportion drift (a stratified split keeps proportions stable).

``audit_partition`` is pure (unit-testable without data); ``audit_grid`` reads a committed split +
strata from the mounted dataset for the CLI freeze gate.

    python -m evemtbench.splits.audit                 # every grid with a committed held_out split
    python -m evemtbench.splits.audit --grids double_line --stratify-by event_type
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from ..data.roots import get_grid
from ..data.windows import read_episode_labels
from .loader import read_held_out_splits, read_meta, read_splits, split_dir
from .protocols import HELD_OUT_FRACTIONS

_UNLABELED = "__unlabeled__"
_NO_EVENT = "no_event"  # event_type of the fault-free episodes (fault episodes = everything else)


def audit_partition(
    parts: dict[str, list[int]],
    strata: dict[int, str],
    *,
    fractions: tuple[float, float, float] = HELD_OUT_FRACTIONS,
    min_per_class: int = 7,
    frac_tol: float = 0.03,
) -> dict:
    """Audit a {train,val,test} episode-id partition against ``strata``. Returns a report dict with
    ``ok`` plus per-check detail. ``min_per_class`` is the episode count above which a class must
    appear in both val and test (below it, a 70/15/15 split cannot guarantee coverage)."""
    sets = {k: set(parts.get(k, [])) for k in ("train", "val", "test")}
    all_ids = sets["train"] | sets["val"] | sets["test"]
    n = len(all_ids)
    issues: list[str] = []

    disjoint = (
        not (sets["train"] & sets["val"])
        and not (sets["train"] & sets["test"])
        and not (sets["val"] & sets["test"])
    )
    if not disjoint:
        issues.append("train/val/test overlap (episode leakage)")

    fractions_ok = True
    for name, target in zip(("train", "val", "test"), fractions, strict=True):
        actual = len(sets[name]) / n if n else 0.0
        if abs(actual - target) > frac_tol:
            fractions_ok = False
            issues.append(f"{name} fraction {actual:.3f} off target {target:.3f}")

    total = Counter(strata.get(i, _UNLABELED) for i in all_ids)
    per_split = {
        name: Counter(strata.get(i, _UNLABELED) for i in sets[name])
        for name in ("train", "val", "test")
    }
    coverage_ok = True
    for cls, tot in sorted(total.items()):
        if tot >= min_per_class:
            for name in ("val", "test"):
                if per_split[name].get(cls, 0) == 0:
                    coverage_ok = False
                    issues.append(f"class {cls!r} (n={tot}) absent from {name}")

    return {
        "ok": disjoint and fractions_ok and coverage_ok,
        "n": n,
        "disjoint": disjoint,
        "fractions_ok": fractions_ok,
        "coverage_ok": coverage_ok,
        "issues": issues,
        "class_totals": dict(sorted(total.items())),
        "histograms": {k: dict(sorted(v.items())) for k, v in per_split.items()},
    }


def _overlap(a: set, b: set) -> int:
    return len(a & b)


def audit_grouped_partition(
    parts: dict[str, list[int]],
    *,
    strata: dict[int, str],
    simulations: dict[int, str],
    topologies: dict[int, str],
    grouping: str,
    fractions: tuple[float, float, float] = HELD_OUT_FRACTIONS,
    frac_tol: float = 0.05,
    min_per_class: int = 7,
) -> dict:
    """Audit a grouped {train,val,test} episode partition of a multi_grid source corpus.

    Pure (unit-testable without data). Beyond :func:`audit_partition`'s episode-level checks it
    reports, per partition: topology / simulation / episode / fault-episode counts and class
    support; and pairwise **overlap counts** for topology ids, simulation ids and episode ids.
    ``ok`` requires zero overlap at (and below) the declared ``grouping`` level and complete
    group metadata for every episode. Topology overlap under ``grouping="simulation"`` is
    reported but not failing (simulation grouping cannot prevent it — that is exactly the
    residual the topology-grouped split removes).

    ``frac_tol`` is looser than the episode-level audit's 0.03: whole groups are atomic, so a
    grouped split cannot hit the target fractions arbitrarily precisely.
    """
    names = ("train", "val", "test")
    sets = {k: set(parts.get(k, [])) for k in names}
    all_ids = sets["train"] | sets["val"] | sets["test"]
    issues: list[str] = []

    missing_meta = [i for i in sorted(all_ids) if i not in simulations or i not in topologies]
    if missing_meta:
        issues.append(
            f"{len(missing_meta)} episode(s) missing simulation/topology metadata "
            f"(first: {missing_meta[:5]})"
        )

    sim_sets = {k: {str(simulations[i]) for i in v if i in simulations} for k, v in sets.items()}
    topo_sets = {k: {str(topologies[i]) for i in v if i in topologies} for k, v in sets.items()}

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    overlaps = {
        "episode": {f"{a}&{b}": _overlap(sets[a], sets[b]) for a, b in pairs},
        "simulation": {f"{a}&{b}": _overlap(sim_sets[a], sim_sets[b]) for a, b in pairs},
        "topology": {f"{a}&{b}": _overlap(topo_sets[a], topo_sets[b]) for a, b in pairs},
    }
    # zero-overlap requirement at the declared grouping level and every level below it
    enforced = (
        ("episode", "simulation")
        if grouping == "simulation"
        else (
            "episode",
            "simulation",
            "topology",
        )
    )
    for level in enforced:
        bad = {k: v for k, v in overlaps[level].items() if v}
        if bad:
            issues.append(f"{level} overlap across partitions: {bad}")

    n = len(all_ids)
    fractions_ok = True
    for name, target in zip(names, fractions, strict=True):
        actual = len(sets[name]) / n if n else 0.0
        if abs(actual - target) > frac_tol:
            fractions_ok = False
            issues.append(f"{name} fraction {actual:.3f} off target {target:.3f}")

    per_split_hist = {
        name: Counter(strata.get(i, _UNLABELED) for i in sets[name]) for name in names
    }
    total = Counter(strata.get(i, _UNLABELED) for i in all_ids)
    # class -> number of distinct groups carrying it (the coverage upper bound under grouping)
    group_of = simulations if grouping == "simulation" else topologies
    seen: dict[str, set] = {}
    for i in all_ids:
        seen.setdefault(strata.get(i, _UNLABELED), set()).add(str(group_of.get(i)))
    class_groups = {c: len(g) for c, g in seen.items()}
    coverage_ok = True
    for cls, tot in sorted(total.items()):
        # a class needs >= 3 carrier groups to be able to appear in all three partitions;
        # below that, absence is a data property the audit reports without failing
        if tot >= min_per_class and class_groups.get(cls, 0) >= 3:
            for name in ("val", "test"):
                if per_split_hist[name].get(cls, 0) == 0:
                    coverage_ok = False
                    issues.append(
                        f"class {cls!r} (n={tot}, groups={class_groups.get(cls)}) "
                        f"absent from {name}"
                    )

    counts = {
        name: {
            "topologies": len(topo_sets[name]),
            "simulations": len(sim_sets[name]),
            "episodes": len(sets[name]),
            "fault_episodes": sum(1 for i in sets[name] if strata.get(i, _UNLABELED) != _NO_EVENT),
        }
        for name in names
    }

    return {
        "ok": not issues,
        "grouping": grouping,
        "n_episodes": n,
        "fractions_ok": fractions_ok,
        "coverage_ok": coverage_ok,
        "issues": issues,
        "counts": counts,
        "overlaps": overlaps,
        "class_totals": dict(sorted(total.items())),
        "class_n_groups": dict(sorted(class_groups.items())),
        "class_support": {k: dict(sorted(v.items())) for k, v in per_split_hist.items()},
    }


def audit_multi_grid(
    grid: str,
    *,
    protocol: str = "multi_grid_pretrain",
    family: str = "multi_grid",
    root=None,
    splits_root=None,
    version: str | None = None,
    stratify_by: str = "event_type",
    **kw,
) -> dict:
    """Full machine-readable audit of a committed multi_grid source-corpus split: the grouped
    partition audit plus the committed-file hashes, seed, grouping strategy and version — the
    per-voltage split-audit artifact. Fails (``ok=False``) if the meta is missing the grouping
    or seed fields, if any partition file is absent, or if any grouped check fails."""
    g = get_grid(grid)
    triples = read_splits(protocol, g.key, version=version, splits_root=splits_root)
    meta = read_meta(protocol, g.key, version=version, splits_root=splits_root)
    parts = {name: [t[2] for t in triples[name]] for name in ("train", "val", "test")}
    strata = read_episode_labels(g, family, column=stratify_by, root=root)
    simulations = read_episode_labels(g, family, column="result_id", root=root)
    topologies = read_episode_labels(g, family, column="graph_id", root=root)

    issues: list[str] = []
    grouping = meta.get("grouping")
    if grouping is None:
        issues.append("meta.json has no 'grouping' (episode-level split — sibling-line leakage)")
        grouping = "episode"
    for field in ("seed", "fractions", "splits_version"):
        if field not in meta:
            issues.append(f"meta.json missing {field!r}")

    d = split_dir(protocol, g.key, version=version, splits_root=splits_root)
    hashes: dict[str, str] = {}
    for name in ("train", "val", "test", "benchmark", "meta"):
        p = d / (f"{name}.txt" if name != "meta" else "meta.json")
        if not p.is_file():
            issues.append(f"partition file missing: {p.name}")
            continue
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()

    if grouping == "episode":
        report = audit_partition(parts, strata, **kw)
        report.setdefault("overlaps", None)
    else:
        report = audit_grouped_partition(
            parts,
            strata=strata,
            simulations=simulations,
            topologies=topologies,
            grouping=grouping,
            fractions=tuple(meta.get("fractions", HELD_OUT_FRACTIONS)),
            **kw,
        )
    report["ok"] = report["ok"] and not issues
    report["issues"] = issues + report["issues"]
    report.update(
        grid=g.key,
        protocol=protocol,
        family=family,
        splits_version=meta.get("splits_version"),
        seed=meta.get("seed"),
        stratify_by=meta.get("stratify_by"),
        group_column=meta.get("group_column"),
        partition_hashes=hashes,
    )
    return report


def write_multi_grid_audits(
    grids: list[str] | None = None,
    *,
    out_dir: str | Path | None = None,
    root=None,
    splits_root=None,
    version: str | None = None,
) -> dict[str, dict]:
    """Write ``audit.json`` next to each committed multi_grid split (the committed, machine-readable
    per-voltage split audit). Deterministic: a pure function of the committed split + the dataset
    metadata, so regeneration byte-compares."""
    from .generator import _GENERATED_PROTOCOLS

    grids = grids or _GENERATED_PROTOCOLS["multi_grid_pretrain"][2]
    reports: dict[str, dict] = {}
    for gk in grids:
        rep = audit_multi_grid(gk, root=root, splits_root=splits_root, version=version)
        d = (
            Path(out_dir) / gk
            if out_dir is not None
            else split_dir("multi_grid_pretrain", gk, version=version, splits_root=splits_root)
        )
        d.mkdir(parents=True, exist_ok=True)
        (d / "audit.json").write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
        reports[gk] = rep
    return reports


def audit_grid(
    grid: str,
    family: str = "adapt_grid",
    *,
    root=None,
    splits_root=None,
    version: str | None = None,
    stratify_by: str = "event_type",
    **kw,
) -> dict:
    """Audit a committed grid split against strata read from the mounted dataset."""
    g = get_grid(grid)
    triples = read_held_out_splits(g.key, version=version, splits_root=splits_root)
    parts = {name: [t[2] for t in triples[name]] for name in ("train", "val", "test")}
    strata = read_episode_labels(g, family, column=stratify_by, root=root)
    report = audit_partition(parts, strata, **kw)
    report["grid"] = g.key
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Audit committed splits (stratification/leakage; grouped for multi_grid)."
    )
    p.add_argument("--protocol", default="held_out", choices=["held_out", "multi_grid_pretrain"])
    p.add_argument(
        "--grids", nargs="*", default=None, help="grid keys (else the protocol's default grids)"
    )
    p.add_argument("--root", default=None)
    p.add_argument("--splits-root", default=None)
    p.add_argument("--splits-version", default=None, help="default: per-protocol resolution")
    p.add_argument("--stratify-by", default="event_type")
    p.add_argument("--json", action="store_true")
    p.add_argument(
        "--write",
        action="store_true",
        help="multi_grid_pretrain only: write audit.json next to each committed split",
    )
    a = p.parse_args(argv)
    all_ok = True
    # the grids that HAVE a committed split under this protocol (the template_* pre-training grids
    # have no held_out split, so iterating every registered grid could only ever fail)
    from .generator import _GENERATED_PROTOCOLS

    if a.protocol == "multi_grid_pretrain":
        grids = a.grids or _GENERATED_PROTOCOLS["multi_grid_pretrain"][2]
        if a.write:
            reports = write_multi_grid_audits(
                grids, root=a.root, splits_root=a.splits_root, version=a.splits_version
            )
        else:
            reports = {
                gk: audit_multi_grid(
                    gk, root=a.root, splits_root=a.splits_root, version=a.splits_version
                )
                for gk in grids
            }
        for gk, rep in reports.items():
            all_ok = all_ok and rep["ok"]
            if a.json:
                print(json.dumps(rep, indent=2, sort_keys=True))
            else:
                status = "OK" if rep["ok"] else "FAIL"
                print(
                    f"[{gk}] {status} grouping={rep.get('grouping')} n={rep.get('n_episodes')} "
                    f"counts={rep.get('counts')}"
                )
                for issue in rep["issues"]:
                    print(f"    - {issue}")
        raise SystemExit(0 if all_ok else 1)
    for gk in a.grids or _GENERATED_PROTOCOLS["held_out"][2]:
        try:
            rep = audit_grid(
                gk,
                root=a.root,
                splits_root=a.splits_root,
                version=a.splits_version,
                stratify_by=a.stratify_by,
            )
        except Exception as e:  # report per grid, keep going
            print(f"[{gk}] AUDIT ERROR: {e}")
            all_ok = False
            continue
        all_ok = all_ok and rep["ok"]
        if a.json:
            print(json.dumps(rep, indent=2))
        else:
            status = "OK" if rep["ok"] else "FAIL"
            print(
                f"[{gk}] {status} n={rep['n']} disjoint={rep['disjoint']} "
                f"fractions_ok={rep['fractions_ok']} coverage_ok={rep['coverage_ok']}"
            )
            for issue in rep["issues"]:
                print(f"    - {issue}")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
