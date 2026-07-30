"""Split-quality audit — the verification gate for stratified, leakage-free splits.

Given a train/val/test partition of episode ids and per-episode strata (e.g. ``event_type``):

* **episode-disjoint** train/val/test (no window/episode leakage across partitions);
* **fractions** within tolerance of the target 70/15/15;
* **class coverage** — every class with enough episodes appears in *both* val and test (so per-class
  metrics like HIF F1 are defined on the held-out sets, not silently absent);
* per-split **class histograms** + proportion drift (a stratified split keeps proportions stable).

``audit_partition`` is pure (unit-testable without data); ``audit_grid`` reads a committed split +
strata from the mounted dataset for the CLI freeze gate.

    python -m evemtbench.splits.audit                 # all registered grids
    python -m evemtbench.splits.audit --grids double_line --stratify-by event_type
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from ..data.roots import GRIDS, get_grid
from ..data.windows import read_episode_labels
from ..version import SPLITS_VERSION
from .loader import read_held_out_splits
from .protocols import HELD_OUT_FRACTIONS

_UNLABELED = "__unlabeled__"


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


def audit_grid(
    grid: str,
    family: str = "adapt_grid",
    *,
    root=None,
    splits_root=None,
    version: str = SPLITS_VERSION,
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


def main() -> None:
    p = argparse.ArgumentParser(
        description="Audit committed held_out splits (stratification/leakage)."
    )
    p.add_argument("--grids", nargs="*", default=None, help="grid keys (else all registered)")
    p.add_argument("--root", default=None)
    p.add_argument("--splits-root", default=None)
    p.add_argument("--stratify-by", default="event_type")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    all_ok = True
    for gk in a.grids or list(GRIDS):
        try:
            rep = audit_grid(gk, root=a.root, splits_root=a.splits_root, stratify_by=a.stratify_by)
        except Exception as e:  # noqa: BLE001 - report per grid, keep going
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
