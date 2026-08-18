"""Scientific consistency checker for the Supplementary Information. Generates NO artifact.

Asserts the invariants the SI's evidentiary value rests on, directly against the committed
artifacts — not against the rendered LaTeX, and not against rounded prose. Exits non-zero if any
critical invariant fails, so it can gate a release.

    PYTHONPATH=src python hpc/check_supplement_consistency.py
    PYTHONPATH=src python hpc/check_supplement_consistency.py --verbose
    make check-supplement

Groups of checks:

* **coverage** — the five primary functions exist at every defined (grid, view, family) cell; the
  per-baseline matrix is complete; transfer covers 3 sources x 4 targets x 2 views x 3 functions;
  every stochastic cell carries the five-seed protocol.
* **selection** — every headline/protection/transfer number is the value of a baseline chosen on a
  VALIDATION signal; no ``test`` or ``eval_grid`` score participates in any selection.
* **leakage** — the committed partitions have zero pairwise overlap at every declared grouping
  level, and the ``eval_grid`` family is disjoint from the whole ``adapt_grid`` family.
* **main-text claims** — the numbers the manuscript states are reproduced from the committed
  results, with tolerances matching the displayed precision.

Claim tolerances are deliberately loose enough for the manuscript's own rounding ("approximately
$0.87$--$0.98$"): a range claim passes if the observed range agrees with the quoted range to the
quoted number of decimals, not to machine precision.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO / "src", _REPO / "hpc", _REPO / "hpc" / "paper_artifacts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import paper_lib as pl  # noqa: E402
import supplement_lib as sl  # noqa: E402

from evemtbench.coverage import LEARNED_BASELINES, PRETRAIN_GRIDS  # noqa: E402
from make_results import BASELINES, load_rows, mean_val  # noqa: E402

PRIMARY = {
    "fault_detection": ("balanced_accuracy", pl.OBS),
    "fault_classification": ("balanced_accuracy", pl.OBS),
    "fault_location": ("mae", pl.OBS),
    "event_detection": ("balanced_accuracy", ("global",)),
    "event_classification": ("balanced_accuracy", ("global",)),
}
GRIDS = [g for g, _ in pl.GRID_ROWS]
#: Baselines whose results are seed-averaged (majority and threshold are deterministic).
STOCHASTIC = ("random_forest", "mlp", "gru", "cnn", "resnet")
N_SEEDS = 5


class Report:
    """Collects pass/fail lines; ``ok`` is false as soon as one check fails."""

    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose
        self.failures: list[str] = []
        self.n = 0

    def check(self, ok: bool, group: str, msg: str) -> bool:
        self.n += 1
        if not ok:
            self.failures.append(f"[{group}] {msg}")
        elif self.verbose:
            print(f"  ok   [{group}] {msg}")
        return ok

    def finish(self) -> int:
        print(f"\n{self.n - len(self.failures)}/{self.n} checks passed.")
        if self.failures:
            print(f"\n{len(self.failures)} FAILED:")
            for f in self.failures:
                print(f"  ✗ {f}")
            return 1
        print("All supplementary consistency invariants hold.")
        return 0


def _in_range(observed: tuple[float, float], quoted: tuple[float, float], nd: int) -> bool:
    """True if an observed [min, max] agrees with a quoted range at ``nd`` decimals.

    Rounding each end to the quoted precision is the right comparison for prose that says
    "approximately $0.87$--$0.98$": it must not be defeated by a value of 0.8701, nor satisfied by
    one of 0.83.
    """
    lo, hi = observed
    qlo, qhi = quoted
    return abs(round(lo, nd) - qlo) <= 10**-nd and abs(round(hi, nd) - qhi) <= 10**-nd


# --------------------------------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------------------------------
def check_coverage(rep: Report, per_grid: dict) -> None:
    for stem, (metric, views) in PRIMARY.items():
        for grid in GRIDS:
            for view in views:
                tid = f"{stem}_{view}"
                for ts in pl.TEST_SETS:
                    choice = pl.best_cell_choice(per_grid[grid], tid, ts, metric)
                    rep.check(
                        choice is not None, "coverage", f"headline cell present: {tid}/{grid}/{ts}"
                    )
    # full per-baseline matrix (S2): every non-structurally-excluded (baseline, cell) is populated
    missing = []
    for stem, (metric, views) in PRIMARY.items():
        for grid in GRIDS:
            for view in views:
                for baseline in BASELINES:
                    if pl._held_unsupported(stem, view, baseline):
                        continue
                    for ts in pl.TEST_SETS:
                        doc = per_grid[grid].get((f"{stem}_{view}", baseline), {}).get(ts)
                        if mean_val(doc, metric) is None:
                            missing.append(f"{stem}_{view}/{grid}/{baseline}/{ts}")
    rep.check(not missing, "coverage", f"S2 per-baseline matrix complete (missing: {missing[:5]})")
    # five-seed protocol on every stochastic cell
    bad_seeds = []
    for grid in GRIDS:
        for (tid, baseline), by_set in per_grid[grid].items():
            if baseline not in STOCHASTIC:
                continue
            for ts, doc in by_set.items():
                if ts == "val":
                    continue
                if len(doc.get("seeds") or []) != N_SEEDS:
                    bad_seeds.append(f"{tid}/{grid}/{baseline}/{ts}={len(doc.get('seeds') or [])}")
    rep.check(
        not bad_seeds, "coverage", f"five seeds on every stochastic cell (bad: {bad_seeds[:5]})"
    )


def check_transfer_coverage(rep: Report, idx: dict, val_idx: dict) -> None:
    missing = []
    for key, _lab, stem, metric, _arrow in sl.TRANSFER_FUNCTIONS:
        for grid in GRIDS:
            for source in PRETRAIN_GRIDS:
                for proto in ("transfer_zeroshot", "transfer_finetune"):
                    for view in sl.TRANSFER_VIEWS:
                        row = pl.best_transfer_row(
                            idx, f"{stem}_{view}", grid, proto, metric, source, val_idx
                        )
                        if row is None:
                            missing.append(f"{key}/{grid}/{source}/{proto}/{view}")
    n_expected = len(sl.TRANSFER_FUNCTIONS) * len(GRIDS) * len(PRETRAIN_GRIDS) * 2 * 2
    rep.check(
        not missing,
        "coverage",
        f"transfer matrix complete: {n_expected} cells "
        f"(3 functions x 4 targets x 3 sources x 2 protocols x 2 views); missing {missing[:5]}",
    )


# --------------------------------------------------------------------------------------------------
# Selection: nothing reported was chosen on the set it is reported on
# --------------------------------------------------------------------------------------------------
def check_selection(rep: Report, per_grid: dict, idx: dict, val_idx: dict) -> None:
    on_test = []
    for stem, (metric, views) in PRIMARY.items():
        for grid in GRIDS:
            for view in views:
                for ts in pl.TEST_SETS:
                    choice = pl.best_cell_choice(per_grid[grid], f"{stem}_{view}", ts, metric)
                    if choice and choice["selected_on"] != "val":
                        on_test.append(f"{stem}_{view}/{grid}/{ts}")
    rep.check(
        not on_test,
        "selection",
        f"every held_out headline selected on validation (select-on-report: {on_test[:5]})",
    )
    # the selected architecture must not depend on which held-out set is reported
    drift = []
    for stem, (metric, views) in PRIMARY.items():
        for grid in GRIDS:
            for view in views:
                picks = {
                    (pl.best_cell_choice(per_grid[grid], f"{stem}_{view}", ts, metric) or {}).get(
                        "baseline"
                    )
                    for ts in pl.TEST_SETS
                }
                picks.discard(None)
                if len(picks) > 1:
                    drift.append(f"{stem}_{view}/{grid}: {sorted(picks)}")
    rep.check(
        not drift,
        "selection",
        "the same validation-selected architecture is reported on both held-out sets "
        f"({drift[:3]})",
    )
    # transfer: zero-shot selects on the SOURCE corpus val, fine-tune on the TARGET val
    fallbacks = []
    for _key, _lab, stem, _metric, _arrow in sl.TRANSFER_FUNCTIONS:
        for grid in GRIDS:
            for source in PRETRAIN_GRIDS:
                for view in sl.TRANSFER_VIEWS:
                    tid = f"{stem}_{view}"
                    if not any(
                        val_idx.get(("multi_grid_pretrain", source, tid, b))
                        for b in LEARNED_BASELINES
                    ):
                        fallbacks.append(f"zeroshot {tid}/{source}")
                    if not any(
                        val_idx.get(("transfer_finetune", grid, tid, b, source))
                        for b in LEARNED_BASELINES
                    ):
                        fallbacks.append(f"finetune {tid}/{grid}/{source}")
    rep.check(
        not fallbacks,
        "selection",
        f"every transfer cell has a validation signal, so none falls back to "
        f"select-on-eval_grid (missing: {sorted(set(fallbacks))[:5]})",
    )
    # scratch reference: validation-selected, reported on eval_grid
    non_val = [
        f"{r['stem']}_{r['view']}/{r['grid']}"
        for r in sl.transfer_scratch_rows()
        if r["choice"] is None or r["choice"]["selected_on"] != "val"
    ]
    rep.check(
        not non_val,
        "selection",
        f"every from-scratch transfer reference is validation-selected (bad: {non_val[:5]})",
    )
    del idx


# --------------------------------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------------------------------
def check_leakage(rep: Report) -> None:
    for r in sl.partition_audit_rows():
        for level, ov in r["overlaps"].items():
            if ov is None:
                continue
            for pair, n in sorted(ov.items()):
                rep.check(n == 0, "leakage", f"{r['key']}: {level} overlap {pair} = {n}")
        if r["protocol"] == "held_out":
            rep.check(
                r["adapt_vs_eval_overlap"] == 0,
                "leakage",
                f"{r['key']}: adapt_grid ∩ eval_grid = {r['adapt_vs_eval_overlap']}",
            )
        else:
            rep.check(r["ok"], "leakage", f"{r['key']}: committed audit ok (issues: {r['issues']})")


# --------------------------------------------------------------------------------------------------
# Main-text claims
# --------------------------------------------------------------------------------------------------
def check_main_text(rep: Report, per_grid: dict, idx: dict, val_idx: dict) -> None:
    def held(grid, tid, ts, metric):
        c = pl.best_cell_choice(per_grid[grid], tid, ts, metric)
        return None if c is None else c["mean"]

    def xfer(stem, view, grid, proto, metric, source):
        row = pl.best_transfer_row(idx, f"{stem}_{view}", grid, proto, metric, source, val_idx)
        return None if row is None else row.mean(metric)

    # --- FL: Double Line line-view from-scratch ~5.2% vs fine-tuned ~8% -------------------------
    scratch = held("double_line", "fault_location_line", "benchmark", "mae")
    rep.check(
        scratch is not None and abs(scratch - 5.2) < 0.05,
        "main-text",
        f"Double Line line-view FL from-scratch eval_grid = {scratch:.3f}% (claimed ~5.2%)",
    )
    ft = [
        xfer("fault_location", "line", "double_line", "transfer_finetune", "mae", s)
        for s in PRETRAIN_GRIDS
    ]
    ft = [v for v in ft if v is not None]
    rep.check(
        bool(ft) and abs(min(ft) - 8.0) < 0.5,
        "main-text",
        f"Double Line line-view FL fine-tuned = {min(ft):.3f}-{max(ft):.3f}% (claimed ~8%)",
    )
    rep.check(
        bool(ft) and min(ft) > scratch,
        "main-text",
        "fine-tuning does not beat the from-scratch reference on Double Line line-view FL",
    )
    in_dist = held("double_line", "fault_location_line", "test", "mae")
    rep.check(
        in_dist is not None and abs(in_dist - 16.1) < 0.05,
        "main-text",
        f"Double Line line-view FL in-distribution = {in_dist:.3f}% (claimed 16.1%)",
    )
    trainmean = mean_val(
        per_grid["double_line"].get(("fault_location_line", "majority"), {}).get("benchmark"), "mae"
    )
    rep.check(
        trainmean is not None and abs(trainmean - 31) < 0.5,
        "main-text",
        f"Double Line eval_grid train-mean reference = {trainmean:.3f}% (claimed ~31%)",
    )

    # --- FD transfer ranges, excluding the large-voltage-gap failure regime --------------------
    zs = {}
    for grid in GRIDS:
        for source in PRETRAIN_GRIDS:
            for view in sl.TRANSFER_VIEWS:
                v = xfer(
                    "fault_detection", view, grid, "transfer_zeroshot", "balanced_accuracy", source
                )
                if v is not None:
                    zs[(grid, source, view)] = v
    collapsed = {k: v for k, v in zs.items() if v < 0.6}
    working = [v for v in zs.values() if v >= 0.6]
    rep.check(
        _in_range((min(working), max(working)), (0.87, 0.98), 2),
        "main-text",
        f"FD zero-shot outside the failure regime = {min(working):.3f}-{max(working):.3f} "
        "(claimed ~0.87-0.98)",
    )
    rep.check(
        all(abs(v - 0.5) < 0.02 for v in collapsed.values()),
        "main-text",
        f"the {len(collapsed)} collapsed FD zero-shot cells sit at chance: "
        f"{sorted(f'{g}/{s}/{vw}={v:.3f}' for (g, s, vw), v in collapsed.items())}",
    )
    ftd = [
        xfer("fault_detection", view, grid, "transfer_finetune", "balanced_accuracy", source)
        for grid in GRIDS
        for source in PRETRAIN_GRIDS
        for view in sl.TRANSFER_VIEWS
    ]
    ftd = [v for v in ftd if v is not None]
    rep.check(
        _in_range((min(ftd), max(ftd)), (0.95, 0.99), 2),
        "main-text",
        f"FD fine-tuned = {min(ftd):.3f}-{max(ftd):.3f} (claimed ~0.95-0.99)",
    )

    # --- FC: 345 kV-source line-view zero-shot on the three lower-voltage targets ---------------
    lower = ["cigre_mv", "double_line", "testgrid_110kv"]
    fc_zs = [
        xfer(
            "fault_classification",
            "line",
            g,
            "transfer_zeroshot",
            "balanced_accuracy",
            "template_345kv",
        )
        for g in lower
    ]
    fc_ft = [
        xfer(
            "fault_classification",
            "line",
            g,
            "transfer_finetune",
            "balanced_accuracy",
            "template_345kv",
        )
        for g in lower
    ]
    rep.check(
        _in_range((min(fc_zs), max(fc_zs)), (0.27, 0.36), 2),
        "main-text",
        f"FC 345 kV-source line zero-shot = {min(fc_zs):.3f}-{max(fc_zs):.3f} (claimed 0.27-0.36)",
    )
    rep.check(
        _in_range((min(fc_ft), max(fc_ft)), (0.66, 0.83), 2),
        "main-text",
        f"FC 345 kV-source line fine-tuned = {min(fc_ft):.3f}-{max(fc_ft):.3f} (claimed 0.66-0.83)",
    )

    # --- FL zero-shot under local observability, and its train-mean references -----------------
    fl_zs = [
        xfer("fault_location", "local", g, "transfer_zeroshot", "mae", s)
        for g in GRIDS
        for s in PRETRAIN_GRIDS
    ]
    fl_zs = [v for v in fl_zs if v is not None]
    rep.check(
        27.0 <= min(fl_zs) and max(fl_zs) <= 34.0,
        "main-text",
        f"FL local zero-shot = {min(fl_zs):.1f}-{max(fl_zs):.1f}% (claimed ~27-34%)",
    )
    tm = [
        mean_val(
            per_grid[g].get(("fault_location_local", "majority"), {}).get("benchmark"),
            "mae",
        )
        for g in GRIDS
    ]
    rep.check(
        30.0 <= min(tm) and max(tm) <= 31.5,
        "main-text",
        f"eval_grid train-mean references = {min(tm):.1f}-{max(tm):.1f}% (claimed ~30-31%)",
    )
    fl_ft = [
        xfer("fault_location", "line", g, "transfer_finetune", "mae", s)
        for g in ("double_line", "testgrid_110kv", "ieee39")
        for s in PRETRAIN_GRIDS
    ]
    fl_ft = [v for v in fl_ft if v is not None]
    rep.check(
        8.0 <= min(fl_ft) and max(fl_ft) <= 14.0,
        "main-text",
        f"FL HV/EHV line fine-tuned = {min(fl_ft):.1f}-{max(fl_ft):.1f}% (claimed ~8-14%)",
    )

    # --- protection grade: in-distribution MDR band and the always-positive collapse cells -----
    mdr = []
    for grid in ("cigre_mv", "double_line"):
        for view in pl.OBS:
            c = pl.best_cell_choice(
                per_grid[grid], f"fault_detection_{view}", "test", "balanced_accuracy"
            )
            doc = per_grid[grid].get((f"fault_detection_{view}", c["baseline"]), {}).get("test")
            mdr.append(mean_val(doc, "missed_fault_rate"))
    rep.check(
        _in_range((100 * min(mdr), 100 * max(mdr)), (19.0, 26.0), 0),
        "main-text",
        f"in-distribution FD MDR on MV+Double Line = {100 * min(mdr):.1f}-{100 * max(mdr):.1f}% "
        "(claimed ~19-26%)",
    )
    collapse_cells = [
        ("double_line", "fault_detection_local"),
        ("double_line", "fault_detection_global"),
        ("ieee39", "fault_detection_global"),
        ("ieee39", "event_detection_global"),
    ]
    for grid, tid in collapse_cells:
        mk, fk = pl._BINARY_RATE_KEYS[tid.rsplit("_", 1)[0]]
        c = pl.best_cell_choice(per_grid[grid], tid, "benchmark", "balanced_accuracy")
        doc = per_grid[grid].get((tid, c["baseline"]), {}).get("benchmark")
        bacc, m, f = c["mean"], mean_val(doc, mk), mean_val(doc, fk)
        rep.check(
            abs(bacc - 0.5) <= 0.01 and m <= 0.01 and f >= 0.98,
            "main-text",
            f"always-positive collapse {tid}/{grid}: bal.acc={bacc:.3f} (~0.5), "
            f"MDR={m:.4f} (~0), FAR={f:.4f} (~1)",
        )

    # --- the conventional threshold's robustness in shifted local FD cells ---------------------
    for grid, thr_claim, learned_claim in (("double_line", 0.916, 0.499), ("ieee39", 0.914, 0.886)):
        thr = mean_val(
            per_grid[grid].get(("fault_detection_local", "threshold"), {}).get("benchmark"),
            "balanced_accuracy",
        )
        learned = held(grid, "fault_detection_local", "benchmark", "balanced_accuracy")
        rep.check(
            abs(thr - thr_claim) < 0.001 and abs(learned - learned_claim) < 0.001,
            "main-text",
            f"{grid} local FD eval_grid: threshold={thr:.3f} (claimed {thr_claim}), "
            f"learned={learned:.3f} (claimed {learned_claim})",
        )

    # --- RF is the validation-selected reference in 8 of the 12 in-distribution FD cells --------
    n_rf = sum(
        1
        for grid in GRIDS
        for view in pl.OBS
        if (
            pl.best_cell_choice(
                per_grid[grid], f"fault_detection_{view}", "test", "balanced_accuracy"
            )
            or {}
        ).get("baseline")
        == "random_forest"
    )
    rep.check(n_rf == 8, "main-text", f"RF selected in {n_rf} of 12 in-distribution FD cells (8)")


# --------------------------------------------------------------------------------------------------
# Composition invariants (only where the dataset-backed artifacts have been generated)
# --------------------------------------------------------------------------------------------------
def check_composition(rep: Report, per_grid: dict, root=None) -> None:
    counts = sl.valid_counts()
    ec_counts, _vocab = sl.ec_support_counts(root=root)
    for grid in GRIDS:
        for ts in pl.TEST_SETS:
            n_ec = counts.get(("event_classification_global", grid, ts))
            rep.check(
                n_ec == sum(ec_counts[(grid, ts)].values()),
                "composition",
                f"EC class support sums to the evaluated n for {grid}/{ts}: "
                f"{sum(ec_counts[(grid, ts)].values())} vs {n_ec}",
            )
    comp = sl.ed_negative_composition(root=root)
    for (grid, ts), c in sorted(comp.items()):
        rep.check(
            c["pre-event"] + c["post-event"] + c["quiescent episode"] == c["negative"],
            "composition",
            f"ED negatives decompose by position for {grid}/{ts}",
        )
        rep.check(
            c["neg. in fault episodes"] + c["neg. in switching episodes"] + c["quiescent episode"]
            == c["negative"],
            "composition",
            f"ED negatives decompose by episode family for {grid}/{ts}",
        )
        rep.check(
            c["positive"] + c["negative"] == c["total"],
            "composition",
            f"ED positives + negatives = total for {grid}/{ts}",
        )
        rep.check(
            c["total"] == counts.get(("event_detection_global", grid, ts)),
            "composition",
            f"ED total matches the evaluated n for {grid}/{ts}",
        )
    del per_grid


#: What the pre-specified diagnostic reproduces, and what both the SI and the main text now state:
#: the steady-state current LEVEL is essentially unchanged between evaluation families while its
#: dispersion contracts sharply.
SI_CURRENT_RATIO_RANGE = (0.7, 1.1)
#: A retired, unreproducible claim: the manuscript previously asserted a "6--13x" raw current-scale
#: difference, which no reading of the shipped statistic supports (max observed ratio 3.61x over 48
#: grid x window-subset x aggregation combinations). Guarded so it cannot creep back in.
_RETIRED_CURRENT_CLAIM = re.compile(r"6\s*\$?--\$?\s*13\s*\\times")


def check_current_shift(rep: Report, root=None) -> None:
    """The raw current-scale diagnostic, and the retired claim it replaced.

    Asserts what the SI and the manuscript now say — the steady-state current level is comparable
    between families while its spread contracts — and guards against the reintroduction of the
    unreproducible "6--13x" figure that this diagnostic retired."""
    st = sl.current_scale_stats(root=root)
    ratios = {g: st[(g, "benchmark")]["median"] / st[(g, "train")]["median"] for g in GRIDS}
    lo, hi = min(ratios.values()), max(ratios.values())
    per_grid = ", ".join(f"{g}={r:.2f}x" for g, r in sorted(ratios.items()))
    rep.check(
        SI_CURRENT_RATIO_RANGE[0] <= lo and hi <= SI_CURRENT_RATIO_RANGE[1],
        "si-claim",
        f"steady-state current level is comparable between families ({lo:.2f}x-{hi:.2f}x, "
        f"SI states {SI_CURRENT_RATIO_RANGE[0]:g}-{SI_CURRENT_RATIO_RANGE[1]:g}x); per grid: "
        f"{per_grid}",
    )
    iqr = {
        g: (st[(g, "benchmark")]["q3"] - st[(g, "benchmark")]["q1"])
        / (st[(g, "train")]["q3"] - st[(g, "train")]["q1"])
        for g in GRIDS
    }
    rep.check(
        max(iqr.values()) < 1.0,
        "si-claim",
        "the operating-point interquartile range contracts on eval_grid for every grid: "
        + ", ".join(f"{g}={r:.2f}x" for g, r in sorted(iqr.items())),
    )


def check_retired_claims(rep: Report) -> None:
    """Guard the manuscript against reintroducing claims this analysis retired.

    Cheap (a text scan, no dataset), so it runs unconditionally rather than behind --dataset."""
    main_tex = _REPO / "paper" / "manuscript" / "main.tex"
    if not main_tex.is_file():
        return
    hits = _RETIRED_CURRENT_CLAIM.findall(main_tex.read_text())
    rep.check(
        not hits,
        "main-text",
        "the retired '6--13x' raw current-scale claim is absent from main.tex "
        f"(no reading of the shipped statistic exceeds 3.61x); found {len(hits)}",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verbose", action="store_true", help="print every passing check")
    ap.add_argument(
        "--dataset",
        action="store_true",
        help="also run the checks that read the released dataset (composition, current scale); "
        "needs $EVEMT_ROOT and, on the cluster, a SLURM allocation",
    )
    ap.add_argument("--root", default=None, help="dataset root override for --dataset")
    args = ap.parse_args(argv)

    rep = Report(args.verbose)
    per_grid = {g: load_rows(g) for g in GRIDS}
    idx = pl._transfer_index("outputs")
    val_idx = pl._transfer_val_index("outputs")

    check_coverage(rep, per_grid)
    check_transfer_coverage(rep, idx, val_idx)
    check_selection(rep, per_grid, idx, val_idx)
    check_leakage(rep)
    check_main_text(rep, per_grid, idx, val_idx)
    check_retired_claims(rep)
    if args.dataset:
        check_composition(rep, per_grid, root=args.root)
        check_current_shift(rep, root=args.root)
    else:
        print("(dataset-backed checks skipped; rerun with --dataset where the data is mounted)")
    return rep.finish()


if __name__ == "__main__":
    raise SystemExit(main())
