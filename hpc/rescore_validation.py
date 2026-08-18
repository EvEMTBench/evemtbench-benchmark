#!/usr/bin/env python3
"""Offline validation re-scoring (review item F06), generalised across protocols.

The sweep exports metrics only for ``test`` and ``benchmark``; the validation split is used for
early stopping but never scored. Selecting the reported "best learned baseline" on the very test
set it reports is a winner's-curse (selection-on-test) bias. This pass computes the missing
VALIDATION metrics so the paper can select the reported baseline on ``val`` instead. Use
``--protocol`` to target the cells whose selection is being fixed:

  * ``held_out`` (default) — the main-text reference-baseline tables (F06);
  * ``transfer_finetune`` — the fine-tune transfer cells (select on the target grid's val);
  * ``multi_grid_pretrain`` — the pre-training cells whose val selects the grid-independent
    ``transfer_zeroshot`` winner (zero-shot has no target train/val of its own).

For each cell:

  * deep baselines (mlp/gru/cnn/resnet) are rebuilt from the saved best-val checkpoint via
    ``_transfer.reload_held_out`` (recomputes transform/stats/dims from ``task.train``; NO
    training);
  * ``random_forest`` is refit on ``task.train`` (deterministic, ``random_state=seed``) because it
    has no checkpoint.

POSITIVE CONTROL: every reloaded/refit model must reproduce the committed ``test`` headline metric
(per seed, within ``--tol``) before its ``val`` metric is accepted. A cell failing the control is
reported and SKIPPED (never written) — a wrong transform/dim/stat silently changes predictions and
the control is what catches it. Writes ``<cell>/val/result.json`` mirroring the test/benchmark
schema (``test_set="val"``), so ``make_results.load_rows`` picks it up automatically. Additive and
reversible; the frozen runner contract is untouched.

Compute — run via ``srun``, never the login node. Example (one cell, smoke test):

    srun --gres=gpu:a40:1 --time=20:00 bash -c \
      'PYTHONPATH=src python hpc/rescore_validation.py --task fault_detection_local \
       --grid double_line --limit 1'
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evemtbench.baselines import _transfer  # noqa: E402
from evemtbench.baselines.runner import _make_baseline  # noqa: E402
from evemtbench.coverage import LEARNED_BASELINES  # noqa: E402
from evemtbench.evaluation.evaluate import _nan_to_none, evaluate  # noqa: E402
from evemtbench.evaluation.significance import ci95_halfwidth  # noqa: E402
from evemtbench.tasks.task import load_task  # noqa: E402
from evemtbench.version import results_version  # noqa: E402

_DEEP = {"mlp", "gru", "cnn", "resnet"}


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _run_config(cell: Path) -> tuple[str, bool]:
    """The *resolved* preprocessing config the sweep actually used for this cell, read from its
    ``run_record.json``. ``args.normalize`` is often ``"auto"``, which the runner resolves
    data-dependently (``none`` for held_out, ``per_window`` for pretrain/transfer); reload does
    NOT re-resolve, so the reloaded model must be built with the RESOLVED value (top-level
    ``normalize``), or the positive control rejects every cell. Returns
    ``(normalize, zero_seq)``."""
    rec = json.load(open(cell / "run_record.json"))
    normalize = rec.get("normalize") or (rec.get("args") or {}).get("normalize") or "none"
    zero_seq = bool((rec.get("args") or {}).get("zero_seq", False))
    return normalize, zero_seq


def _committed_test_headline(cell: Path, headline: str) -> dict[int, float]:
    """{seed -> committed test headline metric} from the cell's test/result.json."""
    doc = json.load(open(cell / "test" / "result.json"))
    out = {}
    for r in doc.get("per_seed") or []:
        v = (r.get("metrics") or {}).get(headline)
        if v is not None:
            out[int(r["seed"])] = float(v)
    return out


def _aggregate(per_seed: list[dict]) -> dict:
    """Mirror the runner's aggregate schema: metric -> {mean,std,ci95,n_seeds,n_defined}."""
    keys: set[str] = set()
    for r in per_seed:
        keys |= set((r.get("metrics") or {}).keys())
    agg = {}
    n_seeds = len(per_seed)
    for k in sorted(keys):
        vals = [(r.get("metrics") or {}).get(k) for r in per_seed]
        defined = [float(v) for v in vals if v is not None]
        if defined:
            agg[k] = {
                "mean": float(np.mean(defined)),
                "std": float(np.std(defined, ddof=1)) if len(defined) > 1 else 0.0,
                "ci95": ci95_halfwidth(defined) if len(defined) > 1 else None,
                "n_seeds": n_seeds,
                "n_defined": len(defined),
            }
        else:
            agg[k] = {"mean": None, "std": None, "ci95": None, "n_seeds": n_seeds, "n_defined": 0}
    return agg


def rescore_cell(
    cell: Path, task_id: str, grid: str, baseline: str, *, tol: float, protocol: str = "held_out"
) -> str:
    """Re-score one (task, grid, baseline) cell on validation. Returns a status string.

    Supports ``held_out`` and the two protocols the transfer tables select on: ``transfer_finetune``
    (target ``adapt_grid`` val, used for its fine-tuning early stopping) and ``multi_grid_pretrain``
    (the pre-training-corpus val on which the grid-independent zero-shot winner is chosen). All
    three have a non-empty ``train`` split, so ``reload_held_out`` can recompute stats;
    ``transfer_zeroshot`` itself has NO target train/val and is never re-scored here — its winner
    is selected on the pretrain val."""
    if baseline not in LEARNED_BASELINES:
        return "skip (not a learned baseline)"
    # load task once; headline metric from its definition
    task = load_task(task_id, protocol, grid=grid)
    headline = task.definition.metrics[0]
    committed = _committed_test_headline(cell, headline)
    if not committed:
        return "skip (no committed test per-seed)"
    normalize, zero_seq = _run_config(cell)
    device = _device()
    per_seed_val: list[dict] = []
    for seed in sorted(committed):
        model = _make_baseline(
            baseline,
            seed=seed,
            device=device,
            epochs=60,
            batch_size=256,
            lr=1e-3,
            zero_seq=zero_seq,
            normalize=normalize,
        )
        if baseline in _DEEP:
            ckpt = cell / "checkpoints" / f"seed{seed}.pt"
            if not ckpt.is_file():
                return f"FAIL seed{seed}: checkpoint missing ({ckpt})"
            import torch

            if protocol == "multi_grid_pretrain":
                # pretrain saves a SELF-CONTAINED transfer bundle (state_dict + numpy stats/dims),
                # not a bare state_dict -> restore it directly (weights_only=False for the numpy
                # payload; the same path the transfer protocols use to reload the pretrained model).
                bundle = torch.load(ckpt, map_location=device, weights_only=False)
                _transfer.restore_pretrained(model, bundle)
            else:  # held_out / transfer_finetune: bare state_dict from runner._save_checkpoint
                state = torch.load(ckpt, map_location=device, weights_only=True)
                _transfer.reload_held_out(model, task, state)
        else:  # random_forest — refit (deterministic)
            model.fit(task)
        # positive control: reproduce committed test headline before trusting val. The tolerance is
        # scale-aware — absolute for O(1) metrics (balanced accuracy) but relative for larger-
        # magnitude ones (MAE %): a GRU's cuDNN-RNN reload wobbles the prediction ~1e-5 relative,
        # which stays <1e-3 on a 0-1 accuracy yet exceeds a fixed 1e-3 abs on a ~22% MAE headline
        # though the reload is exact. A real transform/dim/stat bug shifts it well beyond that.
        m_test = evaluate(task, model.predict(task.test), test_set="test").metrics.get(headline)
        tol_eff = tol * max(1.0, abs(committed[seed]))
        if m_test is None or abs(float(m_test) - committed[seed]) > tol_eff:
            return (
                f"FAIL positive-control seed{seed}: test {headline} reloaded={m_test} "
                f"committed={committed[seed]} (|Δ|>{tol_eff:g})"
            )
        val_metrics = {
            k: _nan_to_none(v)
            for k, v in evaluate(task, model.predict(task.val), test_set="val").metrics.items()
        }
        per_seed_val.append({"seed": seed, "metrics": val_metrics})
    # write val/result.json (mirrors the test/benchmark schema; load_rows picks it up by glob)
    out_dir = cell / "val"
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        # Stamped with the CELL TREE's version (outputs/<results_version(protocol)>/...), so the
        # val row stays version-consistent with the committed test/benchmark rows beside it.
        "benchmark_version": results_version(protocol),
        "task": task_id,
        "protocol": protocol,
        "grid": grid,
        "baseline": baseline,
        "test_set": "val",
        "seeds": [r["seed"] for r in per_seed_val],
        "aggregate": _aggregate(per_seed_val),
        "per_seed": per_seed_val,
        "note": "validation re-score (F06); reload/refit + positive control vs committed test.",
    }
    (out_dir / "result.json").write_text(json.dumps(doc, indent=1))
    val_mean = doc["aggregate"][headline]["mean"]
    return f"OK ({len(per_seed_val)} seeds, val {headline}={val_mean:.3f})"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default=None, help="filter: only this task_id")
    p.add_argument("--grid", default=None, help="filter: only this grid")
    p.add_argument("--baseline", default=None, help="filter: only this baseline")
    p.add_argument(
        "--source",
        default=None,
        help="filter: only this transfer pre-training source grid (e.g. template_110kv). Empty "
        "string matches the legacy source-less cells (the 20 kV default).",
    )
    p.add_argument("--limit", type=int, default=None, help="stop after N cells (smoke test)")
    p.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="positive-control tolerance: absolute for |metric|<=1 (accuracy), relative for larger "
        "magnitudes (MAE %); a fixed 1e-3 abs is too tight for MAE-scale headlines",
    )
    p.add_argument("--force", action="store_true", help="re-score even if val/result.json exists")
    p.add_argument(
        "--protocol",
        default="held_out",
        choices=["held_out", "transfer_finetune", "multi_grid_pretrain"],
        help="protocol whose cells to re-score on validation (transfer_zeroshot has no target "
        "train/val and is selected on the multi_grid_pretrain val instead)",
    )
    args = p.parse_args(argv)

    base = f"outputs/{results_version(args.protocol)}"  # the protocol's CURRENT results tree
    pats = [f"{base}/*/{args.protocol}/*/*/test/result.json"]  # legacy source-less path
    if args.protocol == "transfer_finetune":
        # non-default transfer sources live under a pretrain-<src> subdir (see coverage.cell_dir)
        pats.append(f"{base}/*/{args.protocol}/*/*/pretrain-*/test/result.json")
    cells = []
    for f in sorted({g for p in pats for g in glob.glob(p)}):
        parts = Path(f).parts
        if parts[-3].startswith("pretrain-"):
            # .../<task>/<protocol>/<grid>/<baseline>/pretrain-<source>/test/result.json
            source = parts[-3].removeprefix("pretrain-")
            baseline, grid, task_id = parts[-4], parts[-5], parts[-7]
        else:
            # .../<task>/<protocol>/<grid>/<baseline>/test/result.json
            source = None
            baseline, grid, task_id = parts[-3], parts[-4], parts[-6]
        if args.task and task_id != args.task:
            continue
        if args.grid and grid != args.grid:
            continue
        if args.baseline and baseline != args.baseline:
            continue
        if args.source is not None and (source or "") != args.source:
            continue
        if baseline not in LEARNED_BASELINES:
            continue
        cells.append((Path(f).parents[1], task_id, grid, baseline))

    done = fails = 0
    for cell, task_id, grid, baseline in cells:
        if not args.force and (cell / "val" / "result.json").is_file():
            print(f"[skip existing] {task_id}/{grid}/{baseline}")
            continue
        try:
            status = rescore_cell(
                cell, task_id, grid, baseline, tol=args.tol, protocol=args.protocol
            )
        except Exception as e:  # noqa: BLE001
            status = f"ERROR {type(e).__name__}: {e}"
        flag = "FAIL" in status or "ERROR" in status
        fails += flag
        done += not flag
        print(f"[{'FAIL' if flag else 'ok'}] {task_id}/{grid}/{baseline}: {status}", flush=True)
        if args.limit and (done + fails) >= args.limit:
            break
    print(f"\nrescore_validation: {done} ok, {fails} failed/error")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
