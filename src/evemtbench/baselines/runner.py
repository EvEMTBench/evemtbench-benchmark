"""``evemtbench-run`` — train a baseline over seeds, evaluate, emit reproducible artifacts.

For each seed: fit the baseline on ``task.train`` (never test), predict ``task.test``, evaluate
with the official metrics. Across seeds, report mean ± std + a 95% CI (t-based). Emits, under
``outputs/<benchmark_version>/<task>/<protocol>/<grid>/``:

* ``predictions_seed{n}.parquet`` — the canonical predictions file (auditable by the CLI),
* ``result.json``                 — per-seed + aggregated metrics + stratified breakdowns,
* ``run_record.json``             — seeds, git sha, lockfile hash, env versions, args, and the
  no-tuning attestation (fit on train, select on val, held-out sets predict-only).

The same entrypoint is invoked by the LME and NHR SLURM scripts; only site headers differ.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..coverage import PRETRAIN_GRID
from ..evaluation.evaluate import EvalResult, _nan_to_none, evaluate
from ..evaluation.metrics import METRIC_DIRECTION
from ..evaluation.predictions_io import predictions_frame
from ..evaluation.significance import ci95_halfwidth
from ..splits.protocols import TRANSFER_PROTOCOLS, default_input_normalize
from ..tasks.task import load_task
from ..version import BENCHMARK_VERSION, SPLITS_VERSION
from .interface import Baseline

# ---- baseline registry (BYO plugin point, built on the Baseline Protocol) --------------------
# name -> factory(seed=..., device=..., epochs=..., batch_size=..., lr=..., zero_seq=...).
# A factory receives every runner hyperparameter as a keyword and may ignore any (accept **_).
# Third parties call register_baseline() BEFORE runner.main() — see examples/byo_model.py.
BASELINE_FACTORIES: dict[str, Callable[..., Baseline]] = {}


def register_baseline(name: str, factory: Callable[..., Baseline]) -> None:
    """Register a third-party baseline under ``name`` for ``--baseline name``.

    ``factory(**hyperparams)`` must return an object satisfying the
    :class:`~evemtbench.baselines.interface.Baseline` protocol (``fit(task)`` on train/val only;
    ``predict(view)`` returning the canonical predictions frame). Duplicate names raise —
    the built-in names are reserved."""
    if name in BASELINE_FACTORIES:
        raise ValueError(f"baseline {name!r} is already registered")
    BASELINE_FACTORIES[name] = factory


def _deep_factory(module: str, cls_name: str) -> Callable[..., Baseline]:
    """Factory for the deep baselines (shared signature; torch import stays lazy)."""

    def make(*, seed, device, epochs, batch_size, lr, zero_seq, normalize="none", **_):
        import importlib

        cls = getattr(importlib.import_module(f".{module}", __package__), cls_name)
        return cls(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            seed=seed,
            zero_seq=zero_seq,
            normalize=normalize,
        )

    return make


def _majority_factory(**_) -> Baseline:
    from .majority import MajorityBaseline

    return MajorityBaseline()


def _threshold_factory(*, seed, **_) -> Baseline:
    from .threshold import ThresholdBaseline

    return ThresholdBaseline(seed=seed)


def _random_forest_factory(*, seed, **_) -> Baseline:
    from .random_forest import RandomForestBaseline

    return RandomForestBaseline(seed=seed)


BASELINE_FACTORIES.update(
    {
        "majority": _majority_factory,
        "threshold": _threshold_factory,
        "random_forest": _random_forest_factory,
        "mlp": _deep_factory("mlp", "MLPBaseline"),
        "gru": _deep_factory("gru", "GRUBaseline"),
        "cnn": _deep_factory("cnn", "CNNBaseline"),
        "resnet": _deep_factory("resnet", "ResNetBaseline"),
    }
)


def _make_baseline(
    name: str,
    *,
    seed: int,
    device: str | None,
    epochs: int,
    batch_size: int,
    lr: float,
    zero_seq: bool = False,
    normalize: str = "none",
) -> Baseline:
    factory = BASELINE_FACTORIES.get(name)
    if factory is None:
        raise ValueError(
            f"unknown baseline {name!r} (known: {', '.join(sorted(BASELINE_FACTORIES))})"
        )
    return factory(
        seed=seed,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        zero_seq=zero_seq,
        normalize=normalize,
    )


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _lockfile_hash() -> str | None:
    lock = Path(__file__).resolve().parents[3] / "requirements.lock"
    if lock.is_file():
        return hashlib.sha256(lock.read_bytes()).hexdigest()
    return None


def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _aggregate(per_seed_metrics: list[dict[str, float]]) -> dict[str, dict[str, float | None]]:
    keys = per_seed_metrics[0].keys()
    summary: dict[str, dict[str, float | None]] = {}
    for k in keys:
        vals = np.array([m[k] for m in per_seed_metrics], dtype=float)
        finite = vals[np.isfinite(vals)]
        n_defined, n_seeds = int(finite.size), int(vals.size)
        if n_defined == 0:
            # Metric is structurally undefined for this whole partition across every seed
            # (e.g. balanced_accuracy on a single-class subset). Record nulls + n_defined=0
            # so the result distinguishes "undefined" from a genuine 0.0, and skip the
            # all-NaN nanmean/nanstd that would otherwise warn "Mean of empty slice".
            summary[k] = {
                "mean": None,
                "std": None,
                "ci95": None,
                "n_defined": 0,
                "n_seeds": n_seeds,
            }
        else:
            # Average over the defined seeds only (no nan-aware reducer, hence no warning).
            # ci95: t-based half-width over the defined seeds (ddof=1 — the stored std stays
            # population/ddof=0). None when < 2 seeds: no seed-variance estimate, never a fake ±0.
            summary[k] = {
                "mean": float(finite.mean()),
                "std": float(finite.std()),
                "ci95": ci95_halfwidth(finite),
                "n_defined": n_defined,
                "n_seeds": n_seeds,
            }
    return summary


def _save_checkpoint(model, base_dir: Path, seed: int) -> dict | None:
    """Persist a deep baseline's best-val weights (already restored in ``fit()``); ``None`` for a
    classical baseline (no ``.model`` / ``state_dict``).

    Writes ``<cell>/checkpoints/seed{seed}.pt`` (fp32 — small vs. the matrix, reloads without a
    cast) and returns ``{path, sha256}`` (path relative to the cell dir where ``run_record.json``
    lives) for provenance. Released checkpoints let robustness/latency/probing analyses be added
    later WITHOUT re-training. ``outputs/`` is git-ignored, so the binaries are never committed."""
    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "state_dict"):
        return None
    import torch

    (base_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    rel = f"checkpoints/seed{seed}.pt"
    path = base_dir / rel
    torch.save(inner.state_dict(), path)
    return {"path": rel, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _save_transfer_bundle(model, base_dir: Path, seed: int) -> dict | None:
    """Persist a self-contained transfer bundle (weights + dims/preprocessing) for a fitted deep
    baseline, so the transfer protocols can rebuild it on another grid. ``None`` for classical
    baselines (nothing transferable). Written to the same ``checkpoints/seed{seed}.pt`` path the
    transfer runner resolves."""
    if not hasattr(model, "_build_inner") or getattr(model, "model", None) is None:
        return None
    import torch

    from . import _transfer  # lazy: pulls in torch; classical runs never reach here

    (base_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    rel = f"checkpoints/seed{seed}.pt"
    path = base_dir / rel
    torch.save(_transfer.transfer_bundle(model), path)
    return {"path": rel, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _pretrained_bundle_path(args, seed: int, baseline_tag: str) -> Path:
    """Resolve the multi_grid_pretrain checkpoint the transfer protocols initialise from: an
    explicit ``--pretrained-from`` (a .pt file or a checkpoints/ dir), else the matching
    ``multi_grid_pretrain`` cell for this task/baseline/seed on ``--pretrain-grid``."""
    if args.pretrained_from:
        base = Path(args.pretrained_from)
        return base if base.suffix == ".pt" else base / f"seed{seed}.pt"
    return (
        Path(args.out)
        / BENCHMARK_VERSION
        / args.task
        / "multi_grid_pretrain"
        / args.pretrain_grid
        / baseline_tag
        / "checkpoints"
        / f"seed{seed}.pt"
    )


def _harness_control(task, ts: str) -> dict:
    """The per-run positive control (UnderstandingVCS runner-selftest idiom): feed the TRUE
    labels through the identical evaluate() path and require the headline metric to hit its
    ceiling. If the oracle cannot score perfectly, the harness — alignment, label casting,
    metric wiring — is broken, and the cell must fail loudly rather than persist results.

    Returns {headline, value, ceiling, ok}; raises RuntimeError when the ceiling is missed."""
    view = task.view(ts)
    headline = task.definition.metrics[0]
    frame = predictions_frame(view.metadata["sample_id"], view.metadata["window_idx"], view.labels)
    value = evaluate(task, frame, test_set=ts).metrics[headline]
    ceiling = 0.0 if METRIC_DIRECTION.get(headline, 1) < 0 else 1.0
    ok = value is not None and abs(float(value) - ceiling) <= 1e-9
    if not ok:
        raise RuntimeError(
            f"harness positive control FAILED for {task.definition.task_id}/{ts}: true labels "
            f"scored {headline}={value!r}, expected {ceiling} — the scoring harness (not any "
            "model) is broken; refusing to write results"
        )
    return {"headline": headline, "value": float(value), "ceiling": ceiling, "ok": True}


def _agg_compute(per_seed: list[dict], n_params: int | None) -> dict:
    """Aggregate per-seed compute telemetry for a cell: constant ``n_params`` (architecture fixed) +
    training cost across seeds (mean/total wall-time, mean effective epochs, max peak GPU mem)."""

    def _vals(key: str) -> list:
        return [r[key] for r in per_seed if r.get(key) is not None]

    fit_s, eff, peak = _vals("fit_seconds"), _vals("effective_epochs"), _vals("peak_mem_bytes")
    return {
        "n_params": n_params,
        "n_seeds": len(per_seed),
        "fit_seconds_mean": float(np.mean(fit_s)) if fit_s else None,
        "fit_seconds_total": float(np.sum(fit_s)) if fit_s else None,
        "effective_epochs_mean": float(np.mean(eff)) if eff else None,
        "peak_mem_bytes_max": int(max(peak)) if peak else None,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train + evaluate a benchmark baseline over seeds.")
    p.add_argument("--task", default="fault_detection_local")
    p.add_argument("--protocol", default="held_out")
    p.add_argument("--grid", default=None)
    # choices are resolved at main() call time, so baselines registered via register_baseline()
    # before main() (the BYO plugin path) are accepted alongside the built-ins.
    p.add_argument("--baseline", default="majority", choices=sorted(BASELINE_FACTORIES))
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default=None)
    p.add_argument("--root", default=None, help="data root override")
    p.add_argument("--splits-root", default=None)
    p.add_argument("--directory", default=None, help="explicit windows dir (tests/fixtures)")
    p.add_argument("--out", default="outputs", help="output root directory")
    p.add_argument(
        "--zero-seq",
        action="store_true",
        default=False,
        help="append per-cubicle zero-sequence (I0, U0) channels before feeding the model",
    )
    p.add_argument(
        "--normalize",
        choices=("auto", "none", "per_window"),
        default="auto",
        help="input normalization. 'per_window' = per-window z-score (voltage-invariant, the "
        "cross-grid transfer default); 'auto' resolves to per_window for the transfer/pretrain "
        "protocols and none for held_out.",
    )
    p.add_argument(
        "--pretrain-grid",
        default="template_20kv",
        help="grid whose multi_grid_pretrain checkpoint the transfer protocols initialise from",
    )
    p.add_argument(
        "--pretrained-from",
        default=None,
        help="explicit path to the pretrained checkpoint (.pt file or a checkpoints/ dir); "
        "default resolves the multi_grid_pretrain cell for the same task/baseline/seed",
    )
    args = p.parse_args(argv)
    normalize = (
        default_input_normalize(args.protocol) if args.normalize == "auto" else args.normalize
    )

    task = load_task(
        args.task,
        args.protocol,
        grid=args.grid,
        root=args.root,
        splits_root=args.splits_root,
        directory=args.directory,
    )
    # Projection cache (if enabled): materialize ALL views — including the eval views (test,
    # benchmark) — so the per-seed predict passes read the small cached projection instead of
    # re-reading the full on-disk channels (the benchmark family lives on the NFS mount) once per
    # seed per baseline. Build once here; cheap no-op on subsequent seeds/baselines (file exists).
    cdir = os.environ.get("EVEMTBENCH_PROJECTION_CACHE")
    if cdir:
        for v in (task.train, task.val, task.test, task.benchmark):
            v.materialize_cache(cdir)
    # The held-out test sets: in-distribution (adapt_grid) + the benchmark family. A grid whose
    # benchmark isn't windowed yet has an empty benchmark view -> evaluate in-distribution only.
    test_sets = [ts for ts in ("test", "benchmark") if len(task.view(ts)) > 0]
    print(f"evaluating test sets: {test_sets}")
    baseline_tag = args.baseline + ("+zero_seq" if args.zero_seq else "")
    base_dir = (
        Path(args.out)
        / BENCHMARK_VERSION
        / task.definition.task_id
        / task.protocol
        / task.grid
        / baseline_tag
    )
    # Transfer from a NON-default source lands in a source-qualified subdir (matches
    # coverage.cell_dir); the default (PRETRAIN_GRID) source keeps the legacy source-less path, so
    # the frozen 20 kV-sourced transfer results stay byte-identical.
    if args.protocol in TRANSFER_PROTOCOLS and args.pretrain_grid != PRETRAIN_GRID:
        base_dir = base_dir / f"pretrain-{args.pretrain_grid}"

    # --- Logging redundancy (v1.0): a per-cell durable train.log (OFFLINE, cell-addressable — the
    # SLURM .out is job-named and interleaves many cells) plus an optional W&B run (ONLINE). The run
    # maps 1:1 to this cell; its deterministic id/url is recorded in run_record.json below for the
    # offline<->online cross-link. Both hooks no-op safely when W&B is disabled/unavailable. ---
    base_dir.mkdir(parents=True, exist_ok=True)
    from . import _tracking
    from ._log import set_log_file

    set_log_file(base_dir / "train.log")
    cell_key = (
        f"{BENCHMARK_VERSION}/{task.definition.task_id}/{task.protocol}/{task.grid}/{baseline_tag}"
    )
    _cluster = os.environ.get("SLURM_CLUSTER_NAME") or os.environ.get("CLUSTER")
    wandb_info = _tracking.start_run(
        cell_key=cell_key,
        name=f"{task.definition.task_id}/{task.protocol}/{task.grid}/{baseline_tag}",
        config={
            **vars(args),
            "benchmark_version": BENCHMARK_VERSION,
            "splits_version": SPLITS_VERSION,
            "git_sha": _git_sha(),
            "lockfile_sha256": _lockfile_hash(),
            "normalize": normalize,
            "cluster": _cluster,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        group=task.definition.task_id,
        job_type=baseline_tag,
        tags=[
            task.definition.task_id,
            task.protocol,
            task.grid,
            baseline_tag,
            f"v{BENCHMARK_VERSION}",
            _cluster,
            (_git_sha() or "")[:8],
        ],
    )

    # Protocol dispatch. Transfer protocols initialise from a multi_grid_pretrain checkpoint:
    # zero-shot rebuilds + predicts (NO target training); fine-tune loads the weights then trains
    # on the target adapt_grid. multi_grid_pretrain/held_out train from scratch.
    is_transfer = args.protocol in TRANSFER_PROTOCOLS
    is_zeroshot = args.protocol == "transfer_zeroshot"
    is_pretrain = args.protocol == "multi_grid_pretrain"
    pretrained_from_used: str | None = None

    per_seed_metrics: dict[str, list[dict[str, float]]] = {ts: [] for ts in test_sets}
    per_seed_results: dict[str, list[dict]] = {ts: [] for ts in test_sets}
    last_result: dict[str, EvalResult | None] = {ts: None for ts in test_sets}
    per_seed_compute: list[dict] = []
    checkpoints: dict[str, dict] = {}
    n_params: int | None = None
    for seed in args.seeds:
        model = _make_baseline(
            args.baseline,
            seed=seed,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            zero_seq=args.zero_seq,
            normalize=normalize,
        )
        if is_transfer:
            import torch

            from . import _transfer  # lazy: pulls in torch; transfer implies a deep baseline

            if not hasattr(model, "_build_inner"):
                raise ValueError(
                    f"protocol {args.protocol!r} requires a deep baseline with transferable "
                    f"weights; {args.baseline!r} has none."
                )
            bundle_path = _pretrained_bundle_path(args, seed, baseline_tag)
            pretrained_from_used = str(bundle_path)
            if not bundle_path.is_file():
                raise FileNotFoundError(
                    f"pretrained checkpoint not found: {bundle_path}. Run the "
                    f"multi_grid_pretrain cell for this task/baseline/seed first."
                )
            bundle = torch.load(
                bundle_path, map_location=(args.device or "cpu"), weights_only=False
            )
            if is_zeroshot:
                _transfer.restore_pretrained(model, bundle)  # rebuild + load weights; NO training
                fit_seconds = 0.0
                stats: dict = {}
            else:  # transfer_finetune: init from pretrained weights, then fit on target adapt_grid
                # deep baseline (guarded above): stash weights for fit() to load post-build
                model._pretrained_state = bundle["state_dict"]  # type: ignore[attr-defined]
                t_fit = time.perf_counter()
                model.fit(task)
                fit_seconds = time.perf_counter() - t_fit
                stats = getattr(model, "train_stats_", {}) or {}
        else:
            t_fit = time.perf_counter()
            model.fit(task)  # trains on task.train only
            fit_seconds = time.perf_counter() - t_fit
            stats = getattr(model, "train_stats_", {}) or {}
        n_params = getattr(model, "n_params_", None)  # constant across seeds (fixed architecture)
        per_seed_compute.append(
            {
                "seed": seed,
                "fit_seconds": fit_seconds,
                "effective_epochs": stats.get("effective_epochs"),
                "peak_mem_bytes": stats.get("peak_mem_bytes"),
                "best_val": stats.get("best_val"),
            }
        )
        # multi_grid_pretrain persists the transferable bundle; other protocols keep the plain
        # best-val state_dict. Zero-shot trains nothing, so it writes no checkpoint.
        if is_pretrain:
            ckpt = _save_transfer_bundle(model, base_dir, seed)
        elif is_zeroshot:
            ckpt = None
        else:
            ckpt = _save_checkpoint(model, base_dir, seed)  # deep baselines only; None otherwise
        if ckpt is not None:
            checkpoints[str(seed)] = ckpt
        for ts in test_sets:
            out_dir = base_dir / ts
            out_dir.mkdir(parents=True, exist_ok=True)
            preds = model.predict(task.view(ts))
            preds.to_parquet(out_dir / f"predictions_seed{seed}.parquet", index=False)
            result = evaluate(task, preds, test_set=ts)
            per_seed_metrics[ts].append(result.metrics)
            per_seed_results[ts].append({"seed": seed, "metrics": result.metrics})
            last_result[ts] = result
            _tracking.log_seed_final(seed, ts, result.metrics)
            print(
                f"[seed {seed}] {args.baseline} {task.definition.task_id}/{task.protocol}"
                f" [{ts}]: {result.metrics}"
            )

    compute_summary = _agg_compute(per_seed_compute, n_params)
    # Positive control BEFORE persisting anything: the true labels must score the ceiling
    # through the same evaluate() path the models were scored with (raises on failure).
    harness_control = {ts: _harness_control(task, ts) for ts in test_sets}
    aggs: dict[str, dict] = {}
    for ts in test_sets:
        out_dir = base_dir / ts
        agg = _aggregate(per_seed_metrics[ts])
        aggs[ts] = agg
        lr_ = last_result[ts]
        result_doc = {
            "benchmark_version": BENCHMARK_VERSION,
            "task": task.definition.task_id,
            "protocol": task.protocol,
            "grid": task.grid,
            "baseline": baseline_tag,
            # pre-training source grid for transfer cells (None otherwise) — lets the leaderboard /
            # paper builders distinguish 20/110/345 kV-sourced transfer at the same target.
            "pretrain_grid": (args.pretrain_grid if args.protocol in TRANSFER_PROTOCOLS else None),
            "test_set": ts,
            "n_test": lr_.n_samples if lr_ else 0,
            "seeds": args.seeds,
            "n_params": n_params,  # capacity: trainable params (None for classical)
            "compute": compute_summary,  # per-cell training cost (GPU-hours reporting)
            "per_seed": per_seed_results[ts],
            "aggregate": agg,
            "breakdowns": lr_.breakdowns if lr_ else {},
        }
        sanitized = _nan_to_none(json.loads(json.dumps(result_doc, default=float)))
        (out_dir / "result.json").write_text(
            json.dumps(sanitized, indent=2, sort_keys=True, allow_nan=False)
        )

    # No-tuning attestation (the benchmark rule the paper cites): the held-out target views are
    # always predict-only. Zero-shot never trains on the target at all; fine-tune trains only on the
    # target adapt_grid train split, initialised from the (target-blind) multi_grid pretrain.
    if is_zeroshot:
        attestation = {
            "fit_on": "none (pretrained on the multi_grid corpus; target unseen)",
            "model_selection_on": "none",
            "held_out_untouched": ["train", "val", "test", "benchmark"],
            "statement": (
                "zero-shot: model initialised from the multi_grid_pretrain checkpoint; the target "
                "grid was predict-only, never used for training or model/hyperparameter selection"
            ),
            "pretrained_from": pretrained_from_used,
        }
    elif is_transfer:  # transfer_finetune
        attestation = {
            "fit_on": "train (target adapt_grid; initialised from the multi_grid pretrain)",
            "model_selection_on": "val",
            "held_out_untouched": ["test", "benchmark"],
            "statement": (
                "fine-tuned on the target adapt_grid train split from the multi_grid_pretrain "
                "checkpoint (val for early stopping only); test/benchmark were predict-only"
            ),
            "pretrained_from": pretrained_from_used,
        }
    else:
        attestation = {
            "fit_on": "train",
            "model_selection_on": "val",
            "held_out_untouched": ["test", "benchmark"],
            "statement": (
                "fit() consumed task.train (val used only for early stopping); test/benchmark "
                "were predict-only and never used for model or hyperparameter selection"
            ),
        }

    run_record = {
        "benchmark_version": BENCHMARK_VERSION,
        "splits_version": SPLITS_VERSION,
        "git_sha": _git_sha(),
        "lockfile_sha256": _lockfile_hash(),
        "seeds": args.seeds,
        "args": vars(args),
        "normalize": normalize,  # resolved input normalization (per_window for transfer/pretrain)
        "attestation": attestation,
        "wandb": wandb_info,  # ONLINE-channel cross-link (deterministic run id/url; enabled flag)
        "harness_control": harness_control,
        "versions": {d: _dist_version(d) for d in ("numpy", "pandas", "scikit-learn", "scipy")},
        "n_params": n_params,
        "compute": compute_summary,
        "per_seed_compute": per_seed_compute,
        "checkpoints": checkpoints,  # {seed: {path, sha256}} — best-val weights (deep baselines)
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "timestamp": time.time(),
        # Executing host + scheduler identity — makes the hardware column of the results audit
        # artifact-backed instead of asserted.
        "host": {
            "hostname": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
            "cluster": os.environ.get("SLURM_CLUSTER_NAME") or os.environ.get("CLUSTER"),
        },
        # Input-data provenance: sha256 of the split files read + manifest fingerprints of the
        # windowed partitions actually loaded (content-based, stable across $TMPDIR staging),
        # plus the resolved data-root environment so staged vs vault reads are distinguishable.
        "data": {
            **task.provenance,
            "env": {
                k: os.environ.get(k)
                for k in (
                    "EVEMTBENCH_DATA_ROOT",
                    "EVEMT_ROOT",
                    "EVEMTBENCH_PROJECTION_CACHE",
                    "EVEMTBENCH_CACHE_READONLY",
                )
            },
        },
    }
    try:
        import torch

        run_record["torch"] = torch.__version__
        run_record["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            run_record["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        run_record["torch"] = None
    (base_dir / "run_record.json").write_text(json.dumps(run_record, indent=2, sort_keys=True))

    for ts in test_sets:
        print(f"\nAggregate over seeds {args.seeds} [{ts}]:")
        for k, v in aggs[ts].items():
            if v["mean"] is None:
                print(f"  {k}: undefined (0/{v['n_seeds']} seeds)")
            else:
                print(f"  {k}: {v['mean']:.4f} +/- {v['std']:.4f}")
    print(f"\nArtifacts written under {base_dir}")

    # Mirror the small provenance artifacts to the W&B run (metrics/config/system already logged;
    # the big checkpoints/predictions stay OFFLINE-only, cross-linked via run_record). Then finish
    # the run with a compact metric summary and close the per-cell train.log mirror.
    _artifacts = [base_dir / "run_record.json", base_dir / "train.log"]
    for ts in test_sets:
        _artifacts.append(base_dir / ts / "result.json")
    _summary: dict[str, float] = {}
    for ts in test_sets:
        for _k, _v in aggs[ts].items():
            if isinstance(_v, dict) and _v.get("mean") is not None:
                _summary[f"{ts}/{_k}_mean"] = _v["mean"]
                if _v.get("ci95") is not None:
                    _summary[f"{ts}/{_k}_ci95"] = _v["ci95"]
    _tracking.log_artifacts(_artifacts, name=f"cell-{_tracking.run_id_for(cell_key)}")
    _tracking.finish(summary=_summary)
    set_log_file(None)


if __name__ == "__main__":
    main()
