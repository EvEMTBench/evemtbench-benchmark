"""Optional Weights & Biases experiment tracking for the deep-baseline runner.

REDUNDANT logging, ONLINE channel. The OFFLINE channel is the committed artifact tree
(``result.json`` / ``run_record.json`` / ``predictions`` / ``checkpoints`` / ``train.log``); this
module mirrors the same information to W&B. Everything here is a **best-effort no-op unless
enabled**, and every W&B call is wrapped so a tracking failure (no network/login/version skew)
degrades to offline-only — tracking must never change or break a result, and must not perturb CI or
the torch-free eval layer.

Enable with ``EVEMTBENCH_WANDB=1``. Transport is W&B's own ``WANDB_MODE`` (``online`` on Alex via
the NHR proxy, or ``offline`` + later ``wandb sync``). Entity/project come from ``WANDB_ENTITY`` /
``WANDB_PROJECT``. A W&B run maps 1:1 to a runner cell; its id is a deterministic hash of the cell
key, so re-runs resume the same run and the run URL is known before any upload (recorded into
``run_record.json`` for the offline<->online cross-link).
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

from ._log import log, set_epoch_sink

_RUN: Any = None  # active wandb.Run, or None


def enabled() -> bool:
    return os.environ.get("EVEMTBENCH_WANDB", "").lower() in ("1", "true", "yes")


def run_id_for(cell_key: str) -> str:
    """Deterministic 16-hex W&B run id from the cell key (stable across re-runs)."""
    return hashlib.sha1(cell_key.encode()).hexdigest()[:16]


def start_run(
    *,
    cell_key: str,
    name: str,
    config: dict,
    tags: list,
    group: str | None = None,
    job_type: str | None = None,
) -> dict:
    """Init the W&B run for a cell and register the per-epoch sink. Returns a small provenance dict
    for ``run_record.json`` — populated with the deterministic id/url even when disabled or on
    failure, so the offline record always names the intended run."""
    global _RUN
    run_id = run_id_for(cell_key)
    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT", "evemtbench")
    mode = os.environ.get("WANDB_MODE", "online")
    info = {
        "enabled": False,
        "mode": mode,
        "entity": entity,
        "project": project,
        "run_id": run_id,
        "run_url": (f"https://wandb.ai/{entity}/{project}/runs/{run_id}" if entity else None),
    }
    if not enabled():
        return info
    try:
        import wandb

        _RUN = wandb.init(
            id=run_id,
            resume="allow",
            entity=entity,
            project=project,
            name=name,
            group=group,
            job_type=job_type,
            tags=[t for t in tags if t],
            config=config,
        )
        set_epoch_sink(log_epoch)
        info["enabled"] = True
        try:
            info["run_url"] = _RUN.url
            info["run_dir"] = _RUN.dir
        except Exception:
            pass
        log(f"wandb: run {run_id} ({mode}) -> {info.get('run_url')}")
    except Exception as exc:  # network/login/version — keep the offline artifacts, drop online
        _RUN = None
        log(f"wandb: init failed ({type(exc).__name__}: {exc}); continuing offline-only")
    return info


def log_epoch(metrics: dict) -> None:
    """Per-epoch sink (registered on _log). Seed-namespaced so multi-seed curves don't collide."""
    if _RUN is None:
        return
    seed = metrics.get("seed")
    pref = f"s{seed}/" if seed is not None else ""
    data = {f"{pref}{k}": v for k, v in metrics.items() if k != "seed" and v is not None}
    data["epoch"] = metrics.get("epoch")
    if seed is not None:
        data["seed"] = seed
    try:
        _RUN.log(data)  # let W&B auto-increment its step (monotonic + resume-safe)
    except Exception:
        pass


def log_seed_final(seed: Any, test_set: str, metrics: dict) -> None:
    if _RUN is None:
        return
    try:
        payload = {f"final/{test_set}/{k}": v for k, v in metrics.items() if v is not None}
        payload["seed"] = seed
        _RUN.log(payload)
    except Exception:
        pass


def log_artifacts(paths: list, *, name: str, art_type: str = "cell") -> None:
    if _RUN is None:
        return
    try:
        import wandb

        art = wandb.Artifact(name=name, type=art_type)
        added = 0
        for p in paths:
            sp = str(p)
            if os.path.isfile(sp):
                art.add_file(sp)
                added += 1
        if added:
            _RUN.log_artifact(art)
    except Exception:
        pass


def finish(summary: dict | None = None) -> None:
    global _RUN
    set_epoch_sink(None)
    if _RUN is None:
        return
    try:
        if summary:
            _RUN.summary.update(summary)
        _RUN.finish()
    except Exception:
        pass
    _RUN = None
