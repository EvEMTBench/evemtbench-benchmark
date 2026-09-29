"""Bring-your-own-model walkthrough — the two supported ways to evaluate a custom model.

**Path A (recommended for papers): plug into the official runner.** Implement the
:class:`~evemtbench.baselines.interface.Baseline` protocol, ``register_baseline()`` it, and let
``runner.main()`` produce the same auditable artifacts as the built-ins (``result.json`` with
per-seed metrics + mean/std/ci95, ``predictions_seed{n}.parquet``, ``run_record.json`` with the
no-tuning attestation).

**Path B (quick look): evaluate a predictions frame directly.** ``load_task`` → predict →
``evaluate``. Shown at the bottom.

Run from the repository root on any machine with the dataset available:

    EVEMTBENCH_DATA_ROOT=/path/to/evemtbench PYTHONPATH=src \
        python examples/byo_model.py --grid double_line

On a shared cluster, use a compute node rather than the login node — e.g. the maintainers'
NHR@FAU Alex SLURM invocation (site-specific partition/GRES names):

    srun --partition=a100mig --gres=gpu:a100small:1 --time=30:00 \
        bash -lc 'PYTHONPATH=src python examples/byo_model.py --grid double_line'

The demo model is an SGD logistic classifier trained via the **streaming** view API
(``iter_batches``), so it works unchanged on views too large to materialize. Artifacts go to a
scratch ``--out`` (default ``outputs_byo``) — never the official ``outputs/`` coverage tree.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from evemtbench import evaluate, get_task, list_tasks, load_task
from evemtbench.baselines import runner
from evemtbench.baselines.interface import Baseline
from evemtbench.evaluation.predictions_io import predictions_frame
from evemtbench.tasks.task import Task, TaskView


class SGDLogisticBaseline:
    """A minimal third-party model: logistic regression on flattened windows, trained by
    streaming mini-batches (``partial_fit``) — never materializes the full feature array."""

    name = "sgd_logistic"

    def __init__(self, *, seed: int = 0, epochs: int = 3, batch_size: int = 512) -> None:
        self.seed, self.epochs, self.batch_size = seed, epochs, batch_size
        self.model = None
        self._stats = None

    def _sx(self, x: np.ndarray) -> np.ndarray:
        mean, std = self._stats  # type: ignore[misc]
        return ((x - mean) / std).reshape(len(x), -1)

    def fit(self, task: Task) -> None:  # fit on train ONLY (the no-tuning rule)
        from sklearn.linear_model import SGDClassifier

        view = task.train
        self._stats = view.channel_stats()
        classes = np.unique(view.labels)
        self.model = SGDClassifier(loss="log_loss", random_state=self.seed)
        rng = np.random.default_rng(self.seed)
        for _ in range(self.epochs):
            order = rng.permutation(len(view))
            for xb, yb in view.iter_batches(self.batch_size, order=order):
                self.model.partial_fit(self._sx(xb), yb, classes=classes)

    def predict(self, view: TaskView) -> pd.DataFrame:
        assert self.model is not None, "call fit() first"
        preds = np.concatenate(
            [self.model.predict(self._sx(xb)) for xb, _ in view.iter_batches(self.batch_size)]
        )
        return predictions_frame(view.metadata["sample_id"], view.metadata["window_idx"], preds)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", default="fault_detection_local", choices=list_tasks())
    p.add_argument("--grid", default="double_line")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--root", default=None, help="data root override (e.g. the test fixture)")
    p.add_argument("--splits-root", default=None)
    p.add_argument("--out", default="outputs_byo", help="scratch output dir (NOT outputs/)")
    args = p.parse_args()

    print(f"task card: {get_task(args.task)}")

    # ---- Path A: register + run through the official runner (auditable artifacts) ----------
    runner.register_baseline(
        SGDLogisticBaseline.name,
        lambda *, seed, batch_size, **_: SGDLogisticBaseline(seed=seed, batch_size=batch_size),
    )
    argv = [
        *("--task", args.task, "--grid", args.grid, "--baseline", SGDLogisticBaseline.name),
        *("--seeds", *map(str, args.seeds)),
        *("--out", args.out),
    ]
    if args.root:
        argv += ["--root", args.root]
    if args.splits_root:
        argv += ["--splits-root", args.splits_root]
    runner.main(argv)

    # ---- Path B: direct evaluate() on a predictions frame (no artifacts) -------------------
    task = load_task(
        args.task, "held_out", grid=args.grid, root=args.root, splits_root=args.splits_root
    )
    model = SGDLogisticBaseline(seed=args.seeds[0])
    model.fit(task)
    result = evaluate(task, model.predict(task.test), test_set="test")
    print(f"\nPath B direct evaluate [{args.task}/{args.grid}/test]: {result.metrics}")
    assert isinstance(model, Baseline)  # the protocol is runtime-checkable


if __name__ == "__main__":
    main()
