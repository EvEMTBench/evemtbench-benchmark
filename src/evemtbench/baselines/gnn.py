"""Graph-aware baseline — the cross-topology seam (designed, NOT built).

The strongest generalization for protection ML is train-on-some-grids, protect-an-unseen-one;
that requires models consuming *sets of cubicles* wired by the topology graph rather than a
fixed channel vector. The aux layer already serves each grid's networkx graph
(``data/graphs.py``, the ``graphs`` extra). Design: ``docs/design/cross_topology.md``.

``GNNBaseline`` freezes the interface: an ordinary :class:`~evemtbench.baselines.interface.
Baseline` (per-cubicle waveform encoder + message passing over the grid graph) intended for the
future ``cross_topology`` protocol. Prerequisites (train family windowed, cubicle-set view API)
are post-v0.2.0 — every method raises with a pointer to the design doc.
"""

from __future__ import annotations

import pandas as pd

from ..tasks.task import Task, TaskView

_DESIGN = "designed but not built - see docs/design/cross_topology.md"


class GNNBaseline:
    """Waveform-encoder-per-cubicle + message passing over the topology graph (interface stub)."""

    name = "gnn"

    def __init__(self, *, seed: int = 0, device: str | None = None) -> None:
        self.seed, self.device = seed, device

    def fit(self, task: Task) -> None:
        raise NotImplementedError(f"GNNBaseline.fit: {_DESIGN}")

    def predict(self, view: TaskView) -> pd.DataFrame:
        raise NotImplementedError(f"GNNBaseline.predict: {_DESIGN}")
