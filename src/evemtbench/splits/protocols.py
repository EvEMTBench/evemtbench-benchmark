"""Official benchmark protocols (Benchmark Plan §7).

Splits are at the **simulation-episode level** (one ``sample_id`` = one EMT simulation; all its
windows share a split), so no window crosses a partition boundary.

* ``held_out`` — train/val/test are disjoint episode splits of the **adapt_grid** family of one
  grid (in-distribution generalization), and the model is *additionally* evaluated on the entire
  **benchmark** family of the same grid — a second, never-trained-on held-out test (the
  cross-distribution / official test). Supports interpolation + held-out-generalization claims.

The **pre-training / cross-grid transfer** protocols train on the diverse ``multi_grid``
pre-training corpus and generalize to the reference grids' held-out families, at the portable
LOCAL/LINE observability views only (fixed 6/12 channels). They map to the paper's goals G2
(zero-shot) and G3 (adaptation); see ``docs/protocol.md`` / ``docs/design/cross_topology.md``.

* ``multi_grid_pretrain`` — train/val/test on an episode split of the ``multi_grid`` family of the
  pre-training grid (in-corpus). Reports in-corpus performance AND produces the checkpoints that
  the transfer protocols initialise from.
* ``transfer_zeroshot`` — initialise from a ``multi_grid_pretrain`` checkpoint and evaluate,
  WITHOUT any target training, on a reference grid's ``adapt_grid`` held-out ``test`` and its
  ``benchmark`` family (the target topology is never seen).
* ``transfer_finetune`` — initialise from that checkpoint, fine-tune on the reference grid's
  ``adapt_grid`` train split, then evaluate on its ``benchmark`` family (target adaptation).
"""

from __future__ import annotations

from dataclasses import dataclass

# Three-way episode split names (used by the generic helpers).
SPLIT_NAMES = ("train", "val", "test")

# held_out has a 4th "benchmark" partition (the whole benchmark family, test-only).
HELD_OUT_SPLIT_NAMES = ("train", "val", "test", "benchmark")

# held_out fractions applied to the adapt_grid family episodes (train/val/test).
HELD_OUT_FRACTIONS = (0.70, 0.15, 0.15)


@dataclass(frozen=True)
class Protocol:
    name: str
    description: str
    target_topology_seen: bool  # is the target grid seen during training/model selection?
    claim: str


PROTOCOLS: dict[str, Protocol] = {
    "held_out": Protocol(
        "held_out",
        "Train/val/test on the adapt_grid family; also test on the held-out benchmark family.",
        target_topology_seen=True,
        claim="in-distribution generalization + held-out benchmark test",
    ),
    "multi_grid_pretrain": Protocol(
        "multi_grid_pretrain",
        "Train/val/test on an episode split of the multi_grid pre-training corpus (in-corpus); "
        "produces the checkpoints the transfer protocols initialise from.",
        target_topology_seen=True,
        claim="in-corpus multi-grid pre-training performance",
    ),
    "transfer_zeroshot": Protocol(
        "transfer_zeroshot",
        "Initialise from a multi_grid_pretrain checkpoint and evaluate a reference grid's held-out "
        "families with NO target training (target topology unseen).",
        target_topology_seen=False,
        claim="zero-shot cross-grid generalization from the pre-training corpus",
    ),
    "transfer_finetune": Protocol(
        "transfer_finetune",
        "Initialise from a multi_grid_pretrain checkpoint, fine-tune on the reference grid's "
        "adapt_grid train split, then test on its benchmark family (target adaptation).",
        target_topology_seen=True,
        claim="pretrain broadly -> fine-tune on target -> held-out target benchmark test",
    ),
}

# Protocols that pre-train on / transfer from the multi_grid corpus. They default to per-window
# input normalization (voltage-invariant) and are the cross-grid generalization track.
TRANSFER_PROTOCOLS = ("transfer_zeroshot", "transfer_finetune")
PRETRAIN_TRANSFER_PROTOCOLS = ("multi_grid_pretrain", *TRANSFER_PROTOCOLS)


def get_protocol(name: str) -> Protocol:
    if name not in PROTOCOLS:
        raise KeyError(f"Unknown protocol {name!r}; known: {sorted(PROTOCOLS)}")
    return PROTOCOLS[name]


def default_input_normalize(protocol: str) -> str:
    """Default input normalization for a protocol: per-window z-score for the pre-training/transfer
    protocols (voltage-invariant, leakage-free), else ``none`` (preserves held_out behavior)."""
    return "per_window" if protocol in PRETRAIN_TRANSFER_PROTOCOLS else "none"
