# Design: cross-topology transfer & graph-aware models (GNN)

**Status: designed, not built** — a paper Open Challenge. Code seams:
`evemtbench.baselines.gnn`, the `cross_topology` protocol slot, the aux graph layer
(`data/graphs.py` — built in M2).

> **Now built (the portable-view slice of this):** the `local`/`line` cross-grid transfer track —
> pre-train on the `multi_grid` corpus, zero-shot / fine-tune on the reference grids — is
> implemented as the `multi_grid_pretrain` / `transfer_zeroshot` / `transfer_finetune` protocols.
> See [pretrain_transfer.md](pretrain_transfer.md). What remains open here is the **`global`-view**
> variable-topology case below, which genuinely needs the GNN (fixed 6/12-ch views do not).

## Why

The strongest form of generalization for protection ML is **train on some grids, protect an
unseen one**. Today's tasks are per-grid (`held_out` within a topology); the `benchmark` family
shifts the *scenario distribution*, not the *topology*. The `multi_grid` family (formerly `train`;
20/110/345 kV template variants) is windowed and already used for pre-training by the built
transfer protocols, though stored in the portable 12-channel double-line view, which yields only the `local` / `line` inputs — and the M2 aux layer
already serves each grid's networkx topology graph, so graph-conditioned models have their input.

## Protocol sketch (`cross_topology`)

- **Train:** all episodes of grids G \ {g} (+ the `multi_grid` family, which is stored in the
  portable 12-channel double-line view, which yields only the `local` / `line` inputs).
- **Test:** grid g, both held-out families — zero-shot; a small-adaptation variant
  (few episodes of g's `adapt_grid`) is the second axis.
- Leave-one-grid-out over the 4 reference grids ⇒ 4 folds; per-fold reporting (never pooled —
  class spaces differ across voltage levels).
- Channel heterogeneity (48–420 ch) is the point: models must consume *sets of cubicles*, not a
  fixed channel vector. Local/line views are naturally portable; `global` requires the graph.

## Model seam (`baselines/gnn.py`)

- Nodes = cubicles (waveform encoder per cubicle — reuse the CNN/ResNet trunk), edges from the
  topology graph (`load_grid_graph` + `graph_edge_features`: line params, trafo links).
- A `GNNBaseline` implements the ordinary `Baseline` protocol; the only new requirement is
  access to the task's grid graph — resolved via the aux overlay at `fit()`/`predict()` time.
- Stays in the demo layer (`baselines/`); product layer remains torch-free (`graphs` extra pulls
  networkx only).

## Prerequisites (why this is post-freeze)

1. ~~`train` family windowed~~ — done: the `multi_grid` family (formerly `train`) is windowed and
   used for pre-training, but stored in the portable 12-channel double-line view, which yields only the `local` / `line` inputs; the global-view case still needs
   the cubicle-set view of item 3.
2. Single-config window read (the W0p020/W0p050 coexistence fix — also an M7 prerequisite).
3. A cubicle-set view API (`feature_groups` already names per-cubicle channel blocks).

## Open questions

- Per-grid normalization under transfer (fit stats on train grids only; per-unit vs. z-score).
- Whether `cross_topology` scores enter the leaderboard at v0.3 or live as a challenge track.

## Acceptance criteria

- [ ] A `cross_topology` protocol loads with an eval grid disjoint from every training grid
      (asserted by a test in the style of `test_transfer_protocols.py`).
- [ ] `tests/integration/test_leakage.py` is extended to the new protocol's split files.
- [ ] The GNN baseline registers via `register_baseline`, respects the golden import rule, and
      its cells appear in the coverage matrix (N/A rules updated deliberately, not by accident).
- [ ] At least one full grid-holdout column is green in `audit/coverage.html` and the GNN clears
      the majority floor on it (`coverage --check-floor` stays green).
- [ ] `BENCHMARK_VERSION` minor bump + CHANGELOG entry.
