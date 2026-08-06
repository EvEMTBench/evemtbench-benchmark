# Design: pre-training → cross-grid transfer (multi_grid)

**Status: built.** Adds three protocols on top of `held_out` for the benchmark's headline
generalization goal — train on a diverse pre-training corpus, generalize to an out-of-distribution
grid — at the portable `local`/`line` views. Canonical contract: [../protocol.md](../protocol.md).
The `global`-view / variable-topology track (a GNN over cubicle sets) stays open —
[cross_topology.md](cross_topology.md).

## The representation that makes transfer well-posed

The `multi_grid` family (renamed from `train`; producer's on-disk `multi_grid/`) stores windows in a
portable **12-channel double-line view**: `EndA/EndB × [Isec L1,L2,L3, Usec L1,L2,L3]`, with the
line/grid identity factored into `y` (`graph_id, line_id, bus_a, bus_b`) instead of the channel axis
(`meta.extra.view == "double_line"`). That layout is byte-identical to a reference grid's **`line`**
observability (the faulted line's two terminal cubicles, 12 ch) — and `EndA` alone is **`local`**
(6 ch). So a model pretrained on `multi_grid` consumes the exact tensor shape/semantics of any
reference grid's projected local/line view → transfer needs no reshape. `global` is grid-specific
width (48–420 ch) and impossible on the double-line representation, hence local/line only.

**Unit of a `multi_grid` episode.** Because the identity is factored into `y`, one episode (one
`sample_id`) is **one line of one simulation**, not a whole simulation: `sample_id = sim_idx × 10⁴ +
line_idx`. Measured from the committed splits: 9 753 / 9 879 / 9 895 simulations (≈10 000 per
voltage level, as the dataset paper states) expand to 119 153 / 138 512 / 173 407 episodes at
20 / 110 / 345 kV — a mean of 12.2 / 14.0 / 17.5 observed lines per simulation, rising with
generated grid size. Quote the right unit when comparing corpus sizes with the dataset paper, and
see the split-granularity deviation in [../protocol.md](../protocol.md).

- Projection: `tasks/line_view.py::project_double_line` (slice, no oracle). Terminals are
  **bus-canonicalized** (`_double_line_near_end` in `task.py`, using `y.bus_a/bus_b`) so channel *k*
  matches the reference oracle's `[near, far]` ordering (`_termkey`, smaller bus = near).
- Detection: `line_view.is_double_line_view(feature_names, ws.view)`.

## Protocols (`splits/protocols.py`)

| protocol | train | eval | init |
|---|---|---|---|
| `multi_grid_pretrain` | multi_grid split (in-corpus) | in-corpus `test` | scratch |
| `transfer_zeroshot` | — (none) | target `test` + `benchmark` | pretrain ckpt |
| `transfer_finetune` | target `adapt_grid` | target `benchmark` | pretrain ckpt |

Split assembly: `task.py::_protocol_triples`. `multi_grid_pretrain` reads a committed split
(`splits/generator.py`, `train_family="multi_grid", benchmark_family=None`); the transfer protocols
reuse the target's committed `held_out` split (zero-shot empties train/val). Registered on the
fault-family `*_local`/`*_line` tasks only (`registry.py::_PORTABLE_PROTOCOLS`).

## Normalization

Per-window z-score (`task.py::normalize_per_window`), the default for all three protocols
(`default_input_normalize`), applied via the existing transform seam (`build_input_transform`,
alongside `zero_seq`). Instance normalization is leakage-free (no target statistics) and
voltage-invariant, which is what lets a 20 kV-pretrained model score a 345 kV grid.

## Checkpoint transfer (`baselines/_transfer.py`)

The pre-training run saves a self-contained **bundle** (weights + dims + preprocessing) per
`(task, baseline, seed)` at `<pretrain cell>/checkpoints/seed{seed}.pt`
(`runner._save_transfer_bundle`). Because the portable view fixes `in_ch` and the class vocab is
frozen, the architecture is identical across grids, so the `state_dict` loads without surgery. The
runner resolves it (`--pretrained-from` / `--pretrain-grid`, default `template_20kv`) and either
`restore_pretrained` (zero-shot, no training) or attaches it as `_pretrained_state` for `fit()` to
continue from (fine-tune). Only the deep baselines (`mlp/gru/cnn/resnet`) participate; the
`run_record.json` attestation records `fit_on` / `pretrained_from` per protocol.

## Coverage & reporting

`coverage.py` restricts these protocols to deep baselines (`DEEP_BASELINES`) and the right grids
(`protocol_grid_applies`: pretrain → `template_20kv`; transfer → the 4 reference grids);
`expected_test_sets` is in-corpus (`test` only) for `multi_grid_pretrain`. The leaderboard groups
transfer targets by voltage match vs the 20 kV source.

## Caveats

- The multi_grid pre-training corpus spans **20/110/345 kV** template grids (`template_20kv`,
  `template_110kv`, `template_345kv`); the transfer results currently source from the **20 kV**
  template (`--pretrain-grid template_20kv`), so same-voltage (`cigre_mv`) is the clean
  positive-transfer test and 110/345 kV remain the hard cross-voltage OOD. Using the higher-voltage
  templates as transfer sources (matched-voltage EHV -> IEEE-39) is the next step.
- `fault_location` reference terminal: `y_fault_location` is % on both sides (verified ~1..99); the
  bus-canonicalization aligns the 0%-end. If FL transfer looks degenerate, re-check that convention.

## Acceptance criteria (met — kept for the record)

- [x] Three transfer protocols in the registry; applicability rules in `coverage.py` (deep
      baselines only) render as N/A, not gaps.
- [x] `tests/integration/test_transfer_protocols.py` covers pretrain → zero-shot → fine-tune
      round-trip and the zero-shot no-target-training invariant.
- [x] Transfer cells appear in `coverage_snapshot.json` / the audit matrix with reproduce
      arg-lines.
- [ ] Remaining: all 384 transfer cells green and the transfer section of the paper written
      against them.
