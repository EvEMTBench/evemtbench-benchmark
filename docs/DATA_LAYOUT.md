# Data layout — the on-disk contract

Everything the benchmark reads, where it lives, and how it joins. The data root is
**producer-owned and actively written** — the benchmark treats it read-only; all derived
assets go to the *auxiliary overlay* (separate root) or `outputs/`.

**Where the windows come from.** The published download (FAUDataCloud) contains the producer's
raw simulation exports (`raw/`); the `windows/` partitions below are produced from them by the
producer pipeline, which is not public yet (release planned). A third-party windowing must
satisfy the pinned contract in `src/evemtbench/contract/`: `load_windows` verifies each manifest
against it, and `python -m evemtbench.data.input_digests --verify` checks the partitions against
the committed reference digests.

## Roots (resolved from the environment, never hardcoded)

| root | resolution | contents |
|---|---|---|
| data root | `root=` arg → `$EVEMTBENCH_DATA_ROOT` → `$EVEMT_ROOT` | families → windows + raw |
| aux overlay | `$EVEMTBENCH_AUX_ROOT` → sibling `<data-root>/../evemtbench-aux` | extracted graphs + settings (additive, benchmark-owned) |

## Families × grids

Families (`evemtbench.data.roots.FAMILIES`): **`benchmark`** (cross-distribution held-out;
scored, never trained on), **`adapt_grid`** (in-distribution: train/val/test), **`multi_grid`**
(multi-topology pre-training corpus — `template_20kv` / `template_110kv` / `template_345kv` — for
the `multi_grid_pretrain` and transfer protocols). `fine_tune` is accepted as a read alias for
`adapt_grid` and `train` for `multi_grid` (the pre-v0.2 names).

Grids (`GRIDS`): `double_line` (110 kV HV) · `cigre_mv` (20 kV MV) · `testgrid_110kv` (110 kV HV)
· `ieee39` (345 kV EHV, 60 Hz). Each maps to a `raw_foldername` (directory name, e.g.
`DoubleLine`) and a `topology` id (file-name stem, e.g. `hv_double_line_110kv`).

## Per-family tree

```text
<data-root>/<family>/
├── windows/<RawFolder>/                  # the consumption path (contract C-X-2)
│   ├── X_<topology>_W0p050_S0p005.raw    #   float32 memmap (n, l=480, f)
│   ├── X_....raw.manifest.json           #   shape, feature_hash, config_fingerprint, status
│   ├── X_....raw.meta.json               #   feature_names, feature_groups, channels
│   └── y_<topology>_W0p050_S0p005.parquet#   per-window labels incl. sample_id (episode id)
└── raw/<RawFolder>/                      # producer simulation output (+ per-grid tars)
    ├── data/ · labels/                   #   raw EMT episodes
    └── graphs/                           #   networkx topology pickle(s) + settings.csv
```

The benchmark profile pins **W = 50 ms, step = 5 ms** (`W0p050_S0p005`, l = 480 @ 9.6 kHz).
Some windows dirs also carry other configs (e.g. `W0p020`) side by side; `find_window_files`
(`data/windows.py`) narrows the `X_<topology>_*.raw` candidates to the pinned contract's
`W…_S…` tag when given a `contract` (as the loaders do), and raises unless exactly one file remains
and its `.manifest.json` / `.meta.json` / `y_*.parquet` companions exist.

## Auxiliary overlay (additive; never writes the data root)

```text
$EVEMTBENCH_AUX_ROOT/<family>/<RawFolder>/
├── graph.pickle       # networkx MultiGraph of the topology (nodes=buses, edges=lines/trafos)
│                      # TRUSTED first-party artifact: read with pickle.load (executes code on
│                      # unpickle), so it is safe only because extract_aux produced it from the
│                      # official tarballs. Never load one from an untrusted root or a submission
│                      # — it is used for topology reporting only, never on the scoring path.
└── settings.csv       # one row per episode: event parameters incl. fault resistance
```

Populated by `python -m evemtbench.data.extract_aux --grid <g> --family <f>` (pulls only
`*/graphs/*.pickle` + `*settings*.csv` from the per-grid tars; idempotent) or
`hpc/extract_aux.sh`. Inventory: `python -m evemtbench.data.manifest` reports
windows/graph/settings presence per family × grid.

Consumers: `data/graphs.py` (`load_grid_graph`, node/edge feature frames — `graphs` extra pulls
networkx) and `data/aux_labels.py` (`load_settings`, `join_to_metadata`).

## Join keys (the one rule to remember)

```text
windows y.parquet: sample_id  ==  settings.csv: general/sim_idx   (episode id, int)
```

Every window carries its episode's `sample_id`; splits are **episode-grouped** on it (no window
of an episode crosses partitions), and `join_to_metadata` attaches settings columns (e.g.
`event_flt_shc_resistance` → the `resistance_bin` stratum) by the same key. Labels are derived
**only** from the committed per-window columns / settings vocabularies (pure derivers,
golden-tested) — never from waveforms.

## What is committed vs. derived

| artifact | where | status |
|---|---|---|
| splits | `src/evemtbench/splits/v1.0/held_out/<grid>/` (held_out; transfer reuses it) · `src/evemtbench/splits/v1.1/multi_grid_pretrain/<grid>/` (pre-training; `version.splits_version`) | **committed, frozen** (regenerate bit-for-bit; changing = major bump) |
| results (raw) | `outputs/<version>/<task>/<protocol>/<grid>/<baseline>/<test_set>/` | derived, git-ignored; coverage manifest is the source of truth |
| results (public bundle) | `results/<version>/` — `aggregated_results.{csv,json}` + `tables/*.tex` | committed, **builder-generated** (`hpc/make_results_bundle.py`, freshness-gated); the self-contained public tier between raw `outputs/` and the private `paper/manuscript/` (never hand-edit) |
| rendered tables | `RESULTS.md`, `leaderboard.html` | committed but **builder-generated** (freshness-gated; never hand-edit) |
| projection caches | `$EVEMTBENCH_PROJECTION_CACHE` | derived, content-addressed, safe to delete |
