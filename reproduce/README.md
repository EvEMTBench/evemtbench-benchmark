# Reproducing the paper

One guide per group of results in *EvEMTBench: An Open Benchmark for Machine Learning in Power
System Protection*. Every guide has the same steps: where the numbers are, how to retrain them,
and which script rebuilds the table or figure. Table and figure numbers follow the journal version
of the paper. The paper's results come from release v1.3.0, archived at <https://doi.org/10.5281/zenodo.24463496>.

| Guide | Reproduces |
|---|---|
| [1. Reference results](1_reference_results.md) | Fig. 2; SI Tables S1–S5, Figs. S1–S5 |
| [2. Pre-training and transfer](2_transfer.md) | Fig. 3; SI Tables S6–S9 |
| [3. Input-normalization control](3_normalization_control.md) | SI Table S10 |
| [4. Diagnostics](4_diagnostics.md) | SI Tables S11–S15, Fig. S6 |
| [5. Benchmark configuration](5_configuration.md) | Tables 2–4; SI Tables S16–S24 |


### Names in the paper and in this repository

| Paper | Repository and result files | Dataset archive |
|---|---|---|
| `eval_grid` family | `benchmark` (test set `benchmark`) | `benchmark-*.tar.gz` |
| `adapt_grid` family | `adapt_grid` (test set `test`) | `adapt_grid-*.tar.gz` |
| `multi_grid` family | `multi_grid` | `multigrid-*.tar.gz` |
| CIGRÉ MV grid (20 kV) | `cigre_mv` | `CigreMVGrid` |
| Double Line grid (110 kV) | `double_line` | `DoubleLine` |
| Protection Reference grid (110 kV) | `testgrid_110kv` | `TestGrid110kV` |
| IEEE-39 (345 kV) | `ieee39` | `IEEE39BusSystem` |
## Before you start

**Saved results.** You need nothing below to read the numbers: every result table is in
[`results/1.3.0/tables/`](../results/1.3.0/tables/), and every number (mean, std, 95 % CI over
five seeds, per task, protocol, grid, baseline and test set) is in
[`results/1.3.0/aggregated_results.csv`](../results/1.3.0/aggregated_results.csv). The
[provenance dashboard](https://evemtbench.github.io/evemtbench-benchmark/audit/index.html) lists
every cell with the exact command that produced it.

**Install.**

```bash
git clone https://github.com/EvEMTBench/evemtbench-benchmark.git
cd evemtbench-benchmark
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,baselines]"   # torch; the deep baselines need a CUDA GPU
```

`env/results-v1.3.0.lock` pins the environment the published results ran in.

**Data.** The results use EvEMTBench dataset v0.1
([doi:10.48742/fau.abpd-r521](https://doi.org/10.48742/fau.abpd-r521)).
The download holds the raw simulation exports. The benchmark reads windowed data, and the tool
that windows the exports is not public yet. Until it is, retraining needs windows built to the
contract in `src/evemtbench/contract/`; `python -m evemtbench.data.input_digests --verify` checks
them against the reference windows. Then:

```bash
export EVEMTBENCH_DATA_ROOT=/path/to/windowed/EvEMTBench
```

**Not public yet:** the windowing tool, the per-cell run outputs (`outputs/`, which the table and
figure scripts read) and the model checkpoints. Until they are, the "rebuild" steps work only on a
tree you trained yourself.
