# EvEMTBench results — v1.0.0

The public, self-contained results tier: the aggregated numbers and the LaTeX result
tables for this benchmark version. Derived from the raw per-cell artifacts under
`outputs/` (git-ignored) by `hpc/make_results_bundle.py`; **do not edit by hand** —
regenerate with:

```
PYTHONPATH=src python hpc/make_results_bundle.py
```

Every file here is byte-checked against a fresh rebuild by the freshness gate
(`hpc/check_results_fresh.py`), so the committed bundle provably equals the data.

## `aggregated_results.csv` / `aggregated_results.json`

One record per (task × protocol × grid × baseline × test_set × metric) — 31472
records across 4164 cells for this version. Columns / keys:

- `version`
- `task`
- `protocol`
- `grid`
- `baseline`
- `source`
- `test_set`
- `metric`
- `mean`
- `std`
- `ci95`
- `n_seeds`
- `n_test`
- `n_params`
- `fit_seconds`

`mean`/`std`/`ci95` are aggregated over the `n_seeds` random seeds; a null (JSON) /
empty (CSV) value marks a metric with no support on that eval set (e.g. an absent
per-class F1), reported honestly rather than as a misleading 0. The CSV shows 6
significant figures for readability; the JSON preserves the exact values.

## `tables/`

The LaTeX result tables, copy-paste ready: `tab_res_<fn>.tex` (5 main, best learned
baseline per grid × observability view × held_out test set) and
`supp_tab_<fn>_<test_set>.tex` (10 full per-baseline supplement tables). Each cell is
`mean $\pm$ 95% CI`; `\res` = no result yet. Compile standalone by prepending
`\input{_preamble.tex}` (it stubs `\res`/`\ac`/`\acs`).

## Full provenance

For per-number traceability — output file, evaluation code, training log, training code,
config, W&B run and the exact reproduce command behind each number — see the audit
site: `audit/paper_numbers.html` and `audit/provenance/1.0.0/index.json`.
