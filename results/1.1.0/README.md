# EvEMTBench results — v1.1.0

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

One record per (task × protocol × grid × baseline × test_set × metric) — 35144
records across 4548 cells for this version. Columns / keys:

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

Rows are each protocol's **current** results: `held_out` rows carry `version`
1.0.0 (their artifacts are unchanged by the v1.1.0 scoped
revision, so they were not re-run; see `docs/versioning.md`), while the
pretrain/transfer rows carry 1.1.0.

`mean`/`std`/`ci95` are aggregated over the `n_seeds` random seeds; a null (JSON) /
empty (CSV) value marks a metric with no support on that eval set (e.g. an absent
per-class F1), reported honestly rather than as a misleading 0. The CSV shows 6
significant figures for readability; the JSON preserves the exact values.

One convention split, stated because the column is shared: rows with `test_set=val`
carry a **sample** std (`ddof=1`, written by `hpc/rescore_validation.py`), while
`test`/`benchmark` rows carry a **population** std (`ddof=0`, written by the runner).
`ci95` is t-based (`ddof=1`) in both. Validation rows are published so that baseline
selection is auditable — the reported results are always the `test`/`benchmark` rows.

## `tables/`

The LaTeX result and evidence tables, copy-paste ready — a byte-identical copy of the
manuscript's generated fragments: `tab_res_<fn>.tex` (best learned baseline per grid ×
observability view × held_out test set), `tab_transfer_<fn>.tex` (cross-grid transfer),
`supp_tab_<fn>_<test_set>.tex` (full per-baseline breakdowns) and the supplementary
evidence tables (partition/leakage audit, valid-sample counts, EC support, ED negative
composition, current-scale diagnostic, transfer-vs-scratch reference, selection audit).
Each table names its owning generator and inputs in a header comment. Result cells are
`mean $\pm$ 95% CI`; `\res` = no result yet. Compile standalone by prepending
`\input{_preamble.tex}` (it stubs `\res`/`\ac`/`\acs`).

## Full provenance

For per-number traceability — output file, evaluation code, training log, training code,
config, W&B run and the exact reproduce command behind each number — see the audit
site: `audit/paper_numbers.html` and `audit/provenance/1.1.0/index.json`.
