# EvEMTBench results — v1.3.0

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

One record per (task × protocol × grid × baseline × test_set × metric) — 44312
records across 5724 cells for this version. Columns / keys:

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

Rows are each cell's **current** results, so the `version` column is not constant:
`held_out` rows carry 1.0.0 and pretrain/transfer rows carry
1.1.0 (their artifacts are unchanged by the later scoped
revisions, so they were not re-run; see `docs/versioning.md`), except the fault-location
(`fault_location_*`) `held_out` / `transfer_zeroshot` / `transfer_finetune` rows, which
carry 1.2.0: they were re-run under the corrected fault-location
valid-sample rule (line faults only) and are not comparable to the same cells in the
older bundles. The faulted-phase (`fault_phase_*`) rows of the cells that read a
label repaired by label patch LP-1 carry 1.3.0: `held_out` on cigre_mv, `multi_grid_pretrain`
on template_20kv, and the transfer cells that target cigre_mv or start from the
template_20kv source (the high-impedance-with-arc faults regained their faulted phase;
`docs/known_issues.md` §4).

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
`\input{_preamble.tex}` (it stubs the manuscript macros the tables use).

## Full provenance

For per-number traceability — output file, evaluation code, training log, training code,
config, W&B run and the exact reproduce command behind each number — see the audit
site: `audit/paper_numbers.html` and `audit/provenance/1.3.0/index.json`.
