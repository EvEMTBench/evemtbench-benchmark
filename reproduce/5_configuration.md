# 5. Benchmark configuration

Covers **Tables 2–4** and **SI Tables S16–S24**: what the benchmark is made of. No training.

| Paper | Saved copy (`results/1.3.0/tables/`) | Rebuild (`hpc/paper_artifacts/`) | Needs |
|---|---|---|---|
| Table 2 (dataset families) | `tab_dataset_families.tex` | `build_tab_dataset_families.py` | — |
| Table 3 (task specification) | `tab_task_spec.tex` | `build_tab_task_spec.py` | — |
| Table 4 (baselines) | `tab_baselines.tex` | `build_tab_baselines.py` | — |
| Table S16 (grid topologies) | `tab_topology_overview.tex` | `build_tab_topology_overview.py` | — |
| Table S17 (splits) | `supp_tab_splits.tex` | `build_supp_tab_splits.py` | committed splits |
| Table S18 (valid windows per test set) | `supp_tab_primary_valid_counts.tex` | `build_supp_tab_primary_valid_counts.py` | `outputs/` |
| Tables S19–S20 (FC, EC class support) | `supp_tab_{fc,ec}_support.tex` | `build_supp_tab_{fc,ec}_support.py` | `outputs/`; EC: data |
| Table S21 (FL locator breakdown) | `supp_tab_fl_locator_breakdown.tex` | `build_supp_tab_fl_locator_breakdown.py` | data |
| Table S22 (baseline configuration) | `supp_tab_baseline_cfg.tex` | `build_supp_tab_baseline_cfg.py` | — |
| Table S23 (partition audit) | `supp_tab_partition_audit.tex` | `build_supp_tab_partition_audit.py` | committed splits |
| Table S24 (validation selection) | `supp_tab_val_selection.tex` | `build_supp_tab_val_selection.py` | `outputs/` |

Run as `PYTHONPATH=src python hpc/paper_artifacts/<script>`. The splits are committed under
`src/evemtbench/splits/` and regenerate with `evemtbench-splits`.
