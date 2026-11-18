# 4. Diagnostics

Covers **SI Tables S11–S15** and **Fig. S6**. None of them trains a model: they re-read the stored
predictions of the reference runs (guide 1), and three also read the dataset.

| Paper | Saved copy (`results/1.3.0/tables/`) | Rebuild | Needs |
|---|---|---|---|
| Table S11 | `supp_tab_diagnostics.tex` | `build_supp_tab_diagnostics.py` | `outputs/` |
| Fig. S6 | — | `build_fig_supp_protection.py` | `outputs/` |
| Table S12 | `supp_tab_fc_perclass.tex` | `build_supp_tab_fc_perclass.py` | `outputs/` |
| Table S13 | — | `hpc/diag_fault_vs_switching.py --out <dir>` | `outputs/`, data |
| Table S14 | `supp_tab_current_shift.tex` | `build_supp_tab_current_shift.py` | data |
| Table S15 | `supp_tab_ed_negative_support.tex` | `build_supp_tab_ed_negative_support.py` | data |

`build_*` scripts are in `hpc/paper_artifacts/`; run them as
`PYTHONPATH=src python hpc/paper_artifacts/<script>`. `make supp-artifacts-dataset` runs the
dataset-backed ones. Table S13 (fault vs switching) uses the `event_state_global` predictions; its
script writes CSVs and the table into `<dir>`, and its definitions are in the script's header.
