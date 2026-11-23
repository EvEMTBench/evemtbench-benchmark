# Fault-versus-switching diagnostic (SI Table S13)

Output of `hpc/diag_fault_vs_switching.py` for the reference runs: how often the stored
`event_state_global` predictions (`held_out` protocol, six learned baselines, five seeds) confuse
faults with switching, on the in-distribution `test` split and on the `benchmark` family (the
paper's `eval_grid`). The definitions are frozen in the script's header and repeated in
`mapping.json`. No model is trained or loaded and no threshold is set.

| File | Content |
|---|---|
| `si_table.tex` | SI Table S13, byte-identical to the table in the paper |
| `rates_summary.csv` | the three rates per grid, baseline and test set: seed mean, s.d., min, max, window and episode support; overall and broken down by event type and switching category |
| `rates_per_seed.csv` | the same rates for each seed |
| `confusion_summary.csv`, `confusion_per_seed.csv` | 3×3 true-by-predicted state shares (normal / switching / fault) |
| `selection.csv` | the baseline the validation rule selects per grid and test set (marked † in the table) |
| `controls.csv` | the positive control per cell: split hashes, seeds, and the headline metric recomputed from each stored prediction file |
| `mapping.json` | the class mapping and rate definitions |
| `inputs.sha256` | SHA-256 of the 312 input files that were read (stored predictions and result records) |

Rates: (a) `switch_to_fault`, fault predictions among active switching windows; (b)
`missed_fault`, non-fault predictions among active fault windows; (c) `inactive_to_fault`, fault
predictions among inactive windows. A rate without support is empty, never 0. Windows overlap in
time and are not independent, so no window-level confidence intervals are given.

## Provenance

- Code: release v1.3.0 (`hpc/diag_fault_vs_switching.py`, SHA-256
  `763b9f13fc01cf326f21949eb250fa520ccd5f992d14561d3beae82bb2c3f750`), run on 2026-11-23.
- Controls: 48 baseline–grid–set cells checked, 0 failed, tolerance 0 (240 seed files).
- The script's `summary.json` is not included, because it records machine-specific paths.

## Rebuild

Needs the stored predictions (`outputs/`) and the windowed dataset; see
[reproduce/4_diagnostics.md](../../../reproduce/4_diagnostics.md).

```bash
PYTHONPATH=src python hpc/diag_fault_vs_switching.py --out <dir>
```
