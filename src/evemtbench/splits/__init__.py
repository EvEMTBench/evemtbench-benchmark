"""Layer 3 — splits as versioned, committed artifacts.

Splits are generated at the **simulation-episode level**: every window from one EMT
simulation (one ``sample_id`` / ``sim_idx``) stays in a single partition (Benchmark Plan
§7.6). The committed ``v{SPLITS_VERSION}/<protocol>/{train,val,test}.txt`` files are the
source of truth; :mod:`evemtbench.splits.generator` can regenerate them deterministically,
and :mod:`evemtbench.splits.leakage` validates them in CI.
"""
