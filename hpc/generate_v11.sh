#!/usr/bin/env bash
# Generate the corrected v1.1 multi_grid source-corpus splits + machine-readable audits.
# Topology-grouped = the final benchmark split (committed); simulation-grouped variant is
# generated to a scratch root and only its audits are kept (documented alternative).
set -euo pipefail
cd "$HOME/Repositories/evemtbench-benchmark"
source hpc/clusters/local.env
export PYTHONPATH=src

SIMROOT="${SIMROOT:-local/simulation_variant}"
mkdir -p "$SIMROOT"

echo "=== topology-grouped (final) -> packaged splits dir, v1.1 ==="
"$PY" -m evemtbench.splits.generator \
  --protocol multi_grid_pretrain --stratify-by event_type \
  --grouping topology --splits-version 1.1

echo "=== committed-split audits (written next to the split files) ==="
"$PY" -m evemtbench.splits.audit --protocol multi_grid_pretrain --write

echo "=== simulation-grouped (required variant, scratch root) ==="
"$PY" -m evemtbench.splits.generator \
  --protocol multi_grid_pretrain --stratify-by event_type \
  --grouping simulation --splits-version 1.1 --out "$SIMROOT"

echo "=== simulation-variant audits ==="
"$PY" -m evemtbench.splits.audit --protocol multi_grid_pretrain \
  --splits-root "$SIMROOT" --write

echo "=== regeneration verification (committed topology split) ==="
"$PY" - <<'EOF'
from evemtbench.splits.leakage import verify_protocol_regenerates
for g in ("template_20kv", "template_110kv", "template_345kv"):
    verify_protocol_regenerates("multi_grid_pretrain", g)
    print(f"{g}: regenerates byte-identically OK")
EOF
echo "ALL DONE"
