#!/usr/bin/env bash
# Smoke tier (operating.md): pretrain -> zero-shot -> fine-tune on the corrected v1.1 splits,
# 1 epoch x 1 seed, throwaway --out. Proves the end-to-end path (grouped split load, per-window
# normalization, bundle save/restore) before the full waves. Reads the persistent projection
# cache read-only.
set -euo pipefail
cd "$HOME/Repositories/evemtbench-benchmark"
source hpc/clusters/local.env
export PYTHONPATH=src
export EVEMTBENCH_PROJECTION_CACHE="$WORK/evemtbench/projcache/1.1.0"
export EVEMTBENCH_CACHE_READONLY=1
export EVEMTBENCH_WANDB=0
OUT="${TMPDIR:-/tmp}/evemt_smoke_$$"
mkdir -p "$OUT"

run() { echo "--- runner $*"; "$PY" -m evemtbench.baselines.runner "$@" \
  --seeds 0 --epochs 1 --device cuda --out "$OUT"; }

run --task fault_detection_local --protocol multi_grid_pretrain --grid template_20kv --baseline mlp
run --task fault_detection_local --protocol transfer_zeroshot --grid double_line --baseline mlp \
  --pretrain-grid template_20kv
run --task fault_detection_local --protocol transfer_finetune --grid double_line --baseline mlp \
  --pretrain-grid template_20kv

echo "=== result aggregates ==="
find "$OUT" -name result.json | while read -r f; do
  echo "$f"
  "$PY" - "$f" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(" ", d["task"], d["protocol"], d["grid"], d["test_set"],
      {k: round(v["mean"], 4) for k, v in d["aggregate"].items()
       if isinstance(v, dict) and isinstance(v.get("mean"), float)})
EOF
done
echo "=== split provenance of the pretrain run ==="
"$PY" - "$OUT" <<'EOF'
import json, sys, pathlib
rec = next(pathlib.Path(sys.argv[1]).rglob("multi_grid_pretrain/**/run_record.json"))
d = json.loads(rec.read_text())
print("splits_version:", d["splits_version"], "| dir:", d["data"]["splits"]["dir"])
EOF
echo "SMOKE OK (out: $OUT)"
