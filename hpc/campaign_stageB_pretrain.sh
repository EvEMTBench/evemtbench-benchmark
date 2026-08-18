#!/usr/bin/env bash
# Stage B — the 144 multi_grid_pretrain cells (12 portable tasks x 4 deep baselines x 3 grids)
# under the corrected v1.1 topology-grouped splits. One chained deep wave per template grid;
# sweep.sh re-queries the coverage manifest per link, so links are idempotent and self-terminate.
set -euo pipefail
cd "$HOME/Repositories/evemtbench-benchmark"
TASKS12="fault_category_line fault_category_local fault_classification_line fault_classification_local fault_detection_line fault_detection_local fault_grounded_line fault_grounded_local fault_location_line fault_location_local fault_phase_line fault_phase_local"
for g in template_20kv template_110kv template_345kv; do
  PROTOCOL=multi_grid_pretrain GRIDS="$g" TASKS="$TASKS12" \
  BASELINES="mlp gru cnn resnet" JOBTAG=deep CHAIN="${CHAIN:-4}" \
    bash hpc/submit.sh --submit
done
