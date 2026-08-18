#!/usr/bin/env bash
# Stage C — the 1152 transfer cells (576 zero-shot + 576 fine-tune: 12 tasks x 4 baselines x
# 4 targets x 3 sources), initialised from the Stage-B checkpoints. Run ONLY after coverage
# reports all 144 multi_grid_pretrain cells done (a transfer cell hard-fails without its
# pretrain bundle). Zero-shot is eval-only (short); fine-tune trains, hence the longer chain.
set -euo pipefail
cd "$HOME/Repositories/evemtbench-benchmark"
source hpc/clusters/local.env
pending_pretrain=$(PYTHONPATH=src "$PY" -m evemtbench.coverage --summary \
  --protocols multi_grid_pretrain | awk '/pending|failed/{s+=$2} END{print s+0}')
if [ "$pending_pretrain" -ne 0 ]; then
  echo "REFUSING: $pending_pretrain multi_grid_pretrain cells not done yet" >&2
  exit 1
fi
TASKS12="fault_category_line fault_category_local fault_classification_line fault_classification_local fault_detection_line fault_detection_local fault_grounded_line fault_grounded_local fault_location_line fault_location_local fault_phase_line fault_phase_local"
for g in double_line cigre_mv testgrid_110kv ieee39; do
  PROTOCOL=transfer_zeroshot GRIDS="$g" TASKS="$TASKS12" \
  BASELINES="mlp gru cnn resnet" JOBTAG=zs CHAIN="${ZS_CHAIN:-2}" \
    bash hpc/submit.sh --submit
  PROTOCOL=transfer_finetune GRIDS="$g" TASKS="$TASKS12" \
  BASELINES="mlp gru cnn resnet" JOBTAG=ft CHAIN="${FT_CHAIN:-4}" \
    bash hpc/submit.sh --submit
done
