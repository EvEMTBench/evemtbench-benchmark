#!/bin/bash
# Build (or --check) label patch LP-1 from the 1.x source labels — see hpc/build_label_patch.py.
# CPU-only (label tables, no X bytes); Alex needs a --gres on every job and forbids --mem:
#
#   MG_SETTINGS=<extracted Template20kV/labels/settings_clean.csv> \
#     sbatch -p a100mig --gres=gpu:a100small:1 hpc/build_label_patch.sbatch            # write
#   MODE=--check MG_SETTINGS=... sbatch -p a100mig --gres=gpu:a100small:1 hpc/build_label_patch.sbatch
#
# MG_SETTINGS comes from the 1.x archive multi_grid/multigrid-Template20kV.tar.gz (member
# Template20kV/labels/settings_clean.csv), e.g.
#   pigz -dc multigrid-Template20kV.tar.gz | tar -x -C <dir> --wildcards '*/labels/settings_clean.csv'
# Submit from the repo root (SLURM_SUBMIT_DIR anchors the checkout the patch is written into).
#SBATCH --job-name=evemt-label-patch
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=./hpc/hpc_logs/%x-%j.out
#SBATCH --error=./hpc/hpc_logs/%x-%j.out
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_DIR"
[ -f hpc/build_label_patch.py ] || { echo "ERROR: '$PROJECT_DIR' is not the repo root" >&2; exit 1; }
mkdir -p ./hpc/hpc_logs/revision
# shellcheck disable=SC1091
if [ -f hpc/clusters/local.env ]; then source hpc/clusters/local.env; fi
PY="${PY:-python}"
: "${EVEMT_ROOT:?set EVEMT_ROOT (or define it in hpc/clusters/local.env)}"
: "${MG_SETTINGS:?set MG_SETTINGS to the extracted Template20kV settings_clean.csv}"
export EVEMT_ROOT EVEMTBENCH_DATA_ROOT="$EVEMT_ROOT" CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
MODE="${MODE:-}"
echo "===== label patch LP-1 ${MODE:-write} $(date -Is) git $(git rev-parse --short HEAD) ====="
exec "$PY" hpc/build_label_patch.py --mg-settings "$MG_SETTINGS" \
  --audit hpc/hpc_logs/revision/LP-1_audit.md ${MODE:+$MODE}
