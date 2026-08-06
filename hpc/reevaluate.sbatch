#!/bin/bash
# Offline re-evaluation: rebuild every result.json from the stored per-seed predictions.
# NO retraining, NO GPU compute, NO projection cache, NO raw window bytes — evaluate() touches only
# labels + metadata, so this is a label-table read plus metric arithmetic (see
# src/evemtbench/evaluation/reevaluate.py for the controls that gate every write).
#
# Alex is GPU-only with no cpu partition and forbids --mem on GPU jobs, so this CPU-only pass takes
# the smallest MIG slice, exactly like hpc/build_cache.sh:
#
#   # 1) full pass, one array element per (task, protocol, grid) work unit
#   sbatch --array=0-227%24 --gres=gpu:a100small:1 --partition=a100mig hpc/reevaluate.sbatch
#   # 2) idempotency gate — must report "0 would change" and exit 0
#   MODE=--check sbatch --array=0-227%24 --gres=gpu:a100small:1 --partition=a100mig hpc/reevaluate.sbatch
#
# NUM_SHARDS must match the array width. Each result.json belongs to exactly one work unit, so no
# two elements ever write the same file; writes are tmp + atomic replace.
#SBATCH --job-name=evemt-reeval
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=./hpc/hpc_logs/%x-%A_%a.out
#SBATCH --error=./hpc/hpc_logs/%x-%A_%a.out
set -euo pipefail
# SLURM copies the batch script to a node-local spool dir, so $BASH_SOURCE points at the copy, not
# at hpc/ — SLURM_SUBMIT_DIR (the dir sbatch was run from) is the only reliable anchor here. The
# guard below fails loudly rather than running the shards against the wrong tree.
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_DIR"
[ -f hpc/reevaluate.sbatch ] || { echo "ERROR: '$PROJECT_DIR' is not the repo root — submit from there, or set PROJECT_DIR" >&2; exit 1; }
mkdir -p ./hpc/hpc_logs
# shellcheck disable=SC1091
if [ -f hpc/clusters/local.env ]; then source hpc/clusters/local.env; fi
PY="${PY:-python}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
: "${EVEMT_ROOT:?set EVEMT_ROOT (or define it in hpc/clusters/local.env)}"
export EVEMT_ROOT
export EVEMTBENCH_DATA_ROOT="$EVEMT_ROOT"
export EVEMTBENCH_WANDB=0
# Keep this off the GPU and single-threaded: the sklearn/numpy reductions are what must reproduce
# bit-for-bit against the recorded per-seed metrics, and threaded reductions do not guarantee that.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

NUM_SHARDS="${NUM_SHARDS:-228}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
MODE="${MODE:-}"   # set MODE=--check for the verify-only idempotency gate
REPORT="./hpc/hpc_logs/reeval-report-${SLURM_ARRAY_JOB_ID:-local}-${SHARD}.json"

echo "===== reevaluate shard $SHARD/$NUM_SHARDS ${MODE:-write} $(date -Is) ====="
exec "$PY" -m evemtbench.evaluation.reevaluate \
  --shard "$SHARD" --num-shards "$NUM_SHARDS" \
  --report "$REPORT" \
  ${MODE:+$MODE}
