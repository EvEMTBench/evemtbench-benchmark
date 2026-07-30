#!/bin/bash -l
#SBATCH --job-name=evemt-extract-aux
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100small:1
#SBATCH --time=04:00:00
#SBATCH --output=./hpc/hpc_logs/%x-%j.out
#SBATCH --error=./hpc/hpc_logs/%x-%j.out
# ============================================================================
# NHR@FAU Alex — extract aux artifacts (topology graph + settings.csv) from the per-grid data
# tarballs into the additive aux overlay ($EVEMTBENCH_AUX_ROOT). GPU-less pre-processing, routed to
# a MIG slice (CPU/IO only); large tarballs decompress slowly (ieee39 benchmark is ~142 G, minutes).
# The producer-owned data root is never modified.
#
# Env knobs: GRID (key or 'all', default all), FAMILY (benchmark|adapt_grid, default benchmark).
#   GRID=double_line FAMILY=benchmark sbatch hpc/extract_aux.sh
#   GRID=all FAMILY=adapt_grid sbatch hpc/extract_aux.sh
# ============================================================================
set -eo pipefail
mkdir -p ./hpc/hpc_logs
module load python 2>/dev/null || echo "WARN: 'module load python' failed (run on Alex)"
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate "${CONDA_ENV:-py_dl}" 2>/dev/null \
  || conda activate "${CONDA_ENV_PATH:-$WORK/software/private/conda/envs/py_dl}" 2>/dev/null || true

PROJECT_DIR="${PROJECT_DIR:-$HOME/Repositories/evemtbench-benchmark}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
# Machine profile: sourced unconditionally (caller-wins `:=` defaults) — gating it on one
# variable would leave the others half-configured when that one is preset.
# shellcheck disable=SC1091
if [ -f "$PROJECT_DIR/hpc/clusters/local.env" ]; then source "$PROJECT_DIR/hpc/clusters/local.env"; fi
: "${EVEMT_ROOT:?set EVEMT_ROOT to the EvEMTBench dataset root (or define it in hpc/clusters/local.env)}"
export EVEMT_ROOT
export EVEMTBENCH_AUX_ROOT="${EVEMTBENCH_AUX_ROOT:-$(dirname "$EVEMT_ROOT")/evemtbench-aux}"
GRID="${GRID:-all}"
FAMILY="${FAMILY:-benchmark}"

echo "extract_aux: GRID=$GRID FAMILY=$FAMILY  data=$EVEMT_ROOT  aux=$EVEMTBENCH_AUX_ROOT"
python -m evemtbench.data.extract_aux --grid "$GRID" --family "$FAMILY"
echo "--- manifest ---"
python -m evemtbench.data.manifest
