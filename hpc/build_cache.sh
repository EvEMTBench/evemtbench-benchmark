#!/bin/bash -l
#SBATCH --job-name=evemt-buildcache
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=./hpc/hpc_logs/%x-%j.out
#SBATCH --error=./hpc/hpc_logs/%x-%j.out
# ============================================================================
# I/O prep job — build the persistent projection cache OFF the training GPUs.
#
# Stage a grid's raw windows to node-local /scratch (sequential, bandwidth-limited, locked), build
# the small content-addressed projection cache ONCE for every view, and publish it to the persistent
# cache root on $WORK. After this, GPU training jobs read the small cache read-only and never touch
# raw — so GPUs are never held idle for I/O (the #1 thing cluster admins flag). See the plan /
# docs/internal/operating.md.
#
# Alex is GPU-only with NO cpu partition and FORBIDS --mem on GPU jobs, so this prep runs on the
# SMALLEST accelerator (a MIG a100small slice) rather than a full A40 — pass the gres at submit:
#   GRID=double_line sbatch --gres=gpu:a100small:1 --partition=a100mig hpc/build_cache.sh
#   GRID=ieee39 TASKS="..." sbatch --gres=gpu:a100small:1 --partition=a100mig --time=24:00:00 hpc/build_cache.sh
#   # $TMPDIR is 7-14 TB node-local SSD, so even ieee39's wide global caches fit trivially.
#
# hpc/sweep.sh reads this persistent cache when present (EVEMTBENCH_PROJECTION_CACHE_ROOT /
# EVEMTBENCH_CACHE_READONLY): it node-local-copies the small per-view cache and SKIPS the raw stage.
# ============================================================================
set -eo pipefail
mkdir -p ./hpc/hpc_logs
module load python 2>/dev/null || true
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate "${CONDA_ENV:-py_dl}" 2>/dev/null \
  || conda activate "${CONDA_ENV_PATH:-$WORK/software/private/conda/envs/py_dl}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/Repositories/evemtbench-benchmark}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
# Machine profile: sourced unconditionally (caller-wins `:=` defaults) — gating it on one
# variable would leave the others (PY, ...) half-configured when that one is preset.
# shellcheck disable=SC1091
if [ -f "$PROJECT_DIR/hpc/clusters/local.env" ]; then source "$PROJECT_DIR/hpc/clusters/local.env"; fi
PY="${PY:-python}"
: "${EVEMT_ROOT:?set EVEMT_ROOT to the EvEMTBench dataset root (or define it in hpc/clusters/local.env)}"
export EVEMT_ROOT

GRID="${GRID:?set GRID}"
PROTOCOL="${PROTOCOL:-held_out}"
TASKS="${TASKS:-fault_detection_local fault_detection_line fault_detection_global \
fault_classification_local fault_classification_line fault_classification_global \
fault_location_local fault_location_line fault_location_global}"
# Persistent cache lives on $WORK (survives the job); pin it BEFORE stage_grid repoints
# EVEMTBENCH_DATA_ROOT to node-local $TMPDIR, else persistent_cache_root() resolves to the
# ephemeral staged dir and the published cache vanishes at job end.
export EVEMTBENCH_PROJECTION_CACHE_ROOT="${EVEMTBENCH_PROJECTION_CACHE_ROOT:-$WORK/evemtbench/projcache}"
SCRATCH="${TMPDIR:-/tmp}"
echo "build_cache ${SLURM_JOB_ID:-$$} on $(hostname) | GRID=$GRID | GPUs=${CUDA_VISIBLE_DEVICES:-none} | start $(date)"

# 1) Stage raw to local scratch so the projection's random reads hit local SSD, not GlusterFS.
#    Stage the family the protocol reads (mirrors sweep.sh): multi_grid for multi_grid_pretrain,
#    else adapt_grid+benchmark (held_out). Without this, pretrain-grid cache builds fail (no
#    adapt_grid/benchmark family exists for the multi_grid template grids).
source hpc/lib_stage.sh
if [ "$PROTOCOL" = "multi_grid_pretrain" ]; then STAGE_FAMS="multi_grid"; else STAGE_FAMS="adapt_grid benchmark"; fi
stage_grid "$GRID" "$STAGE_FAMS"
export EVEMTBENCH_DATA_ROOT="$STAGE"
echo "data root for build: $EVEMTBENCH_DATA_ROOT"

# 2) Build every view's cache to LOCAL scratch (CPU), dedup by content key.
LOCAL_CACHE="$SCRATCH/projcache"; mkdir -p "$LOCAL_CACHE"
export EVEMTBENCH_PROJECTION_CACHE="$LOCAL_CACHE"
unset EVEMTBENCH_CACHE_READONLY   # this is the ONE place allowed to build
# shellcheck disable=SC2086
$PY - "$PROTOCOL" "$GRID" $TASKS <<'PY'
import os, sys
from evemtbench.tasks.task import load_task
proto, grid, *tasks = sys.argv[1:]
cdir = os.environ["EVEMTBENCH_PROJECTION_CACHE"]
built, skipped = set(), 0
for t in tasks:
    task = load_task(t, proto, grid=grid)
    # Defensive only: every split currently resolves for both documented protocols (the
    # multi_grid_pretrain builds report "0 view(s) absent"). A protocol that later omits one should
    # skip that view rather than abort a multi-hour build, and the count below makes it visible
    # instead of silent. NB the real constraint is TASKS: multi_grid_pretrain supports the 12
    # local/line tasks only, so passing the held_out default (which includes *_global) fails at
    # load_task — pass the protocol's own task set (python -m evemtbench.coverage lists them).
    for vn in ("train", "val", "test", "benchmark"):
        try:
            v = task.view(vn)
        except (KeyError, ValueError, FileNotFoundError) as e:
            skipped += 1
            print(f"skip {t}/{vn}: {type(e).__name__}: {e}", flush=True)
            continue
        fn = v.cache_filename()
        if fn and fn not in built:
            v.materialize_cache(cdir)
            built.add(fn)
            print("built", fn, flush=True)
print(f"total {len(built)} cache file(s) ({skipped} view(s) absent for {proto})", flush=True)
PY

# 3) Publish caches to the persistent root on /cluster (one sequential, bandwidth-limited write).
CROOT="$($PY -m evemtbench.data.staging cache-root)"
BW="$($PY -m evemtbench.data.staging bwlimit)"
mkdir -p "$CROOT"
ionice -c3 rsync -a --bwlimit="$BW" "$LOCAL_CACHE"/ "$CROOT"/
echo "published $(ls "$CROOT"/proj_*_"${GRID}"_*.f32 2>/dev/null | wc -l) cache file(s) for $GRID -> $CROOT"
echo "build_cache done at $(date)"
