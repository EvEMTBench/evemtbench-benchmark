#!/bin/bash -l
#SBATCH --job-name=evemt-sweep
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=./hpc/hpc_logs/%x-%j.out
#SBATCH --error=./hpc/hpc_logs/%x-%j.out
#
# ============================================================================
# NHR@FAU Alex — multi-cell baseline worker (one allocation = one (grid, baseline group)).
#
# Runs only the cells the coverage manifest reports NOT done for the requested (GRID, TASKS,
# BASELINES), so re-submitting never repeats a completed cell. Each cell fits once per seed and
# evaluates BOTH held-out sets (adapt_grid `test` + `benchmark`), writing
# outputs/<version>/<task>/<protocol>/<grid>/<baseline>/<test_set>/result.json.
#
# Alex conventions: module-load conda on $WORK, $EVEMT_ROOT data root,
# $TMPDIR staging (canonical W0p050 only), a40/a100 gres set at sbatch time (NO --partition, NO
# --mem). The gres is overridable by the submit driver. Lets num_workers() self-tune (it caps at
# 4 without / 8 with the projection cache — do NOT force EVEMTBENCH_NUM_WORKERS here).
# Prebuilt projection caches are staged node-local LAZILY, PER CELL, with a one-task-ahead
# background prefetch (2026-07-17 audit: the old up-front full copy idled the GPU up to 2.4 h/link).
#
# Env knobs: CLUSTER (alex|tinygpu, selects hpc/clusters/<CLUSTER>.env), GRID (required), TASKS,
#            BASELINES, PROTOCOL, SEEDS, EPOCHS, DEVICE, ZERO_SEQ, OUT.
# ============================================================================
set -eo pipefail
mkdir -p ./hpc/hpc_logs
echo "sweep job ${SLURM_JOB_ID:-$$} on $(hostname) | start $(date) | GPU=${CUDA_VISIBLE_DEVICES:-none}"

PROJECT_DIR="${PROJECT_DIR:-$HOME/Repositories/evemtbench-benchmark}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

# --- Cluster profile: MODULE_CMD / CONDA_ENV[_PATH] (+ any shared OUT/EVEMT_ROOT defaults) come from
#     hpc/clusters/<CLUSTER>.env. CLUSTER is exported by hpc/submit.sh (default alex). ---
CLUSTER="${CLUSTER:-alex}"
CLUSTER_ENV="$PROJECT_DIR/hpc/clusters/$CLUSTER.env"
[ -f "$CLUSTER_ENV" ] || { echo "ERROR: unknown CLUSTER=$CLUSTER (no $CLUSTER_ENV)"; exit 1; }
# shellcheck disable=SC1090
source "$CLUSTER_ENV"

# --- Environment (module -> conda, per cluster profile) ---
eval "${MODULE_CMD:-}" 2>/dev/null || echo "WARN: MODULE_CMD='${MODULE_CMD:-}' failed"
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate "${CONDA_ENV:-py_dl}" 2>/dev/null \
  || conda activate "${CONDA_ENV_PATH:-$WORK/software/private/conda/envs/py_dl}" 2>/dev/null \
  || { echo "ERROR: conda activate failed — create the env first (deps: pyproject.toml; name/path knobs: CONDA_ENV / CONDA_ENV_PATH)"; exit 1; }

# Set in the caller's env or hpc/clusters/local.env (untracked machine profile) — never hardcoded.
: "${EVEMT_ROOT:?set EVEMT_ROOT to the EvEMTBench dataset root (or define it in hpc/clusters/local.env)}"
export EVEMT_ROOT
[ -d "$EVEMT_ROOT" ] || { echo "ERROR: \$EVEMT_ROOT not found: $EVEMT_ROOT"; exit 1; }

GRID="${GRID:?set GRID}"
PROTOCOL="${PROTOCOL:-held_out}"
SEEDS="${SEEDS:-0 1 2 3 4}"
EPOCHS="${EPOCHS:-60}"
DEVICE="${DEVICE:-$([ -n "${CUDA_VISIBLE_DEVICES:-}" ] && echo cuda || echo cpu)}"
DEST_OUT="${OUT:-$PROJECT_DIR/outputs}"; mkdir -p "$DEST_OUT"   # written directly (shared $HOME FS)

# --- Experiment tracking (W&B) — ONLINE channel, redundant with the offline artifact tree. Alex
#     compute nodes reach the internet via the NHR proxy, so we log live. WANDB_ENTITY + WANDB_API_KEY
#     propagate from the login-node env via submit.sh --export=ALL (a key is NEVER committed). Disable
#     with EVEMTBENCH_WANDB=0; WANDB_MODE=offline falls back to a syncable local dir (hpc/wandb_sync.sh). ---
export EVEMTBENCH_WANDB="${EVEMTBENCH_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
[ -n "${WANDB_ENTITY:-}" ] && export WANDB_ENTITY  # personal entity comes from the env, never a default
export WANDB_PROJECT="${WANDB_PROJECT:-evemtbench}"
export WANDB_DIR="${WANDB_DIR:-$DEST_OUT/_wandb}"; mkdir -p "$WANDB_DIR"
if [ "$WANDB_MODE" = "online" ]; then
  export http_proxy="${http_proxy:-http://proxy.nhr.fau.de:80}"
  export https_proxy="${https_proxy:-http://proxy.nhr.fau.de:80}"
fi
ZERO_SEQ="${ZERO_SEQ:-0}"; ZS_FLAG=$([ "$ZERO_SEQ" = 1 ] && echo "--zero-seq" || echo "")
# Stop starting new cells past this elapsed time so a link exits on a whole-cell boundary before the
# 24h wall kills it mid-training; the next afterany chain link continues the rest (default 23h).
SOFT_DEADLINE_SEC="${SOFT_DEADLINE_SEC:-82800}"

# Full task suite default (registry-defined); the submit driver narrows per batch.
TASKS="${TASKS:-\
fault_detection_local fault_detection_line fault_detection_global \
fault_classification_local fault_classification_line fault_classification_global \
fault_location_local fault_location_line fault_location_global \
event_state_global event_detection_global switch_detection_global \
switch_type_global event_classification_global \
fault_grounded_local fault_grounded_line fault_grounded_global \
fault_phase_local fault_phase_line fault_phase_global \
fault_category_local fault_category_line fault_category_global \
fault_origin_global}"
BASELINES="${BASELINES:-majority threshold random_forest mlp gru cnn resnet}"

# --- Ask the manifest for pending cells BEFORE staging, so a fully-done chain link exits instantly
#     without paying the stage copy. Coverage reads outputs/ + packaged splits (no staged data root).
#     Fail loud: an empty PENDING from a coverage crash is indistinguishable from "all done". ---
# shellcheck disable=SC2086
if ! pending_out="$(python -m evemtbench.coverage \
  --out "$DEST_OUT" --tasks $TASKS --grids "$GRID" --protocols "$PROTOCOL" \
  --baselines $BASELINES --emit-pending)"; then
  echo "ERROR: 'coverage --emit-pending' failed (env/import/registry?) — aborting instead of running 0 cells"
  exit 1
fi
mapfile -t PENDING <<< "$pending_out"
n_left=0; for _l in "${PENDING[@]}"; do [ -n "$_l" ] && n_left=$((n_left + 1)); done
if [ "$n_left" -eq 0 ]; then
  echo "0 pending for grid=$GRID protocol=$PROTOCOL baselines=[$BASELINES] — nothing to do (chain terminates)."
  exit 0
fi

# --- GPU-utilisation fix (audit F2/F3, reworked 2026-07-17): prefer a PREBUILT persistent
#     projection cache (created off-GPU by hpc/build_cache.sh), read read-only, SKIPPING the
#     ~479 GB raw window stage. The caches are staged node-local LAZILY, PER CELL (see the
#     stage-cell helpers below): the old up-front copy of every pending task's views idled the
#     GPU 15 min (56 GB, double_line) to 2.4 h (923 GB, ieee39) at the start of every chain link.
#     Fallback (no cache / any error): the original raw stage + per-job cache build, unchanged. ---
SCRATCH="${TMPDIR:-/tmp}"
export EVEMTBENCH_PROJECTION_CACHE="$SCRATCH/projcache"; mkdir -p "$EVEMTBENCH_PROJECTION_CACHE"
# Pin the persistent projection-cache root to match hpc/build_cache.sh's publish location
# ($WORK/evemtbench/projcache). Without this, persistent_cache_root() (data/staging.py) falls back to
# <data-root>/projcache and never finds the prebuilt caches -> USE_CACHE=0 -> per-job GPU-idle
# stage(~479 GB)+inline cache build. Keep in sync with build_cache.sh:45.
export EVEMTBENCH_PROJECTION_CACHE_ROOT="${EVEMTBENCH_PROJECTION_CACHE_ROOT:-$WORK/evemtbench/projcache}"
CROOT="$(python -m evemtbench.data.staging cache-root 2>/dev/null || true)"
USE_CACHE=0
# Unique tasks among the pending cells, order-preserving: the cache gate and the per-cell staging
# only ever touch what THIS link will actually run (not the full TASKS suite).
PEND_TASKS="$(printf '%s\n' "${PENDING[@]}" \
  | awk '{for(i=1;i<NF;i++) if($i=="--task") print $(i+1)}' | awk '!seen[$0]++')"
# multi_grid_pretrain is now cacheable too (hpc/build_cache.sh is family-aware as of the v1.0
# release). The `staging status` check below is the safety gate: USE_CACHE stays 0 (fallback raw
# stage + inline build) unless EVERY pending view's prebuilt cache is present, so enabling the fast
# path here can only help — it never serves an incomplete cache.
# shellcheck disable=SC2086
if [ -n "$CROOT" ] && [ -n "$PEND_TASKS" ] && \
   timeout 600 python -m evemtbench.data.staging status \
     --grid "$GRID" --tasks $PEND_TASKS --protocol "$PROTOCOL" --root "$EVEMT_ROOT" >/dev/null 2>&1; then
  USE_CACHE=1
fi
if [ "$USE_CACHE" = 1 ]; then
  echo "projection cache present ($CROOT) — staging per cell (lazy), SKIPPING raw stage"
  export EVEMTBENCH_CACHE_READONLY=1            # never build on the GPU allocation
  export EVEMTBENCH_DATA_ROOT="$EVEMT_ROOT"     # raw memmap opened but batches served from the cache
else
  source hpc/lib_stage.sh
  # Families to stage depend on the protocol: multi_grid_pretrain trains on the multi_grid corpus;
  # held_out + the transfer protocols read the grid's adapt_grid (+benchmark). Transfer additionally
  # loads the pretrain checkpoint from outputs/ (shared FS, not staged), so no override is needed.
  if [ "$PROTOCOL" = "multi_grid_pretrain" ]; then STAGE_FAMS="multi_grid"; else STAGE_FAMS="adapt_grid benchmark"; fi
  stage_grid "$GRID" "$STAGE_FAMS" || { echo "ERROR: staging failed for $GRID"; exit 1; }
  export EVEMTBENCH_DATA_ROOT="$STAGE"
fi
echo "Data root: $EVEMTBENCH_DATA_ROOT | proj_cache=$EVEMTBENCH_PROJECTION_CACHE | use_cache=$USE_CACHE | zero_seq=$ZERO_SEQ"

# --- stage-cell helpers (USE_CACHE=1 only): copy exactly ONE task's <=4 view-cache files
#     node-local, atomically (write .tmp.$$ then rename, so an existing file is ALWAYS complete —
#     the old `cp -n || true` could mistake a partial file from a killed copy for a finished one),
#     and prefetch the NEXT task's files in the background while the current cell trains. PENDING
#     is task-major, so cells 2..N of a task hit the .ok_<task> marker without another python call. ---
_copy_one() {  # $1=cache filename; loud on failure, never silent
  local fn="$1" tmp   # NB: $fn must not be referenced in the same `local` statement that sets it
  local dst="$EVEMTBENCH_PROJECTION_CACHE/$fn"
  [ -e "$dst" ] && return 0
  tmp="$dst.tmp.$BASHPID"   # BASHPID (not $$): unique per background copy — two concurrent
                            # copies of a view shared across tasks must never share a tmp path
  if cp "$CROOT/$fn" "$tmp" 2>/dev/null && mv -f "$tmp" "$dst"; then
    # Carry the identity sidecar along with the payload so the node-local copy can still be
    # identity-checked (a missing sidecar is tolerated by the loader, but then only the size check
    # runs). Best-effort: entries built before sidecars existed simply have none.
    local sfn="${fn%.f32}.json"
    if [ -e "$CROOT/$sfn" ] && [ ! -e "$EVEMTBENCH_PROJECTION_CACHE/$sfn" ]; then
      cp "$CROOT/$sfn" "$EVEMTBENCH_PROJECTION_CACHE/$sfn.tmp.$BASHPID" 2>/dev/null \
        && mv -f "$EVEMTBENCH_PROJECTION_CACHE/$sfn.tmp.$BASHPID" \
                 "$EVEMTBENCH_PROJECTION_CACHE/$sfn" 2>/dev/null \
        || rm -f "$EVEMTBENCH_PROJECTION_CACHE/$sfn.tmp.$BASHPID"
    fi
    return 0
  fi
  rm -f "$tmp"
  echo "!! stage-cell: copy failed for $fn (src present: $([ -e "$CROOT/$fn" ] && echo yes || echo NO);" \
       "dest avail: $(df -h --output=avail "$EVEMTBENCH_PROJECTION_CACHE" 2>/dev/null | tail -1 | tr -d ' '))" >&2
  return 1
}
ensure_views() {  # $1=task — make the task's view caches available node-local (memoized)
  local task="$1" fn files rc=0 t0=$SECONDS
  local marker="$EVEMTBENCH_PROJECTION_CACHE/.ok_$task"
  if [ -f "$marker" ]; then  # marker holds the file list: revalidate by existence (survives /tmp reaping)
    local ok=1
    while IFS= read -r fn; do [ -n "$fn" ] && [ ! -e "$EVEMTBENCH_PROJECTION_CACHE/$fn" ] && ok=0; done < "$marker"
    [ "$ok" = 1 ] && return 0
    rm -f "$marker"
  fi
  files="$(timeout 600 python -m evemtbench.data.staging files \
    --grid "$GRID" --tasks "$task" --protocol "$PROTOCOL" --root "$EVEMT_ROOT")" || return 1
  local -a pids=()
  while IFS= read -r fn; do [ -n "$fn" ] && { _copy_one "$fn" & pids+=("$!"); }; done <<< "$files"
  local p; for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ "$rc" -eq 0 ] || return 1
  printf '%s\n' "$files" > "$marker"
  echo "[stage-cell] $task: view caches ready in $((SECONDS - t0))s"
}
PREFETCH_PID=""; PREFETCH_TASK=""
start_prefetch() {  # $1=task ('' = none) — at most one prefetch in flight, never re-launched
  [ -z "$1" ] || [ "$1" = "$PREFETCH_TASK" ] && return 0
  [ -n "$PREFETCH_PID" ] && { wait "$PREFETCH_PID" 2>/dev/null || true; }
  ( ensure_views "$1" 2>&1 | sed 's/^/[prefetch] /' ) & PREFETCH_PID=$!; PREFETCH_TASK="$1"
}
reap_prefetch() {
  [ -n "$PREFETCH_PID" ] && { kill "$PREFETCH_PID" 2>/dev/null || true; wait "$PREFETCH_PID" 2>/dev/null || true; }
  PREFETCH_PID=""
}
trap reap_prefetch EXIT
task_of() { local w prev=""; for w in $1; do [ "$prev" = "--task" ] && { echo "$w"; return; }; prev="$w"; done; }
next_distinct_task() {  # $1=index of the current cell -> first later pending task that differs
  local i cur t; cur="$(task_of "${PENDING[$1]}")"
  for ((i = $1 + 1; i < ${#PENDING[@]}; i++)); do
    t="$(task_of "${PENDING[$i]}")"
    if [ -n "$t" ] && [ "$t" != "$cur" ]; then echo "$t"; return; fi
  done
}

echo "Sweep: grid=$GRID protocol=$PROTOCOL seeds=[$SEEDS] epochs=$EPOCHS device=$DEVICE | pending=$n_left"

fail=0; stage_fail=0
for idx in "${!PENDING[@]}"; do
  line="${PENDING[$idx]}"
  [ -z "$line" ] && continue
  if [ "$SECONDS" -ge "$SOFT_DEADLINE_SEC" ]; then
    echo "soft deadline ${SOFT_DEADLINE_SEC}s reached at ${SECONDS}s — stopping before next cell; the afterany chain link continues the rest"
    break
  fi
  if [ "$USE_CACHE" = 1 ]; then
    ctask="$(task_of "$line")"
    # if the background prefetch targeted this task, let it finish instead of racing it
    if [ "$PREFETCH_TASK" = "$ctask" ] && [ -n "$PREFETCH_PID" ]; then
      wait "$PREFETCH_PID" 2>/dev/null || true; PREFETCH_PID=""
    fi
    if ! ensure_views "$ctask"; then
      echo "!! FAILED (staging): $line (continuing)"; fail=1; stage_fail=$((stage_fail + 1))
      if [ "$stage_fail" -ge 3 ]; then
        echo "ERROR: $stage_fail staging failures (disk full / cache root mutated?) — aborting this link"
        exit 1
      fi
      continue
    fi
    start_prefetch "$(next_distinct_task "$idx")"   # overlaps the training below
  fi
  echo "===== $(date '+%H:%M:%S')  $line $ZS_FLAG ====="
  # shellcheck disable=SC2086
  if ! python -m evemtbench.baselines.runner $line $ZS_FLAG \
      --seeds $SEEDS --epochs "$EPOCHS" --device "$DEVICE" --out "$DEST_OUT"; then
    echo "!! FAILED: $line (continuing)"; fail=1
  fi
done
reap_prefetch

echo "sweep done at $(date) (fail=$fail). Results in $DEST_OUT"
exit 0
