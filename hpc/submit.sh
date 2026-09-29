#!/bin/bash -l
# ============================================================================
# NHR@FAU Alex / TinyGPU — submit driver (run on the LOGIN node — no compute here).
#
# CLUSTER=alex (default) | tinygpu selects a profile in hpc/clusters/<CLUSTER>.env that supplies the
# per-cluster gres/partition/MIG/--mem/conda defaults, so this one driver federates a single sweep
# across both clusters (they share the data root + outputs on the shared OUT). Everything below
# is cluster-agnostic.
#
# Alex is a GPU-only cluster: EVERY training/eval allocation gets a GPU, so there is NO cpu/gpu
# job split — that only made sense on LME. By default we submit ONE GPU job per grid running ALL
# baselines. Each job runs hpc/sweep.sh, which re-checks the coverage manifest and skips done cells
# (idempotent). gres routes the partition (a40 default; GRES=gpu:a100:1 for A100). No --partition,
# no node excludes, no --mem (Alex allocates ~16 cores + 60/120/240 G host RAM per GPU).
# AUTO-ROUTING: an all-classical BASELINES set (majority/threshold/random_forest — CPU-only) is sent to
# a MIG slice on a100mig with MIG_CPUS cores instead of a full A40, so CPU work doesn't burn a whole
# GPU (GRES= disables it). The autonomous loop exploits this by submitting TWO waves per grid
# (BASELINES + JOBTAG=cls|deep) — see autonomous_loop.sh::submit_waves. JOBTAG appends a per-grid
# job-name suffix (multi-grid safe, unlike the full JOBNAME override).
#
# 24h WALL / >24h WORK. The cluster caps a job at 24h. Two knobs finish a large sweep under that cap
# (a cell = one (task,baseline) over all seeds is the atomic, resumable unit — coverage skips done
# cells, so re-running only continues what's left):
#   SPLIT_BASELINES=1  -> one job(-chain) PER baseline, not per grid (fewer cells/job + parallelism).
#   CHAIN=K            -> K links per job chained by --dependency=afterany:<prev>; when a link ENDS
#                         (incl. TIMEOUT) the next starts, re-queries pending, and continues past 24h.
#                         Self-terminating (a 0-pending link exits fast). K default 1 (no chain).
# Example (finish the timed-out deep sweep):
#   SPLIT_BASELINES=1 CHAIN=3 GRIDS="double_line testgrid_110kv cigre_mv ieee39" \
#     BASELINES="mlp gru cnn resnet" hpc/submit.sh --submit
#
# NORMALIZE=auto|none|per_window -> forces the input normalization (default auto = per-protocol:
#                         none for held_out, per_window for pretrain/transfer). Only for ablations.
# Ablation example — held_out (random init) WITH the transfer protocols' per-window normalization,
# into an ISOLATED tree, W&B off, and job names that do NOT match autonomous_loop.sh's '^held_out'
# idle-guard (JOBNAME takes one grid at a time; see the guard below):
#   OUT=$ABL/outputs NORMALIZE=per_window EVEMTBENCH_WANDB=0 GRIDS=double_line \
#     JOBNAME=pwctrl.double_line TASKS="fault_location_line fault_classification_line" \
#     BASELINES="mlp gru cnn resnet" SPLIT_BASELINES=1 hpc/submit.sh --submit
#
# DRY-RUN BY DEFAULT: prints the plan + sbatch lines. Pass --submit to actually queue.
#   hpc/submit.sh                                    # dry-run, ready grids, full task suite
#   GRIDS="double_line testgrid_110kv" hpc/submit.sh --submit
#   GRIDS=double_line TASKS=fault_detection_local BASELINES=mlp hpc/submit.sh --submit
# ============================================================================
set -eo pipefail
PROJECT_DIR="${PROJECT_DIR:-$HOME/Repositories/evemtbench-benchmark}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

# Cluster profile: the per-cluster SLURM/env differences (gres, partition, MIG, --mem, conda) live in
# hpc/clusters/<CLUSTER>.env so ONE driver serves Alex + TinyGPU.
# Detect an explicit GRES override BEFORE loading the profile so the profile default never counts as one.
if [ -n "${GRES:-}" ]; then GRES_OVERRIDDEN=1; else GRES_OVERRIDDEN=0; fi
CLUSTER="${CLUSTER:-alex}"
CLUSTER_ENV="$PROJECT_DIR/hpc/clusters/$CLUSTER.env"
[ -f "$CLUSTER_ENV" ] || { echo "ERROR: unknown CLUSTER=$CLUSTER (no $CLUSTER_ENV)"; exit 2; }
# shellcheck disable=SC1090
source "$CLUSTER_ENV"

eval "${MODULE_CMD:-}" 2>/dev/null || true
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate "${CONDA_ENV:-py_dl}" 2>/dev/null \
  || conda activate "${CONDA_ENV_PATH:-$WORK/software/private/conda/envs/py_dl}" 2>/dev/null || true
PY="${PY:-python}"

SUBMIT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --submit) SUBMIT=1 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac; shift
done

PROTOCOL="${PROTOCOL:-held_out}"
# Default to the grids that are windowed on Alex right now (double_line + testgrid_110kv).
GRIDS="${GRIDS:-double_line testgrid_110kv}"
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
SEEDS="${SEEDS:-0 1 2 3 4}"; EPOCHS="${EPOCHS:-60}"
BASELINES="${BASELINES:-majority threshold random_forest mlp gru cnn resnet}"

# Resource spec (override per run). Defaults come from the cluster profile: DEFAULT_GRES, and
# DEFAULT_PARTITION (empty on Alex — gres selects the partition; set on TinyGPU — -p picks the type).
# GRES_OVERRIDDEN was detected above, before the profile was sourced.
GRES="${GRES:-$DEFAULT_GRES}"
TIME="${TIME:-24:00:00}"
# CPUs per task (data-loader workers). Default from the cluster profile; overriding raises throughput
# for the CPU-bound input pipeline (audit F2) but requesting > cores-per-GPU strands GPUs — the
# projection cache is the better fix. Empty -> use sweep.sh's #SBATCH default.
CPUS_PER_TASK="${CPUS_PER_TASK:-}"
# Optional SLURM dependency, passed straight to sbatch (e.g. DEP=afterok:<jobid> to chain a sweep
# behind a re-windowing job — the sweep sits PENDING (Dependency) and only starts if that job exits 0).
DEP="${DEP:-}"
# >24h controls (see header): afterany-chain length and per-baseline fan-out. SOFT_DEADLINE_SEC is
# passed to sweep.sh so a link stops on a whole-cell boundary before the 24h wall. The default is
# PER-GRID (grid_soft_deadline below): a deadline must leave at least the grid's longest cell as
# margin, or a cell started just under the deadline dies at the wall and its work is lost (job
# 3861691, testgrid_110kv: ~2.25h resnet cell + flat 23h deadline -> TIMEOUT mid-cell). An explicit
# SOFT_DEADLINE_SEC=... still overrides every job of this invocation.
CHAIN="${CHAIN:-1}"
SPLIT_BASELINES="${SPLIT_BASELINES:-0}"
SOFT_DEADLINE_OVERRIDE="${SOFT_DEADLINE_SEC:-}"

# CPU-only classical baselines (majority/threshold/random_forest) never touch the GPU, so a job whose
# BASELINES are ALL classical is auto-routed to a MIG slice on the a100mig partition instead of holding
# a full A40 (raises acc_utilization; the streamed threshold/rf comfortably fit a slice's RAM). Deep or
# mixed baseline sets stay on a full GPU. An explicit GRES= override disables the auto-routing. The
# decision is made per submitted baseline SUBSET, so it is correct under SPLIT_BASELINES too.
CLASSICAL_SET=" majority threshold random_forest "
# MIG_GRES / MIG_PARTITION / SUPPORTS_MIG come from the cluster profile (Alex: on; TinyGPU: off).
is_classical() { local b; for b in $1; do case "$CLASSICAL_SET" in *" $b "*) ;; *) return 1 ;; esac; done; return 0; }

# Per-grid soft deadline (seconds): ~24h minus 1.5x the grid's longest observed cell (2026-07-17
# audit of the v1.0.0 sweep logs), rounded to 30 min. Classical-only jobs have short cells (<=1h)
# on every grid. Unknown grids get a conservative 20h until measured.
grid_soft_deadline() {  # $1=grid  $2=baselines
  if is_classical "$2"; then echo 81000; return; fi   # 22.5h (longest classical cell ~59min)
  case "$1" in
    double_line)    echo 81000 ;;  # 22.5h (longest cell ~51min, resnet)
    ieee39)         echo 77400 ;;  # 21.5h (longest cell ~90min, mlp)
    testgrid_110kv) echo 73800 ;;  # 20.5h (longest cell ~135min, resnet)
    cigre_mv)       echo 66600 ;;  # 18.5h (longest cell ~195min, cnn)
    *)              echo 72000 ;;  # 20h
  esac
}

# JOBNAME is a full --job-name override, meant for a SINGLE grid. With multiple grids it would name
# every job identically, so refuse it (SPLIT_BASELINES still disambiguates by appending .<baseline>).
# JOBTAG is the multi-grid-safe alternative: a SUFFIX appended per grid ('<protocol>.<grid>.<tag>'),
# used by autonomous_loop.sh to disambiguate its two waves (JOBTAG=cls / JOBTAG=deep) while keeping
# the 'held_out' prefix the loop's idle-guard greps.
if [ -n "${JOBNAME:-}" ] && [ "$(echo "$GRIDS" | wc -w)" -gt 1 ]; then
  echo "ERROR: JOBNAME=$JOBNAME set with multiple GRIDS ($GRIDS) — jobs would share a name." >&2
  echo "       Submit one grid per JOBNAME (or unset JOBNAME to use the default '<protocol>.<grid>')." >&2
  exit 2
fi

n_pending() {  # $1=grid  $2=baselines
  # Read the SAME outputs root the sweep writes to ($OUT), so under a shared OUT one cluster never
  # re-submits cells the other already finished. Unset -> repo-local "outputs" (single-cluster default).
  # shellcheck disable=SC2086
  $PY -m evemtbench.coverage --out "${OUT:-outputs}" --tasks $TASKS --grids "$1" \
      --protocols "$PROTOCOL" --baselines $2 --emit-pending | grep -c . || true
}

submit_job() {  # $1=grid  $2=baselines  $3=job-name  — submits a CHAIN of afterany-linked jobs
  local grid="$1" bl="$2" name="$3" pend
  pend="$(n_pending "$grid" "$bl")"
  if [ "$pend" -eq 0 ]; then echo "  skip $name (0 pending)"; return; fi
  local gres="$GRES" device=cuda; local -a part=()
  local cpt="${CPUS_PER_TASK:-}"
  # TinyGPU needs an explicit -p to select the GPU type; Alex leaves DEFAULT_PARTITION empty (gres picks it).
  [ -n "${DEFAULT_PARTITION:-}" ] && part=(--partition="$DEFAULT_PARTITION")
  # GPU-utilisation fix (audit F1): CPU-only baselines must NEVER hold a full GPU (they run the card at
  # ~0% util). On a MIG-capable cluster (Alex) force them to the smallest accelerator (a MIG slice) EVEN
  # when GRES is overridden; the only escape is an explicit ALLOW_CLASSICAL_GPU=1. On a cluster without
  # MIG (TinyGPU) there is no cheap slice, so warn — classical belongs on Alex.
  if is_classical "$bl" && [ "${ALLOW_CLASSICAL_GPU:-0}" != 1 ]; then
    if [ "${SUPPORTS_MIG:-0}" = 1 ]; then
      [ "$GRES_OVERRIDDEN" = 1 ] && echo "  note: $name is CPU-only -> forcing MIG ($MIG_GRES), ignoring GRES=$GRES (set ALLOW_CLASSICAL_GPU=1 to override)"
      gres="$MIG_GRES"; part=(--partition="$MIG_PARTITION"); device=cpu
      cpt="${MIG_CPUS:-8}"   # a MIG slice's fair CPU share is smaller than a full GPU's 16
    else
      echo "  WARN: $name is CPU-only but CLUSTER=$CLUSTER has no MIG -> it would idle a $gres GPU. Run classical on Alex (or set ALLOW_CLASSICAL_GPU=1 to force here)."
      device=cpu
    fi
  fi
  # --mem: forbidden on Alex GPU jobs (USE_MEM=0); allowed on TinyGPU if MEM is set.
  local -a mem=()
  [ "${USE_MEM:-0}" = 1 ] && [ -n "${MEM:-}" ] && mem=(--mem="$MEM")
  # --cpus-per-task: overrides sweep.sh's #SBATCH default when the profile/caller sets it.
  local -a cpus=()
  [ -n "$cpt" ] && cpus=(--cpus-per-task="$cpt")
  local sdl; sdl="${SOFT_DEADLINE_OVERRIDE:-$(grid_soft_deadline "$grid" "$bl")}"
  # NORMALIZE/OUT/EVEMTBENCH_WANDB are named EXPLICITLY rather than left to ride on `ALL`, so the
  # dry-run plan line below shows them before anything is queued. Both of the unnamed ones were
  # silent, irreversible hazards for an ablation wave: a wrong OUT overwrites published cells (the
  # cell dir does not encode `normalize`), and W&B resumes by a cell key that ignores --out entirely
  # (runner.cell_identity + _tracking.run_id_for, resume="allow"), so it would overwrite the
  # official run's config/summary even with a perfectly isolated offline tree.
  local exp; exp="ALL,CLUSTER=$CLUSTER,GRID=$grid,PROTOCOL=$PROTOCOL,TASKS=$TASKS,BASELINES=$bl,SEEDS=$SEEDS,EPOCHS=$EPOCHS,DEVICE=$device,ZERO_SEQ=${ZERO_SEQ:-0},NORMALIZE=${NORMALIZE:-auto},OUT=${OUT:-},EVEMTBENCH_WANDB=${EVEMTBENCH_WANDB:-1},SOFT_DEADLINE_SEC=$sdl"
  local prev="" link jid lname
  for link in $(seq 1 "$CHAIN"); do
    local -a dep=()
    if [ -n "$prev" ]; then dep=(--dependency="afterany:$prev")   # link 2..K: continue after prev ENDS (incl TIMEOUT)
    elif [ -n "$DEP" ]; then dep=(--dependency="$DEP"); fi         # link 1: honour caller DEP (e.g. afterok:pretrain)
    lname="$name"; [ "$CHAIN" -gt 1 ] && lname="$name.l$link"
    local -a flags=(--job-name="$lname" --time="$TIME" --gres="$gres" "${part[@]}" "${cpus[@]}" "${mem[@]}" "${dep[@]}")
    echo "  $lname : $pend pending${prev:+ (afterany:$prev)}  ->  sbatch ${flags[*]} --export=$exp hpc/sweep.sh"
    if [ "$SUBMIT" = 1 ]; then
      jid="$(sbatch --parsable "${flags[@]}" --export="$exp" hpc/sweep.sh)"
      echo "    submitted $jid"; prev="$jid"
    else
      prev="<${lname}>"   # dry-run placeholder so the next link's dependency prints sensibly
    fi
  done
}

submit_grid() {  # $1=grid
  local grid="$1" base b
  base="${JOBNAME:-${PROTOCOL}.${grid}${JOBTAG:+.$JOBTAG}}"
  if [ "$SPLIT_BASELINES" = 1 ]; then
    for b in $BASELINES; do submit_job "$grid" "$b" "$base.$b"; done
  else
    submit_job "$grid" "$BASELINES" "$base"
  fi
}

echo "=== submit plan (SUBMIT=$SUBMIT cluster=$CLUSTER) protocol=$PROTOCOL grids=[$GRIDS] baselines=[$BASELINES]"
echo "    gres=$GRES partition=${DEFAULT_PARTITION:-<from-gres>} time=$TIME chain=$CHAIN split_baselines=$SPLIT_BASELINES ==="
for grid in $GRIDS; do
  echo "[$grid]"
  submit_grid "$grid"
done
[ "$SUBMIT" = 0 ] && echo "(dry-run — pass --submit to queue these jobs)"
exit 0
