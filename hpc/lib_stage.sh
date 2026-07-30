#!/bin/bash -l
# ============================================================================
# Sourced helper — Alex staging of a grid's windows to node-local SSD, SHARED across co-located jobs.
#
# Provides:  stage_grid <grid>    (sets $STAGE — caller exports EVEMTBENCH_DATA_ROOT="$STAGE")
#
# Follows NHR@FAU's documented node-local staging pattern (https://doc.nhr.fau.de/data/staging/):
# instead of copying into the per-job $TMPDIR (wiped per job -> every co-located job re-copies the
# same ~GBs), we stage into a node-shared, JOB_CLASS-keyed dir  /tmp/$USER-<JOB_CLASS>  guarded by a
# flock + a `.complete` marker. The FIRST job of a class on a node stages; later jobs of the same
# class BLOCK on the lock, then reuse the staged data READ-ONLY (they copy nothing). It's
# opportunistic — jobs that land on different nodes each stage independently (we never force
# co-location; NHR warns that hurts scheduling). Safe because the runner writes only into the
# per-job projection cache ($EVEMTBENCH_PROJECTION_CACHE, kept in $TMPDIR by sweep.sh), never into
# the data root. No cleanup (mirrors NHR's template; node teardown reclaims /tmp).
#
# Each windows dir on Alex holds TWO configs (W0p020 + W0p050); the loader
# (windows.find_window_files) requires exactly one, so we stage ONLY the canonical W0p050_S0p005
# config (windows.py DEFAULT_CONTRACT). Stages the in-distribution `adapt_grid` family
# (train/val/test) and the held-out `benchmark` family.
#
# Requires: $EVEMT_ROOT (source mount), $USER, python on PATH. ($SCRATCH/$TMPDIR is NOT used for the
# shared stage — only for the per-job projection cache, which the caller sets.)
# ============================================================================
WIN_CONFIG="${WIN_CONFIG:-W0p050_S0p005}"     # canonical window contract (W0.05 / S0.005)

_stage_family() {  # $1=family $2=raw_foldername -> copies only *$WIN_CONFIG* files
  # Return code is meaningful: 0 = staged >=1 file OK; 1 = source ABSENT (benign — e.g. a grid with
  # no benchmark family); 2 = source PRESENT but staging incomplete (mkdir/cp failed, or zero
  # matching files) — a hard error the caller must NOT mask, else `.complete` would cache a broken
  # stage that co-located jobs then reuse.
  local fam="$1" raw="$2" src dst n=0 f
  src="$EVEMT_ROOT/$fam/windows/$raw"; dst="$STAGE/$fam/windows/$raw"
  [ -d "$src" ] || { echo "[stage] no '$fam' windows for $raw ($src)"; return 1; }
  mkdir -p "$dst" || { echo "[stage] ERROR: mkdir $dst failed" >&2; return 2; }
  for f in "$src"/*"$WIN_CONFIG"*; do
    [ -e "$f" ] || continue
    cp "$f" "$dst/" || { echo "[stage] ERROR: cp failed for $f" >&2; return 2; }
    n=$((n + 1))
  done
  echo "[stage] $fam/$raw: staged $n files ($WIN_CONFIG)"
  [ "$n" -gt 0 ] || { echo "[stage] ERROR: no $WIN_CONFIG files found under $src" >&2; return 2; }
  return 0
}

stage_grid() {  # $1=grid  [$2="fam1 fam2..." first mandatory]  -> sets $STAGE (node-shared reuse)
  local grid="$1" raw JOB_CLASS rc=0
  # Families to stage: default = held_out's adapt_grid (mandatory) + benchmark (optional). The
  # multi_grid_pretrain protocol passes "multi_grid"; the first family is mandatory, the rest
  # optional. The transfer protocols read the target's adapt_grid+benchmark (default) and the
  # pretrain checkpoint from outputs/ (shared FS, not staged), so they need no override.
  local families="${2:-adapt_grid benchmark}"
  local fam_arr mandatory fam
  read -r -a fam_arr <<< "$families"
  mandatory="${fam_arr[0]}"
  raw="$(python -c "from evemtbench.data.roots import get_grid; print(get_grid('$grid').raw_foldername)")" \
    || { echo "[stage] ERROR: could not resolve raw foldername for grid '$grid'" >&2; return 1; }
  [ -n "$raw" ] || { echo "[stage] ERROR: empty raw foldername for grid '$grid'" >&2; return 1; }
  # Class-key on grid + window config + mandatory family: baseline-split jobs of the same grid+family
  # share one stage; different grids/configs/family-sets/users never collide (path carries $USER).
  JOB_CLASS="evemt_${grid}_${WIN_CONFIG}_${mandatory}"
  STAGE="/tmp/${USER:?USER must be set}-${JOB_CLASS}"
  (umask 0077; mkdir -p "$STAGE") || { echo "[stage] ERROR: creating $STAGE failed" >&2; return 1; }
  # Only one job may stage; others block on the lock, then reuse (avoids a copy race). See NHR docs.
  (
    exec {FD}>"$STAGE/.lock"
    flock "$FD"
    # Reuse only if the marker is present AND the staged mandatory-family windows still exist —
    # guards a stale `.complete` left behind after /tmp cleanup reaped the data files.
    if [ -f "$STAGE/.complete" ] && compgen -G "$STAGE/$mandatory/windows/$raw/*$WIN_CONFIG*" >/dev/null; then
      echo "[stage] $grid: reusing shared stage at $STAGE (.complete + data present) — skipping copy"
      exit 0
    fi
    rm -f "$STAGE/.complete"   # incomplete/stale -> re-stage cleanly, only re-mark on success
    echo "[stage] $grid ($raw, $WIN_CONFIG): staging $EVEMT_ROOT -> $STAGE [families: $families]"
    # First family is mandatory: absent (1) or incomplete (2) are both fatal.
    _stage_family "$mandatory" "$raw" \
      || { echo "[stage] ERROR: $mandatory staging failed for $raw" >&2; exit 1; }
    # Remaining families are optional: absent (rc 1) -> that view is empty; present-but-incomplete
    # (rc 2) is a real error we must NOT silently mark complete. (|| keeps `set -e` from aborting.)
    for fam in "${fam_arr[@]:1}"; do
      rc=0
      _stage_family "$fam" "$raw" || rc=$?
      case "$rc" in
        0) : ;;  # staged
        1) echo "[stage] (no $fam windows -> that held-out view is empty)" ;;
        *) echo "[stage] ERROR: $fam windows present but staging incomplete for $raw" >&2; exit 1 ;;
      esac
    done
    : > "$STAGE/.complete"
  ) || return 1
}
