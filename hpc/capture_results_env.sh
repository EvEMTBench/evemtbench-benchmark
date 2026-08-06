#!/bin/bash -l
# ============================================================================
# Capture the RESULTS environment into env/results-v<VERSION>.lock.
#
# The results environment is a mutable conda prefix on the cluster, so the only honest pin we can
# ship is a capture of what it contained when the sweep ran. This script produces that capture
# deterministically (sorted, header-stamped) so re-running it on an unchanged prefix is a no-op and
# any drift shows up as a reviewable diff.
#
# NOT the CI pin: CI installs requirements-ci-py3.*.lock (hash-enforced, torch-free). See
# docs/determinism.md for why the two are separate files.
#
# Runs via SLURM, never on the login node (CLAUDE.md hard rule). Alex is GPU-only and forbids
# --mem on GPU jobs, so this metadata-only job takes the smallest MIG slice:
#   srun --gres=gpu:a100small:1 --partition=a100mig --cpus-per-task=2 --time=00:10:00 \
#     bash hpc/capture_results_env.sh
#
# After a capture that changes the file, run hpc/backfill_results_env.py so existing run records
# either still match (no-op) or are reported as mismatching — never silently re-stamped.
# ============================================================================
set -eo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"
# shellcheck disable=SC1091
if [ -f "$PROJECT_DIR/hpc/clusters/local.env" ]; then source "$PROJECT_DIR/hpc/clusters/local.env"; fi
PY="${PY:-python}"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
# Keep the capture off the GPU: this job only reads package metadata.
export CUDA_VISIBLE_DEVICES=""

VERSION="${VERSION:-$("$PY" -c 'import evemtbench; print(evemtbench.BENCHMARK_VERSION)')}"
OUT="${OUT:-env/results-v${VERSION}.lock}"
mkdir -p "$(dirname "$OUT")"

# `pip list --format=freeze` rather than `pip freeze`: in a conda prefix, freeze emits
# `name @ file:///<conda-build-root>/feedstock_root/...` for conda-built wheels, which pins nothing a third
# party can act on. --format=freeze normalises every distribution to `name==version`.
#
# Two deliberate exclusions, both needed for the capture to be deterministic:
#   * PYTHONPATH is cleared for this one call. pip discovers distributions along sys.path, so an
#     inherited PYTHONPATH=src lets the source tree's egg-info shadow the installed dist and the
#     captured version flips between runs (observed: 0.2.0.dev0 vs 0.1.0).
#   * `evemtbench` itself is filtered out. It is the package being pinned, not one of its pins;
#     including it is circular, and its version is the least stable line in the file. The identity
#     of the code that produced the results is `git_sha` in every run_record.json.
pkgs="$(PYTHONPATH= "$PY" -m pip list --format=freeze \
  | grep -E '^[A-Za-z0-9._-]+==' | grep -vE '^evemtbench==' | sort -f)"
pyvv="$("$PY" -c 'import sys; print(sys.version.replace(chr(10), " "))')"
prefix="$("$PY" -c 'import sys; print(sys.prefix)')"
plat="$("$PY" -c 'import platform; print(platform.platform())')"
torchv="$("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '<not installed>')"

{
  cat <<HDR
# EvEMTBench v${VERSION} — the RESULTS environment.
#
# This is the environment that actually produced every number in outputs/${VERSION} (and therefore in
# RESULTS.md, the leaderboard, the audit site and the manuscript). It is captured, not authored:
# \`python -m pip list --format=freeze\` inside the conda prefix below, run on an Alex compute node.
#
# It is NOT the CI environment. CI installs requirements-ci-py3.10.lock / requirements-ci-py3.12.lock,
# which pin a deliberately torch-free subset for the lint+test gate. Do not conflate the two:
#   * this file              -> what the published results were computed with (provenance)
#   * requirements-ci-*.lock -> what CI installs (hash-enforced, reproducible test gate)
# See docs/determinism.md.
#
# Provenance of this capture:
#   conda prefix : ${prefix}
#   python       : ${pyvv}
#   torch        : ${torchv}
#   platform     : ${plat}
#   cluster      : NHR@FAU Alex (a100mig slice; results ran on A40 / A100-MIG)
#
# Caveat, stated plainly: the prefix is a mutable conda environment, so this freeze is a faithful
# record of its contents at capture time rather than a hash-verified, byte-reconstructable
# specification. The four packages whose versions move metric values (numpy, pandas, scikit-learn,
# scipy) are independently recorded per cell in every run_record.json \`versions\` block, and all 1876
# records agree with this file exactly — that cross-check is the real guarantee. \`results_env_sha256\`
# in each run record is the sha256 of this file.
#
# Regenerate with: bash hpc/capture_results_env.sh
HDR
  printf '%s\n' "$pkgs"
} > "$OUT"

echo "wrote $OUT ($(grep -cE '^[A-Za-z0-9._-]+==' "$OUT") packages)"
"$PY" - "$OUT" <<'EOF'
import hashlib, sys
print("sha256", hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())
EOF
