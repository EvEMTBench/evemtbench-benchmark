#!/bin/bash
# make_repro_pack.sh — build the reviewer repro pack, gated by a leak scan.
#
#   bash hpc/make_repro_pack.sh [OUT_DIR]     # default OUT_DIR: dist/
#
# Adapted from UnderstandingVCS tools/make_supplement_zip.sh: stages a CLEAN `git archive`
# export (never the working tree — untracked files can't leak), keeps the reviewer-facing
# artifacts, captures live gate transcripts (the BATTERY_RESULT.txt idiom), then runs a
# FORBIDDEN-STRING SCAN over every staged file. Any hit => exit 1 and the zip is NOT written.
# On success: zip + SHA256SUMS.
set -uo pipefail

# The scan: shared fail-closed core (hpc/lib_leak_scan.sh) + this pack's stricter policy — the
# reviewer pack additionally forbids cluster hostnames and ANY private-forge reference (unlike the
# public-release scan, which allows hostnames and the producer-schema $id namespace). --scan-only
# <dir> exposes it to the test suite so the refusal tests exercise THESE patterns, not a copy.
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/lib_leak_scan.sh"
leak_scan() {
  local bad=1
  if leak_scan_core "$1" "${LEAK_COMMON_PATTERNS[@]}" \
    '[a-z0-9.-]+\.nhr\.fau\.de'; then bad=0; fi
  if leak_scan_occurrences "$1" 'git5\.cs\.fau\.de[A-Za-z0-9./_-]*'; then bad=0; fi
  return "$bad"
}
if [ "${1:-}" = "--scan-only" ]; then
  if leak_scan "$2"; then
    echo "LEAK SCAN FAILED — forbidden strings above." >&2
    exit 1
  fi
  echo "leak scan clean."
  exit 0
fi

cd "$(git rev-parse --show-toplevel)" || exit 1
# Machine profile: sourced unconditionally (caller-wins `:=` defaults).
# shellcheck disable=SC1091
if [ -f hpc/clusters/local.env ]; then source hpc/clusters/local.env; fi
PY="${PY:-python3}"
OUT_DIR="${1:-dist}"
VERSION="$(PYTHONPATH=src "$PY" -c 'from evemtbench.version import BENCHMARK_VERSION; print(BENCHMARK_VERSION)')"
PACK="evemtbench_repro_pack_${VERSION}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "staging clean export of HEAD ($(git rev-parse --short=12 HEAD)) ..."
git archive HEAD | tar -x -C "$STAGE"

# Keep only the reviewer-facing artifacts (the audit is the entry point).
# `env` carries results-v<version>.lock — the environment the numbers were actually computed in.
# requirements.lock ships too, but only because its sha256 is the lockfile_sha256 in every v1.0.0
# run record; it is NOT the reproduction recipe (no torch). See docs/determinism.md.
KEEP=(audit RESULTS.md coverage_snapshot.json env requirements.lock requirements-ci-py3.10.lock
      requirements-ci-py3.12.lock README.md LICENSE docs CITATION.cff)
mkdir -p "$STAGE/$PACK"
for item in "${KEEP[@]}"; do
  [ -e "$STAGE/$item" ] && mv "$STAGE/$item" "$STAGE/$PACK/"
done
find "$STAGE" -mindepth 1 -maxdepth 1 ! -name "$PACK" -exec rm -rf {} +
# internal ops docs (cluster paths, runbooks) are not reviewer-facing — never ship them
rm -rf "$STAGE/$PACK/docs/internal"
rm -f "$STAGE/$PACK/docs/design/cross_cluster.md"  # federation ops doc (cluster paths)

# Live gate transcripts (only meaningful where outputs/ lives; recorded honestly either way).
{
  echo "== evemtbench repro pack transcript =="
  echo "built: $(date -u '+%F %T UTC')  commit: $(git rev-parse HEAD)  version: $VERSION"
  echo
  echo "-- evaluator self-test (dataless) --"
  PYTHONPATH=src "$PY" -m evemtbench.evaluation.cli --self-test 2>&1 || true
  echo
  echo "-- coverage snapshot self-check --"
  PYTHONPATH=src "$PY" -m evemtbench.coverage --check-snapshot 2>&1 || true
  echo
  echo "-- analytic floor gate --"
  PYTHONPATH=src "$PY" -m evemtbench.coverage --check-floor 2>&1 || true
  echo
  echo "-- artifact integrity (fast) --"
  PYTHONPATH=src "$PY" -m evemtbench.integrity 2>&1 || true
} > "$STAGE/$PACK/PACK_TRANSCRIPT.txt"

# ---- THE LEAK-SCAN GATE: any hit below refuses the pack (UVCS: "zip NOT written") ----------
# Absolute home/cluster paths, key/token material, account names, hostnames, forge references —
# see the shared core (hpc/lib_leak_scan.sh) + this pack's policy at the top of this script.
if leak_scan "$STAGE/$PACK"; then
  echo
  echo "LEAK SCAN FAILED — forbidden strings above; pack NOT written." >&2
  exit 1
fi
echo "leak scan clean."

mkdir -p "$OUT_DIR"
(cd "$STAGE" && zip -qr "$PACK.zip" "$PACK")
mv "$STAGE/$PACK.zip" "$OUT_DIR/"
(cd "$OUT_DIR" && sha256sum "$PACK.zip" > "$PACK.zip.SHA256SUMS")
echo "wrote $OUT_DIR/$PACK.zip ($(du -h "$OUT_DIR/$PACK.zip" | cut -f1)) + SHA256SUMS"
