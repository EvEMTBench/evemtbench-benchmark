#!/bin/bash
# make_public_release.sh — build the public GitHub export, gated by a leak scan.
#
#   bash hpc/make_public_release.sh [OUT_DIR]      # default OUT_DIR: dist/
#   bash hpc/make_public_release.sh --update       # sync the EXISTING public clone (new commit
#                                                  #   on top of its main — for every push after
#                                                  #   the first; never rewrite public history)
#   bash hpc/make_public_release.sh --scan-only <dir>    # run just the scan (test hook)
#   bash hpc/make_public_release.sh --print-prune        # list the prune set (test hook)
#
# The public repository (github.com/EvEMTBench/evemtbench-benchmark) is a curated FRESH-HISTORY
# export of the private development repo: development history contains the private manuscript and
# internal ops docs, so it is never pushed. This script stages a CLEAN `git archive` export of
# HEAD (untracked files can't leak), prunes the private/operator-only paths below, verifies the
# public tree is self-consistent, then runs a FORBIDDEN-STRING SCAN (hpc/lib_leak_scan.sh —
# fail-closed, occurrence-level allowlists, self-safe patterns so this script and its tests ship
# with no scan exclusions) over every staged file. Any hit => exit 1 and nothing is written. On
# success it emits a ready-to-push git repo with a single release commit.
set -euo pipefail

# --- The prune set -----------------------------------------------------------------------------
# (a) private content: the manuscript + reviews, internal ops/agent docs, agent config;
# (b) operator-only machinery whose value is running OUR sweep (daemons, watchers, mailers,
#     one-time migrations, machine-specific job wrappers) — the scientific machinery
#     (sweep/submit/rescore/builders/gates) all ships.
PRUNE=(
  paper
  docs/internal
  docs/design/cross_cluster.md
  docs/results_audit/02-understandingvcs-case-study.md
  docs/results_audit/03-evemtbench-design.md
  CLAUDE.md
  .claude
  ONBOARDING.md
  .gitlab
  .gitlab-ci.yml
  .github/workflows/paper.yml
  hpc/autonomous_loop.sh
  hpc/autocommit_overview.sh
  hpc/autocommit_progress.sh
  hpc/refresh_results.sh
  hpc/notify.sh
  hpc/heartbeat_check.sh
  hpc/watch_ieee39_windows.sh
  hpc/nightly_integrity.sh
  hpc/setup_shared_outputs.sh
  hpc/flip_split_stratified.sh
  hpc/train_nhr.sh
  hpc/build_paper.py
  hpc/rescore_val.sbatch
  hpc/rescore_transfer_val.sbatch
)
if [ "${1:-}" = "--print-prune" ]; then
  printf '%s\n' "${PRUNE[@]}"
  exit 0
fi

# --- The scan: shared fail-closed core + this script's policy -----------------------------------
# Policy vs the repro-pack scan: NHR HOSTNAMES ARE ALLOWED — node and proxy names are public
# infrastructure and appear in provenance and docs. The private forge is forbidden, except the
# producer-contract schema $id namespace — allowlisted per matched OCCURRENCE, so it can never
# mask another forbidden string on the same line.
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/lib_leak_scan.sh"
leak_scan() {
  local bad=1
  if leak_scan_core "$1" "${LEAK_COMMON_PATTERNS[@]}"; then bad=0; fi
  if leak_scan_occurrences "$1" 'git5\.cs\.fau\.de[A-Za-z0-9./_-]*' \
    'git5\.cs\.fau\.de/juoelhaf/power-grid-and-ai-helper-functions'; then bad=0; fi
  return "$bad"
}
if [ "${1:-}" = "--scan-only" ]; then
  # Explicit usage error: under `set -u` a missing target otherwise dies with "$2: unbound
  # variable", which reads like a script bug rather than a missing argument. Exit 2 keeps it
  # distinguishable from exit 1 = "scan found something".
  [ -n "${2:-}" ] || { echo "usage: $0 --scan-only <dir-or-file>" >&2; exit 2; }
  if leak_scan "$2"; then
    echo "LEAK SCAN FAILED — forbidden strings above." >&2
    exit 1
  fi
  echo "leak scan clean."
  exit 0
fi

cd "$(git rev-parse --show-toplevel)"
# shellcheck disable=SC1091
if [ -f hpc/clusters/local.env ]; then source hpc/clusters/local.env; fi
PY="${PY:-python3}"
UPDATE=0
if [ "${1:-}" = "--update" ]; then
  UPDATE=1
  shift
fi
OUT_DIR="${1:-dist}"
VERSION="$(PYTHONPATH=src "$PY" -c 'from evemtbench.version import BENCHMARK_VERSION; print(BENCHMARK_VERSION)')"
[ -n "$VERSION" ] || { echo "ERROR: could not resolve BENCHMARK_VERSION (is \$PY usable?)" >&2; exit 1; }
SHA="$(git rev-parse --short=12 HEAD)"
NAME="evemtbench-benchmark"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "staging clean export of HEAD ($SHA, version $VERSION) ..."
mkdir -p "$STAGE/$NAME"
git archive HEAD | tar -x -C "$STAGE/$NAME"

echo "pruning private/operator-only paths ..."
for item in "${PRUNE[@]}"; do
  rm -rf "$STAGE/$NAME/${item:?}"
  if [ -e "$STAGE/$NAME/$item" ]; then
    echo "ERROR: prune failed for $item" >&2
    exit 1
  fi
done

# --- Self-consistency: everything the public repo advertises must actually ship -----------------
REQUIRED=(
  LICENSE LICENSES/DATA-LICENSE.md CITATION.cff README.md CONTRIBUTING.md CHANGELOG.md
  pyproject.toml requirements.lock requirements-ci-py3.10.lock requirements-ci-py3.12.lock
  env/results-v1.0.0.lock env/results-scoring.constraints
  src/evemtbench/version.py src/evemtbench/splits/v1.0
  RESULTS.md leaderboard.html coverage_snapshot.json
  "results/$VERSION" audit/index.html "audit/provenance/$VERSION/index.json"
  .github/workflows/ci.yml .github/workflows/nightly.yml
  index.html .nojekyll
  docs/quickstart.md docs/protocol.md docs/DATA_LAYOUT.md
  hpc/sweep.sh hpc/submit.sh hpc/make_results.py hpc/make_results_bundle.py hpc/make_audit.py
  hpc/check_results_fresh.py hpc/rescore_validation.py hpc/paper_artifacts/paper_lib.py
  hpc/lib_leak_scan.sh tests examples
)
for item in "${REQUIRED[@]}"; do
  [ -e "$STAGE/$NAME/$item" ] || { echo "ERROR: required public path missing: $item" >&2; exit 1; }
done

# ---- THE LEAK-SCAN GATE: any hit (or scan error — fail closed) refuses the export ---------------
if leak_scan "$STAGE/$NAME"; then
  echo
  echo "LEAK SCAN FAILED — forbidden strings above; export NOT written." >&2
  exit 1
fi
echo "leak scan clean."

GIT_NAME="$(git config user.name)"
GIT_EMAIL="$(git config user.email)"
DEST="$OUT_DIR/${NAME}-public"

if [ "$UPDATE" = 1 ]; then
  # --- Sync the EXISTING public clone: replace its tree with the staged export, commit on top ---
  # Public history is append-only (forks/clones/stars depend on it) — never a rewritten root.
  [ -d "$DEST/.git" ] || { echo "ERROR: --update needs an existing clone at $DEST (first release: run without --update, push, keep the dir)" >&2; exit 1; }
  rsync -a --delete --exclude=.git "$STAGE/$NAME/" "$DEST/"
  git -C "$DEST" add -A
  if git -C "$DEST" diff --cached --quiet; then
    echo "public clone already matches the export of $SHA — nothing to commit."
    exit 0
  fi
  git -C "$DEST" -c user.name="$GIT_NAME" -c user.email="$GIT_EMAIL" commit -q \
    -m "sync with development repository @ ${SHA}" \
    -m "Curated, leak-scanned export (v${VERSION})."
  echo "committed sync in $DEST — push with:"
  echo "  git -C $DEST push origin main"
  exit 0
fi

# --- First release: emit a ready-to-push repo with a single release commit ----------------------
git -C "$STAGE/$NAME" init -q -b main
git -C "$STAGE/$NAME" add -A
git -C "$STAGE/$NAME" -c user.name="$GIT_NAME" -c user.email="$GIT_EMAIL" commit -q \
  -m "EvEMTBench v${VERSION} — public release export" \
  -m "Curated, leak-scanned export of the development repository at ${SHA}."

mkdir -p "$OUT_DIR"
rm -rf "$DEST"
mv "$STAGE/$NAME" "$DEST"
echo "wrote $DEST ($(du -sh "$DEST" | cut -f1)) — single-commit git repo on branch 'main'."
echo "push with:"
echo "  git -C $DEST remote add origin git@github.com:EvEMTBench/evemtbench-benchmark.git"
echo "  git -C $DEST push -u origin main"
