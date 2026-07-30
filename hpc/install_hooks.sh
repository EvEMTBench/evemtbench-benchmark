#!/bin/bash
# Install the repo's git hooks by pointing core.hooksPath at the version-controlled .githooks/ dir.
# Run once per clone (on the loop/HPC host especially, where outputs/ lives and the freshness gate
# is meaningful). Mirrors akmaier/Agent4CT's install-registry-hook.sh.
#
#   bash hpc/install_hooks.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
chmod +x .githooks/* 2>/dev/null || true
git config core.hooksPath .githooks
echo "Installed git hooks: core.hooksPath -> .githooks"
echo "Active hooks: $(ls .githooks | tr '\n' ' ')"
echo "The pre-commit gate runs hpc/check_results_fresh.py when RESULTS.md/leaderboard.html/audit/*.html are staged."
