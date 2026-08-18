#!/usr/bin/env bash
# Full verification: fixture regen + ruff (CI pin) + ruff format --check + mypy + pytest.
# Run on a compute node (CPU-only is fine: CUDA_VISIBLE_DEVICES= for rtxpro6k/Blackwell).
set -uo pipefail
cd "$HOME/Repositories/evemtbench-benchmark"
source hpc/clusters/local.env
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=
export http_proxy=http://proxy.nhr.fau.de:80 https_proxy=http://proxy.nhr.fau.de:80

RUFFVENV="${RUFFVENV:-local/ruff-ci-venv}"
if [ ! -x "$RUFFVENV/bin/ruff" ]; then
  "$PY" -m venv "$RUFFVENV" && "$RUFFVENV/bin/pip" -q install ruff==0.16.1
fi

rc=0
echo "=== fixture regeneration ==="
"$PY" tests/fixtures/generate_fixture.py || rc=1

echo "=== ruff check (CI pin $("$RUFFVENV/bin/ruff" --version)) ==="
"$RUFFVENV/bin/ruff" check src tests hpc || rc=1
echo "=== ruff format --check ==="
"$RUFFVENV/bin/ruff" format --check src tests hpc || rc=1

echo "=== mypy src ==="
"$PY" -m mypy src || rc=1

echo "=== pytest ==="
"$PY" -m pytest -q 2>&1 | tail -30 || rc=1

echo "VERIFY_EXIT=$rc"
exit $rc
