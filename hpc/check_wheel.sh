#!/bin/bash
# check_wheel.sh — build the evemtbench wheel and prove it works OUTSIDE the repository (roadmap W3).
#
# Builds the wheel, installs it into a fresh virtual environment, and runs the evaluation layer from
# an empty directory, so nothing can be imported from the checkout. It checks:
#   * the wheel contains the package data (committed splits, contract, digests, schemas, py.typed)
#     and no repo-only trees (tests/, hpc/, paper/, outputs/, audit/);
#   * the installed package (not the checkout) imports, lists the tasks and reads a committed split;
#   * `evemtbench-evaluate --self-test` passes from the installed console script.
#
# Two modes:
#   --clean    CI (GitHub / GitLab runners have PyPI access): a plain venv; the wheel is built with
#              build isolation and installed WITH its declared dependencies from the index. This
#              also proves the dependency list in pyproject.toml is sufficient.
#   (default)  offline (HPC compute nodes have no internet): the venv borrows the third-party
#              packages of $PY (--system-site-packages); build without isolation, install --no-deps.
#              Run via srun, never on a login node:
#                srun -p a100mig --gres=gpu:1 --cpus-per-task=4 --time=20:00 bash hpc/check_wheel.sh
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="offline"
[ "${1:-}" = "--clean" ] && MODE="clean"
if [ "$MODE" = offline ] && [ -f hpc/clusters/local.env ]; then
  # shellcheck disable=SC1091
  source hpc/clusters/local.env
fi
PY="${PY:-python3}"
REPO="$PWD"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/evemt-wheel.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT
VENV="$WORK_DIR/venv"

if [ "$MODE" = clean ]; then
  "$PY" -m venv "$VENV"
  "$VENV/bin/python" -m pip install -q --upgrade pip
  "$VENV/bin/python" -m pip wheel --no-deps -q -w "$WORK_DIR/dist" "$REPO"
else
  "$PY" -m venv --system-site-packages "$VENV"
  "$PY" -m pip wheel --no-deps --no-build-isolation -q -w "$WORK_DIR/dist" "$REPO"
fi
WHEEL="$(ls "$WORK_DIR"/dist/evemtbench-*.whl)"
echo "built $(basename "$WHEEL") ($MODE mode)"

"$VENV/bin/python" - "$WHEEL" <<'PYEOF'
import sys, zipfile
names = zipfile.ZipFile(sys.argv[1]).namelist()
need = ["evemtbench/contract/contract.json", "evemtbench/data/input_digests.json",
        "evemtbench/py.typed", "evemtbench/baselines/run_record.schema.json"]
missing = [n for n in need if n not in names]
splits = [n for n in names if n.startswith("evemtbench/splits/v") and n.endswith(".txt")]
leaked = [n for n in names if n.split("/")[0] in ("tests", "hpc", "paper", "outputs", "audit")]
print(f"wheel: {len(names)} files, {len(splits)} split files")
assert not missing, f"missing from wheel: {missing}"
assert splits, "no committed split files in the wheel"
assert not leaked, f"repo-only files in the wheel: {leaked[:5]}"
PYEOF

if [ "$MODE" = clean ]; then
  "$VENV/bin/python" -m pip install -q "$WHEEL"
else
  "$VENV/bin/python" -m pip install -q --no-deps --no-index "$WHEEL"
fi

cd "$WORK_DIR"   # an empty directory: the checkout is not importable from here
unset PYTHONPATH
"$VENV/bin/python" - <<'PYEOF'
import pathlib
from importlib.metadata import version

import evemtbench
from evemtbench.contract import CONTRACT_SHA256
from evemtbench.splits.loader import read_held_out_splits
from evemtbench.tasks.registry import list_tasks

loc = pathlib.Path(evemtbench.__file__).resolve()
assert "site-packages" in str(loc), f"imported from {loc}, not the installed wheel"
parts = read_held_out_splits("double_line")  # committed split files, read from the wheel
assert all(parts[k] for k in ("train", "val", "test")), "empty committed split"
print(f"installed evemtbench {version('evemtbench')} from {loc.parent}")
print(f"{len(list_tasks())} tasks; contract sha {CONTRACT_SHA256[:12]}; double_line held_out: "
      + ", ".join(f"{k}={len(v)}" for k, v in parts.items()))
PYEOF
"$VENV/bin/evemtbench-evaluate" --self-test
echo "check_wheel: OK ($MODE mode)"
