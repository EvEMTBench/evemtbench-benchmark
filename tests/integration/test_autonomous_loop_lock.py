"""Singleton-lock regression test for hpc/autonomous_loop.sh.

The script must let exactly one instance run at a time: a second concurrent
launch has to exit cleanly while the first holds an flock on
``hpc/hpc_logs/autonomous_loop.lock``. Without that guard two instances both
tick hourly, doubling the log and -- far worse -- racing on SLURM submission
and on ``git push origin main``.

These tests drive the *real* script in a hermetic sandbox: SLURM (``squeue``,
``submit.sh``), ``git``, and the python entrypoint are all replaced by PATH/``PY``
stubs, so a tick never touches the cluster or the git remote. We assert on the
process exit code and on the log file the script writes.
"""

from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "hpc" / "autonomous_loop.sh"

# util-linux flock + advisory flock(2) semantics; linux-only. The script under test is
# operator-only machinery pruned from the public release export — skip cleanly there.
pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("flock") is None or not SCRIPT.exists(),
    reason="autonomous_loop.sh relies on util-linux flock (linux-only)",
)


def _stub(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)


def _sandbox(tmp_path: Path):
    """Build a fake PROJECT_DIR + stub bin so the script runs with no cluster/remote.

    Returns (proj_dir, env, calls_log). Every stub appends to ``calls_log`` so a
    test can assert exactly which side effects were (not) triggered.
    """
    proj = tmp_path / "proj"
    (proj / "hpc").mkdir(parents=True)
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    calls = tmp_path / "calls.log"

    # SLURM queue always idle (no stdout) -> the idle-guard lets a tick proceed.
    _stub(stub_bin / "squeue", f'echo "squeue $*" >> {calls}\n')
    # git: record the subcommand and succeed. `git diff --quiet` returning 0 means
    # "no changes", so regen_and_push never reaches add/commit/push.
    _stub(stub_bin / "git", f'echo "git $*" >> {calls}\nexit 0\n')
    # submit.sh: if it ever runs, record it -- a 0-pending tick must NOT submit.
    _stub(proj / "hpc" / "submit.sh", f'echo "submit $*" >> {calls}\n')
    # python entrypoint: coverage reports 0 pending; everything else is a noop.
    py = stub_bin / "fakepy"
    _stub(
        py,
        f'echo "py $*" >> {calls}\n'
        'case "$*" in\n'
        '  *coverage*--summary*) echo "pending 0" ;;\n'
        "esac\n"
        "exit 0\n",
    )

    env = dict(os.environ)
    env.update(
        PROJECT_DIR=str(proj),
        SRC_DATA=str(tmp_path / "nodata"),  # ieee39 trigger sees no manifests -> skipped
        PY=str(py),
        ITERS="1",
        SLEEP="0",
        PATH=f"{stub_bin}:{env['PATH']}",
        USER=env.get("USER", "tester"),  # script uses $USER under `set -u`
    )
    return proj, env, calls


def _run(env, proj):
    # `bash <script>` (not `-l`) so the env/PATH we pass is used verbatim and no
    # login profile re-shadows the stubs.
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        cwd=str(proj),
        capture_output=True,
        text=True,
        timeout=60,
    )


def _log(proj: Path) -> str:
    return (proj / "hpc" / "hpc_logs" / "autonomous_loop.log").read_text()


def test_runs_a_tick_when_lock_is_free(tmp_path):
    proj, env, calls = _sandbox(tmp_path)

    res = _run(env, proj)

    assert res.returncode == 0, res.stderr
    log = _log(proj)
    assert "autonomous loop start" in log
    assert "autonomous loop end" in log  # ITERS=1 completed the loop
    assert "all cells done (0 pending)" in log  # idle-guard path, no submission
    # hermetic: never submitted a SLURM wave, never pushed to the remote
    calls_txt = calls.read_text() if calls.exists() else ""
    assert "submit" not in calls_txt
    assert "git push" not in calls_txt


def test_second_instance_exits_when_lock_held(tmp_path):
    proj, env, calls = _sandbox(tmp_path)
    lock = proj / "hpc" / "hpc_logs" / "autonomous_loop.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("424242\n")  # sentinel "holder" pid for the diagnostic message

    # Hold the lock like a live first instance. fcntl.flock uses flock(2), the same
    # advisory lock the script's `flock -n` takes -- so they genuinely contend.
    fd = os.open(str(lock), os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        res = _run(env, proj)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    assert res.returncode == 0, res.stderr  # clean no-op, not an error
    log = _log(proj)
    assert "another autonomous_loop instance is running (pid 424242)" in log
    assert "autonomous loop start" not in log  # bailed before doing any work
    # nothing ran past the guard: no SLURM, no git
    calls_txt = calls.read_text() if calls.exists() else ""
    assert "submit" not in calls_txt
    assert "git" not in calls_txt


def test_lock_is_released_on_exit(tmp_path):
    """Back-to-back runs must both succeed: the first releases the lock when it exits."""
    proj, env, _ = _sandbox(tmp_path)

    first = _run(env, proj)
    second = _run(env, proj)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    log = _log(proj)  # both runs append to the same log file
    assert log.count("autonomous loop start") == 2
    assert "another autonomous_loop instance is running" not in log
