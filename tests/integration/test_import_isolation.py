"""The golden rule: the user-facing layers must NOT import the baselines (demo) layer.

A static source scan of ``data``, ``tasks``, ``splits``, ``evaluation`` for any import of
``evemtbench.baselines``. If this fails, a user could no longer evaluate their own model
without pulling in baseline code — comparability dies and people fork instead of comply.
"""

import ast
from pathlib import Path

import evemtbench

SRC = Path(evemtbench.__file__).resolve().parent
USER_LAYERS = ["data", "tasks", "splits", "evaluation"]


def _imports_baselines(path: Path) -> bool:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            # relative (from ..baselines / from .baselines) or absolute
            if mod.endswith("baselines") or "baselines" in mod.split("."):
                return True
            if node.level > 0 and any(a.name == "baselines" for a in node.names):
                return True
        elif isinstance(node, ast.Import):
            if any("baselines" in a.name.split(".") for a in node.names):
                return True
    return False


def test_no_user_layer_imports_baselines():
    offenders = []
    for layer in USER_LAYERS:
        for py in (SRC / layer).rglob("*.py"):
            if _imports_baselines(py):
                offenders.append(str(py.relative_to(SRC)))
    assert not offenders, f"user layers must not import baselines: {offenders}"


def test_evaluate_path_does_not_require_baselines(monkeypatch):
    """Importing the public eval path must not transitively import baselines.

    Purge via monkeypatch.delitem so the ORIGINAL module objects come back at teardown —
    a raw ``del sys.modules[...]`` makes every later ``import evemtbench.baselines...`` in
    the session create fresh module objects, silently detached from bindings other test
    modules took at collection time."""
    import sys

    for m in list(sys.modules):
        if m.startswith("evemtbench.baselines"):
            monkeypatch.delitem(sys.modules, m)
    import importlib

    importlib.import_module("evemtbench.evaluation.evaluate")
    importlib.import_module("evemtbench.tasks.task")
    assert not any(m.startswith("evemtbench.baselines") for m in sys.modules)
