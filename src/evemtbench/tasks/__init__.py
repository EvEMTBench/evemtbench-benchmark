"""Layer 2 — meaning: task definitions, label derivation, valid-sample rules.

A *task* binds a protection objective (fault detection, trip/no-trip, fault-type, faulted
line) to an input setting (local / line / global), a protocol
(``held_out`` — v0.1's single protocol), and a deterministic, golden-tested
label derivation from ``settings.csv`` rows. ``load_task`` is the primary user entry point.
"""

__all__ = ["load_task", "get_task", "list_tasks"]


def load_task(*args, **kwargs):
    """Build a :class:`~evemtbench.tasks.task.Task` (implemented in Step 3)."""
    from .task import load_task as _load_task

    return _load_task(*args, **kwargs)


def get_task(*args, **kwargs):
    """Look up a task definition by id (implemented in Step 3)."""
    from .registry import get_task as _get_task

    return _get_task(*args, **kwargs)


def list_tasks(*args, **kwargs):
    """List registered task ids (implemented in Step 3)."""
    from .registry import list_tasks as _list_tasks

    return _list_tasks(*args, **kwargs)
