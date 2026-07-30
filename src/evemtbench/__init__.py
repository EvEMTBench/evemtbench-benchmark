"""EvEMTBench — a protocol-driven ML benchmark for power system protection.

Public API (the only entry points a third party needs to evaluate their own model):

    from evemtbench import list_tasks, get_task, load_task, load_arrays, evaluate

    list_tasks()                                        # the 24 registered task ids
    get_task("fault_detection_local")                   # its frozen TaskDefinition
    task = load_task("fault_detection_local", "held_out", grid="double_line")
    X, y, meta = load_arrays(task, "train")             # small/medium views only (guarded)
    preds = my_model.predict(task.test)                 # or stream task.train.iter_batches(...)
    result = evaluate(task, preds)

See ``examples/byo_model.py`` for the full bring-your-own-model walkthrough (including
plugging a custom baseline into the official runner via ``register_baseline``).

The layers a user touches — ``evemtbench.data``, ``evemtbench.tasks``, ``evemtbench.splits``,
``evemtbench.evaluation`` — are import-isolated from ``evemtbench.baselines`` (the demo layer).
CI enforces that nothing outside ``baselines`` imports it.
"""

from .version import BENCHMARK_VERSION, SPLITS_VERSION

__all__ = [
    "BENCHMARK_VERSION",
    "SPLITS_VERSION",
    "TaskDefinition",
    "evaluate",
    "get_task",
    "list_tasks",
    "load_arrays",
    "load_task",
]


def load_task(*args, **kwargs):
    """Lazy re-export of :func:`evemtbench.tasks.task.load_task`."""
    from .tasks import load_task as _load_task

    return _load_task(*args, **kwargs)


def load_arrays(*args, **kwargs):
    """Lazy re-export of :func:`evemtbench.tasks.task.load_arrays`."""
    from .tasks.task import load_arrays as _load_arrays

    return _load_arrays(*args, **kwargs)


def list_tasks(*args, **kwargs):
    """Lazy re-export of :func:`evemtbench.tasks.registry.list_tasks`."""
    from .tasks import list_tasks as _list_tasks

    return _list_tasks(*args, **kwargs)


def get_task(*args, **kwargs):
    """Lazy re-export of :func:`evemtbench.tasks.registry.get_task`."""
    from .tasks import get_task as _get_task

    return _get_task(*args, **kwargs)


def evaluate(*args, **kwargs):
    """Lazy re-export of :func:`evemtbench.evaluation.evaluate.evaluate`."""
    # Import from the MODULE, not the package attribute: the attribute shares its name with the
    # submodule and would be shadowed by the module object after `import ...evaluation.evaluate`.
    from .evaluation.evaluate import evaluate as _evaluate

    return _evaluate(*args, **kwargs)


def __getattr__(name: str):
    """PEP 562 lazy attribute: classes re-exported without importing pandas/numpy at package
    import time (the function re-exports above follow the same lazy discipline)."""
    if name == "TaskDefinition":
        from .tasks.taxonomy import TaskDefinition

        return TaskDefinition
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
