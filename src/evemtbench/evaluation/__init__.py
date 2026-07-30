"""The product layer — frozen, versioned metrics.

``evaluate(task, predictions)`` computes the official metrics for a task's protection
objective plus stratified robustness breakdowns, aligning predictions to the test set by
``sample_id``/``window_id`` (never by order). Also exposed as the ``evemtbench-evaluate`` CLI
so third-party results can be audited from a predictions file.

The eager re-import below is deliberate: the package attribute ``evaluate`` collides with the
``evaluation.evaluate`` *submodule* name, and Python's import machinery overwrites the package
attribute with the module object whenever the submodule is imported (e.g. by the runner). A
lazy wrapper here gets silently shadowed — ``from evemtbench.evaluation import evaluate`` then
returns a module and crashes at call time. Binding the real function at package-import time
wins permanently (later submodule imports hit sys.modules and never re-set the attribute).
"""

from .evaluate import EvalResult, evaluate

__all__ = ["EvalResult", "evaluate"]
