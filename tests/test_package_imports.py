"""Step 0 smoke test: the package and all layers import cleanly."""

import importlib


def test_top_level_api_present():
    import evemtbench

    assert isinstance(evemtbench.BENCHMARK_VERSION, str)
    assert hasattr(evemtbench, "load_task")
    assert hasattr(evemtbench, "evaluate")


def test_all_layers_import():
    for mod in [
        "evemtbench.data",
        "evemtbench.tasks",
        "evemtbench.tasks.labels",
        "evemtbench.splits",
        "evemtbench.evaluation",
        "evemtbench.baselines",
    ]:
        assert importlib.import_module(mod) is not None
