"""Demo layer — reference baseline implementations.

IMPORTANT: this package is import-isolated. Nothing in ``data``, ``tasks``, ``splits``, or
``evaluation`` may import from ``baselines`` — a user must be able to evaluate their own model
without it. ``tests/integration/test_import_isolation.py`` enforces this.
"""
