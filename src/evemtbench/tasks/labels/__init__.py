"""Derived labels — *code is part of the benchmark*.

Each function here is a pure, deterministic ``settings_row -> label`` (or window-row -> label)
mapping covered by golden tests in ``tests/golden/``. Any change to a derivation bumps the
benchmark **major** version, because it can silently change scores.
"""
