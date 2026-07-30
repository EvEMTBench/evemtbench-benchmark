# Security Policy

EvEMTBench is a research benchmark (synthetic power-grid EMT data + evaluation code); it processes
no secrets and has no network/authentication surface. Still, we take defects seriously.

## Reporting a vulnerability

Please report suspected security issues (e.g. a crafted input file that triggers unsafe file writes,
code execution during data loading, or a dependency CVE that affects this package) **privately** to:

- **Julian Oelhaf** — <julian.oelhaf@fau.de>

Do **not** open a public issue for a security report. Include a description, affected version
(`python -c "import evemtbench; print(evemtbench.__version__)"` / `BENCHMARK_VERSION`), and a minimal
reproduction. We aim to acknowledge within 5 working days.

## Supported versions

The `main` branch and the latest tagged `BENCHMARK_VERSION` are supported. Older frozen benchmark
versions are retained read-only for reproducibility and do not receive fixes.
