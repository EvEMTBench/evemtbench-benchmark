"""Data-availability manifest — the data analog of ``evemtbench.coverage``.

Reports, per (family, grid), which artifacts are on disk: windowed configs (W0p050 / W0p020),
the topology graph, and the per-episode settings.csv (the last two from the aux overlay). Read-only
and cheap (globs + existence checks; no data load).

    python -m evemtbench.data.manifest
    python -m evemtbench.data.manifest --json
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from .roots import FAMILIES, GRIDS, aux_dir, get_grid, windows_dir

# The canonical + secondary window configs (see docs/DATA_LAYOUT.md / windows.py DEFAULT_CONTRACT).
_CONFIGS = {"W0p050": "W0p050_S0p005", "W0p020": "W0p020_S0p005"}


def _windows_status(grid: str, family: str, root: str | os.PathLike | None = None) -> list[str]:
    """Which window configs have an X memmap on disk for this grid/family."""
    try:
        wdir = windows_dir(grid, family, root)
    except Exception:
        return []
    g = get_grid(grid)
    present = []
    for short, cfg in _CONFIGS.items():
        if glob.glob(str(wdir / f"X_{g.topology}_{cfg}.raw")):
            present.append(short)
    return present


def scan(root: str | os.PathLike | None = None, aux_root: str | os.PathLike | None = None) -> dict:
    """Return ``{family: {grid: {windows: [...], graph: bool, settings: bool}}}``."""
    out: dict = {}
    for family in FAMILIES:
        out[family] = {}
        for grid in GRIDS:
            wins = _windows_status(grid, family, root)
            try:
                ad = aux_dir(grid, family, aux_root)
                graph = (ad / "graph.pickle").exists()
                settings = (ad / "settings.csv").exists()
            except Exception:
                graph = settings = False
            out[family][grid] = {"windows": wins, "graph": graph, "settings": settings}
    return out


def _render(m: dict) -> str:
    lines = [f"{'family':<11} {'grid':<15} {'windows':<14} {'graph':<6} settings", "-" * 55]
    for family, grids in m.items():
        for grid, st in grids.items():
            wins = ",".join(st["windows"]) or "—"
            lines.append(
                f"{family:<11} {grid:<15} {wins:<14} "
                f"{'yes' if st['graph'] else '—':<6} {'yes' if st['settings'] else '—'}"
            )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Report on-disk data artifacts per family/grid.")
    p.add_argument("--root", default=None)
    p.add_argument("--aux-root", default=None)
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    m = scan(root=a.root, aux_root=a.aux_root)
    print(json.dumps(m, indent=2) if a.json else _render(m))


if __name__ == "__main__":
    main()
