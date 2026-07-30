"""Extract the small aux artifacts (topology graph + settings.csv) from a grid's data tarball into
the aux overlay (``$EVEMTBENCH_AUX_ROOT``).

Additive + idempotent: pulls only ``*/graphs/*.pickle`` and ``*settings*.csv`` out of the per-grid
``*.tar.gz`` and writes them under ``aux_dir(grid, family)`` as ``graph.pickle`` / ``settings.csv``
(+ ``settings_clean.csv``). Never writes into the producer-owned data root. Decompressing a large
tarball is slow, so this is a GPU-less prep job (``hpc/extract_aux.sh``), not a hot-path import.

    python -m evemtbench.data.extract_aux --grid double_line --family benchmark
    python -m evemtbench.data.extract_aux --grid all --family adapt_grid
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .roots import FAMILIES, GRIDS, aux_dir, get_grid, resolve_data_root


def _find_tar(grid: str, family: str, root: str | os.PathLike | None = None) -> Path:
    """Locate a grid/family tarball. Naming is usually ``<RawFolder>_<family>.tar.gz`` but some
    (e.g. ieee39 benchmark) drop the suffix (``IEEE39BusSystem.tar.gz``)."""
    g = get_grid(grid)
    fam_dir = resolve_data_root(family, root)
    cands = sorted(
        glob.glob(str(fam_dir / f"{g.raw_foldername}*.tar.gz"))
        + glob.glob(str(fam_dir / f"{g.raw_foldername}*.tar"))
    )
    if not cands:
        raise FileNotFoundError(f"no tarball for {g.raw_foldername} in {fam_dir}")
    named = [c for c in cands if family in os.path.basename(c)]
    return Path(named[0] if named else cands[0])


def extract(
    grid: str,
    family: str = "benchmark",
    *,
    root: str | os.PathLike | None = None,
    aux_root: str | os.PathLike | None = None,
    force: bool = False,
) -> dict:
    """Extract graph + settings for one grid/family into the aux overlay. Idempotent."""
    g = get_grid(grid)
    dest = aux_dir(grid, family, aux_root)
    dest.mkdir(parents=True, exist_ok=True)
    graph_out, settings_out = dest / "graph.pickle", dest / "settings.csv"
    if graph_out.exists() and settings_out.exists() and not force:
        return {"grid": g.key, "family": family, "status": "present", "dest": str(dest)}
    tar = _find_tar(grid, family, root)
    comp = "z" if str(tar).endswith(".gz") else ""
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
        subprocess.run(
            ["tar", f"-x{comp}f", str(tar), "--wildcards", "*/graphs/*.pickle", "*settings*.csv"],
            cwd=td,
            check=True,
        )
        for pat, out in (
            ("graphs/*.pickle", graph_out),
            ("settings.csv", settings_out),
            ("settings_clean.csv", dest / "settings_clean.csv"),
        ):
            hits = sorted(Path(td).rglob(pat))
            if hits:
                shutil.copyfile(hits[0], out)
    return {
        "grid": g.key,
        "family": family,
        "status": "extracted",
        "graph": graph_out.exists(),
        "settings": settings_out.exists(),
        "tar": os.path.basename(str(tar)),
        "dest": str(dest),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Extract aux artifacts (graph + settings).")
    p.add_argument("--grid", required=True, help="grid key, or 'all'")
    p.add_argument("--family", default="benchmark", choices=[*FAMILIES, "fine_tune", "train"])
    p.add_argument("--root", default=None, help="data-root override (else env)")
    p.add_argument("--aux-root", default=None, help="aux-root override (else $EVEMTBENCH_AUX_ROOT)")
    p.add_argument("--force", action="store_true", help="re-extract even if present")
    a = p.parse_args()
    grids = list(GRIDS) if a.grid == "all" else [a.grid]
    for gk in grids:
        try:
            print(extract(gk, a.family, root=a.root, aux_root=a.aux_root, force=a.force))
        except Exception as e:  # noqa: BLE001 - report per-grid, keep going
            print({"grid": gk, "family": a.family, "status": "error", "error": str(e)})


if __name__ == "__main__":
    main()
