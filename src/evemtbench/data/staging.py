"""Persistent projection-cache location + status, and staging tunables shared by the HPC scripts.

Why this exists: on LME the dataset lives on a single GlusterFS mount (no Lustre/$WORK/$HPCVAULT).
Re-reading the raw windows every job — and rebuilding the projection cache while holding GPUs —
saturates the shared filesystem and idles GPUs (the two behaviours cluster admins flag). The fix is
to build the small, content-addressed projection cache **once** in a CPU prep job, persist it, and
have every later job read it. This module is the **single source of truth** for *where* that cache
lives, *whether it exists*, and the staging tunables — used by both the Python training path and the
thin shell wrappers (via the CLI at the bottom: ``python -m evemtbench.data.staging <cmd>``).

The cache identity itself is defined by :meth:`evemtbench.tasks.task.TaskView.cache_filename` (not
duplicated here) so Python and the cache builder can never drift.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..version import BENCHMARK_VERSION
from .roots import resolve_data_root

# --- staging tunables (single source of truth; shell reads them via the `config` subcommand) ---
# rsync bandwidth cap for any copy off the shared GlusterFS mount (be a polite tenant).
BWLIMIT = os.environ.get("EVEMTBENCH_STAGE_BWLIMIT", "200m")
VIEWS = ("train", "val", "test", "benchmark")


def persistent_cache_root() -> Path:
    """Persistent projection-cache directory, **version-scoped** so a contract bump (new
    ``BENCHMARK_VERSION``) gets a fresh dir and old caches can be GC'd. Precedence mirrors
    :mod:`evemtbench.data.roots`:

    1. ``$EVEMTBENCH_PROJECTION_CACHE_ROOT`` (then ``/<BENCHMARK_VERSION>``),
    2. ``<data-root>/projcache/<BENCHMARK_VERSION>`` derived from the resolved data root.

    NOTE: the fallback (2) follows ``$EVEMTBENCH_DATA_ROOT``. Any context that **stages** data and
    repoints ``EVEMTBENCH_DATA_ROOT`` to ephemeral scratch (e.g. ``build_cache.sh``) MUST set
    ``$EVEMTBENCH_PROJECTION_CACHE_ROOT`` explicitly so the persistent cache doesn't resolve to (and
    vanish with) scratch — the HPC wrappers pin it from the stable mount.
    """
    env = os.environ.get("EVEMTBENCH_PROJECTION_CACHE_ROOT")
    base = Path(env) if env else resolve_data_root("adapt_grid").parent / "projcache"
    return base / BENCHMARK_VERSION


def view_cache_files(
    grid: str,
    tasks,
    protocol: str = "held_out",
    *,
    root=None,
    splits_root=None,
) -> list[tuple[str, bool]]:
    """Dedup'd ``[(filename, exists)]`` for every cacheable view across ``tasks``. The main reuse is
    across seeds/models/re-runs of the *same* task (same content key); task families keep different
    window subsets (detection=fault+normal, classification/location=faulted) so they do NOT share.
    Loading a task reads only labels/metadata + the memmap *header* (cheap) — never the ``.raw``
    payload."""
    from ..tasks.task import load_task

    croot = persistent_cache_root()
    seen: dict[str, bool] = {}
    for t in tasks:
        task = load_task(t, protocol, grid=grid, root=root, splits_root=splits_root)
        for vname in VIEWS:
            fn = task.view(vname).cache_filename()
            if fn and fn not in seen:
                seen[fn] = (croot / fn).exists()
    return sorted(seen.items())


def cache_status(
    grid: str, tasks, protocol: str = "held_out", *, root=None, splits_root=None
) -> dict:
    """``{total, present, missing}`` — single source of truth for "is prep needed?", used by the
    autonomous loop and the verifier."""
    files = view_cache_files(grid, tasks, protocol, root=root, splits_root=splits_root)
    missing = [fn for fn, ex in files if not ex]
    return {"total": len(files), "present": len(files) - len(missing), "missing": missing}


def prune_stale_versions(*, dry_run: bool = True) -> list[Path]:
    """GC: list (and, unless ``dry_run``, delete) sibling ``projcache/<ver>`` dirs for non-current
    versions — keeps the cache from re-filling the 91%-full mount.

    Safety: only dirs that actually *look like a projection cache* (contain ``proj_*.f32``) are
    eligible, so a mis-set ``EVEMTBENCH_PROJECTION_CACHE_ROOT`` pointing at a shared dir can never
    wipe unrelated siblings."""
    cur = persistent_cache_root()
    parent = cur.parent
    if not parent.is_dir():
        return []
    stale = [
        p
        for p in parent.iterdir()
        if p.is_dir() and p.name != BENCHMARK_VERSION and any(p.glob("proj_*.f32"))
    ]
    if not dry_run:
        import shutil

        for p in stale:
            shutil.rmtree(p, ignore_errors=True)
    return stale


# --- CLI: thin surface the shell wrappers call (no value drift between Python and bash) ---
def _main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m evemtbench.data.staging")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cache-root", help="print the persistent projection-cache dir")
    sub.add_parser("bwlimit", help="print the rsync bandwidth cap")
    sub.add_parser("config", help="print shell-sourceable KEY=VAL tunables")
    s = sub.add_parser("files", help="list view-cache filenames (optionally only the missing ones)")
    st = sub.add_parser("status", help="print total/present/missing cache counts")
    for q in (s, st):
        q.add_argument("--grid", required=True)
        q.add_argument("--tasks", nargs="+", required=True)
        q.add_argument("--protocol", default="held_out")
        q.add_argument("--root", default=None)
        q.add_argument("--splits-root", default=None)
    s.add_argument("--only-missing", action="store_true")
    args = p.parse_args(argv)

    if args.cmd == "cache-root":
        print(persistent_cache_root())
    elif args.cmd == "bwlimit":
        print(BWLIMIT)
    elif args.cmd == "config":
        print(f"EVEMTBENCH_STAGE_BWLIMIT={BWLIMIT}")
        print(f"PROJECTION_CACHE_ROOT={persistent_cache_root()}")
    elif args.cmd == "files":
        for fn, ex in view_cache_files(
            args.grid, args.tasks, args.protocol, root=args.root, splits_root=args.splits_root
        ):
            if not args.only_missing or not ex:
                print(fn)
    elif args.cmd == "status":
        status_info = cache_status(
            args.grid, args.tasks, args.protocol, root=args.root, splits_root=args.splits_root
        )
        print(
            f"total={status_info['total']} present={status_info['present']} "
            f"missing={len(status_info['missing'])}"
        )
        for fn in status_info["missing"]:
            print(f"  MISSING {fn}")
        return 0 if not status_info["missing"] else 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
