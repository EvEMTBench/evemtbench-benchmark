"""Freshness gate — committed rendered views must byte-match a fresh rebuild from outputs/.

This is the cure for "rendered views drift from the data" (e.g. the absent-class F1 that read
0.000 for weeks): the deterministic builders — ``make_results.build_results``,
``leaderboard.render_html`` and ``make_audit.build_audit_pages`` — are the single source of
truth, and the committed artifacts must equal their output. The one volatile field (the
"updated <ts>[ · git <sha>]" line) is normalized away before comparison.

Covered views: ``RESULTS.md``, ``coverage_snapshot.json``, ``leaderboard.html``, ``audit/*.html``,
the committed provenance index ``audit/provenance/<version>/index.json``, and the public results
bundle ``results/<version>/`` (aggregated CSV+JSON + the LaTeX tables).

Behaviour:
* Exit 0 if every committed view matches a fresh build.
* Exit 0 (skip) if ``outputs/`` is absent/empty — dev machines / CI cannot rebuild the
  gitignored data, so the gate is a no-op there. It is meaningful only where ``outputs/``
  lives (the loop host).
* Exit 1 on drift (including a committed audit page a rebuild no longer produces), printing a
  short unified diff and how to fix.
* ``--fix`` regenerates every view (via the normal builders) instead of failing.
* ``--staged`` compares what is STAGED (the git index) instead of the working tree. The pre-commit
  hook uses it: a check of the working tree passes after ``--fix`` even when only some of the
  regenerated files were staged, and the commit would ship the stale rest.

Run from anywhere; it operates at the repo root.

    python hpc/check_results_fresh.py          # verify
    python hpc/check_results_fresh.py --fix     # regenerate
    python hpc/check_results_fresh.py --staged  # verify the staged views (pre-commit hook)
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# The code always lives in the real repo (kept on sys.path for imports). The repo whose ARTIFACTS
# we check is overridable via EVEMTBENCH_REPO — used by tests; production leaves it unset.
_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO / "src", _REPO / "hpc", _REPO / "hpc" / "paper_artifacts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _repo() -> Path:
    return Path(os.environ.get("EVEMTBENCH_REPO") or _REPO)


# The only non-deterministic line in the HTML views: the render timestamp (leaderboard) /
# timestamp + build sha (audit). Strip it before diffing.
_VOLATILE = re.compile(r"<div class=v>updated [^<]*</div>")


def _normalize(text: str) -> str:
    return _VOLATILE.sub("", text)


class _Views:
    """Reads the committed views from the working tree, or with ``staged`` from the git index."""

    def __init__(self, repo: Path, staged: bool):
        self.repo, self.staged = repo, staged
        self._index: set[str] | None = None
        if staged:
            out = subprocess.run(
                ["git", "ls-files", "-z"], cwd=repo, capture_output=True, text=True, check=True
            ).stdout
            self._index = {f for f in out.split("\0") if f}

    def read(self, path: Path) -> str:
        if not self.staged:
            return path.read_text() if path.exists() else ""
        rel = path.relative_to(self.repo).as_posix()
        if rel not in (self._index or set()):
            return ""
        return subprocess.run(
            ["git", "show", f":{rel}"], cwd=self.repo, capture_output=True, text=True, check=True
        ).stdout

    def files(self, base: Path, *, suffix: str = "", recursive: bool = False) -> list[Path]:
        """Files under ``base`` ending in ``suffix`` -- directly in it, or at any depth with
        ``recursive`` (from the git index when staged, else from disk)."""
        if not self.staged:
            found = base.rglob("*") if recursive else base.glob("*")
            paths = [p for p in found if p.is_file()]
        else:
            prefix = base.relative_to(self.repo).as_posix() + "/"
            paths = [
                self.repo / f
                for f in (self._index or set())
                if f.startswith(prefix) and (recursive or "/" not in f[len(prefix) :])
            ]
        return sorted(p for p in paths if p.name.endswith(suffix))


def _outputs_present() -> bool:
    out = _repo() / "outputs"
    return out.is_dir() and next(out.rglob("result.json"), None) is not None


def _fresh_results() -> str:
    from make_results import build_results

    return build_results()


def _fresh_leaderboard() -> str:
    from evemtbench.leaderboard import _bibtex_from_cff, render_html, scan_results
    from evemtbench.version import BENCHMARK_VERSION

    rows = scan_results("outputs")
    # generated_at=None => no timestamp line: the comparison is against deterministic content.
    return render_html(
        rows, version=BENCHMARK_VERSION, generated_at=None, citation=_bibtex_from_cff()
    )


# Audit pages that are committed but deliberately NOT byte-checked: they derive from git
# HISTORY (which grows with every snapshot commit), not from outputs/ — gating them would make
# them perpetually one tick stale. Rebuilt by the loops; labelled "informational" on-page.
_UNGATED = {"progress.html"}


def _fresh_audit_views() -> tuple[dict[str, str], dict[str, str]]:
    """(audit pages, provenance index) from make_audit.fresh_views — ONE outputs scan for both
    gated views (the provenance index is a separate JSON view, keyed by its audit-relative
    path). The gate runs on every rendered-view commit, so the shared scan matters."""
    from make_audit import fresh_views

    return fresh_views("outputs", generated_at=None)


def _fresh_results_bundle() -> dict[str, str]:
    """The committed public results bundle (results/<version>/…: aggregated CSV+JSON + the LaTeX
    tables + README) — a separate gated view, keyed by its repo-relative path, no timestamp."""
    from make_results_bundle import build_results_bundle

    return build_results_bundle("outputs")


def _fresh_snapshot() -> str:
    from evemtbench.coverage import render_snapshot

    return render_snapshot("outputs")


_GEN_REL = Path("paper") / "manuscript" / "generated"


def _paper_fragment_checks(repo: Path) -> list[tuple[str, Path, str]]:
    """The manuscript's generated fragments are a LIVE gated view (like RESULTS.md): every
    committed fragment — held_out result tables, supplement tables, cross-grid transfer tables,
    and the static spec tables — must byte-match a fresh rebuild from outputs/, and ``--fix``
    regenerates them. They populate from the current sweep, so the manuscript numbers are
    provisional (they update as the sweep completes) rather than frozen."""
    gen = repo / _GEN_REL
    if not gen.is_dir():
        return []
    from paper_lib import (
        build_paper_tables,
        build_spec_tables,
        build_supplement_tables,
        build_transfer_tables,
    )
    from supplement_lib import dataset_free_artifacts

    fresh = {
        **build_paper_tables(),
        **build_supplement_tables(),
        **build_transfer_tables(),
        **build_spec_tables(),
        # SI evidence tables derived from committed repo artefacts only. The three DATASET_BACKED
        # fragments (EC support, ED negative composition, current-scale) need the mounted dataset,
        # so they are gated by `make check-supp-artifacts-dataset` on a machine that has it rather
        # than by this commit-time hook.
        **dataset_free_artifacts(),
    }
    return [
        (f"paper/generated/{name}", gen / name, content) for name, content in sorted(fresh.items())
    ]


def _short_diff(name: str, committed: str, fresh: str, limit: int = 40) -> str:
    d = list(
        difflib.unified_diff(
            committed.splitlines(),
            fresh.splitlines(),
            fromfile=f"committed/{name}",
            tofile=f"fresh/{name}",
            lineterm="",
        )
    )
    body = "\n".join(d[:limit])
    return body + (f"\n… (+{len(d) - limit} more diff lines)" if len(d) > limit else "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify committed results match a fresh rebuild.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--fix", action="store_true", help="regenerate the files instead of failing")
    mode.add_argument(
        "--staged", action="store_true", help="check the staged (git index) views, not the disk"
    )
    args = ap.parse_args(argv)
    repo = _repo()

    if not _outputs_present():
        print(
            "check_results_fresh: outputs/ absent or empty — skipping "
            "(gate runs only where data lives)."
        )
        return 0

    os.chdir(repo)  # the builders glob 'outputs/...' relative to cwd
    views = _Views(repo, staged=args.staged)

    # The snapshot's embedded self-check gates here (the loops' --write-snapshot calls are
    # `|| true`-guarded, so THIS is where a failed invariant actually blocks publication).
    snapshot_fresh = _fresh_snapshot()
    self_check = json.loads(snapshot_fresh).get("self_check", {})
    if not self_check.get("ok", True):
        for c in self_check.get("checks", []):
            if not c.get("ok"):
                print(f"✗ snapshot self-check FAILED: {c['id']} — {c['detail']}")
        print(
            "check_results_fresh: the coverage snapshot's embedded invariants failed — "
            "refusing to certify the views (fix the plumbing, then rerun)."
        )
        return 1

    from evemtbench.version import BENCHMARK_VERSION

    audit_fresh, prov_fresh = _fresh_audit_views()
    results_fresh = _fresh_results_bundle()
    audit_dir = repo / "audit"
    if audit_dir.is_dir() or args.staged:
        stray = sorted(
            p.name
            for p in views.files(audit_dir, suffix=".html")
            if p.name not in audit_fresh and p.name not in _UNGATED
        )
        # The provenance index (audit/provenance/<version>/index.json) is keyed by its
        # audit-relative path. Only the CURRENT version's tree is gated: any JSON there a rebuild
        # does not emit is stray. audit/provenance/<older>/ indices are frozen history (they
        # document a superseded revision's results) — never touched, never reported stale.
        stray += sorted(
            str(p.relative_to(audit_dir))
            for p in views.files(
                audit_dir / "provenance" / BENCHMARK_VERSION, suffix=".json", recursive=True
            )
            if str(p.relative_to(audit_dir)) not in prov_fresh
        )
    else:
        stray = []
    # results/<BENCHMARK_VERSION>/ strays: committed files a rebuild no longer emits. results/
    # <older>/ trees are frozen pre-revision bundles kept for comparison; the gate owns only the
    # current version's bundle. Keyed by repo-relative path.
    results_root = repo / "results" / BENCHMARK_VERSION
    results_stray = (
        sorted(
            str(p.relative_to(repo))
            for p in views.files(results_root, recursive=True)
            if str(p.relative_to(repo)) not in results_fresh
        )
        if results_root.is_dir() or args.staged
        else []
    )

    if args.fix:
        from evemtbench.leaderboard import main as lb_main
        from make_audit import main as audit_main
        from make_results_bundle import main as results_main

        (repo / "RESULTS.md").write_text(_fresh_results())
        (repo / "coverage_snapshot.json").write_text(_fresh_snapshot())
        lb_main(["--outputs", "outputs", "--out", "leaderboard.html"])  # writes WITH a timestamp
        audit_main(["--outputs", "outputs", "--out", str(audit_dir)])  # ditto (ts + git sha)
        # paper fragments BEFORE the bundle: results/<version>/tables/ copies them from disk, so the
        # reverse order left the bundle one regeneration behind (a second --fix was needed)
        for _name, path, content in _paper_fragment_checks(repo):
            path.write_text(content)  # post-freeze: fragments regenerate with the other views
        results_main(["--outputs", "outputs", "--out", str(repo)])  # results/<version>/ bundle
        for name in stray:  # a rebuild no longer produces these pages
            (audit_dir / name).unlink()
            print(f"check_results_fresh: removed stray audit page {name}")
        for rel in results_stray:  # a rebuild no longer produces these results files
            (repo / rel).unlink()
            print(f"check_results_fresh: removed stray results file {rel}")
        print(
            "check_results_fresh: regenerated RESULTS.md + coverage_snapshot.json + "
            "leaderboard.html + audit/ + results/."
        )
        return 0

    checks: list[tuple[str, Path, str]] = [
        ("RESULTS.md", (repo / "RESULTS.md"), _fresh_results()),
        ("coverage_snapshot.json", (repo / "coverage_snapshot.json"), snapshot_fresh),
        ("leaderboard.html", (repo / "leaderboard.html"), _fresh_leaderboard()),
    ]
    checks += [
        (f"audit/{name}", audit_dir / name, content)
        for name, content in sorted(audit_fresh.items())
    ]
    checks += [
        (f"audit/{name}", audit_dir / name, content) for name, content in sorted(prov_fresh.items())
    ]
    checks += [
        (name, repo / name, content) for name, content in sorted(results_fresh.items())
    ]  # results/<version>/… keyed by repo-relative path
    checks += _paper_fragment_checks(repo)  # the manuscript's live table view
    stale: list[str] = []
    for name, path, fresh in checks:
        committed = views.read(path)
        if _normalize(committed) != _normalize(fresh):
            stale.append(name)
            print(f"\n✗ {name} is STALE (committed != fresh rebuild):")
            print(_short_diff(name, _normalize(committed), _normalize(fresh)))
    for name in stray:
        stale.append(f"audit/{name}")
        print(f"\n✗ audit/{name} is STALE: committed but no longer produced by a rebuild.")
    for rel in results_stray:
        stale.append(rel)
        print(f"\n✗ {rel} is STALE: committed but no longer produced by a rebuild.")

    if stale:
        print(
            f"\ncheck_results_fresh: {', '.join(stale)} disagree with a fresh build.\n"
            "Fix: `python hpc/check_results_fresh.py --fix` (or rerun make_results.py + "
            "evemtbench-leaderboard + make_audit.py), then stage the regenerated files"
            + (" (this check reads what is STAGED)." if args.staged else ".")
        )
        return 1
    print(
        "check_results_fresh: RESULTS.md + coverage_snapshot.json + leaderboard.html + audit/ + "
        "results/ match a fresh rebuild. ✓"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
