"""Compile the Supplementary Information PDF and QA the LaTeX log.

Counterpart to ``hpc/build_paper.py`` for ``paper/manuscript/supplementary.tex``. It compiles
ONLY — the generated fragments are produced by their individual owners under
``hpc/paper_artifacts/`` and orchestrated by the repository ``Makefile``:

    make supp-artifacts          # regenerate every fragment (one generator per artifact)
    make check-supp-artifacts    # verify the committed fragments are fresh
    make supplementary           # check-supp-artifacts, then this script

Tectonic compiles to convergence internally, so no manual second pass is needed; the QA step
re-reads the log for undefined references/labels/citations, missing files and overfull boxes.
Exits non-zero on a compile error or an unresolved reference/citation.

Runs as a short SLURM mini-job on Alex (GPU-only cluster -> needs a ``--gres`` allocation; never
the login node)::

    srun --gres=gpu:a100small:1 -p a100mig --time=15:00 -A <acct> \
        bash -c 'cd <repo>; PYTHONPATH=src python hpc/build_supplement.py'
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANUSCRIPT = REPO / "paper" / "manuscript"
sys.path.insert(0, str(REPO / "hpc"))

from build_paper import _find_tectonic  # noqa: E402  (single source for the tectonic lookup)

#: Log lines that must fail the build.
_FATAL = (
    re.compile(r"undefined (reference|citation|label)", re.I),
    re.compile(r"there were undefined references", re.I),
    re.compile(r"file .* not found", re.I),
)
#: Log lines reported but not fatal (layout warnings a human should look at).
_WARN = (re.compile(r"overfull \\[hv]box", re.I),)


def _compile(tectonic: str) -> str:
    proc = subprocess.run(
        [
            tectonic,
            "-X",
            "compile",
            "supplementary.tex",
            "--outdir",
            str(MANUSCRIPT),
            "--keep-logs",
        ],
        cwd=MANUSCRIPT,
        capture_output=True,
        text=True,
    )
    log = proc.stdout + proc.stderr
    print(log)
    if proc.returncode != 0:
        sys.exit(f"ERROR: tectonic failed (exit {proc.returncode}).")
    return log


def _qa_sources() -> None:
    """Static pre-flight the LaTeX log does NOT catch.

    A ``\\cite`` in a fragment shared with ``main.tex`` renders as a silent ``[?]`` here, because
    the Supplementary carries no bibliography — tectonic emits no error, so the defect only shows
    up in the rendered PDF. Fail loudly instead."""
    si = MANUSCRIPT / "supplementary.tex"
    has_bib = "\\bibliography{" in si.read_text()
    if has_bib:
        return
    offenders = []
    for tex in [si, *sorted((MANUSCRIPT / "generated").glob("*.tex"))]:
        for i, line in enumerate(tex.read_text().splitlines(), 1):
            if re.search(r"\\cite[a-z]*\{", line):
                offenders.append(f"{tex.name}:{i}: {line.strip()[:90]}")
    if offenders:
        print("[qa] citations in a document with no bibliography (would render as [?]):")
        for o in offenders:
            print(f"    {o}")
        sys.exit("ERROR: unresolvable citations — remove them or add a bibliography.")
    print("[qa] no citations without a bibliography.")


def _qa(log: str) -> None:
    lines = log.splitlines()
    fatal = [ln for ln in lines if any(p.search(ln) for p in _FATAL)]
    warn = [ln for ln in lines if any(p.search(ln) for p in _WARN)]
    if warn:
        print(f"[qa] {len(warn)} layout warning(s):")
        for ln in warn:
            print(f"    {ln.strip()}")
    if fatal:
        print("[qa] FATAL log lines:")
        for ln in fatal:
            print(f"    {ln.strip()}")
        sys.exit("ERROR: unresolved references/labels/files — fix before relying on the PDF.")
    print("[qa] no undefined references, labels, citations or missing files.")


def main() -> None:
    tectonic = _find_tectonic()
    print(f"[build] tectonic: {tectonic}")
    _qa_sources()
    log = _compile(tectonic)
    _qa(log)
    pdf = MANUSCRIPT / "supplementary.pdf"
    size_kb = pdf.stat().st_size / 1024 if pdf.exists() else 0
    print(f"[build] OK -> {pdf} ({size_kb:.0f} KiB)")


if __name__ == "__main__":
    main()
