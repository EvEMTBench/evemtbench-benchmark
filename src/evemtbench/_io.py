"""Small, dependency-free file-writing helpers shared by the runner, coverage and the view builders.

* :func:`atomic_write_text` / :func:`atomic_replace` — write to a temporary sibling, then
  ``os.replace`` it into place. A reader (the coverage scan, the builders, a second job) sees the
  old file or the complete new one, never a truncated one. This matters most on the shared NFS
  outputs tree: before it, a job killed mid-write could leave a half-written ``result.json``
  that the builders skipped with only a warning.
* :func:`write_view_if_changed` — the idempotent writer for the rendered HTML views. A rebuild
  whose content is unchanged apart from the volatile ``updated <ts>[ · git <sha>]`` line leaves the
  file untouched (its old timestamp included), so rebuilding with no data change produces no
  diff at all (the Agent4CT ``build_registry.py`` timestamp-restore idiom).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

#: The one non-deterministic line in the HTML views: the render timestamp (leaderboard) /
#: timestamp + build sha (audit). The freshness gate strips it before comparing; the view writer
#: ignores it when deciding whether a page changed. Single source for both.
VOLATILE_VIEW_LINE = re.compile(r"<div class=v>updated [^<]*</div>")


def normalize_view(text: str) -> str:
    """``text`` without its volatile ``updated …`` line."""
    return VOLATILE_VIEW_LINE.sub("", text)


def _tmp_sibling(path: Path) -> Path:
    # same directory => same filesystem => os.replace is atomic; pid-qualified so two writers
    # never share a temporary file
    return path.with_name(f".{path.name}.tmp-{os.getpid()}")


def atomic_replace(tmp: Path, path: Path) -> None:
    """Move a fully written ``tmp`` into place at ``path`` (atomic on one filesystem)."""
    os.replace(tmp, path)


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write ``text`` to ``path`` so readers never observe a partial file."""
    path = Path(path)
    tmp = _tmp_sibling(path)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def tmp_path_for(path: str | Path) -> Path:
    """A temporary sibling for writers that need a path (e.g. ``DataFrame.to_parquet``); pair it
    with :func:`atomic_replace`."""
    return _tmp_sibling(Path(path))


def write_view_if_changed(path: str | Path, content: str) -> bool:
    """Write a rendered view unless the file already holds the same content modulo the volatile
    ``updated …`` line. Returns ``True`` when the file was written."""
    path = Path(path)
    if path.is_file():
        try:
            if normalize_view(path.read_text(encoding="utf-8")) == normalize_view(content):
                return False
        except (OSError, UnicodeDecodeError):
            pass  # unreadable -> rewrite it
    atomic_write_text(path, content)
    return True
