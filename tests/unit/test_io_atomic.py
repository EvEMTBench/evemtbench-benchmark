"""Atomic writes and the idempotent view writer (evemtbench._io)."""

from evemtbench._io import atomic_write_text, normalize_view, write_view_if_changed

STAMP_A = "<div class=v>updated 2026-01-01 00:00 UTC · git aaaaaaa</div>"
STAMP_B = "<div class=v>updated 2026-02-02 12:00 UTC · git bbbbbbb</div>"


def test_atomic_write_replaces_and_leaves_no_temp_file(tmp_path):
    p = tmp_path / "result.json"
    atomic_write_text(p, "old")
    atomic_write_text(p, "new")
    assert p.read_text() == "new"
    assert [f.name for f in tmp_path.iterdir()] == ["result.json"]


def test_view_unchanged_apart_from_timestamp_is_not_rewritten(tmp_path):
    p = tmp_path / "page.html"
    p.write_text(f"<h1>x</h1>{STAMP_A}<p>1</p>")
    assert write_view_if_changed(p, f"<h1>x</h1>{STAMP_B}<p>1</p>") is False
    assert STAMP_A in p.read_text()  # old bytes, old timestamp kept


def test_view_with_a_content_change_is_rewritten_with_the_new_stamp(tmp_path):
    p = tmp_path / "page.html"
    p.write_text(f"<h1>x</h1>{STAMP_A}<p>1</p>")
    assert write_view_if_changed(p, f"<h1>x</h1>{STAMP_B}<p>2</p>") is True
    assert p.read_text() == f"<h1>x</h1>{STAMP_B}<p>2</p>"


def test_missing_view_is_written(tmp_path):
    p = tmp_path / "new.html"
    assert write_view_if_changed(p, "<p>1</p>") is True
    assert p.read_text() == "<p>1</p>"


def test_normalize_strips_only_the_volatile_line():
    assert normalize_view(f"a{STAMP_A}b") == "ab"
