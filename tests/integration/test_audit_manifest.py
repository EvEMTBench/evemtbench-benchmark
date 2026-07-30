"""Integrity of the audit manifest (hpc/audit_manifest.py) — the ledger⇄repo⇄paper sync
mechanism. Data-free and torch-free: checks that every claim row is well-formed, every referenced
path exists, every {placeholder} resolves, every gate anchor still matches its source, and every
paper_anchor / Contributions bullet is still backed by the manuscript. A reword in main.tex or a
moved gate assertion fails here, naming the exact row to update."""

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "hpc"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import audit_manifest as MAN  # noqa: E402
import make_audit  # noqa: E402
from evemtbench.coverage import ALL_BASELINES  # noqa: E402
from evemtbench.tasks.registry import get_task, list_tasks  # noqa: E402

MAIN_TEX = REPO / "paper" / "manuscript" / "main.tex"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s)


def test_claim_rows_complete_and_statuses_legal():
    assert len(MAN.CLAIMS) >= 15
    for c in MAN.CLAIMS:
        assert c.status in MAN.STATUSES, f"{c.claim!r}: illegal status {c.status!r}"
        assert c.claim and c.value and c.detail, f"{c.claim!r}: degenerate row"
        if c.status in ("deferred", "note"):
            assert c.command == "—", f"{c.claim!r}: deferred/note rows must not claim a command"
        else:
            assert c.command and c.command != "—", f"{c.claim!r}: needs a reproduce command"
            assert c.script, f"{c.claim!r}: needs a producing script"
        if c.status == "auto":
            assert c.cells is not None, f"{c.claim!r}: status=auto needs a cell selector"
            assert set(c.cells) <= {"task", "protocol", "grid", "baseline"}


def test_script_and_repo_artifact_paths_exist():
    paths: set[str] = set()
    for c in MAN.CLAIMS:
        paths.add(c.script)
        paths |= set(c.artifact.split(" · "))  # a row may name several artifacts
    paths |= {g.file for g in MAN.GATES}
    paths |= {p for p, _ in MAN.DOC_LINKS}
    paths |= {p["source"] for p in MAN.PILLARS}
    for path in sorted(paths):
        if not path or path == "—" or path.startswith("outputs/"):
            continue  # runtime artifacts are gitignored; their pattern is checked below
        if path.startswith("paper/") and not (REPO / "paper").is_dir():
            continue  # the manuscript tree is pruned from the public release export
        assert (REPO / path).exists(), f"manifest references a missing repo path: {path}"
    for c in MAN.CLAIMS:
        if c.artifact.startswith("outputs/"):
            assert "<version>" in c.artifact, f"{c.claim!r}: outputs artifact must be a pattern"


def test_every_placeholder_resolves():
    used = make_audit.manifest_placeholders()
    unknown = used - set(make_audit.VALUE_KEYS)
    assert not unknown, f"manifest placeholders missing from make_audit.VALUE_KEYS: {unknown}"
    # and compute_values always emits the full key set, even with nothing measured
    values = make_audit.compute_values(
        [], make_audit.CoverageView("v", {}, {}), make_audit.RunRecordStats()
    )
    assert set(values) == set(make_audit.VALUE_KEYS)


def test_gate_anchor_snippets_exist_exactly_once():
    for g in MAN.GATES:
        line = make_audit._line_of(g.file, g.anchor_snippet)  # raises if missing/ambiguous
        assert line > 0


def test_gate_call_stacks_resolve():
    """Every step of every gate's guided call stack must anchor to exactly one source line —
    a moved or duplicated snippet fails here, naming the gate and step."""
    for g in MAN.GATES:
        assert len(g.stack) >= 3, f"{g.gate_id}: call stack too shallow for the code tour"
        for file, snippet, computes in g.stack:
            assert (REPO / file).exists(), f"{g.gate_id}: stack file missing {file}"
            assert computes, f"{g.gate_id}: empty explanation for {snippet!r}"
            assert make_audit._line_of(file, snippet) > 0


def test_pipeline_diagrams_name_real_modules():
    for diagram in (MAN.DATA_PIPELINE, MAN.PUBLISH_PIPELINE):
        for ref in ("result.json", "run_record.json"):
            assert ref in diagram or ref in MAN.DATA_PIPELINE
    assert "check_results_fresh" in MAN.PUBLISH_PIPELINE


def test_verified_by_names_real_gates_or_tests():
    gate_ids = {g.gate_id for g in MAN.GATES}
    for c in MAN.CLAIMS:
        for entry in c.verified_by:
            if entry in gate_ids:
                continue
            path, _, func = entry.partition("::")
            assert (REPO / path).exists(), f"{c.claim!r}: verified_by path missing: {path}"
            if func:
                assert func in (REPO / path).read_text(), f"{c.claim!r}: {func} not found in {path}"


@pytest.mark.skipif(
    not MAIN_TEX.exists(), reason="private manuscript not present (public release clone)"
)
def test_ledger_tracks_paper_anchors_and_contributions():
    tex = _norm(MAIN_TEX.read_text())
    for c in MAN.CLAIMS:
        if c.paper_anchor:
            assert _norm(c.paper_anchor) in tex, (
                f"{c.claim!r}: paper_anchor no longer occurs verbatim in main.tex — the "
                "manuscript changed; update the ledger row to match"
            )
    # every Contributions \item must be backed by at least one claim row
    raw = MAIN_TEX.read_text()
    block = raw.split(r"\textbf{Contributions.}")[1].split(r"\end{itemize}")[0]
    n_items = len(re.findall(r"\\item", block))
    covered = {c.contribution for c in MAN.CLAIMS if c.contribution}
    assert covered >= set(range(1, n_items + 1)), (
        f"Contributions bullets {set(range(1, n_items + 1)) - covered} have no ledger row — "
        "the paper's contribution list changed; add/retag claims in hpc/audit_manifest.py"
    )


def test_lay_prose_covers_the_registry():
    functions = {get_task(t).function.value for t in list_tasks()}
    views = {get_task(t).observability.value for t in list_tasks()}
    assert functions <= set(MAN.FUNCTION_ABOUT), (
        f"objective functions without FUNCTION_ABOUT prose: {functions - set(MAN.FUNCTION_ABOUT)}"
    )
    assert views <= set(MAN.VIEW_ABOUT)
    assert set(ALL_BASELINES) <= set(MAN.BASELINE_ABOUT), (
        f"baselines without BASELINE_ABOUT prose: {set(ALL_BASELINES) - set(MAN.BASELINE_ABOUT)}"
    )


def test_na_reasons_and_pillars_wellformed():
    assert {"threshold_scope", "transfer_deep_only", "pretrain_grid_only"} <= set(MAN.NA_REASONS)
    gate_ids = {g.gate_id for g in MAN.GATES}
    for p in MAN.PILLARS:
        assert p["title"] and p["text"] and p["source"]
        if p["gate_id"]:
            assert p["gate_id"] in gate_ids
