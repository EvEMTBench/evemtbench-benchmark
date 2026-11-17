"""The leaderboard lists valid outside submissions with their tier badge, skips invalid ones, and
keeps them out of the in-house coverage cards and the validation-selected transfer table."""

import json

from evemtbench.evaluation.aggregate import (
    aggregate_seed_metrics,
    build_result_doc,
    render_result_json,
)
from evemtbench.leaderboard import render_html, scan_submissions
from evemtbench.version import results_version

TASK, GRID = "fault_detection_line", "cigre_mv"
METRICS = ("balanced_accuracy", "macro_f1", "missed_fault_rate", "false_alarm_rate", "accuracy")


def _submission(root, model, *, reproduced=False, tuned_on="val"):
    d = root / model
    meta = {
        "schema_version": 1, "model": model, "org": "Lab", "model_type": "deep",
        "code_link": "https://example.org/x", "benchmark_version": "1.1.0", "track": "partial",
        "trained_on": ["adapt_grid:train"], "tuned_on": tuned_on, "test_data_leakage": False,
        "reproduced": reproduced,
        **({"reproduced_by": "maintainer", "reproduced_commit": "abc1234"} if reproduced else {}),
    }  # fmt: skip
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps(meta))
    per_seed = [{"seed": s, "metrics": {m: 0.87 for m in METRICS}} for s in (0, 1, 2)]
    doc = build_result_doc(
        benchmark_version=results_version("held_out"), task=TASK, protocol="held_out",
        grid=GRID, baseline=model, pretrain_grid=None, test_set="test", n_test=100,
        seeds=[0, 1, 2], n_params=None, compute={}, per_seed=per_seed,
        aggregate=aggregate_seed_metrics([p["metrics"] for p in per_seed]), breakdowns={},
    )  # fmt: skip
    p = d / "results" / TASK / "held_out" / GRID / "test"
    p.mkdir(parents=True)
    (p / "result.json").write_text(render_result_json(doc))


def test_valid_submissions_render_with_their_tier(tmp_path, capsys):
    root = tmp_path / "submissions"
    _submission(root, "acme-net")
    _submission(root, "beta-net", reproduced=True)
    _submission(root, "cheat-net", tuned_on="test")  # invalid -> skipped
    rows = scan_submissions(root, reference=tmp_path / "none.csv")
    assert {r.baseline for r in rows} == {"acme-net", "beta-net"}
    assert "skipping invalid submission cheat-net" in capsys.readouterr().err
    page = render_html(rows, version="1.1.0")
    assert 'acme-net <span class="tier self"' in page
    assert 'beta-net <span class="tier repro"' in page
    assert "cheat-net" not in page


def test_no_submissions_dir_is_fine(tmp_path):
    assert scan_submissions(tmp_path / "absent") == []
