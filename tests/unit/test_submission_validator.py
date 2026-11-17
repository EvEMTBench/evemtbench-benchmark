"""Submission validator (evemtbench.submission): a well-formed submission passes, and each kind of
broken submission fails with a specific error (the counter-proof for every check)."""

import json

import pytest

from evemtbench import submission as S
from evemtbench.evaluation.aggregate import (
    aggregate_seed_metrics,
    build_result_doc,
    render_result_json,
)
from evemtbench.version import results_version

TASK, GRID = "fault_detection_line", "cigre_mv"
METRICS = ("balanced_accuracy", "macro_f1", "missed_fault_rate", "false_alarm_rate", "accuracy")


def _meta(**over):
    meta = {
        "schema_version": 1,
        "model": "acme-net",
        "org": "ACME Lab",
        "model_type": "deep",
        "code_link": "https://example.org/acme-net",
        "benchmark_version": "1.1.0",
        "track": "partial",
        "trained_on": ["adapt_grid:train"],
        "tuned_on": "val",
        "test_data_leakage": False,
        "reproduced": False,
    }
    meta.update(over)
    return meta


def _result(model="acme-net", seeds=(0, 1, 2), test_set="test", n_test=100, task=TASK):
    per_seed = [
        {"seed": s, "metrics": {m: 0.8 + 0.01 * i for m in METRICS}} for i, s in enumerate(seeds)
    ]
    agg = aggregate_seed_metrics([p["metrics"] for p in per_seed])
    return build_result_doc(
        benchmark_version=results_version("held_out"),
        task=task,
        protocol="held_out",
        grid=GRID,
        baseline=model,
        pretrain_grid=None,
        test_set=test_set,
        n_test=n_test,
        seeds=list(seeds),
        n_params=1000,
        compute={},
        per_seed=per_seed,
        aggregate=agg,
        breakdowns={},
    )


def _write(root, meta=None, docs=None, name="acme-net"):
    d = root / name
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps(meta or _meta(model=name)))
    for doc in docs if docs is not None else [_result(model=name)]:
        p = d / "results" / doc["task"] / doc["protocol"] / doc["grid"] / doc["test_set"]
        p.mkdir(parents=True, exist_ok=True)
        (p / "result.json").write_text(render_result_json(doc))
    return d


def _errors(d, **kw):
    return S.validate_submission(d, **kw).errors


def test_valid_submission_passes(tmp_path):
    rep = S.validate_submission(_write(tmp_path))
    assert rep.ok, rep.errors
    assert rep.n_results == 1


@pytest.mark.parametrize(
    "override, fragment",
    [
        ({"tuned_on": "test"}, "tuned_on"),
        ({"test_data_leakage": True}, "test_data_leakage"),
        ({"trained_on": ["adapt_grid:test"]}, "names a held-out split"),
        ({"trained_on": ["benchmark"]}, "not allowed"),
        ({"model_type": "magic"}, "model_type"),
        ({"code_link": "not a url"}, "code_link"),
        ({"reproduced": True}, "reproduced_by"),
        ({"track": "everything"}, "track"),
    ],
)
def test_bad_metadata_fails(tmp_path, override, fragment):
    errs = _errors(_write(tmp_path, meta=_meta(**override)))
    assert any(fragment in e for e in errs), errs


def test_in_house_name_is_reserved(tmp_path):
    d = _write(tmp_path, meta=_meta(model="mlp"), docs=[_result(model="mlp")], name="mlp")
    assert any("collides" in e for e in _errors(d))


def test_tampered_aggregate_fails(tmp_path):
    doc = _result()
    doc["aggregate"]["balanced_accuracy"]["mean"] = 0.99  # not what per_seed says
    errs = _errors(_write(tmp_path, docs=[doc]))
    assert any("recomputed from per_seed" in e for e in errs), errs


def test_too_few_seeds_fails(tmp_path):
    errs = _errors(_write(tmp_path, docs=[_result(seeds=(0, 1))]))
    assert any("at least 3" in e for e in errs), errs


def test_baseline_must_equal_model(tmp_path):
    errs = _errors(_write(tmp_path, docs=[_result(model="other-net")]))
    assert any("must equal the model name" in e for e in errs), errs


def test_unscored_test_set_fails(tmp_path):
    errs = _errors(_write(tmp_path, docs=[_result(test_set="val")]))
    assert any("not scored" in e for e in errs), errs


def test_missing_official_metric_fails(tmp_path):
    doc = _result()
    del doc["aggregate"]["missed_fault_rate"]
    errs = _errors(_write(tmp_path, docs=[doc]))
    assert any("official metrics missing" in e for e in errs), errs


def test_n_test_checked_against_reference(tmp_path):
    ref = tmp_path / "aggregated_results.csv"
    ref.write_text(
        "version,task,protocol,grid,baseline,source,test_set,metric,mean,std,ci95,n_seeds,"
        "n_test,n_params,fit_seconds\n"
        f"1.0.0,{TASK},held_out,{GRID},mlp,,test,accuracy,0.9,0,0,5,123,1,1\n"
    )
    errs = _errors(_write(tmp_path / "s"), reference=ref)
    assert any("n_test 100 != reference 123" in e for e in errs), errs


def test_full_track_requires_every_held_out_cell(tmp_path):
    errs = _errors(_write(tmp_path, meta=_meta(track="full")))
    assert any("track 'full' is missing" in e for e in errs), errs


def test_run_record_attestation_is_checked(tmp_path):
    d = _write(tmp_path)
    rec = {"attestation": {"model_selection_on": "test", "held_out_untouched": ["benchmark"]}}
    (d / "results" / TASK / "held_out" / GRID / "run_record.json").write_text(json.dumps(rec))
    errs = _errors(d)
    assert any("held_out_untouched" in e for e in errs), errs
    assert any("model_selection_on" in e for e in errs), errs


def test_misplaced_result_fails(tmp_path):
    d = _write(tmp_path)
    wrong = d / "results" / "elsewhere" / "result.json"
    wrong.parent.mkdir(parents=True)
    wrong.write_text(render_result_json(_result(test_set="benchmark")))
    errs = _errors(d)
    assert any("path does not match" in e for e in errs), errs


def test_cli_exit_codes(tmp_path, capsys):
    good = _write(tmp_path / "ok")
    bad = _write(tmp_path / "bad", meta=_meta(tuned_on="benchmark"))
    ref = ["--reference", str(tmp_path / "no-reference.csv")]  # the fixtures' n_test is synthetic
    assert S.main([*ref, str(good)]) == 0
    assert S.main([*ref, str(bad)]) == 1
    assert S.main([*ref, "--all", str(tmp_path / "ok"), str(tmp_path / "bad")]) == 1
    assert "FAIL" in capsys.readouterr().out
