"""The evaluator self-test (evemtbench.evaluation.selftest): all controls pass on the healthy
harness, and a sabotaged metric path makes the relevant control FAIL — proving the self-test is
non-vacuous (the UnderstandingVCS 'the harness is broken, not the method' idiom)."""

import pytest

from evemtbench.evaluation import selftest


def test_selftest_passes_on_healthy_harness():
    ok, checks = selftest.run_selftest()
    assert ok, [c for c in checks if not c[1]]
    assert len(checks) >= 12
    # every control routes through the real evaluate() (spot: the ceiling checks name tasks)
    names = [c[0] for c in checks]
    assert any("fault_detection_local" in n for n in names)
    assert any("alignment" in n for n in names)


def test_selftest_catches_a_broken_metric(monkeypatch):
    """Sabotage the binary metric leaf the way a real bug would (wrong scale) — the perfect-
    predictions ceiling control must FAIL and the overall result must flip."""
    import importlib

    # the package re-exports the evaluate FUNCTION, shadowing the submodule attribute
    eval_mod = importlib.import_module("evemtbench.evaluation.evaluate")

    real = eval_mod.compute_binary_metrics

    def broken(y_true, y_pred, metrics):
        out = real(y_true, y_pred, metrics)
        return {k: (v * 0.9 if isinstance(v, float) else v) for k, v in out.items()}

    monkeypatch.setattr(eval_mod, "compute_binary_metrics", broken)
    ok, checks = selftest.run_selftest()
    assert not ok
    failed = [name for name, passed, _ in checks if not passed]
    assert any("fault_detection_local" in n and "ceiling" in n for n in failed)


def test_selftest_catches_a_silenced_alignment(monkeypatch):
    """If alignment stops raising on malformed frames, the alignment controls must FAIL."""
    import importlib

    # the package re-exports the evaluate FUNCTION, shadowing the submodule attribute
    eval_mod = importlib.import_module("evemtbench.evaluation.evaluate")

    def permissive(metadata, predictions):
        import numpy as np

        merged = metadata.merge(
            predictions.drop_duplicates(["sample_id", "window_idx"]),
            on=["sample_id", "window_idx"],
            how="left",
        )
        if "prediction" not in merged:
            return np.zeros(len(metadata))
        return merged["prediction"].fillna(0).to_numpy()

    monkeypatch.setattr(eval_mod, "align_to_view", permissive)
    ok, checks = selftest.run_selftest()
    assert not ok
    assert any("alignment" in name for name, passed, _ in checks if not passed)


def test_cli_flag_exits_zero(capsys):
    from evemtbench.evaluation import cli

    with pytest.raises(SystemExit) as e:
        cli.main(["--self-test"])
    assert e.value.code == 0
    assert "evaluator self-test: OK" in capsys.readouterr().out
