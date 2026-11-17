"""Impedance-based fault locator on a synthetic line with a known fault position.

The test line is built from first principles in phase coordinates: a lumped R-X line (positive-
and zero-sequence series impedance, no shunt capacitance) between two Thevenin sources, with a
fault of resistance ``rf`` at the fraction ``m`` of its length. Solving the three-node network
gives the steady-state phasors at both terminals; they are turned into sampled waveforms with the
benchmark's window geometry (480 samples at 9.6 kHz) and channel order, so the tests exercise the
whole chain: waveform -> full-cycle DFT -> loop selection -> location.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from evemtbench.baselines._phasor import fundamental_phasors, samples_per_cycle
from evemtbench.baselines.impedance import (
    LOOPS,
    ImpedanceBaseline,
    LineParameterError,
    LineParameters,
    fit_impedance_scale,
    graph_nominal_frequencies,
    line_parameters_from_graph,
    lines_measured_at_far_end,
    one_terminal_distance,
    select_fault_loop,
    two_terminal_distance,
)
from evemtbench.tasks.line_view import OneEndedLineWarning
from evemtbench.tasks.registry import get_task
from evemtbench.tasks.task import Task

FS = 9600.0
WINDOW = 480
A = np.exp(2j * np.pi / 3)

LINE = LineParameters("MainLn1-2A", 40.0, 0.0334 + 0.26j, 0.118 + 0.979j)
E_PHASE = 110e3 / np.sqrt(3)

BOLTED = 1e-6  # ohm; "bolted" without a singular fault admittance
POSITIONS = (20.0, 50.0, 80.0)
FAULTS = ("abc", "ag", "bc", "bcg")
EXPECTED_LOOP = {"abc": "ABC", "ag": "AG", "bc": "BC", "bcg": "BC"}

# Tolerances, in percent of the line length.
TOL_EXACT = 0.02  # where the method is exact for the lumped line (float32 waveforms + DFT)


def _phase_matrix(z1: complex, z0: complex) -> np.ndarray:
    """3x3 phase-coordinate impedance of a symmetrical element from its sequence impedances."""
    self_z, mutual = (z0 + 2 * z1) / 3, (z0 - z1) / 3
    return np.full((3, 3), mutual) + np.eye(3) * (self_z - mutual)


def _fault_admittance(fault: str, rf: float) -> np.ndarray:
    g = 1.0 / rf
    y = np.zeros((3, 3), dtype=complex)
    if fault == "abc":  # three-phase (symmetrical; no ground current flows)
        y += np.eye(3) * g
    elif fault == "ag":  # phase A to ground
        y[0, 0] = g
    elif fault == "bc":  # phase B to phase C
        y[1, 1] = y[2, 2] = g
        y[1, 2] = y[2, 1] = -g
    elif fault == "bcg":  # phases B and C to ground
        y[1, 1] = y[2, 2] = g
    else:
        raise ValueError(fault)
    return y


def synthetic_terminal_phasors(
    m_percent: float,
    fault: str,
    rf: float = BOLTED,
    *,
    line: LineParameters = LINE,
    far_angle_deg: float = -8.0,
    far_source_scale: float = 1.0,
):
    """Steady-state phasors ``(v_s, i_s, v_r, i_r)`` (each ``(3,)``; both currents into the line)
    for a fault at ``m_percent`` of the line from terminal S."""
    m = m_percent / 100.0
    z_line = _phase_matrix(line.z1_total, line.z0_ohm_per_km * line.length_km)
    z_src_s = _phase_matrix(1.0 + 8.0j, 2.0 + 12.0j)
    z_src_r = _phase_matrix(2.0 + 15.0j, 3.0 + 20.0j) * far_source_scale
    balanced = np.array([1.0, A**2, A])
    e_s = E_PHASE * balanced
    e_r = E_PHASE * np.exp(1j * np.radians(far_angle_deg)) * balanced
    y_s = np.linalg.inv(z_src_s + m * z_line)
    y_r = np.linalg.inv(z_src_r + (1 - m) * z_line)
    v_f = np.linalg.solve(y_s + y_r + _fault_admittance(fault, rf), y_s @ e_s + y_r @ e_r)
    i_s, i_r = y_s @ (e_s - v_f), y_r @ (e_r - v_f)
    return e_s - z_src_s @ i_s, i_s, e_r - z_src_r @ i_r, i_r


def _waveform(phasors: np.ndarray, f0: float) -> np.ndarray:
    """RMS phasors ``(c,)`` -> sampled steady-state waveforms ``(WINDOW, c)`` (float32)."""
    t = np.arange(WINDOW) / FS
    wave = np.sqrt(2) * (phasors[None, :] * np.exp(2j * np.pi * f0 * t)[:, None]).real
    return wave.astype(np.float32)


def synthetic_window(m_percent, fault, rf=BOLTED, *, f0=50.0, vtr=1.0, ctr=1.0, **kwargs):
    """One 12-channel line-view window: [I_S(3), V_S(3), I_R(3), V_R(3)], divided by the
    instrument-transformer ratios."""
    v_s, i_s, v_r, i_r = synthetic_terminal_phasors(m_percent, fault, rf, **kwargs)
    return _waveform(np.concatenate([i_s / ctr, v_s / vtr, i_r / ctr, v_r / vtr]), f0)


def _locate_window(window, f0, *, scale=1.0):
    """(one-terminal %, loop name, two-terminal %) for one synthetic line-view window."""
    p = fundamental_phasors(window[None], samples_per_cycle(FS, f0))
    z1, k0 = np.array([LINE.z1_total]), np.array([LINE.k0])
    one, loop = one_terminal_distance(p[:, 3:6], p[:, 0:3], z1, k0, impedance_scale=scale)
    two = two_terminal_distance(
        p[:, 3:6], p[:, 0:3], p[:, 9:12], p[:, 6:9], z1, impedance_scale=scale
    )
    return float(one[0]), LOOPS[loop[0]], float(two[0])


# --- the two locators on the synthetic line ---------------------------------------------------
@pytest.mark.parametrize("f0", [50.0, 60.0])
@pytest.mark.parametrize("fault", FAULTS)
@pytest.mark.parametrize("m", POSITIONS)
def test_bolted_fault_is_located_by_both_methods(m, fault, f0):
    one, loop, two = _locate_window(synthetic_window(m, fault, f0=f0), f0)
    assert loop == EXPECTED_LOOP[fault]
    assert one == pytest.approx(m, abs=TOL_EXACT)
    assert two == pytest.approx(m, abs=TOL_EXACT)


@pytest.mark.parametrize("f0", [50.0, 60.0])
@pytest.mark.parametrize("fault", ["abc", "ag"])
@pytest.mark.parametrize("m", POSITIONS)
def test_resistive_fault_two_terminal_exact_one_terminal_biased(m, fault, f0):
    """A 10 ohm fault with infeed from both ends: the two-terminal method is unaffected, the
    one-terminal reactance method is off by several percent of the line length (the far-end
    current through the fault resistance is not in phase with the local current)."""
    one, _, two = _locate_window(synthetic_window(m, fault, rf=10.0, f0=f0), f0)
    assert two == pytest.approx(m, abs=TOL_EXACT)
    assert 1.0 < abs(one - m) < 40.0


@pytest.mark.parametrize("m", POSITIONS)
def test_resistive_three_phase_fault_without_far_infeed_is_located(m):
    """With (practically) no infeed from the far end the fault-resistance drop is in phase with
    the local current and the reactance is unaffected: that is why the method uses Im(Z)."""
    window = synthetic_window(m, "abc", rf=10.0, far_source_scale=1e7)
    one, loop, _ = _locate_window(window, 50.0)
    assert loop == "ABC"
    assert one == pytest.approx(m, abs=0.05)


def test_far_current_sign_convention():
    """With the far-end current counted out of the line, the default sign is wrong and -1 fixes
    it: the sign is a property of the data, not of the method."""
    window = synthetic_window(35.0, "ag", rf=5.0)
    window[:, 6:9] *= -1.0
    p = fundamental_phasors(window[None], samples_per_cycle(FS, 50.0))
    args = (p[:, 3:6], p[:, 0:3], p[:, 9:12], p[:, 6:9], np.array([LINE.z1_total]))
    assert abs(two_terminal_distance(*args)[0] - 35.0) > 20.0
    assert two_terminal_distance(*args, far_current_sign=-1)[0] == pytest.approx(35.0, abs=0.02)


def test_secondary_quantities_need_the_impedance_scale():
    vtr, ctr = 1100.0, 400.0  # VT 110 kV / 100 V, CT 400 A / 1 A
    window = synthetic_window(50.0, "ag", vtr=vtr, ctr=ctr)
    one, _, two = _locate_window(window, 50.0)
    assert one == pytest.approx(50.0 * ctr / vtr, abs=0.05)  # uncalibrated: off by VTR/CTR
    assert abs(two - 50.0) > 5.0
    one, _, two = _locate_window(window, 50.0, scale=vtr / ctr)
    assert one == pytest.approx(50.0, abs=TOL_EXACT)
    assert two == pytest.approx(50.0, abs=TOL_EXACT)


# --- phase selector ----------------------------------------------------------------------------
def test_phase_selector_table():
    def phasors(a, b, c):
        return np.array([[a, b * A**2, c * A]], dtype=complex)

    assert LOOPS[select_fault_loop(phasors(10, 10, 10))[0]] == "ABC"
    assert LOOPS[select_fault_loop(phasors(10, 1, 1))[0]] == "AG"
    assert LOOPS[select_fault_loop(phasors(1, 10, 1))[0]] == "BG"
    assert LOOPS[select_fault_loop(phasors(1, 1, 10))[0]] == "CG"
    # two phases in opposition (phase-to-phase): no residual current
    assert LOOPS[select_fault_loop(np.array([[10, -10, 0.5]], dtype=complex))[0]] == "AB"
    assert LOOPS[select_fault_loop(np.array([[0.5, 10, -10]], dtype=complex))[0]] == "BC"
    assert LOOPS[select_fault_loop(np.array([[-10, 0.5, 10]], dtype=complex))[0]] == "CA"
    # three similar currents with a residual: ground loop of the largest phase
    assert LOOPS[select_fault_loop(np.array([[10, 9, 8]], dtype=complex))[0]] == "AG"
    assert LOOPS[select_fault_loop(np.zeros((1, 3), complex))[0]] == "AG"  # degenerate, defined


# --- scale fit ----------------------------------------------------------------------------------
def test_fit_impedance_scale_is_robust_to_gross_outliers():
    rng = np.random.default_rng(0)
    target = rng.uniform(5, 95, 400)
    uncalibrated = target / 2.75 + rng.normal(0, 0.05, 400)
    uncalibrated[:120] = rng.uniform(200, 5000, 120)  # windows where the method does not apply
    fit = fit_impedance_scale(uncalibrated, target)
    assert fit["scale"] == pytest.approx(2.75, rel=0.01)
    assert fit["plain_least_squares"] < 0.5  # what plain least squares would have returned
    assert fit["n_windows"] == 400 and fit["n_kept"] == 200
    with pytest.raises(ValueError, match="usable windows"):
        fit_impedance_scale(np.ones(3), np.ones(3))


# --- line parameters from a topology graph ------------------------------------------------------
def _graph():
    nx = pytest.importorskip("networkx")
    g = nx.MultiGraph()
    good = {"length": 20.0, "rline": 0.03, "xline": 0.26, "rline0": 0.12, "xline0": 0.98}
    g.add_edge("MainBus1", "MainBus2", **{"MainLn1-2A_param_dict": {**good, "cubicle_1": "Cub_2"}})
    g["MainBus1"]["MainBus2"][0]["MainLn1-2A_typ_param_dict"] = {"is_cable": False}
    g.add_edge("MainBus1", "MainBus2", **{"MainLn1-2B_param_dict": {"length": 25.0, "xline": 0.2}})
    g.add_edge("MainBus2", "MainBus3", **{"MainLn2-3A_param_dict": {**good, "length": 0.0}})
    g.add_edge("BusExt", "MainBus1", **{"Trf1_typ_param_dict": {"frnom": 50.0}})
    g.add_edge("MainBus2", "BusX", **{"LnOHLSwitch_param_dict": dict(good)})  # not a main line
    return g


def test_line_parameters_from_graph_reads_and_reports():
    params, unusable = line_parameters_from_graph(_graph())
    assert set(params) == {"MainLn1-2A"}
    line = params["MainLn1-2A"]
    assert line.length_km == 20.0
    assert line.z1_total == pytest.approx(20.0 * (0.03 + 0.26j))
    assert line.k0 == pytest.approx(((0.12 + 0.98j) - (0.03 + 0.26j)) / (3 * (0.03 + 0.26j)))
    assert set(unusable) == {"MainLn1-2B", "MainLn2-3A"}  # reported, not guessed
    assert "rline" in unusable["MainLn1-2B"] and "non-physical" in unusable["MainLn2-3A"]
    assert graph_nominal_frequencies(_graph()) == {50.0}


# --- fit / predict on an in-memory task ---------------------------------------------------------
class _View:
    """Minimal in-memory stand-in for a TaskView (the attributes the baseline uses)."""

    def __init__(self, x, labels, lines, cubicles, grid):
        self.x, self.labels, self.grid = x, np.asarray(labels, dtype=float), grid
        self.metadata = pd.DataFrame(
            {"sample_id": np.arange(len(x)), "window_idx": 0, "y_fault_line": lines}
        )
        self._feature_names = [
            f"{cub}_{q}_{ph}_{unit}"
            for cub in cubicles
            for q, unit in (("Isec", "A"), ("Usec", "V"))
            for ph in ("L1", "L2", "L3")
        ]

    def __len__(self):
        return len(self.x)

    @property
    def feature_shape(self):
        return self.x.shape

    def iter_batches(self, batch_size, *, order=None):
        for i in range(0, len(self.x), batch_size):
            yield self.x[i : i + batch_size], self.labels[i : i + batch_size]

    def feature_axes(self):
        return [(self.grid, self._feature_names, None)]


_TWO_ENDED = ("Bus_1_Line_1-2A", "Bus_2_Line_1-2A")


def _task(task_id, view, grid="double_line"):
    return Task(get_task(task_id), "held_out", grid, "test", view, view, view, view)


def _dataset(n, *, observability, f0=50.0, vtr=1.0, ctr=1.0, seed=0):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(5, 95, n)
    faults = [FAULTS[i % len(FAULTS)] for i in range(n)]
    x = np.stack(
        [
            synthetic_window(m, f, f0=f0, vtr=vtr, ctr=ctr)
            for m, f in zip(positions, faults, strict=True)
        ]
    )
    if observability == "local":
        x = x[:, :, :6]
    return x, positions


@pytest.mark.parametrize("task_id", ["fault_location_local", "fault_location_line"])
def test_fit_predict_recovers_positions_with_the_default_scale(task_id):
    """Primary quantities (the benchmark's case): nothing is fitted, the scale is 1."""
    x, positions = _dataset(48, observability=task_id.rsplit("_", 1)[1])
    view = _View(x, positions, LINE.name, _TWO_ENDED, "synthetic_two_ended")
    model = ImpedanceBaseline(line_parameters={LINE.name: LINE})
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # in particular: no "not primary quantities" warning
        model.fit(_task(task_id, view))
    record = model.calibration_
    assert record["impedance_scale"]["grid"] == 1.0
    assert record["impedance_scale"]["fit"] == {"source": "given"}
    assert record["voltage_level"]["measured_to_nominal"] == pytest.approx(1.0, abs=0.2)
    assert record["samples_per_cycle"] == 192 and record["nominal_hz"] == 50.0
    assert record["one_ended_lines"] == []
    preds = model.predict(view)
    assert list(preds.columns) == ["sample_id", "window_idx", "prediction"]
    assert np.abs(preds["prediction"].to_numpy() - positions).max() < 0.1
    expected = "two_terminal" if task_id.endswith("line") else "one_terminal"
    assert set(model.locate(view)["method"]) == {expected}


@pytest.mark.parametrize("task_id", ["fault_location_local", "fault_location_line"])
def test_secondary_quantities_scale_fitted_on_train(task_id):
    """Secondary quantities: ``impedance_scale="fit"`` recovers VT ratio / CT ratio from the
    training view (bolted faults, where the one-terminal estimate is unbiased)."""
    vtr, ctr = 1100.0, 400.0
    x, positions = _dataset(48, observability=task_id.rsplit("_", 1)[1], vtr=vtr, ctr=ctr)
    view = _View(x, positions, LINE.name, _TWO_ENDED, "synthetic_two_ended")
    model = ImpedanceBaseline(line_parameters={LINE.name: LINE}, impedance_scale="fit")
    model.fit(_task(task_id, view))
    record = model.calibration_["impedance_scale"]
    assert record["grid"] == pytest.approx(vtr / ctr, rel=1e-3)
    assert record["fit"]["by_line"][LINE.name]["scale"] == pytest.approx(vtr / ctr, rel=1e-3)
    assert np.abs(model.predict(view)["prediction"].to_numpy() - positions).max() < 0.1
    per_line = ImpedanceBaseline(
        line_parameters={LINE.name: LINE}, impedance_scale="fit", scale_per="line"
    )
    per_line.fit(_task(task_id, view))
    assert per_line.calibration_["impedance_scale"]["per"] == "line"
    assert per_line.calibration_["impedance_scale"]["by_line"][LINE.name] == pytest.approx(
        vtr / ctr, rel=1e-3
    )


def test_default_scale_warns_when_the_quantities_are_not_primary():
    x, positions = _dataset(12, observability="local", vtr=1100.0, ctr=400.0)
    view = _View(x, positions, LINE.name, _TWO_ENDED, "synthetic_two_ended")
    model = ImpedanceBaseline(line_parameters={LINE.name: LINE})
    with pytest.warns(UserWarning, match="assumes primary quantities"):
        model.fit(_task("fault_location_local", view))
    assert model.calibration_["voltage_level"]["measured_to_nominal"] < 0.01


def _write_overlay_graph(root, rated_frequencies):
    """A minimal aux overlay: ``<root>/adapt_grid/DoubleLine/graph.pickle`` holding LINE."""
    import pickle

    nx = pytest.importorskip("networkx")
    g = nx.MultiGraph()
    values = {
        "length": LINE.length_km,
        "rline": LINE.z1_ohm_per_km.real,
        "xline": LINE.z1_ohm_per_km.imag,
        "rline0": LINE.z0_ohm_per_km.real,
        "xline0": LINE.z0_ohm_per_km.imag,
    }
    g.add_edge("MainBus1", "MainBus2", **{f"{LINE.name}_param_dict": values})
    for i, frnom in enumerate(rated_frequencies):
        g.add_edge(f"BusExt{i}", "MainBus1", **{f"Trf{i}_typ_param_dict": {"frnom": frnom}})
    directory = root / "adapt_grid" / "DoubleLine"
    directory.mkdir(parents=True)
    with open(directory / "graph.pickle", "wb") as fh:
        pickle.dump(g, fh)
    return str(root)


def test_fit_reads_the_line_parameters_from_the_overlay_graph(tmp_path):
    x, positions = _dataset(12, observability="line")
    view = _View(x, positions, LINE.name, _TWO_ENDED, "synthetic_two_ended")
    # a graph may carry element types rated for another frequency (as the IEEE-39 graph does)
    model = ImpedanceBaseline(aux_root=_write_overlay_graph(tmp_path / "a", [50.0, 60.0]))
    model.fit(_task("fault_location_line", view))
    record = model.calibration_
    assert "topology graph" in record["line_parameters"]["source"]
    assert record["line_parameters"]["assumed_units"] == "ohm/km and km"
    assert record["graph_rated_frequencies_hz"] == [50.0, 60.0]
    assert model.lines_[LINE.name].z1_total == pytest.approx(LINE.z1_total)
    assert np.abs(model.predict(view)["prediction"].to_numpy() - positions).max() < 0.1
    # ... but the grid's nominal frequency (50 Hz for double_line) must be one the graph knows
    other = ImpedanceBaseline(aux_root=_write_overlay_graph(tmp_path / "b", [60.0]))
    with pytest.raises(ValueError, match="disagrees with the topology graph"):
        other.fit(_task("fault_location_line", view))


def test_60_hz_override_and_given_per_line_scale():
    x, positions = _dataset(12, observability="line", f0=60.0)
    view = _View(x, positions, LINE.name, _TWO_ENDED, "synthetic_two_ended")
    model = ImpedanceBaseline(
        line_parameters={LINE.name: LINE}, impedance_scale={LINE.name: 1.0}, nominal_hz=60.0
    )
    model.fit(_task("fault_location_line", view))
    assert model.calibration_["samples_per_cycle"] == 160
    assert model.calibration_["impedance_scale"]["fit"] == {"source": "given per line"}
    assert np.abs(model.predict(view)["prediction"].to_numpy() - positions).max() < 0.1


def test_one_ended_line_falls_back_and_mirrors():
    """A line measured only at its higher-numbered bus (CIGRE MV 8-14): the line view repeats
    that terminal, the two-terminal method is undefined, and the one-terminal estimate counts
    from the wrong end, so it is mirrored to the label's reference end."""
    line = LineParameters("MainLn8-14", LINE.length_km, LINE.z1_ohm_per_km, LINE.z0_ohm_per_km)
    rng = np.random.default_rng(1)
    from_measured_end = rng.uniform(5, 95, 16)
    near = np.stack(
        [synthetic_window(m, "ag", line=line)[:, :6] for m in from_measured_end]
    )  # the measured terminal
    x = np.concatenate([near, near], axis=2)  # far slot = near terminal repeated
    labels = 100.0 - from_measured_end  # counted from bus 8, the unmeasured end
    view = _View(x, labels, line.name, ("Bus_MainBus14_Line_8-14",), "synthetic_one_ended")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OneEndedLineWarning)
        assert lines_measured_at_far_end(view) == (("MainLn8-14",), ("MainLn8-14",))
        model = ImpedanceBaseline(line_parameters={line.name: line}, impedance_scale=1.0)
        model.fit(_task("fault_location_line", view))
        frame = model.locate(view)
    assert set(frame["method"]) == {"one_terminal_fallback"}
    assert frame["mirrored"].all()
    assert model.calibration_["one_ended_lines"] == ["MainLn8-14"]
    assert model.calibration_["mirrored_lines"] == ["MainLn8-14"]
    assert np.abs(frame["location"].to_numpy() - labels).max() < 0.1


def test_one_ended_line_measured_at_the_reference_end_is_not_mirrored():
    view = _View(np.zeros((1, WINDOW, 12), np.float32), [50.0], "MainLn8-14",
                 ("Bus_MainBus8_Line_8-14",), "synthetic_one_ended_reference")  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OneEndedLineWarning)
        assert lines_measured_at_far_end(view) == (("MainLn8-14",), ())


def test_predictions_are_clipped_and_undefined_estimates_reported_at_midline():
    x = np.zeros((2, WINDOW, 6), np.float32)  # no signal at all: 0/0
    x[1] = synthetic_window(50.0, "abc", rf=400.0)[:, :6]  # far beyond the line: clipped
    view = _View(x, [50.0, 50.0], LINE.name, _TWO_ENDED, "synthetic_two_ended")
    model = ImpedanceBaseline(line_parameters={LINE.name: LINE}, impedance_scale=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # half the windows carry no voltage at all
        model.fit(_task("fault_location_local", view))
    frame = model.locate(view)
    assert np.isnan(frame["raw"][0]) and frame["location"][0] == 50.0
    assert 0.0 <= frame["location"][1] <= 100.0


def test_line_without_parameters_is_reported_not_guessed():
    x, positions = _dataset(12, observability="local")
    lines = ["MainLn1-2A"] * 11 + ["MainLn1-2B"]
    view = _View(x, positions, lines, _TWO_ENDED, "synthetic_two_ended")
    model = ImpedanceBaseline(line_parameters={LINE.name: LINE}, impedance_scale=1.0)
    with pytest.raises(LineParameterError, match="MainLn1-2B"):
        model.fit(_task("fault_location_local", view))


def test_rejects_other_tasks_and_bad_arguments():
    view = _View(np.zeros((1, WINDOW, 6), np.float32), [50.0], LINE.name, _TWO_ENDED, "synthetic")
    model = ImpedanceBaseline(line_parameters={LINE.name: LINE})
    with pytest.raises(ValueError, match="fault locator"):
        model.fit(_task("fault_detection_local", view))
    with pytest.raises(ValueError, match="local or line"):
        model.fit(_task("fault_location_global", view))
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(view)
    with pytest.raises(ValueError, match="scale_per"):
        ImpedanceBaseline(scale_per="window")
    with pytest.raises(ValueError, match="far_current_sign"):
        ImpedanceBaseline(far_current_sign=0)
    with pytest.raises(ValueError, match="impedance_scale"):
        ImpedanceBaseline(impedance_scale="auto")


# --- review fixes (EXP-06 review) ----------------------------------------------------------------
def test_conflicting_graph_entries_are_reported_not_picked():
    nx = pytest.importorskip("networkx")
    g = _graph()
    other = {"length": 30.0, "rline": 0.03, "xline": 0.26, "rline0": 0.12, "xline0": 0.98}
    g.add_edge("MainBus1", "MainBus9", **{"MainLn1-2A_param_dict": other})  # same label, new value
    g.add_edge("MainBus2", "MainBus9", **{"MainLn2-3A_param_dict": {**other, "length": 12.0}})
    assert isinstance(g, nx.MultiGraph)
    params, unusable = line_parameters_from_graph(g)
    assert "MainLn1-2A" not in params and "conflicting" in unusable["MainLn1-2A"]
    # a broken entry next to a usable one for the same label does not discard the usable one
    assert params["MainLn2-3A"].length_km == 12.0 and "MainLn2-3A" not in unusable


def test_terminals_are_taken_from_the_located_view_not_from_train():
    """A split whose export measures a line at one terminal only (the far one) must fall back and
    mirror even if the training split measured both ends; before the fix the train-time lists
    were used and the repeated terminal went into the two-terminal formula (always 50 %)."""
    x_train, positions = _dataset(16, observability="line")
    train = _View(x_train, positions, LINE.name, _TWO_ENDED, "synthetic_two_ended")
    rng = np.random.default_rng(2)
    from_bus2 = rng.uniform(5, 95, 12)
    near = np.stack([synthetic_window(m, "abc")[:, :6] for m in from_bus2])
    one_ended = _View(
        np.concatenate([near, near], axis=2), 100.0 - from_bus2, LINE.name,
        ("Bus_2_Line_1-2A",), "synthetic_one_ended_at_bus2",
    )  # fmt: skip
    model = ImpedanceBaseline(line_parameters={LINE.name: LINE})
    model.fit(_task("fault_location_line", train))
    assert model.calibration_["one_ended_lines"] == []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OneEndedLineWarning)
        frame = model.locate(one_ended)
    assert set(frame["method"]) == {"one_terminal_fallback"} and frame["mirrored"].all()
    assert np.abs(frame["location"].to_numpy() - (100.0 - from_bus2)).max() < 0.1


def test_terminal_channel_layout_is_checked():
    x, positions = _dataset(4, observability="local")
    view = _View(x, positions, LINE.name, _TWO_ENDED, "synthetic_two_ended")
    names = view._feature_names
    names[0:6] = names[3:6] + names[0:3]  # voltages first: phasors would be of the wrong quantity
    model = ImpedanceBaseline(line_parameters={LINE.name: LINE})
    with pytest.raises(LineParameterError, match="expects"):
        model.fit(_task("fault_location_local", view))


def test_view_width_must_match_the_observability():
    x, positions = _dataset(4, observability="line")
    model = ImpedanceBaseline(line_parameters={LINE.name: LINE})
    model.fit(_task("fault_location_line", _View(x, positions, LINE.name, _TWO_ENDED, "s")))
    local = _View(x[:, :, :6], positions, LINE.name, _TWO_ENDED, "s")
    with pytest.raises(ValueError, match="12 channels"):
        model.predict(local)
