"""Impedance-based fault locator: the conventional reference for the fault-location tasks.

A deterministic, non-learned locator of the kind found in line protection relays and fault
recorders. It answers "how far along the line is the fault" from fundamental-frequency phasors
and the line's series impedance. It is a reference implementation written to be audited, not a
manufacturer-grade algorithm: no load compensation (Takagi), no shunt-capacitance model, no
mutual-coupling compensation, no transient filtering beyond the full-cycle DFT.

Inputs
------
* **Waveforms**: the task view. Per terminal, in this order: three phase currents, three phase
  voltages (``Isec L1..L3``, ``Usec L1..L3``). ``local`` = near terminal (6 channels), ``line`` =
  near then far terminal (12 channels, time-aligned).
* **Line parameters**: positive- and zero-sequence series impedance per unit length and the line
  length, read from the grid topology graph (:func:`line_parameters_from_graph`). The graph is
  loaded from the ``adapt_grid`` family of the auxiliary overlay, because the ``benchmark``
  family lacks the graph file for two of the four reference grids; where both exist they carry
  the same lines. **Units are assumed** to be ohm/km and km (the graph does not state them); the
  assumption is supported by the data, see "Scale" below.
* **Which line**: ``y_fault_line`` from the view metadata selects the line whose parameters are
  used. This is the same oracle the ``local``/``line`` views are built on (the view is already
  centred on that line), so it adds no information beyond what the view encodes. No other label
  column is read at prediction time. ``y_fault_location`` is read only by the optional scale fit
  (``impedance_scale="fit"``), in :meth:`fit`, on ``train``.
* **Nominal frequency**: the grid registry (``GridSpec.frequency_hz``). The topology graph is
  used as a consistency check only (the value must be among its ``frnom`` entries): it is not a
  clean source, because the IEEE-39 graph carries 50 Hz transformer types next to the 60 Hz ones.

Method
------
Phasors are full-cycle DFT estimates on the **last complete cycle** of each window
(``_phasor.py``).

``local`` view, one-terminal reactance method. A phase selector picks the fault loop from the
current phasors (:func:`select_fault_loop`), the loop impedance is formed
(:func:`loop_impedance`; ground loops use zero-sequence compensation with
``k0 = (Z0 - Z1) / (3 Z1)``), and

    location [%] = 100 * scale * Im(Z_loop) / (x1 * length)

Using only the reactance makes the estimate insensitive to a purely resistive fault path when the
fault current comes from this terminal alone. With infeed from the far end the voltage drop across
the fault resistance is not in phase with the local current and appears partly as reactance (the
classical reactance-method error; shown in ``tests/unit/test_impedance_locator.py``).

``line`` view, two-terminal method on positive-sequence quantities with synchronised phasors of
both ends (S = near, R = far, ``Z_L`` the total positive-sequence line impedance):

    m = (V_S - V_R + Z_L * I_R) / (Z_L * (I_S + I_R)),    location [%] = 100 * Re(m)

It needs no fault-loop selection and is independent of the fault resistance for a series-impedance
line. Both currents are taken as flowing **into the line** (``far_current_sign = +1``); see
:class:`ImpedanceBaseline` for how that was established.

``global`` view: not applicable (it adds no physical information for a per-line locator).

Lines with one measured terminal
--------------------------------
A line whose far terminal is not measured (``GridOracle.one_ended_lines``; CIGRE MV line 8-14)
has a ``line`` view that repeats the near terminal, for which the two-terminal method is
undefined. Such a line (determined per located view, from its feature names) falls back to the
one-terminal method and, for the training view, is listed in
``calibration_["one_ended_lines"]``. If its single measured terminal is not the label's
reference end (the lower-numbered bus), the one-terminal estimate is mirrored
(``100 - location``); see docs/known_issues.md section 1.

Scale (instrument-transformer ratios)
-------------------------------------
An impedance formed from secondary voltages and currents equals the primary impedance divided by
``VT ratio / CT ratio``. ``impedance_scale`` is that factor (primary ohm per measured V/A) and is
an explicit parameter.

* **Default ``1.0``: no calibration.** The channels are named ``Isec``/``Usec`` and the dataset
  states no ratios, but the values are at primary level: on all four reference grids the phase
  voltages equal the nominal phase voltage (110 kV / sqrt(3) and so on), and the two-terminal
  method with scale 1 and the graph's impedances taken as ohm/km reproduces the labelled
  position of sustained short circuits to a small fraction of a percent of the line length. The
  two-terminal equation is not homogeneous in the scale, so that agreement confirms the scale
  and the units together. :meth:`fit` re-checks the voltage level on ``train`` without labels,
  records it, and warns if it is far from nominal.
* **A known ratio**: a float, or ``{line: float}``.
* **``"fit"``**: one factor per grid (``scale_per="line"``: per line), estimated on the
  **training split only** by least trimmed squares of the label on the uncalibrated one-terminal
  estimate (:func:`fit_impedance_scale`). It is meant for data with true secondary quantities
  and no stated ratios. It inherits the bias of the one-terminal method: on the reference grids,
  where the true scale is 1, it returns values below 1 that differ from line to line. That is
  why it is not the default.

Whatever is used is kept in ``calibration_`` and written to the run record by the runner.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..evaluation.predictions_io import predictions_frame
from ..tasks.task import Task, TaskView
from ._phasor import fundamental_phasors, samples_per_cycle, symmetrical_components

if TYPE_CHECKING:  # pragma: no cover - typing only
    import networkx as nx

# Fault loops, in the order of the integer codes returned by ``select_fault_loop``.
LOOPS = ("AG", "BG", "CG", "AB", "BC", "CA", "ABC")
_LOOP_PHASES = {
    "AG": (0,), "BG": (1,), "CG": (2,), "AB": (0, 1), "BC": (1, 2), "CA": (2, 0), "ABC": (0, 1, 2),
}  # fmt: skip
# Phase-pair loop for each "the two involved phases are ..." pattern (index = the phase left out).
_PAIR_LOOP_WITHOUT_PHASE = (LOOPS.index("BC"), LOOPS.index("CA"), LOOPS.index("AB"))

GRAPH_FAMILY = "adapt_grid"  # the only family whose aux overlay has a graph for all four grids
_PARAM_SUFFIX = "_param_dict"
_TYPE_SUFFIX = "_typ_param_dict"
_REQUIRED_KEYS = ("length", "rline", "xline", "rline0", "xline0")
_LINE_LABEL = re.compile(r"^MainLn(\d+)-(\d+)")
_BUS_OF_FEATURE = re.compile(r"^Bus_(?:MainBus)?(\d+)_Line_(?P<lineid>.+?)_(?:Isec|Usec)_")
_BATCH = 8192
_MIDLINE = 50.0  # reported when the estimate is not a finite number


class LineParameterError(ValueError):
    """A line the locator is asked about has no usable parameters in the topology graph."""


@dataclass(frozen=True)
class LineParameters:
    """Series impedance of one line. Units as assumed from the graph: ohm/km and km."""

    name: str
    length_km: float
    z1_ohm_per_km: complex  # positive sequence, r + jx
    z0_ohm_per_km: complex  # zero sequence, r0 + jx0

    @property
    def z1_total(self) -> complex:
        """Total positive-sequence series impedance of the line (primary ohm)."""
        return self.z1_ohm_per_km * self.length_km

    @property
    def k0(self) -> complex:
        """Zero-sequence compensation factor ``(Z0 - Z1) / (3 Z1)`` (dimensionless)."""
        return (self.z0_ohm_per_km - self.z1_ohm_per_km) / (3.0 * self.z1_ohm_per_km)


def line_parameters_from_graph(
    graph: nx.Graph,
) -> tuple[dict[str, LineParameters], dict[str, str]]:
    """Read the main-line parameters from a grid topology graph.

    Path: every main line is an edge attribute ``MainLn<id>_param_dict`` holding ``length``,
    ``rline``, ``xline`` (positive sequence) and ``rline0``, ``xline0`` (zero sequence). The
    attribute name without the suffix is the line label used by ``y_fault_line``.

    Returns ``(parameters, unusable)``: ``unusable`` maps a line label to the reason its entry
    cannot be used (a missing key, a non-positive length or reactance, two edges with different
    values for the same label). Nothing is guessed.
    """
    parameters: dict[str, LineParameters] = {}
    unusable: dict[str, str] = {}
    conflicting: dict[str, str] = {}
    for *_, attrs in graph.edges(data=True):
        for key, values in attrs.items():
            if not key.endswith(_PARAM_SUFFIX) or key.endswith(_TYPE_SUFFIX):
                continue
            name = key[: -len(_PARAM_SUFFIX)]
            if not name.startswith("MainLn") or not isinstance(values, dict):
                continue
            missing = [k for k in _REQUIRED_KEYS if not isinstance(values.get(k), (int, float))]
            if missing:
                unusable[name] = f"missing {', '.join(missing)}"
                continue
            length = float(values["length"])
            z1 = complex(float(values["rline"]), float(values["xline"]))
            z0 = complex(float(values["rline0"]), float(values["xline0"]))
            if not (length > 0 and z1.imag > 0 and np.isfinite([length, abs(z1), abs(z0)]).all()):
                unusable[name] = f"non-physical values (length={length}, z1={z1})"
                continue
            entry = LineParameters(name, length, z1, z0)
            if name in conflicting:
                continue
            if name in parameters and parameters[name] != entry:
                # the same label on two edges with different values: do not pick one silently
                conflicting[name] = f"conflicting entries ({parameters.pop(name)} vs {entry})"
                continue
            parameters[name] = entry
    for name, reason in conflicting.items():
        unusable[name] = reason
    # a label with one usable entry is usable; a broken duplicate of it is not a reason to drop it
    return parameters, {k: v for k, v in unusable.items() if k not in parameters}


def graph_nominal_frequencies(graph: nx.Graph) -> set[float]:
    """All ``frnom`` values found in the graph's element parameter dicts (empty if none)."""
    found: set[float] = set()
    containers = [attrs for _, attrs in graph.nodes(data=True)]
    containers += [attrs for *_, attrs in graph.edges(data=True)]
    for attrs in containers:
        for values in attrs.values():
            if isinstance(values, dict) and isinstance(values.get("frnom"), (int, float)):
                found.add(float(values["frnom"]))
    return found


# --------------------------------------------------------------------------------------------
# Phasor-domain building blocks (pure functions; unit-tested on a synthetic line)
# --------------------------------------------------------------------------------------------
def select_fault_loop(
    i_abc: np.ndarray, *, ground_ratio: float = 0.1, phase_ratio: float = 0.5
) -> np.ndarray:
    """Classical overcurrent phase selector. ``i_abc`` complex ``(n, 3)`` -> loop codes ``(n,)``
    indexing :data:`LOOPS`.

    Two comparisons on the current phasors of the measuring terminal:

    * **ground involvement**: residual current ``|Ia + Ib + Ic| > ground_ratio * max|I_phase|``;
    * **involved phases**: every phase with ``|I_phase| >= phase_ratio * max|I_phase|``.

    Loop chosen:

    ===========  ==============  =======================================================
    ground       involved        loop
    ===========  ==============  =======================================================
    yes or no    one phase       that phase to ground (AG / BG / CG)
    yes or no    two phases      between those phases (AB / BC / CA); also correct for a
                                 two-phase-to-ground fault
    no           three phases    three-phase (ABC), evaluated on positive sequence
    yes          three phases    phase with the largest current, to ground
    ===========  ==============  =======================================================

    It works on total (not superimposed) currents, so a fault current that does not clearly
    exceed the load current (high-impedance and incipient faults, weak infeed) is mis-selected.
    """
    mag = np.abs(np.asarray(i_abc))
    largest = mag.max(axis=1)
    safe = np.where(largest > 0, largest, 1.0)
    involved = mag >= phase_ratio * safe[:, None]
    n_involved = involved.sum(axis=1)
    grounded = np.abs(np.asarray(i_abc).sum(axis=1)) > ground_ratio * safe
    strongest = mag.argmax(axis=1)

    loop = np.full(len(mag), LOOPS.index("ABC"), dtype=np.int8)
    single = (n_involved <= 1) | ((n_involved == 3) & grounded)
    loop[single] = strongest[single]  # codes 0..2 are AG, BG, CG
    pair = n_involved == 2
    left_out = np.argmin(involved, axis=1)  # the one phase that is not involved
    loop[pair] = np.asarray(_PAIR_LOOP_WITHOUT_PHASE, dtype=np.int8)[left_out[pair]]
    return loop


def loop_impedance(
    v_abc: np.ndarray, i_abc: np.ndarray, loop: np.ndarray, k0: np.ndarray
) -> np.ndarray:
    """Apparent positive-sequence impedance seen in the selected fault loop, ``(n,)`` complex.

    * phase-to-ground loop of phase p: ``V_p / (I_p + k0 * (Ia + Ib + Ic))``
    * phase-to-phase loop p-q:         ``(V_p - V_q) / (I_p - I_q)``
    * three-phase:                     ``V_1 / I_1`` (positive-sequence components)

    For a bolted fault on a line without infeed from the far end each of these equals the
    positive-sequence line impedance between the terminal and the fault.
    """
    v_abc, i_abc = np.asarray(v_abc), np.asarray(i_abc)
    residual = i_abc.sum(axis=1)
    numerator = np.empty(len(loop), dtype=complex)
    denominator = np.empty(len(loop), dtype=complex)
    for code, name in enumerate(LOOPS):
        rows = loop == code
        if not rows.any():
            continue
        phases = _LOOP_PHASES[name]
        if len(phases) == 1:
            (p,) = phases
            numerator[rows] = v_abc[rows, p]
            denominator[rows] = i_abc[rows, p] + k0[rows] * residual[rows]
        elif len(phases) == 2:
            p, q = phases
            numerator[rows] = v_abc[rows, p] - v_abc[rows, q]
            denominator[rows] = i_abc[rows, p] - i_abc[rows, q]
        else:
            numerator[rows] = symmetrical_components(v_abc[rows])[1]
            denominator[rows] = symmetrical_components(i_abc[rows])[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return numerator / denominator


def one_terminal_distance(
    v_abc: np.ndarray,
    i_abc: np.ndarray,
    z1_total: np.ndarray,
    k0: np.ndarray,
    *,
    impedance_scale: np.ndarray | float = 1.0,
    ground_ratio: float = 0.1,
    phase_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """One-terminal reactance method. Returns ``(distance, loop)``: the distance of the fault
    from the **measuring terminal** in percent of the line length (unclipped), and the loop code.

    ``impedance_scale`` converts a measured V/A ratio into primary ohm (1.0 for primary values).
    """
    loop = select_fault_loop(i_abc, ground_ratio=ground_ratio, phase_ratio=phase_ratio)
    z_loop = loop_impedance(v_abc, i_abc, loop, np.asarray(k0))
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = 100.0 * impedance_scale * z_loop.imag / np.asarray(z1_total).imag
    return distance, loop


def two_terminal_distance(
    v_near: np.ndarray,
    i_near: np.ndarray,
    v_far: np.ndarray,
    i_far: np.ndarray,
    z1_total: np.ndarray,
    *,
    impedance_scale: np.ndarray | float = 1.0,
    far_current_sign: int = 1,
) -> np.ndarray:
    """Two-terminal method on positive-sequence phasors. Returns the distance of the fault from
    the **near** terminal in percent of the line length (unclipped).

    Derivation: the voltage at the fault point, written from either end of a series-impedance
    line, ``V_S - m Z_L I_S = V_R - (1 - m) Z_L I_R`` with both currents flowing into the line,
    gives ``m = (V_S - V_R + Z_L I_R) / (Z_L (I_S + I_R))``. ``far_current_sign = -1`` is for a
    far-end current measured out of the line.
    """
    v_s = symmetrical_components(v_near)[1]
    i_s = symmetrical_components(i_near)[1]
    v_r = symmetrical_components(v_far)[1]
    i_r = far_current_sign * symmetrical_components(i_far)[1]
    z_line = np.asarray(z1_total) / impedance_scale  # primary ohm -> measured V/A
    with np.errstate(divide="ignore", invalid="ignore"):
        m = (v_s - v_r + z_line * i_r) / (z_line * (i_s + i_r))
    return 100.0 * m.real


def fit_impedance_scale(
    uncalibrated: np.ndarray, target: np.ndarray, *, keep: float = 0.5, max_iter: int = 50
) -> dict[str, float]:
    """Least-squares scale ``s`` in ``target ~= s * uncalibrated`` through the origin, trimmed.

    ``uncalibrated`` is the one-terminal distance with ``impedance_scale = 1`` and ``target`` the
    labelled distance from the measuring terminal, both in percent. Plain least squares is
    dominated by the windows where the reactance method does not apply at all (a window whose last
    cycle still holds the fault inception, a high-impedance fault: the apparent reactance is many
    times the line reactance). The fit is therefore *least trimmed squares*: start from the
    median ratio, then repeat {keep the ``keep`` fraction of windows with the smallest absolute
    residual, refit by least squares} until the kept set stops changing. Deterministic.

    Returns the scale with the plain least-squares and median-ratio values for comparison.
    """
    u = np.asarray(uncalibrated, dtype=float)
    t = np.asarray(target, dtype=float)
    ok = np.isfinite(u) & np.isfinite(t) & (u > 0)
    u, t = u[ok], t[ok]
    if len(u) < 10:
        raise ValueError(f"only {len(u)} usable windows to fit the impedance scale; need >= 10")
    median_ratio = float(np.median(t / u))
    plain = float(u @ t / (u @ u))
    scale, n_keep = median_ratio, max(10, int(round(keep * len(u))))
    kept: np.ndarray | None = None
    for _ in range(max_iter):
        order = np.argsort(np.abs(t - scale * u), kind="stable")[:n_keep]
        order.sort()
        if kept is not None and np.array_equal(order, kept):
            break
        kept = order
        scale = float(u[kept] @ t[kept] / (u[kept] @ u[kept]))
    return {
        "scale": scale,
        "plain_least_squares": plain,
        "median_ratio": median_ratio,
        "n_windows": int(len(u)),
        "n_kept": int(n_keep),
    }


# --------------------------------------------------------------------------------------------
# Which terminal a line's measurements come from
# --------------------------------------------------------------------------------------------
def lines_measured_at_far_end(view: TaskView) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(one_ended, mirrored)`` line labels for the grid behind ``view``.

    ``one_ended``: lines with a single measured terminal (``GridOracle.one_ended_lines``).
    ``mirrored``: the subset whose single terminal is **not** at the lower-numbered bus, i.e. not
    at the reference end of ``y_fault_location``; a one-terminal estimate from it measures the
    distance from the other end. Derived from the window feature names only (no labels).

    Also checks that every line cubicle carries its channels in the order the locator indexes
    them (``Isec L1..L3, Usec L1..L3``); a different layout fails loud instead of producing
    phasors of the wrong quantity.
    """
    from ..tasks import line_view

    one_ended: set[str] = set()
    mirrored: set[str] = set()
    for grid, feature_names, channels in view.feature_axes():
        if line_view.is_double_line_view(feature_names):
            continue  # portable EndA/EndB layout: always two terminals, canonicalised by bus id
        _check_terminal_layout(feature_names)
        oracle = line_view.build_oracle(grid, feature_names, channels)
        for label in oracle.one_ended_lines:
            one_ended.add(label)
            reference = _LINE_LABEL.match(label)
            lineid = label[len("MainLn") :]
            buses = {
                int(m.group(1))
                for m in map(_BUS_OF_FEATURE.match, feature_names)
                if m is not None and m.group("lineid") == lineid
            }
            if reference is None or len(buses) != 1:
                raise LineParameterError(
                    f"cannot tell which end of one-ended line {label!r} is measured "
                    f"(buses found: {sorted(buses)})"
                )
            if buses.pop() != min(int(reference.group(1)), int(reference.group(2))):
                mirrored.add(label)
    return tuple(sorted(one_ended)), tuple(sorted(mirrored))


_TERMINAL_LAYOUT = tuple(f"_{q}_{ph}" for q in ("Isec", "Usec") for ph in ("L1", "L2", "L3"))


def _check_terminal_layout(feature_names: list[str]) -> None:
    """Fail loud unless each ``Bus_*_Line_*`` cubicle block is ``Isec L1..L3, Usec L1..L3``."""
    from ..tasks.line_view import CHANNELS_PER_CUBICLE, _cubicle_keys

    if len(feature_names) % CHANNELS_PER_CUBICLE:
        return  # not a cubicle layout; the oracle reports that itself
    for c, key in enumerate(_cubicle_keys(feature_names)):
        if not re.match(r"^Bus_.+_Line_.+$", key):
            continue
        block = feature_names[c * CHANNELS_PER_CUBICLE : (c + 1) * CHANNELS_PER_CUBICLE]
        pairs = zip(block, _TERMINAL_LAYOUT, strict=True)
        if not all(name.startswith(key + token) for name, token in pairs):
            raise LineParameterError(
                f"line cubicle {key!r} has channels {block}; the impedance locator expects "
                "Isec L1, L2, L3 then Usec L1, L2, L3"
            )


# --------------------------------------------------------------------------------------------
# The baseline
# --------------------------------------------------------------------------------------------
class ImpedanceBaseline:
    """Impedance-based fault locator (see the module docstring for the method).

    Parameters
    ----------
    impedance_scale:
        ``VT ratio / CT ratio``. ``1.0`` (default): primary quantities, nothing is fitted. A
        float: a known ratio for the whole grid. A ``{line: float}`` dict: per line. ``"fit"``:
        estimate on ``train`` (module docstring, "Scale").
    scale_per:
        ``"grid"`` (default) or ``"line"``: granularity of the scale when it is fitted.
    far_current_sign:
        ``+1`` (default): the far-terminal current flows into the line, like the near-terminal
        one. Established on the data: in healthy pre-fault windows of all four reference grids the
        positive-sequence currents of the two terminals of a line satisfy ``I_far ~= -I_near``
        (through-flowing load), which only holds if both are counted into the line; and on
        ``train`` the two-terminal error is far smaller with ``+1`` than with ``-1``.
    line_parameters:
        ``{line: LineParameters}`` to use instead of reading the topology graph (tests, grids
        without a graph, corrected parameters).
    nominal_hz, sampling_rate_hz:
        Overrides. Default: the grid registry's nominal frequency, and the window length of the
        benchmark's window contract divided into the number of samples per window.
    ground_ratio, phase_ratio:
        Phase-selector thresholds (:func:`select_fault_loop`). Fixed, not fitted.
    """

    name = "impedance"

    def __init__(
        self,
        *,
        impedance_scale: str | float | dict[str, float] = 1.0,
        scale_per: str = "grid",
        far_current_sign: int = 1,
        line_parameters: dict[str, LineParameters] | None = None,
        nominal_hz: float | None = None,
        sampling_rate_hz: float | None = None,
        ground_ratio: float = 0.1,
        phase_ratio: float = 0.5,
        scale_fit_keep: float = 0.5,
        aux_root: str | None = None,
    ):
        if scale_per not in ("grid", "line"):
            raise ValueError(f"scale_per must be 'grid' or 'line', got {scale_per!r}")
        if far_current_sign not in (1, -1):
            raise ValueError("far_current_sign must be +1 or -1")
        if isinstance(impedance_scale, str) and impedance_scale != "fit":
            raise ValueError("impedance_scale must be 'fit', a float or a {line: float} dict")
        self.impedance_scale = impedance_scale
        self.scale_per = scale_per
        self.far_current_sign = far_current_sign
        self.line_parameters = line_parameters
        self.nominal_hz = nominal_hz
        self.sampling_rate_hz = sampling_rate_hz
        self.ground_ratio = ground_ratio
        self.phase_ratio = phase_ratio
        self.scale_fit_keep = scale_fit_keep
        self.aux_root = aux_root
        # set by fit()
        self.lines_: dict[str, LineParameters] | None = None
        self.scales_: dict[str, float] = {}
        self.default_scale_: float | None = None
        self.n_cycle_: int | None = None
        self.two_terminal_: bool = False
        self.one_ended_lines_: tuple[str, ...] = ()
        self.mirrored_lines_: tuple[str, ...] = ()
        self.calibration_: dict[str, Any] | None = None

    # --- fit --------------------------------------------------------------------------------
    def fit(self, task: Task) -> None:
        """Load the line parameters and (optionally) fit the impedance scale on ``task.train``.
        Reads no other split."""
        from ..tasks.taxonomy import Function, Observability

        td = task.definition
        if td.function is not Function.FAULT_LOCATION:
            raise ValueError(
                f"impedance baseline is a fault locator; task {td.task_id!r} is {td.function.value}"
            )
        if td.observability not in (Observability.LOCAL, Observability.LINE):
            raise ValueError(
                "impedance baseline needs the terminal(s) of the faulted line (local or line "
                f"view); task {td.task_id!r} uses {td.observability.value} input"
            )
        self.two_terminal_ = td.observability is Observability.LINE

        unusable: dict[str, str] = {}
        graph_frequencies: set[float] = set()
        if self.line_parameters is not None:
            self.lines_, source = dict(self.line_parameters), "constructor argument"
        else:
            from ..data.graphs import load_grid_graph

            graph = load_grid_graph(task.grid, family=GRAPH_FAMILY, root=self.aux_root)
            self.lines_, unusable = line_parameters_from_graph(graph)
            graph_frequencies = graph_nominal_frequencies(graph)
            source = f"topology graph ({GRAPH_FAMILY} family of the aux overlay)"

        nominal_hz = self.nominal_hz
        if nominal_hz is None:
            from ..data.roots import get_grid

            nominal_hz = float(get_grid(task.grid).frequency_hz)
        # Consistency check only: element types in a graph may carry another rated frequency
        # (the IEEE-39 graph holds 60 Hz and a few 50 Hz transformer types), so the graph cannot
        # be the source; the nominal frequency must at least be one the graph knows.
        if graph_frequencies and float(nominal_hz) not in graph_frequencies:
            raise ValueError(
                f"nominal frequency {nominal_hz} Hz disagrees with the topology graph of "
                f"{task.grid!r} (frnom values: {sorted(graph_frequencies)})"
            )
        n_samples = int(task.train.feature_shape[1])
        sampling_rate = self.sampling_rate_hz
        if sampling_rate is None:
            from ..data.windows import DEFAULT_CONTRACT

            sampling_rate = n_samples / DEFAULT_CONTRACT.window_length
        self.n_cycle_ = samples_per_cycle(sampling_rate, nominal_hz)
        if self.n_cycle_ > n_samples:
            raise ValueError(f"a {n_samples}-sample window holds no full cycle ({self.n_cycle_})")

        self.one_ended_lines_, self.mirrored_lines_ = lines_measured_at_far_end(task.train)
        train_lines = task.train.metadata["y_fault_line"].astype(str)
        self._require_parameters(train_lines.unique(), unusable)

        fit_record: dict[str, Any]
        if isinstance(self.impedance_scale, dict):
            self.scales_, self.default_scale_ = dict(self.impedance_scale), None
            fit_record = {"source": "given per line"}
        elif isinstance(self.impedance_scale, (int, float)):
            self.scales_, self.default_scale_ = {}, float(self.impedance_scale)
            fit_record = {"source": "given"}
        else:
            fit_record = self._fit_scale(task.train)
        voltage_level = self._voltage_level(task)
        if self.impedance_scale == 1.0 and voltage_level["measured_to_nominal"] is not None:
            if not 0.5 <= voltage_level["measured_to_nominal"] <= 1.5:
                warnings.warn(
                    f"impedance_scale=1.0 assumes primary quantities, but the phase voltages of "
                    f"{task.grid!r} are {voltage_level['measured_to_nominal']:.3g} times the "
                    "nominal phase voltage; pass the VT/CT ratio or impedance_scale='fit'",
                    UserWarning,
                    stacklevel=2,
                )

        self.calibration_ = {
            "method": "two_terminal_positive_sequence"
            if self.two_terminal_
            else "one_terminal_reactance",
            "phasors": "full-cycle DFT on the last complete cycle of the window",
            "nominal_hz": float(nominal_hz),
            "graph_rated_frequencies_hz": sorted(graph_frequencies),
            "sampling_rate_hz": float(sampling_rate),
            "samples_per_cycle": int(self.n_cycle_),
            "line_parameters": {
                "source": source,
                "assumed_units": "ohm/km and km",
                "n_lines": len(self.lines_),
                "unusable": unusable,
            },
            "impedance_scale": {
                "meaning": "primary ohm per measured V/A (VT ratio / CT ratio)",
                "per": "line" if self.scales_ else "grid",
                "grid": self.default_scale_,
                "by_line": dict(sorted(self.scales_.items())),
                "fit": fit_record,
            },
            "voltage_level": voltage_level,
            "far_current_sign": int(self.far_current_sign),
            "phase_selector": {
                "ground_ratio": self.ground_ratio,
                "phase_ratio": self.phase_ratio,
            },
            "one_ended_lines": list(self.one_ended_lines_),
            "mirrored_lines": list(self.mirrored_lines_),
        }

    def _require_parameters(self, lines, unusable: dict[str, str] | None = None) -> None:
        assert self.lines_ is not None
        missing = sorted(str(ln) for ln in lines if str(ln) not in self.lines_)
        if missing:
            reasons = {ln: (unusable or {}).get(ln, "not in the topology graph") for ln in missing}
            raise LineParameterError(
                f"no usable line parameters for {reasons}; the impedance locator does not guess "
                "them (pass line_parameters= to supply them)"
            )

    def _voltage_level(self, task: Task) -> dict[str, float | None]:
        """Label-free check of the "quantities are primary" assumption: the median, over the
        training windows, of the largest near-terminal phase-voltage phasor (a healthy phase stays
        near nominal during a fault), against the grid's nominal phase voltage."""
        largest = [
            np.abs(self._window_phasors(x)[:, 3:6]).max(axis=1)
            for x, _ in task.train.iter_batches(_BATCH)
        ]
        measured = float(np.median(np.concatenate(largest))) if largest else float("nan")
        try:
            from ..data.roots import get_grid

            nominal: float | None = get_grid(task.grid).voltage_kv * 1e3 / np.sqrt(3.0)
        except KeyError:  # a grid outside the registry: nothing to compare with
            nominal = None
        ratio = measured / nominal if nominal and np.isfinite(measured) else None
        return {
            "median_largest_phase_voltage": measured if np.isfinite(measured) else None,
            "nominal_phase_voltage": nominal,
            "measured_to_nominal": ratio,
        }

    def _fit_scale(self, train: TaskView) -> dict[str, Any]:
        """Fit the impedance scale on the training view from the near-terminal one-terminal
        estimate with scale 1 (both views carry the near terminal in channels 0..5)."""
        frame = self._locate(train, scale_override=1.0, one_terminal_only=True)
        target = np.asarray(train.labels, dtype=float)
        mirrored = frame["mirrored"].to_numpy()
        # distance from the measuring terminal, on both sides of the regression
        target = np.where(mirrored, 100.0 - target, target)
        uncalibrated = frame["terminal_distance"].to_numpy()
        grid_fit = fit_impedance_scale(uncalibrated, target, keep=self.scale_fit_keep)
        by_line: dict[str, dict[str, float]] = {}
        lines = frame["line"].to_numpy()
        for line in sorted(set(lines)):
            rows = lines == line
            try:
                by_line[line] = fit_impedance_scale(
                    uncalibrated[rows], target[rows], keep=self.scale_fit_keep
                )
            except ValueError:
                continue  # too few windows on this line: it keeps the grid value
        self.default_scale_ = grid_fit["scale"]
        self.scales_ = (
            {ln: f["scale"] for ln, f in by_line.items()} if self.scale_per == "line" else {}
        )
        return {
            "source": "least trimmed squares on train (one-terminal reactance estimate)",
            "keep_fraction": self.scale_fit_keep,
            "grid": grid_fit,
            # reported for every line also when one grid value is used, to show how stable it is
            "by_line": by_line,
        }

    def _window_phasors(self, x: np.ndarray) -> np.ndarray:
        """Fundamental phasors of the last complete cycle of each window, ``(b, channels)``.

        The last cycle is the one most likely to lie entirely in the fault state, because the
        valid samples are fault-active windows and most of them contain the inception. Choosing
        instead the cycle with the largest phase current (a disturbance-detector style selection)
        was tried on ``train`` and did not reduce the error, so the simpler rule is kept."""
        assert self.n_cycle_ is not None
        return fundamental_phasors(x, self.n_cycle_)

    # --- predict ----------------------------------------------------------------------------
    def predict(self, view: TaskView) -> pd.DataFrame:
        frame = self.locate(view)
        return predictions_frame(
            view.metadata["sample_id"], view.metadata["window_idx"], frame["location"].to_numpy()
        )

    def locate(self, view: TaskView) -> pd.DataFrame:
        """Per-window estimate with its diagnostics, aligned with ``view.metadata``:
        ``location`` (percent from the lower-numbered bus, clipped to [0, 100]; what
        :meth:`predict` returns), ``raw`` (the same before clipping; NaN if undefined),
        ``terminal_distance`` (percent from the measuring terminal), ``loop`` (selected fault
        loop at the near terminal), ``method``, ``mirrored``, ``line``."""
        if self.calibration_ is None:
            raise RuntimeError("call fit() before predict()")
        return self._locate(view)

    def _locate(
        self,
        view: TaskView,
        *,
        scale_override: float | None = None,
        one_terminal_only: bool = False,
    ) -> pd.DataFrame:
        assert self.lines_ is not None and self.n_cycle_ is not None
        lines = view.metadata["y_fault_line"].astype(str).to_numpy()
        self._require_parameters(pd.unique(lines))
        # per-window line constants
        params = {ln: self.lines_[ln] for ln in pd.unique(lines)}
        z1_total = np.array([params[ln].z1_total for ln in lines], dtype=complex)
        k0 = np.array([params[ln].k0 for ln in lines], dtype=complex)
        if scale_override is not None:
            scale: np.ndarray = np.full(len(lines), float(scale_override))
        else:
            default = np.nan if self.default_scale_ is None else self.default_scale_
            by_line = {ln: self.scales_.get(ln, default) for ln in params}
            scale = np.array([by_line[ln] for ln in lines], dtype=float)
            if not np.isfinite(scale).all():
                raise LineParameterError("no impedance scale for some lines of this view")
        # Which terminals are measured is a property of the view being located, not of `train`:
        # derive it here (label-free, from the feature names) so a split whose export differs
        # from the training one is neither paired with a repeated terminal nor mis-mirrored.
        one_ended_lines, mirrored_lines = lines_measured_at_far_end(view)
        one_ended = np.isin(lines, one_ended_lines)
        mirrored = np.isin(lines, mirrored_lines)
        two_terminal = self.two_terminal_ and not one_terminal_only
        n_channels = 12 if self.two_terminal_ else 6
        if view.feature_shape[2] != n_channels:
            raise ValueError(
                f"the {'line' if self.two_terminal_ else 'local'} view should carry "
                f"{n_channels} channels per window, this one has {view.feature_shape[2]}"
            )

        distance = np.empty(len(lines), dtype=float)
        loop = np.empty(len(lines), dtype=np.int8)
        start = 0
        for x, _ in view.iter_batches(_BATCH):
            rows = slice(start, start + len(x))
            start += len(x)
            phasors = self._window_phasors(x)  # (b, 6 or 12)
            i_near, v_near = phasors[:, 0:3], phasors[:, 3:6]
            near, loop[rows] = one_terminal_distance(
                v_near,
                i_near,
                z1_total[rows],
                k0[rows],
                impedance_scale=scale[rows],
                ground_ratio=self.ground_ratio,
                phase_ratio=self.phase_ratio,
            )
            if two_terminal:
                both = two_terminal_distance(
                    v_near,
                    i_near,
                    phasors[:, 9:12],
                    phasors[:, 6:9],
                    z1_total[rows],
                    impedance_scale=scale[rows],
                    far_current_sign=self.far_current_sign,
                )
                # a one-ended line repeats the near terminal in the far slot: undefined
                distance[rows] = np.where(one_ended[rows], near, both)
            else:
                distance[rows] = near
        if start != len(lines):
            raise RuntimeError(f"view yielded {start} windows, metadata has {len(lines)}")

        used_one_terminal = one_ended | (not two_terminal)
        # a one-terminal estimate from the non-reference end measures from the other side
        raw = np.where(mirrored & used_one_terminal, 100.0 - distance, distance)
        raw = np.where(np.isfinite(raw), raw, np.nan)
        location = np.clip(np.where(np.isnan(raw), _MIDLINE, raw), 0.0, 100.0)
        if not two_terminal:
            method: np.ndarray = np.full(len(lines), "one_terminal", dtype=object)
        else:
            method = np.where(one_ended, "one_terminal_fallback", "two_terminal")
        return pd.DataFrame(
            {
                "sample_id": view.metadata["sample_id"].to_numpy(),
                "window_idx": view.metadata["window_idx"].to_numpy(),
                "line": lines,
                "location": location,
                "raw": raw,
                "terminal_distance": distance,
                "loop": np.asarray(LOOPS, dtype=object)[loop],
                "method": method,
                "mirrored": mirrored & used_one_terminal,
            }
        )
