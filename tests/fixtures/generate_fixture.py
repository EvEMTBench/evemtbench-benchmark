"""Generate the tiny synthetic fixture committed under ``tests/fixtures/synthetic/``.

Fabricated (NOT real) DoubleLine-shaped windowed data — `adapt_grid` + `benchmark` families, 66
on-disk channels — that reproduces the v0.1 schema so the data loader, line-view projection,
the FD/FC/FL label derivers, the held_out split generator, and the baselines/runner can run in
CI without the multi-GB real dataset.

Quirks encoded: 66 channels (11 cubicles; the loader projects to the 48 main-line channels),
episode-constant ``y_fault_line`` / ``y_fault_location`` (NaN on bus faults + non-fault
episodes), pre-fault (clean) windows inside fault episodes, the full flt_* taxonomy, and a
couple of non-fault (switching / no_event) episodes the fault tasks must exclude.

Run ``python tests/fixtures/generate_fixture.py`` to (re)write the fixture deterministically.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evemtbench.contract import EXPECTED_CONTRACT_VERSION as _CONTRACT_VERSION

HERE = Path(__file__).resolve().parent
ROOT = HERE / "synthetic"

TOPOLOGY = "hv_double_line_110kv"
RAW_FOLDER = "DoubleLine"
STEM = f"{TOPOLOGY}_W0p050_S0p005"
N_CUBICLES = 11
N_CHANNELS = N_CUBICLES * 6  # 66
WINDOW_LENGTH = 0.05
STEP = 0.005
WINDOWS_PER_SIM = 4  # first 2 pre-fault (clean), last 2 active
WINDOW_SAMPLES = 8  # tiny "l"

_FAULTS = (
    "flt_1phg_shc",
    "flt_1phg_shc_w_arc",
    "flt_2ph_shc",
    "flt_2phg_shc",
    "flt_3ph_shc",
    "flt_1phg_incipient",
    "flt_1phg_incipient_w_arc",
)
_LINES = ("MainLn1-2A", "MainLn1-2B", "MainLn2-3A", "MainLn2-3B")
_BUSES = ("MainBus1", "MainBus2", "MainBus3")


def _rng(tag: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16))


def _attr_cols(et: str, fam: str) -> dict:
    """Episode-constant fault/switching descriptor columns, derived deterministically from the
    event type (mirrors the real windows schema so the attribute/event tasks can load the fixture).
    Non-applicable descriptors take the neutral value the real int8/nullable columns use."""
    is_fault = et.startswith("flt_")
    # fault descriptors
    if is_fault:
        category = "hif" if "hif" in et else ("incipient" if "incipient" in et else "shc")
        grounded = 0 if et in ("flt_2ph_shc", "flt_3ph_shc") else 1  # ungrounded phase faults
        if "3ph" in et:
            pa, pb, pc = 1, 1, 1
        elif "2ph" in et:  # 2ph / 2phg -> A,B
            pa, pb, pc = 1, 1, 0
        else:  # 1phg / incipient / hif -> single phase A
            pa, pb, pc = 1, 0, 0
    else:
        category, grounded, pa, pb, pc = None, 0, 0, 0, 0
    # switching descriptors — map event_type -> the frozen SWITCH_TYPE_CLASSES vocabulary explicitly
    # (a plain split() would give "cap"/"motor"/"gen", not "capacitor"/"motor_start"/"generator").
    if fam == "switching":
        _CAT = (
            ("load", "load"),
            ("cap", "capacitor"),
            ("ohl", "ohl"),
            ("inrush", "inrush"),
            ("ibr", "ibr"),
            ("cable", "cable"),
            ("motor", "motor_start"),
            ("gen", "generator"),
        )
        sw_cat = next((cat for key, cat in _CAT if key in et), None)
        sw_act = (
            "on"
            if et.endswith("_on")
            else "off"
            if et.endswith("_off")
            else "start"
            if et.endswith("start")
            else "trip"
            if et.endswith("trip")
            else None
        )
        sw_bus = "MainBus2"
    else:
        sw_cat, sw_act, sw_bus = None, None, None
    return {
        "y_fault_category": category,
        "y_is_grounded": grounded,
        "y_phase_A": pa,
        "y_phase_B": pb,
        "y_phase_C": pc,
        "y_switch_category": sw_cat,
        "y_switch_action": sw_act,
        "y_switch_target_bus": sw_bus,
    }


def _episodes(family: str, n_fault: int, n_bus: int, n_nonfault: int):
    rng = _rng(family + "ep")
    specs, idx = [], 0
    for i in range(n_fault):  # line faults — cover all classes + lines
        loc = float(round(rng.uniform(5, 95), 3))
        specs.append((idx, _FAULTS[i % len(_FAULTS)], "fault", _LINES[i % len(_LINES)], loc))
        idx += 1
    for i in range(n_bus):  # bus faults — location undefined (NaN)
        specs.append(
            (idx, _FAULTS[i % len(_FAULTS)], "fault", _BUSES[i % len(_BUSES)], float("nan"))
        )
        idx += 1
    for i in range(n_nonfault):  # non-fault episodes (excluded by the fault-task masks)
        et, fam = ("switch_load_on", "switching") if i % 2 == 0 else ("no_event", "normal")
        specs.append((idx, et, fam, None, float("nan")))
        idx += 1
    return specs


def _write_family(family: str, specs) -> None:
    win_dir = ROOT / family / "windows" / RAW_FOLDER
    win_dir.mkdir(parents=True, exist_ok=True)
    rng = _rng(family + "x")

    y_rows, x_blocks = [], []
    for sim_idx, et, fam, line, loc in specs:
        is_fault = et.startswith("flt_")
        active_start = WINDOWS_PER_SIM // 2
        for w in range(WINDOWS_PER_SIM):
            active = w >= active_start
            present = int(active and et != "no_event")
            # Use the contract's WindowStatus vocabulary (no "event"/"active" value exists).
            status = "clean" if not present else ("in_fault" if is_fault else "switch_event")
            y_rows.append(
                {
                    "sample_id": int(sim_idx),
                    "window_idx": w,
                    "event_type": et,
                    "event_family": fam,
                    "status": status,
                    "overlap_ratio": 1.0 if present else 0.0,
                    "y_event_present": present,
                    "y_fault_present": int(active and is_fault),
                    "y_switch_present": int(present and fam == "switching"),
                    "y_fault_line": line,  # episode-constant
                    "y_fault_location": loc,  # episode-constant; NaN on bus / non-fault
                    # episode-constant fault/switching descriptors (mirror the real schema)
                    **_attr_cols(et, fam),
                    # window time relative to event inception (mirrors the real schema): the first
                    # active window sits at dt_start=0; earlier windows are pre-event (negative).
                    "dt_start": float((w - active_start) * WINDOW_LENGTH),
                    "window_start_time": float(1.1 + w * WINDOW_LENGTH),
                    "window_duration": float(WINDOW_LENGTH),
                }
            )
            x_blocks.append(rng.standard_normal((WINDOW_SAMPLES, N_CHANNELS)).astype(np.float32))

    X = np.stack(x_blocks).astype(np.float32)
    n = X.shape[0]
    (win_dir / f"X_{STEM}.raw").write_bytes(X.tobytes())

    # Realistic DoubleLine cubicle names (on-disk order) so the line-view oracle, which derives
    # the main-line universe + differential terminals from feature_names, parses the fixture the
    # same way it parses the real grid. 8 of the 11 are line bays -> the 48-channel universe.
    cubicle_keys = [
        "Bus_1_Line_1-2A",
        "Bus_1_Line_1-2B",
        "Bus_1_Transformer_1-ExtGrid1",
        "Bus_2_Load_2",
        "Bus_2_Line_1-2A",
        "Bus_2_Line_1-2B",
        "Bus_2_Line_2-3A",
        "Bus_2_Line_2-3B",
        "Bus_3_Line_2-3A",
        "Bus_3_Line_2-3B",
        "Bus_3_Transformer_3-ExtGrid3",
    ]
    assert len(cubicle_keys) == N_CUBICLES
    # full real suffix: <Isec|Usec>_<L1|L2|L3>_<A|V> (current in A, voltage in V)
    feature_names = [
        f"{key}_{q}_{ph}_{unit}"
        for key in cubicle_keys
        for q, unit in (("Isec", "A"), ("Usec", "V"))
        for ph in ("L1", "L2", "L3")
    ]
    # Structured channel descriptor (contract C-META-2), current-first per cubicle. Built inline
    # (the benchmark takes no psp_helper dependency); mirrors the real producer's meta.channels.
    channels = []
    ci = 0
    for key in cubicle_keys:
        _, bus, etype, eid = key.split("_", 3)
        for q in ("I", "V"):
            for ph in ("L1", "L2", "L3"):
                channels.append(
                    {
                        "index": ci,
                        "cubicle": key,
                        "bus": bus,
                        "element_type": etype,
                        "element_id": eid,
                        "quantity": q,
                        "phase": ph,
                        "unit": "A" if q == "I" else "V",
                    }
                )
                ci += 1
    (win_dir / f"X_{STEM}.raw.meta.json").write_text(
        json.dumps(
            {
                "shape": list(X.shape),
                "dtype": "float32",
                "feature_names": feature_names,
                "channels": channels,
                "feature_groups": {
                    "lines": list(range(N_CHANNELS)),
                    "currents": [i for i, c in enumerate(channels) if c["quantity"] == "I"],
                    "voltages": [i for i, c in enumerate(channels) if c["quantity"] == "V"],
                },
                "extra": {
                    "topology": TOPOLOGY,
                    "window_length": WINDOW_LENGTH,
                    "step_length_seconds": STEP,
                },
            },
            indent=2,
        )
    )

    fhash = hashlib.sha256()
    for nm in feature_names:
        fhash.update(nm.encode())
        fhash.update(b"\n")
    (win_dir / f"X_{STEM}.raw.manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                # basenames, never absolute paths: the loader locates files by directory (the
                # manifest path fields are producer provenance), and absolute paths would embed
                # the generating machine's home into committed fixtures (leak-scan refusal).
                "x_path": f"X_{STEM}.raw",
                "y_path": f"y_{STEM}.parquet",
                "shape": {"n": n, "l": WINDOW_SAMPLES, "f": N_CHANNELS},
                "feature_hash": fhash.hexdigest(),
                "config_fingerprint": {
                    "topology": TOPOLOGY,
                    "window_length": WINDOW_LENGTH,
                    "step_length_seconds": STEP,
                    # fs chosen so l == round(W*fs): round(0.05 * 160) == 8
                    "sampling_frequency": WINDOW_SAMPLES / WINDOW_LENGTH,
                    "contract_version": _CONTRACT_VERSION,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    pd.DataFrame(y_rows).to_parquet(win_dir / f"y_{STEM}.parquet", index=False)


# --- multi_grid pre-training corpus: portable 12-channel double_line view (Template20kV-shaped) --
MG_TOPOLOGY = "mv_template_20kv"
MG_RAW_FOLDER = "Template20kV"
MG_STEM = f"{MG_TOPOLOGY}_W0p050_S0p005"
MG_CHANNELS = 12  # EndA/EndB x {I,V} x {L1,L2,L3}


def _write_multigrid_family(specs) -> None:
    """Write a tiny ``multi_grid`` double_line partition (fixed 12-ch EndA/EndB view; line/grid
    identity in ``y`` via bus_a/bus_b). Mirrors the real Template20kV schema so the double-line
    projection + the multi_grid_pretrain/transfer protocols can be exercised in CI."""
    win_dir = ROOT / "multi_grid" / "windows" / MG_RAW_FOLDER
    win_dir.mkdir(parents=True, exist_ok=True)
    rng = _rng("multi_gridx")

    y_rows, x_blocks = [], []
    for sim_idx, et, fam, _line, loc in specs:
        is_fault = et.startswith("flt_")
        active_start = WINDOWS_PER_SIM // 2
        # bus pair for this episode's target line (identity lives in y, not on the channel axis)
        bus_a, bus_b = (int(sim_idx) % 5), (int(sim_idx) % 5) + 1
        mg_line = f"MainLn_{bus_a}_{bus_b}_0" if is_fault else None
        for w in range(WINDOWS_PER_SIM):
            active = w >= active_start
            present = int(active and et != "no_event")
            status = "clean" if not present else ("in_fault" if is_fault else "switch_event")
            y_rows.append(
                {
                    "sample_id": int(sim_idx),
                    "window_idx": w,
                    "event_type": et,
                    "event_family": fam,
                    "status": status,
                    "overlap_ratio": 1.0 if present else 0.0,
                    "y_event_present": present,
                    "y_fault_present": int(active and is_fault),
                    "y_switch_present": int(present and fam == "switching"),
                    "y_fault_line": mg_line,
                    "y_fault_location": loc,
                    # simulation/topology identity as in the real corpus: one episode is one line
                    # of one simulation (result_id), each simulation belongs to exactly one
                    # generated topology (graph_id) — the group keys the grouped splits use
                    "result_id": int(sim_idx) // 2,
                    "graph_id": f"g{(int(sim_idx) // 2) % 5}",
                    "line_id": f"L{bus_a}_{bus_b}",
                    "bus_a": bus_a,
                    "bus_b": bus_b,
                    **_attr_cols(et, fam),
                    "dt_start": float((w - active_start) * WINDOW_LENGTH),
                    "window_start_time": float(1.1 + w * WINDOW_LENGTH),
                    "window_duration": float(WINDOW_LENGTH),
                }
            )
            x_blocks.append(rng.standard_normal((WINDOW_SAMPLES, MG_CHANNELS)).astype(np.float32))

    X = np.stack(x_blocks).astype(np.float32)
    n = X.shape[0]
    (win_dir / f"X_{MG_STEM}.raw").write_bytes(X.tobytes())

    feature_names = [
        f"End{end}_{q}_{ph}_{unit}"
        for end in ("A", "B")
        for q, unit in (("Isec", "A"), ("Usec", "V"))
        for ph in ("L1", "L2", "L3")
    ]
    channels = []
    ci = 0
    for end in ("A", "B"):
        for q in ("I", "V"):
            for ph in ("L1", "L2", "L3"):
                channels.append(
                    {
                        "index": ci,
                        "cubicle": f"End{end}",
                        "bus": None,
                        "element_type": None,
                        "element_id": None,
                        "quantity": q,
                        "phase": ph,
                        "unit": "A" if q == "I" else "V",
                    }
                )
                ci += 1
    (win_dir / f"X_{MG_STEM}.raw.meta.json").write_text(
        json.dumps(
            {
                "shape": list(X.shape),
                "dtype": "float32",
                "feature_names": feature_names,
                "channels": channels,
                "feature_groups": {
                    "currents": [i for i, c in enumerate(channels) if c["quantity"] == "I"],
                    "voltages": [i for i, c in enumerate(channels) if c["quantity"] == "V"],
                    "end_a": list(range(6)),
                    "end_b": list(range(6, 12)),
                },
                "extra": {
                    "topology": MG_TOPOLOGY,
                    "view": "double_line",
                    "channels_per_line": 12,
                    "window_length": WINDOW_LENGTH,
                    "step_length_seconds": STEP,
                },
            },
            indent=2,
        )
    )
    fhash = hashlib.sha256()
    for nm in feature_names:
        fhash.update(nm.encode())
        fhash.update(b"\n")
    (win_dir / f"X_{MG_STEM}.raw.manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                # basenames, never absolute paths — see the adapt_grid manifest writer above
                "x_path": f"X_{MG_STEM}.raw",
                "y_path": f"y_{MG_STEM}.parquet",
                "shape": {"n": n, "l": WINDOW_SAMPLES, "f": MG_CHANNELS},
                "feature_hash": fhash.hexdigest(),
                "config_fingerprint": {
                    "topology": MG_TOPOLOGY,
                    "window_length": WINDOW_LENGTH,
                    "step_length_seconds": STEP,
                    "sampling_frequency": WINDOW_SAMPLES / WINDOW_LENGTH,
                    "contract_version": _CONTRACT_VERSION,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    pd.DataFrame(y_rows).to_parquet(win_dir / f"y_{MG_STEM}.parquet", index=False)


def build() -> None:
    if ROOT.exists():
        import shutil

        shutil.rmtree(ROOT)
    _write_family("adapt_grid", _episodes("adapt_grid", n_fault=16, n_bus=3, n_nonfault=2))
    _write_family("benchmark", _episodes("benchmark", n_fault=8, n_bus=2, n_nonfault=0))
    _write_multigrid_family(_episodes("multi_grid", n_fault=20, n_bus=4, n_nonfault=2))
    print(f"fixture written to {ROOT}")


if __name__ == "__main__":
    build()
