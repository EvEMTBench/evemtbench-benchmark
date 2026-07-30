"""Render the audit site's committed visual assets — audit/assets/*.svg + manifest.json.

The UnderstandingVCS ``build_assets.py`` analogue: run MANUALLY when the data or task suite
changes (it needs the data/aux roots, unlike the page builder), commit the rendered SVGs, and let
``hpc/make_audit.py`` embed whatever ``audit/assets/manifest.json`` lists — pages stay pure
functions of committed files, so the freshness gate is untouched. Zero plotting dependencies:
waveforms and topology diagrams are hand-rolled SVG (the burnup-chart precedent); topology layout
uses networkx (already installed / a ``[graphs]`` extra).

Every figure records its provenance in the manifest (grid, family, sample_id, window index,
event type, cubicle, feature hash) — the audit's own figures are traceable like everything else.

    PYTHONPATH=src python hpc/make_audit_assets.py            # -> audit/assets/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO / "src", _REPO / "hpc"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import os  # noqa: E402

# The asset builder needs the real data root from the environment (no hardcoded machine paths).
if not (os.environ.get("EVEMTBENCH_DATA_ROOT") or os.environ.get("EVEMT_ROOT")):
    raise SystemExit("make_audit_assets: set EVEMT_ROOT (or EVEMTBENCH_DATA_ROOT) to the data root")

from evemtbench.data.roots import GRIDS  # noqa: E402
from evemtbench.data.schema import is_fault  # noqa: E402
from evemtbench.data.windows import CHANNELS_PER_CUBICLE, load_windows  # noqa: E402
from evemtbench.splits.loader import read_held_out_splits  # noqa: E402
from evemtbench.tasks.registry import get_task, list_tasks  # noqa: E402

ASSETS = _REPO / "audit" / "assets"
WAVE_GRID = "double_line"  # the illustrative grid for per-function example windows
PHASE_COLORS = ("#d62728", "#2ca02c", "#1f77b4")  # phases A/B/C

# Per objective function: which event-class family illustrates it, and the k-th distinct class of
# that family to use (variety across figures while staying deterministic).
_FAULT, _SWITCH = "fault", "switch"
FUNCTION_EXAMPLES: dict[str, tuple[str, int]] = {
    "fault_detection": (_FAULT, 0),
    "fault_class": (_FAULT, 1),
    "fault_location": (_FAULT, 2),
    "fault_grounded": (_FAULT, 3),
    "fault_phase": (_FAULT, 4),
    "fault_category": (_FAULT, 5),
    "fault_origin": (_FAULT, 6),
    "event_detection": (_FAULT, 0),
    "event_state": (_SWITCH, 0),
    "event_classification": (_SWITCH, 2),
    "switch_detection": (_SWITCH, 1),
    "switch_type": (_SWITCH, 3),
}


# ---------------------------------------------------------------------------------------------
# Waveform SVG (hand-rolled; two panels: 3-phase currents, 3-phase voltages)
# ---------------------------------------------------------------------------------------------
def _panel(sig: np.ndarray, names: list[str], y0: float, h: float, w: float, ml: float) -> str:
    """One 3-trace panel; ``sig`` is (l, 3) float32. Normalized per panel, peak annotated."""
    n = sig.shape[0]
    peak = float(np.abs(sig).max()) or 1.0
    out = [
        f'<line x1="{ml}" y1="{y0 + h / 2:.1f}" x2="{w:.1f}" y2="{y0 + h / 2:.1f}" '
        'stroke="#8884" stroke-dasharray="3 3"/>'
    ]
    for ch in range(3):
        pts = " ".join(
            f"{ml + (w - ml) * i / (n - 1):.1f},{y0 + h / 2 - (h / 2 - 4) * sig[i, ch] / peak:.1f}"
            for i in range(n)
        )
        out.append(
            f'<polyline points="{pts}" fill="none" stroke="{PHASE_COLORS[ch]}" '
            'stroke-width="1.4" opacity="0.9"/>'
        )
    legend = " · ".join(
        f'<tspan fill="{PHASE_COLORS[i]}">{nm}</tspan>' for i, nm in enumerate(names)
    )
    out.append(f'<text x="{ml}" y="{y0 + 13:.1f}" font-size="11" fill="#888">{legend}</text>')
    out.append(
        f'<text x="{w:.1f}" y="{y0 + 13:.1f}" font-size="11" fill="#888" text-anchor="end">'
        f"peak |x| = {peak:.4g}</text>"
    )
    return "".join(out)


def waveform_svg(window: np.ndarray, ch_names: list[str]) -> str:
    """(480, 6) local-view window -> two-panel SVG (I over V), 0-50 ms axis."""
    w, h, ml = 860.0, 340.0, 10.0
    panel_h = (h - 40) / 2
    cur = _panel(window[:, 0:3], ch_names[0:3], 8, panel_h, w, ml)
    vol = _panel(window[:, 3:6], ch_names[3:6], 16 + panel_h, panel_h, w, ml)
    ticks = "".join(
        f'<text x="{ml + (w - ml) * t / 50:.1f}" y="{h - 8}" font-size="10" fill="#888" '
        f'text-anchor="middle">{t} ms</text>'
        for t in (0, 10, 20, 30, 40, 50)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:g} {h:g}" role="img" '
        'style="max-width:100%;height:auto" aria-label="example waveform window">'
        f"{cur}{vol}{ticks}</svg>"
    )


def render_waveforms() -> dict[str, dict]:
    """One example-window SVG per objective function (from WAVE_GRID's committed test split)."""
    test_sids = {sid for _f, _g, sid in read_held_out_splits(WAVE_GRID)["test"]}
    ws = load_windows(WAVE_GRID, "adapt_grid")
    in_test = np.isin(ws.sample_ids, sorted(test_sids))
    etypes = ws.y["event_type"].astype(str).to_numpy()

    def classes(family: str) -> list[str]:
        pred = is_fault if family == _FAULT else lambda e: str(e).startswith("switch")
        return sorted({e for e in etypes[in_test] if pred(e)})

    fam_classes = {_FAULT: classes(_FAULT), _SWITCH: classes(_SWITCH)}
    entries: dict[str, dict] = {}
    for fn, (family, kth) in sorted(FUNCTION_EXAMPLES.items()):
        pool = fam_classes[family]
        if not pool:
            continue
        cls = pool[kth % len(pool)]
        idxs = np.flatnonzero(in_test & (etypes == cls))
        if idxs.size == 0:
            continue
        i = int(idxs[0])
        window = np.asarray(ws.X[i], dtype=np.float32)  # one-window memmap read
        # the cubicle that "sees" the event: max RMS over its current channels
        n_cub = window.shape[1] // CHANNELS_PER_CUBICLE
        rms = [float(np.sqrt((window[:, c * 6 : c * 6 + 3] ** 2).mean())) for c in range(n_cub)]
        cub = int(np.argmax(rms))
        local = window[:, cub * 6 : (cub + 1) * 6]
        names = [n.split("/")[-1] for n in ws.feature_names[cub * 6 : (cub + 1) * 6]]
        (ASSETS / f"wave_{fn}.svg").write_text(waveform_svg(local, names))
        entries[fn] = {
            "figure": f"wave_{fn}.svg",
            "grid": WAVE_GRID,
            "family": "adapt_grid",
            "split": "test",
            "event_type": cls,
            "sample_id": str(ws.y["sample_id"].iloc[i]),
            "window_index": i,
            "cubicle": cub,
            "feature_hash": (ws.provenance or {}).get("feature_hash"),
        }
        print(f"  wave_{fn}.svg  <- {cls} (sample {entries[fn]['sample_id']}, window {i})")
    return entries


# ---------------------------------------------------------------------------------------------
# Topology SVG (networkx layout over the aux graph; honest skip when not extracted)
# ---------------------------------------------------------------------------------------------
def topology_svg(g) -> str:
    import networkx as nx

    pos = nx.kamada_kawai_layout(nx.Graph(g))  # deterministic; collapse multi-edges for layout
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span_x = (max(xs) - min(xs)) or 1.0
    span_y = (max(ys) - min(ys)) or 1.0
    w, h, m = 860.0, 480.0, 60.0

    def xy(n) -> tuple[float, float]:
        return (
            m + (w - 2 * m) * (pos[n][0] - min(xs)) / span_x,
            m + (h - 2 * m) * (pos[n][1] - min(ys)) / span_y,
        )

    parts = []
    seen: dict[tuple, int] = {}
    for u, v, key in sorted(g.edges(keys=True), key=str):
        (x1, y1), (x2, y2) = xy(u), xy(v)
        k = seen.get((u, v), 0)
        seen[(u, v)] = k + 1  # offset parallel edges of the MultiGraph
        off = (k - (g.number_of_edges(u, v) - 1) / 2) * 7
        nxv, nyv = -(y2 - y1), (x2 - x1)
        ln = (nxv**2 + nyv**2) ** 0.5 or 1.0
        ox, oy = off * nxv / ln, off * nyv / ln
        parts.append(
            f'<line x1="{x1 + ox:.1f}" y1="{y1 + oy:.1f}" x2="{x2 + ox:.1f}" '
            f'y2="{y2 + oy:.1f}" stroke="#1890ff" stroke-width="1.6" opacity="0.75"/>'
        )
        parts.append(
            f'<text x="{(x1 + x2) / 2 + ox:.1f}" y="{(y1 + y2) / 2 + oy - 3:.1f}" '
            f'font-size="9" fill="#888" text-anchor="middle">{key}</text>'
        )
    for n in sorted(g.nodes, key=str):
        x, y = xy(n)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#52c41a" stroke="#389e0d"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{y - 10:.1f}" font-size="10" fill="#555" '
            f'text-anchor="middle">{n}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:g} {h:g}" role="img" '
        'style="max-width:100%;height:auto" aria-label="grid topology">' + "".join(parts) + "</svg>"
    )


def render_topologies() -> dict[str, dict]:
    from evemtbench.data.graphs import load_grid_graph

    entries: dict[str, dict] = {}
    for grid in sorted(GRIDS):
        try:
            g = load_grid_graph(grid, family="benchmark")
        except Exception as e:  # aux graph not extracted (yet) -> honest skip
            print(f"  topo_{grid}.svg  SKIPPED ({type(e).__name__}: aux graph not extracted)")
            continue
        (ASSETS / f"topo_{grid}.svg").write_text(topology_svg(g))
        entries[grid] = {
            "figure": f"topo_{grid}.svg",
            "source": f"aux overlay graph.pickle (benchmark/{GRIDS[grid].raw_foldername})",
            "n_nodes": g.number_of_nodes(),
            "n_edges": g.number_of_edges(),
        }
        print(f"  topo_{grid}.svg  ({g.number_of_nodes()} nodes / {g.number_of_edges()} edges)")
    return entries


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    ASSETS.mkdir(parents=True, exist_ok=True)
    # sanity: only render functions the registry actually has
    known = {get_task(t).function.value for t in list_tasks()}
    unknown = set(FUNCTION_EXAMPLES) - known
    if unknown:
        raise SystemExit(f"FUNCTION_EXAMPLES has unknown functions: {unknown}")

    print("waveforms:")
    waves = render_waveforms()
    print("topologies:")
    topos = render_topologies()
    manifest = {
        "generated_by": "hpc/make_audit_assets.py (run manually when data/tasks change)",
        "note": (
            "Committed figure provenance. Pages embed only figures listed here; the freshness "
            "gate ignores audit/assets/ (figures are inputs, not derived views)."
        ),
        "waveforms": waves,
        "topologies": topos,
    }
    (ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {ASSETS / 'manifest.json'} ({len(waves)} waveforms, {len(topos)} topologies)")


if __name__ == "__main__":
    main()
