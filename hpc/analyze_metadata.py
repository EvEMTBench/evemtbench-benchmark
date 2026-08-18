"""Exploratory: multi_grid topology/simulation structure + class support per corpus.

Feeds the grouped-split design (simulation- vs topology-grouped 70/15/15):
- confirms graph_id/result_id are present, consistent, and complete per episode;
- confirms sample_id == result_id*10^4 + line_idx;
- per-topology episode counts + event_type histograms;
- rare-class concentration (how many topologies/simulations carry each class);
- worst-case feasibility for topology-grouped stratification.

Writes JSON to <outdir>/metadata_<grid>.json + a text summary to stdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from evemtbench.data.roots import get_grid
from evemtbench.data.windows import DEFAULT_CONTRACT, find_window_files, windows_dir

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("local")
GRIDS = ("template_20kv", "template_110kv", "template_345kv")
COLS = ["sample_id", "result_id", "graph_id", "event_type", "event_family"]

for grid in GRIDS:
    g = get_grid(grid)
    files = find_window_files(
        windows_dir(g, "multi_grid"), topology=g.topology, contract=DEFAULT_CONTRACT
    )
    y = pd.read_parquet(files["y_path"], columns=COLS)
    ep = y.groupby("sample_id").first().reset_index()  # episode-constant rows

    n_ep = len(ep)
    assert ep["result_id"].notna().all() and ep["graph_id"].notna().all(), "missing ids"
    # sample_id encodes (sim, line): verify the documented arithmetic
    bad = int((ep["sample_id"] // 10_000 != ep["result_id"]).sum())
    # one simulation -> exactly one topology?
    sims_multi_topo = int((ep.groupby("result_id")["graph_id"].nunique() > 1).sum())
    # windows within an episode: constant event_type?
    wct = y.groupby("sample_id")["event_type"].nunique()
    ep_var_event = int((wct > 1).sum())

    topo_sizes = ep.groupby("graph_id")["sample_id"].size()
    sim_sizes = ep.groupby("result_id")["sample_id"].size()
    sims_per_topo = ep.groupby("graph_id")["result_id"].nunique()

    cls_total = ep["event_type"].value_counts().to_dict()
    cls_topos = ep.groupby("event_type")["graph_id"].nunique().to_dict()
    cls_sims = ep.groupby("event_type")["result_id"].nunique().to_dict()
    fam_total = ep["event_family"].value_counts().to_dict()

    # per-topology class histogram (for the greedy generator design)
    topo_cls = {t: d["event_type"].value_counts().to_dict() for t, d in ep.groupby("graph_id")}

    report = {
        "grid": grid,
        "episodes": n_ep,
        "simulations": int(ep["result_id"].nunique()),
        "topologies": int(ep["graph_id"].nunique()),
        "sample_id_arithmetic_violations": bad,
        "sims_spanning_multiple_topologies": sims_multi_topo,
        "episodes_with_varying_event_type_across_windows": ep_var_event,
        "episodes_per_topology": {
            "min": int(topo_sizes.min()),
            "median": float(topo_sizes.median()),
            "max": int(topo_sizes.max()),
        },
        "episodes_per_simulation": {
            "min": int(sim_sizes.min()),
            "median": float(sim_sizes.median()),
            "max": int(sim_sizes.max()),
        },
        "sims_per_topology": {
            "min": int(sims_per_topo.min()),
            "median": float(sims_per_topo.median()),
            "max": int(sims_per_topo.max()),
        },
        "class_totals": {k: int(v) for k, v in sorted(cls_total.items())},
        "class_n_topologies": {k: int(v) for k, v in sorted(cls_topos.items())},
        "class_n_simulations": {k: int(v) for k, v in sorted(cls_sims.items())},
        "family_totals": {k: int(v) for k, v in sorted(fam_total.items())},
        "per_topology_class_histograms": topo_cls,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"metadata_{grid}.json").write_text(json.dumps(report, indent=2, sort_keys=True))

    print(
        f"== {grid}: {n_ep} episodes / {report['simulations']} sims / "
        f"{report['topologies']} topologies =="
    )
    print(
        f"  sample_id arithmetic violations: {bad}; sims spanning >1 topo: {sims_multi_topo}; "
        f"episodes w/ varying event_type: {ep_var_event}"
    )
    print(
        f"  episodes/topo {report['episodes_per_topology']},"
        f" sims/topo {report['sims_per_topology']}"
    )
    print("  class -> (episodes, topologies, sims):")
    for k in sorted(cls_total):
        print(f"    {k:24s} {cls_total[k]:7d} {cls_topos[k]:5d} {cls_sims[k]:6d}")
    print(flush=True)
print("DONE")
