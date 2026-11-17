"""Build label patch LP-1 (faulted phase of ``flt_1phg_hif_w_arc``) from the 1.x source labels.

Deterministic: reads the 1.x windows y (through ``load_windows``, so the manifest is the one the
loader sees) and the producer's per-episode source labels, and writes

* ``src/evemtbench/data/label_patches/fault_phase_hif_w_arc_v1.csv`` — one row per episode,
  sorted by (grid, family, sample_id), LF line endings, no timestamps;
* ``src/evemtbench/data/label_patches/index.json`` — patch id, CSV sha256, per partition the
  ``manifest_sha256`` it applies to, the episode count and the number of patched windows;
* an audit table (Markdown) — per partition the defect, the join, the producer-rule control on
  the other single-phase-to-ground types, the patched windows per split x phase, the
  fault_phase-valid counts before/after, and the zero-remaining check after applying the patch
  through the loader's own :func:`evemtbench.data.label_patches.apply_label_patches`.

Running it twice gives byte-identical files (``--check`` rebuilds into a temporary directory and
compares with the committed ones). It reads the data root and the aux overlay and never writes into
either. Run it as a SLURM job (``hpc/build_label_patch.sbatch``), not on a login node.

The phase rule is the producer's rule for every other ``flt_1phg*`` type:
``events/event_phase_select_1ph`` a/b/c -> (A, B, C) = 100 / 010 / 001.

Sources:

* the reference grid (cigre_mv ``adapt_grid`` / ``benchmark``): ``settings_clean.csv`` of the aux
  overlay (``evemtbench.data.extract_aux``), joined on ``general/sim_idx == sample_id``;
* the ``multi_grid`` Template20kV corpus: ``Template20kV/labels/settings_clean.csv`` from the 1.x
  archive ``multi_grid/multigrid-Template20kV.tar.gz`` (pass the extracted file with
  ``--mg-settings``), joined on ``result_id`` (the ``resultN.csv`` index in
  ``general/result_file_path``); its ``events/event_target`` must equal the window's
  ``y_fault_line``.

    PYTHONPATH=src python hpc/build_label_patch.py --mg-settings <.../settings_clean.csv> \\
        --audit hpc/hpc_logs/revision/LP-1_audit.md
    PYTHONPATH=src python hpc/build_label_patch.py --mg-settings ... --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from evemtbench.data.input_digests import load_table as load_digests  # noqa: E402
from evemtbench.data.label_patches import CSV_COLUMNS, apply_label_patches  # noqa: E402
from evemtbench.data.roots import aux_dir, get_grid  # noqa: E402
from evemtbench.data.windows import load_windows  # noqa: E402
from evemtbench.splits.loader import read_splits  # noqa: E402
from evemtbench.tasks.labels.event_present import derive_fault_active_series  # noqa: E402
from evemtbench.tasks.labels.fault_phase import (  # noqa: E402
    PHASE_CLASSES,
    derive_fault_phase_series,
)
from evemtbench.version import splits_version  # noqa: E402

PATCH_ID = "LP-1"
CSV_NAME = "fault_phase_hif_w_arc_v1.csv"
EVENT_TYPE = "flt_1phg_hif_w_arc"
PHASE_COLS = ["y_phase_A", "y_phase_B", "y_phase_C"]
SELECT_TO_BITS = {"a": (1, 0, 0), "b": (0, 1, 0), "c": (0, 0, 1)}
OUT_DIR = _REPO / "src" / "evemtbench" / "data" / "label_patches"
MG_ARCHIVE_MEMBER = "multigrid-Template20kV.tar.gz:Template20kV/labels/settings_clean.csv"

# (grid, family, split protocol whose committed splits cover the partition, split names)
PARTITIONS = (
    ("cigre_mv", "adapt_grid", "held_out", ("train", "val", "test")),
    ("cigre_mv", "benchmark", "held_out", ("benchmark",)),
    ("template_20kv", "multi_grid", "multi_grid_pretrain", ("train", "val", "test")),
)


class BuildError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bits_str(df: pd.DataFrame) -> pd.Series:
    b = df[PHASE_COLS].fillna(0).astype(np.int64).astype(str)
    return b["y_phase_A"] + b["y_phase_B"] + b["y_phase_C"]


def _expected_1ph(sel: pd.Series) -> pd.Series:
    return (
        sel.str.strip()
        .str.lower()
        .map({k: "".join(map(str, v)) for k, v in SELECT_TO_BITS.items()})
    )


def _read_settings(path: Path) -> pd.DataFrame:
    cols = [
        "general/sim_idx",
        "general/result_file_path",
        "events/event_type",
        "events/event_phase_select_1ph",
        "events/event_target",
    ]
    s = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in cols if c not in s.columns]
    if missing:
        raise BuildError(f"{path}: missing source-label columns {missing}")
    return s[cols]


def _source_join(
    grid: str, family: str, ep: pd.DataFrame, settings: pd.DataFrame
) -> tuple[pd.DataFrame, str]:
    """``ep`` (one row per fault-active episode) joined to its source-label row; the key name."""
    if family == "multi_grid":
        s = settings.assign(
            result_id=settings["general/result_file_path"]
            .str.extract(r"result(\d+)\.csv")[0]
            .astype("Int64")
        )
        if s["result_id"].isna().any() or s["result_id"].duplicated().any():
            raise BuildError(f"{grid}/{family}: result_id is not a unique key of the source labels")
        m = ep.merge(s, on="result_id", how="left", validate="many_to_one", indicator=True)
        key = "result_id"
    else:
        s = settings.assign(sample_id=settings["general/sim_idx"].astype(np.int64))
        if s["sample_id"].duplicated().any():
            raise BuildError(f"{grid}/{family}: general/sim_idx is not a unique key")
        m = ep.merge(s, on="sample_id", how="left", validate="one_to_one", indicator=True)
        key = "general/sim_idx"
    return m, key


def _split_of(grid: str, family: str, protocol: str, names: tuple[str, ...], sid: np.ndarray):
    out = np.full(len(sid), "none", dtype=object)
    splits = read_splits(protocol, grid, version=splits_version(protocol))
    for name in names:
        ids = [s for (fam, _g, s) in splits[name] if fam == family]
        out[np.isin(sid, ids)] = name
    return out


def build_partition(grid: str, family: str, protocol: str, names, settings_path: Path) -> dict:
    ws = load_windows(get_grid(grid), family)
    manifest = ws.provenance["manifest_sha256"]
    pin = load_digests().get("entries", {}).get(f"{grid}/{family}", {})
    if pin.get("manifest_sha256") != manifest:
        raise BuildError(
            f"{grid}/{family}: manifest_sha256 {manifest} != committed input digest "
            f"{pin.get('manifest_sha256')} — not the 1.x windows this patch is for"
        )
    y = ws.y
    fa = derive_fault_active_series(y).astype(bool)
    code = derive_fault_phase_series(y)
    nophase = fa & (code == -1)
    bad_types = sorted(set(y.loc[nophase, "event_type"]) - {EVENT_TYPE})
    if bad_types:
        raise BuildError(
            f"{grid}/{family}: fault-active windows without a phase bit of {bad_types}"
        )
    is_type = fa & (y["event_type"].to_numpy() == EVENT_TYPE)
    if (is_type & ~nophase).any():
        raise BuildError(f"{grid}/{family}: some {EVENT_TYPE} windows already carry a phase bit")

    keep = ["sample_id", "event_type", "y_fault_line", *PHASE_COLS]
    if family == "multi_grid":
        keep.append("result_id")
    yfa = y.loc[fa, keep].copy()
    yfa["bits"] = _bits_str(yfa)
    if (yfa.groupby("sample_id")["bits"].nunique() > 1).any():
        raise BuildError(f"{grid}/{family}: phase bits change within an episode")
    ep = yfa.drop_duplicates("sample_id").reset_index(drop=True)
    if family == "multi_grid":
        ep["result_id"] = ep["result_id"].astype("Int64")

    settings = _read_settings(settings_path)
    m, key = _source_join(grid, family, ep, settings)
    unmatched = int((m["_merge"] != "both").sum())
    type_mismatch = int(
        (m["_merge"] == "both").sum() - (m["event_type"] == m["events/event_type"]).sum()
    )
    if unmatched or type_mismatch:
        raise BuildError(
            f"{grid}/{family}: {unmatched} fault episodes unmatched, {type_mismatch} with a "
            "different source event type"
        )
    one = m["event_type"].str.startswith("flt_1phg")
    exp = _expected_1ph(m["events/event_phase_select_1ph"])
    control = (
        m[one & (m["event_type"] != EVENT_TYPE)]
        .assign(ok=lambda d: d["bits"] == exp[d.index])
        .groupby("event_type")["ok"]
        .agg(["sum", "count"])
    )
    if (control["sum"] != control["count"]).any():
        raise BuildError(f"{grid}/{family}: producer-rule control failed:\n{control}")

    tgt = m[m["event_type"] == EVENT_TYPE].copy()
    sel = tgt["events/event_phase_select_1ph"].str.strip().str.lower()
    if not sel.isin(list(SELECT_TO_BITS)).all():
        raise BuildError(f"{grid}/{family}: {EVENT_TYPE} episodes without an a/b/c phase selector")
    target_ok = None
    if family == "multi_grid":
        target_ok = int((tgt["events/event_target"] == tgt["y_fault_line"]).sum())
        if target_ok != len(tgt):
            raise BuildError(f"{grid}/{family}: events/event_target != y_fault_line")
    bits = np.array([SELECT_TO_BITS[s] for s in sel], dtype=np.int64).reshape(-1, 3)
    if family == "multi_grid":
        source_file = MG_ARCHIVE_MEMBER
        source_key = "result_id=" + tgt["result_id"].astype(str)
    else:
        source_file = f"{family}-{get_grid(grid).raw_foldername}.tar.gz:settings_clean.csv"
        source_key = "general/sim_idx=" + tgt["sample_id"].astype(str)
    rows = pd.DataFrame(
        {
            "grid": grid,
            "family": family,
            "sample_id": tgt["sample_id"].astype(np.int64).to_numpy(),
            "event_type": EVENT_TYPE,
            "source_file": source_file,
            "source_key": source_key.to_numpy(),
            "event_phase_select_1ph": sel.to_numpy(),
            "y_phase_A": bits[:, 0],
            "y_phase_B": bits[:, 1],
            "y_phase_C": bits[:, 2],
        }
    )
    sid = y["sample_id"].to_numpy()
    split = _split_of(grid, family, protocol, names, sid)
    return {
        "grid": grid,
        "family": family,
        "manifest_sha256": manifest,
        "source_sha256": _sha256(settings_path),
        "source_key": key,
        "y": y,
        "fault_active": np.asarray(fa),
        "split": split,
        "rows": rows,
        "windows_patched": int(nophase.sum()),
        "fault_active_windows": int(fa.sum()),
        "control": control,
        "target_ok": target_ok,
        "select_counts": sel.value_counts().sort_index().to_dict(),
        "names": names,
    }


def write_outputs(parts: list[dict], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = (
        pd.concat([p["rows"] for p in parts], ignore_index=True)
        .sort_values(["grid", "family", "sample_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    assert tuple(table.columns) == CSV_COLUMNS
    csv_path = out_dir / CSV_NAME
    csv_path.write_bytes(table.to_csv(index=False, lineterminator="\n").encode())
    index = {
        "schema_version": 1,
        "generated_by": "hpc/build_label_patch.py",
        "patches": [
            {
                "id": PATCH_ID,
                "file": CSV_NAME,
                "sha256": _sha256(csv_path),
                "description": (
                    "Faulted phase of flt_1phg_hif_w_arc restored from the source label "
                    "events/event_phase_select_1ph (a/b/c -> A/B/C), the producer's rule for "
                    "every other flt_1phg type; the 1.x windows carry no phase bit for this type."
                ),
                "event_types": [EVENT_TYPE],
                "affects_columns": PHASE_COLS,
                "affects_functions": ["fault_phase"],
                "from_results_version": "1.3.0",
                "partitions": [
                    {
                        "grid": p["grid"],
                        "family": p["family"],
                        "manifest_sha256": p["manifest_sha256"],
                        "episodes": int(len(p["rows"])),
                        "windows_patched": p["windows_patched"],
                        "source_key": p["source_key"],
                        "source_sha256": p["source_sha256"],
                    }
                    for p in parts
                ],
            }
        ],
    }
    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return csv_path, index_path


def verify_and_audit(parts: list[dict], index_path: Path) -> str:
    """Apply the written patch through the loader's own code path and render the audit table."""
    lines = [
        f"# {PATCH_ID} audit: faulted phase of `{EVENT_TYPE}`",
        "",
        "Generated by `hpc/build_label_patch.py` (deterministic; no timestamps). The patch is",
        "applied through `evemtbench.data.label_patches.apply_label_patches`, the loader's own",
        "code path.",
        "",
        "## Partitions",
        "",
        "| grid / family | manifest_sha256 | fault-active windows | patched windows | episodes "
        "| select a/b/c | remaining without a phase bit |",
        "|---|---|--:|--:|--:|---|--:|",
    ]
    split_lines = [
        "",
        "## Patched windows per split and phase; fault_phase-valid windows before / after",
        "",
        "| grid / family | split | A | B | C | patched | valid before | valid after | added |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    ctl_lines = [
        "",
        "## Control: window bits equal the source selector for the other 1phg types (episodes)",
        "",
        "| grid / family | event type | equal | episodes |",
        "|---|---|--:|--:|",
    ]
    for p in parts:
        y0 = p["y"]
        y1, applied = apply_label_patches(
            y0,
            grid=p["grid"],
            family=p["family"],
            manifest_sha256=p["manifest_sha256"],
            function="fault_phase",
            fault_active=p["fault_active"],
            index_path=str(index_path),
        )
        if len(applied) != 1 or applied[0]["rows_patched"] != p["windows_patched"]:
            raise BuildError(f"{p['grid']}/{p['family']}: applying the patch gave {applied}")
        fa = p["fault_active"]
        c0, c1 = derive_fault_phase_series(y0), derive_fault_phase_series(y1)
        remaining = int((fa & (c1 == -1)).sum())
        if remaining:
            raise BuildError(f"{p['grid']}/{p['family']}: {remaining} windows still without a bit")
        sel = p["select_counts"]
        lines.append(
            f"| {p['grid']} / {p['family']} | `{p['manifest_sha256'][:16]}…` | "
            f"{p['fault_active_windows']:,} | {p['windows_patched']:,} | {len(p['rows']):,} | "
            f"{sel.get('a', 0)} / {sel.get('b', 0)} / {sel.get('c', 0)} | {remaining} |"
        )
        patched = fa & (c0 == -1)
        for name in p["names"]:
            ins = p["split"] == name
            v0 = int((ins & fa & (c0 != -1)).sum())
            v1 = int((ins & fa & (c1 != -1)).sum())
            per = [int((ins & patched & (c1 == PHASE_CLASSES.index(ph))).sum()) for ph in "ABC"]
            pct = f"+{100 * (v1 - v0) / v0:.1f}%" if v0 else "n/a"
            split_lines.append(
                f"| {p['grid']} / {p['family']} | {name} | {per[0]:,} | {per[1]:,} | {per[2]:,} | "
                f"{sum(per):,} | {v0:,} | {v1:,} | {pct} |"
            )
        for et, r in p["control"].iterrows():
            ctl_lines.append(
                f"| {p['grid']} / {p['family']} | `{et}` | {int(r['sum'])} | {int(r['count'])} |"
            )
    tail = [
        "",
        "## Result",
        "",
        "Every fault-active window of every patched partition carries a phase bit after the patch",
        "(remaining = 0), the patched-window counts equal the index pins, and the producer's",
        "one-phase rule reproduces the window bits of every other single-phase-to-ground type on",
        "100% of episodes.",
    ]
    return "\n".join(lines + split_lines + ctl_lines + tail) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--mg-settings",
        required=True,
        help="Template20kV/labels/settings_clean.csv extracted from the 1.x "
        "multi_grid/multigrid-Template20kV.tar.gz",
    )
    ap.add_argument("--aux-root", default=None, help="aux overlay root (default: resolved)")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--audit", default=None, help="write the audit table (Markdown) here")
    ap.add_argument(
        "--check",
        action="store_true",
        help="rebuild into a temporary directory and require byte equality with --out-dir",
    )
    args = ap.parse_args(argv)

    parts = []
    for grid, family, protocol, names in PARTITIONS:
        if family == "multi_grid":
            src = Path(args.mg_settings)
        else:
            src = aux_dir(grid, family, args.aux_root) / "settings_clean.csv"
        print(f"[{PATCH_ID}] {grid}/{family}: source {src}", flush=True)
        parts.append(build_partition(grid, family, protocol, names, src))
        p = parts[-1]
        print(
            f"[{PATCH_ID}] {grid}/{family}: {len(p['rows'])} episodes, "
            f"{p['windows_patched']} windows",
            flush=True,
        )

    if args.check:
        with tempfile.TemporaryDirectory() as td:
            csv_new, idx_new = write_outputs(parts, Path(td))
            out = Path(args.out_dir)
            same = all(
                (out / f.name).is_file() and (out / f.name).read_bytes() == f.read_bytes()
                for f in (csv_new, idx_new)
            )
            print(f"[{PATCH_ID}] check: {'byte-identical' if same else 'DIFFERS'}")
            verify_and_audit(parts, idx_new)
            return 0 if same else 1

    csv_path, index_path = write_outputs(parts, Path(args.out_dir))
    audit = verify_and_audit(parts, index_path)
    if args.audit:
        Path(args.audit).parent.mkdir(parents=True, exist_ok=True)
        Path(args.audit).write_text(audit)
    print(audit)
    print(f"[{PATCH_ID}] wrote {csv_path} (sha256 {_sha256(csv_path)}) and {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
