"""Committed, episode-keyed label patches (y-overlays) for the 1.x windows.

A label patch corrects a label column of a windowed partition WITHOUT touching the windows: the
X arrays, the y parquet and the manifests stay byte-identical, so every integrity pin
(``input_digests.json``, the run-record ``partitions``, the projection-cache identity) still
names the same bytes, and a record computed before the patch re-evaluates exactly (the patch is
off for it). The patch itself is a small committed CSV, pinned by sha256 in ``index.json`` and
built deterministically by ``hpc/build_label_patch.py`` from the producer's source labels.

``LP-1`` (benchmark v1.3.0): the faulted-phase bits of ``flt_1phg_hif_w_arc``. In the 1.x windows
every fault-active window of that type has ``y_phase_A = y_phase_B = y_phase_C = 0`` (the type is
missing from the producer's one-phase rule), so the ``fault_phase`` valid-sample rule dropped the
class. The patch restores the bit from the source label ``events/event_phase_select_1ph``
(``docs/known_issues.md`` §4).

Application rules (:func:`apply_label_patches`), each enforced loudly:

* only the functions a patch names (``affects_functions``; LP-1: ``fault_phase``), and only when
  the caller enables patches (``version.label_patches_enabled``: a v1.3.0+ cell);
* only on the partition whose ``manifest_sha256`` the patch was built for. Any other manifest (a
  re-windowed or 2.0 partition, a test fixture) gets no patch: the overlay is inert there;
* on a matching partition, every targeted window must currently be fault-active, of the patched
  event type and carry no phase bit, every patch episode must be present, the number of patched
  windows must equal the pinned count, and afterwards no fault-active window of the type may be
  left without a bit. Anything else raises :class:`LabelPatchError` ("dataset changed").
"""

from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

PATCH_DIR = Path(__file__).resolve().parent
INDEX_PATH = PATCH_DIR / "index.json"

# Columns of a patch CSV (the builder writes exactly these, in this order).
CSV_COLUMNS = (
    "grid",
    "family",
    "sample_id",
    "event_type",
    "source_file",
    "source_key",
    "event_phase_select_1ph",
    "y_phase_A",
    "y_phase_B",
    "y_phase_C",
)


class LabelPatchError(RuntimeError):
    """A label patch does not fit the data it is applied to, or its file does not match its pin."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@cache
def load_index(path: str | None = None) -> dict:
    """The patch index (``index.json``): ``{"schema_version", "patches": [...]}``."""
    p = Path(path) if path is not None else INDEX_PATH
    if not p.is_file():
        return {"schema_version": 1, "patches": []}
    return json.loads(p.read_text())


@cache
def load_patch_table(patch_id: str, index_path: str | None = None) -> pd.DataFrame:
    """The rows of one patch, after checking its file against the sha256 pinned in the index."""
    entry = get_patch(patch_id, index_path)
    base = Path(index_path).parent if index_path is not None else PATCH_DIR
    path = base / entry["file"]
    got = sha256_file(path)
    if got != entry["sha256"]:
        raise LabelPatchError(
            f"label patch {patch_id}: {path.name} has sha256 {got}, index pins {entry['sha256']} "
            "(edited by hand? rebuild it with hpc/build_label_patch.py)"
        )
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(df.columns) != CSV_COLUMNS:
        raise LabelPatchError(
            f"label patch {patch_id}: columns {list(df.columns)} != {CSV_COLUMNS}"
        )
    df = df.astype({c: np.int64 for c in ("sample_id", "y_phase_A", "y_phase_B", "y_phase_C")})
    if df.duplicated(["grid", "family", "sample_id"]).any():
        raise LabelPatchError(f"label patch {patch_id}: duplicate (grid, family, sample_id) rows")
    return df


def get_patch(patch_id: str, index_path: str | None = None) -> dict:
    for entry in load_index(index_path)["patches"]:
        if entry["id"] == patch_id:
            return entry
    raise KeyError(f"no label patch {patch_id!r} in the index")


def _phase_cols(entry: dict) -> list[str]:
    return list(entry["affects_columns"])


def apply_label_patches(
    y: pd.DataFrame,
    *,
    grid: str,
    family: str,
    manifest_sha256: str | None,
    function: str,
    fault_active: np.ndarray,
    index_path: str | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Apply every patch of the index that targets ``function`` on this exact partition.

    ``fault_active`` is the boolean fault-active mask of ``y`` (the caller derives it with
    ``tasks.labels.event_present.derive_fault_active_series``; passed in so the data layer does
    not import the task layer). Returns ``(y, provenance)``: ``y`` is a patched COPY when a patch
    applied (the input frame is never mutated) and ``provenance`` lists one record per applied
    patch ``{id, sha256, grid, family, manifest_sha256, episodes, rows_patched}``.
    """
    applied: list[dict] = []
    fault_active = np.asarray(fault_active, dtype=bool)
    for entry in load_index(index_path)["patches"]:
        if function not in entry["affects_functions"]:
            continue
        part = next(
            (p for p in entry["partitions"] if p["grid"] == grid and p["family"] == family), None
        )
        if part is None or part["manifest_sha256"] != manifest_sha256:
            continue  # not the partition this patch was built for: inert
        table = load_patch_table(entry["id"], index_path)
        rows = table[(table["grid"] == grid) & (table["family"] == family)]
        if len(rows) != part["episodes"]:
            raise LabelPatchError(
                f"label patch {entry['id']} {grid}/{family}: {len(rows)} rows in the CSV, the "
                f"index pins {part['episodes']} episodes"
            )
        cols = _phase_cols(entry)
        types = set(entry["event_types"])
        sid = y["sample_id"].to_numpy()
        target = np.isin(sid, rows["sample_id"].to_numpy()) & fault_active
        prefix = f"label patch {entry['id']} on {grid}/{family}"
        missing = set(rows["sample_id"].tolist()) - set(sid[target].tolist())
        if missing:
            raise LabelPatchError(
                f"{prefix}: {len(missing)} patch episode(s) have no fault-active window "
                f"(e.g. sample_id {sorted(missing)[:5]}) — dataset changed under the manifest pin"
            )
        et = y["event_type"].to_numpy()[target]
        if not set(pd.unique(et)) <= types:
            raise LabelPatchError(
                f"{prefix}: targeted windows carry event types {sorted(set(pd.unique(et)) - types)}"
                f", the patch covers only {sorted(types)} — dataset changed"
            )
        bits = y.loc[target, cols].fillna(0).astype(np.int64).to_numpy()
        if bits.any():
            raise LabelPatchError(
                f"{prefix}: {int(bits.any(axis=1).sum())} targeted window(s) already carry a phase "
                "bit — patch target already labelled / dataset changed"
            )
        n_target = int(target.sum())
        if n_target != part["windows_patched"]:
            raise LabelPatchError(
                f"{prefix}: {n_target} fault-active windows targeted, the index pins "
                f"{part['windows_patched']} — dataset changed"
            )
        lut = rows.set_index("sample_id")[cols]
        new = lut.loc[sid[target]].to_numpy(dtype=np.int64)
        if (new.sum(axis=1) < 1).any():
            raise LabelPatchError(f"{prefix}: a patch row sets no phase bit")
        y = y.copy()
        for j, c in enumerate(cols):
            vals = y[c].to_numpy().copy()
            vals[target] = new[:, j].astype(vals.dtype, copy=False)
            y[c] = vals
        left = (
            fault_active
            & np.isin(y["event_type"].to_numpy(), list(types))
            & (y[cols].fillna(0).astype(np.int64).to_numpy().sum(axis=1) == 0)
        )
        if left.any():
            raise LabelPatchError(
                f"{prefix}: {int(left.sum())} fault-active window(s) of {sorted(types)} still "
                "carry no phase bit after patching — the patch is incomplete for this partition"
            )
        applied.append(
            {
                "id": entry["id"],
                "sha256": entry["sha256"],
                "grid": grid,
                "family": family,
                "manifest_sha256": manifest_sha256,
                "episodes": int(len(rows)),
                "rows_patched": n_target,
            }
        )
    return y, applied


def expected_provenance(
    grid: str,
    family: str,
    manifest_sha256: str | None,
    function: str,
    index_path: str | None = None,
) -> list[dict]:
    """The provenance records a v1.3.0+ load of this partition must carry (from the index alone,
    no data read): what ``integrity`` and ``reevaluate`` compare a run record against."""
    out = []
    for entry in load_index(index_path)["patches"]:
        if function not in entry["affects_functions"]:
            continue
        for p in entry["partitions"]:
            if (p["grid"], p["family"], p["manifest_sha256"]) == (grid, family, manifest_sha256):
                out.append(
                    {
                        "id": entry["id"],
                        "sha256": entry["sha256"],
                        "grid": grid,
                        "family": family,
                        "manifest_sha256": manifest_sha256,
                        "episodes": int(p["episodes"]),
                        "rows_patched": int(p["windows_patched"]),
                    }
                )
    return out
