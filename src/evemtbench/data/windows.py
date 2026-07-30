"""Load pre-built windowed data: memmap ``X.raw`` + ``y.parquet`` + manifest/meta.

This is the primary consumption path. A windows directory contains, per topology:

* ``X_<topology>_W..._S....raw``        — float32 memmap, C-order, shape ``(n, l, f)``
* ``...raw.manifest.json``              — ``shape``, ``feature_hash``, ``config_fingerprint``
* ``...raw.meta.json``                  — ``feature_names`` + ``feature_groups`` + ``extra``
* ``y_<topology>_W..._S....parquet``    — per-window labels incl. ``sample_id`` (episode id)

The loader verifies the manifest against an expected window contract (fail loud on mismatch),
maps windows to their simulation episode via ``sample_id`` (for episode-level splits), and
exposes feature groups + per-cubicle slicing for the local/wide-area input settings.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .roots import GridSpec, get_grid, windows_dir
from .schema import assert_windows_y_schema

CHANNELS_PER_CUBICLE = 6  # [Isec A,B,C, Usec A,B,C]


@dataclass(frozen=True)
class WindowContract:
    """The window geometry the **EvEMTBench-v0.2 profile** pins for its scored runs.

    Window geometry is a *recorded, self-describing parameter* of the general windowed-dataset
    format (contract C-X-2), not a universal constant — other instances differ (e.g. PROTECT-90
    windows exist at various W, fs=6400). These defaults are the benchmark's chosen values
    (50 ms windows, 5 ms step → l=480 at fs=9600) and are gated against each manifest's
    ``config_fingerprint`` in :func:`_verify_manifest`. Pass ``expected_contract=None`` to load
    windows of a different geometry."""

    window_length: float = 0.05
    step_length_seconds: float = 0.005


DEFAULT_CONTRACT = WindowContract()


@dataclass
class WindowSet:
    """A loaded windowed partition."""

    X: np.ndarray  # memmap (n, l, f), float32
    y: pd.DataFrame  # per-window labels, length n; carries sample_id
    feature_names: list[str]
    feature_groups: dict[str, list[int]]
    contract: WindowContract
    topology: str
    x_path: Path
    channels: list[dict] | None = None  # structured per-channel descriptor (contract C-META-2)
    # Feature-axis layout. ``None``/absent = the default wide per-cubicle layout (bus/line identity
    # in ``feature_names``). ``"double_line"`` = the portable 12-channel EndA/EndB view used by the
    # multi_grid pre-training corpus (identity in ``y``); observable at LOCAL/LINE only. See
    # ``tasks/line_view.py``.
    view: str | None = None
    # Input-data provenance for the run record: {grid, family, x_path, shape, feature_hash,
    # config_fingerprint, manifest_sha256}. Content-based fields are identical between the vault
    # master and a $TMPDIR staged copy, so they identify the input bytes across staging.
    provenance: dict | None = None

    @property
    def n_windows(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[2]

    @property
    def n_cubicles(self) -> int:
        return self.n_features // CHANNELS_PER_CUBICLE

    @property
    def sample_ids(self) -> np.ndarray:
        """Episode id per window (the split grouping key)."""
        return self.y["sample_id"].to_numpy()

    def cubicle_slice(self, cubicle_idx: int) -> slice:
        """Feature-axis slice for one cubicle's 6 channels [Ia,Ib,Ic,Va,Vb,Vc]."""
        start = cubicle_idx * CHANNELS_PER_CUBICLE
        return slice(start, start + CHANNELS_PER_CUBICLE)

    def local_view(self, cubicle_idx: int) -> np.ndarray:
        """Local input setting: windows restricted to a single cubicle -> (n, l, 6)."""
        return self.X[:, :, self.cubicle_slice(cubicle_idx)]


def _contract_tag(contract: WindowContract) -> str:
    """The ``W..._S...`` filename tag for a window geometry (0.05/0.005 -> ``W0p050_S0p005``).

    The producer encodes both values with three decimals; narrowing by this tag is only ever
    *applied* when it selects exactly one file, so an unexpected encoding degrades to the
    fail-loud multi-file error rather than a wrong pick."""

    def num(v: float) -> str:
        return f"{v:.3f}".replace(".", "p")

    return f"W{num(contract.window_length)}_S{num(contract.step_length_seconds)}"


def find_window_files(
    directory: str | os.PathLike,
    topology: str | None = None,
    *,
    contract: WindowContract | None = None,
) -> dict[str, Path]:
    """Locate the ``X``/manifest/meta/``y`` files in a windows directory.

    Returns a dict with keys ``x_path``, ``manifest_path``, ``meta_path``, ``y_path``.
    A windows dir may hold **several window geometries** side by side (e.g. ``W0p020`` and
    ``W0p050`` on the production mount); passing ``contract`` narrows to the pinned geometry's
    file. Raises if zero — or, after any narrowing, more than one — X file matches.
    """
    directory = Path(directory)
    pattern = f"X_{topology}_*.raw" if topology else "X_*.raw"
    x_files = sorted(p for p in directory.glob(pattern) if p.suffix == ".raw")
    if len(x_files) > 1 and contract is not None:
        narrowed = [p for p in x_files if _contract_tag(contract) in p.name]
        if len(narrowed) == 1:
            x_files = narrowed
    if len(x_files) != 1:
        raise FileNotFoundError(
            f"expected exactly one {pattern} in {directory}"
            + (f" for contract {_contract_tag(contract)}" if contract is not None else "")
            + f", found {len(x_files)}: {[p.name for p in x_files]}"
        )
    x_path = x_files[0]
    manifest_path = x_path.with_suffix(x_path.suffix + ".manifest.json")
    meta_path = x_path.with_suffix(x_path.suffix + ".meta.json")
    # y file mirrors the X stem: X_<stem>.raw -> y_<stem>.parquet
    stem = x_path.name[len("X_") :].removesuffix(".raw")
    y_path = directory / f"y_{stem}.parquet"
    for p in (manifest_path, meta_path, y_path):
        if not p.is_file():
            raise FileNotFoundError(f"missing companion file: {p}")
    return {
        "x_path": x_path,
        "manifest_path": manifest_path,
        "meta_path": meta_path,
        "y_path": y_path,
    }


def _verify_manifest(manifest: dict, expected: WindowContract | None, topology: str | None) -> None:
    if manifest.get("status") != "complete":
        raise ValueError(
            f"window manifest status is {manifest.get('status')!r}, expected 'complete'"
        )
    fp = manifest.get("config_fingerprint", {})
    if topology is not None and fp.get("topology") != topology:
        raise ValueError(f"manifest topology {fp.get('topology')!r} != expected {topology!r}")
    if expected is not None:
        if fp.get("window_length") != expected.window_length or (
            fp.get("step_length_seconds") != expected.step_length_seconds
        ):
            raise ValueError(
                "window contract mismatch: manifest "
                f"(W={fp.get('window_length')}, S={fp.get('step_length_seconds')}) "
                f"!= expected (W={expected.window_length}, S={expected.step_length_seconds})"
            )

    # Contract-version gate: datasets written under Core >=1.0 self-declare their
    # ``contract_version``. We reject only a *major* mismatch; legacy data without the field is
    # accepted (it predates the contract). See ``evemtbench.contract``.
    declared = fp.get("contract_version")
    if declared is not None:
        from evemtbench.contract import EXPECTED_CONTRACT_VERSION

        if declared.split(".", 1)[0] != EXPECTED_CONTRACT_VERSION.split(".", 1)[0]:
            raise ValueError(
                f"windows contract_version {declared!r} is major-incompatible with the "
                f"benchmark's expected {EXPECTED_CONTRACT_VERSION!r}"
            )


def read_episode_ids(
    grid: str | GridSpec,
    family: str,
    *,
    root: str | os.PathLike | None = None,
    directory: str | os.PathLike | None = None,
    contract: WindowContract | None = DEFAULT_CONTRACT,
) -> list[int]:
    """Sorted unique ``sample_id`` episode ids for a partition (reads only the y parquet)."""
    g = get_grid(grid)
    directory = Path(directory) if directory is not None else windows_dir(g, family, root)
    files = find_window_files(directory, topology=g.topology, contract=contract)
    y = pd.read_parquet(files["y_path"], columns=["sample_id"])
    return sorted(int(i) for i in y["sample_id"].unique())


def read_episode_labels(
    grid: str | GridSpec,
    family: str,
    *,
    column: str = "event_type",
    root: str | os.PathLike | None = None,
    directory: str | os.PathLike | None = None,
    contract: WindowContract | None = DEFAULT_CONTRACT,
) -> dict[int, str]:
    """Episode-constant label per ``sample_id`` (e.g. ``event_type``) for a partition — the strata
    used by the stratified split generator. Reads only ``[sample_id, column]`` from the y file."""
    g = get_grid(grid)
    directory = Path(directory) if directory is not None else windows_dir(g, family, root)
    files = find_window_files(directory, topology=g.topology, contract=contract)
    y = pd.read_parquet(files["y_path"], columns=["sample_id", column])
    # episode-constant: take the first value per sample_id
    return {int(sid): str(v) for sid, v in y.groupby("sample_id")[column].first().items()}


def load_windows(
    grid: str | GridSpec,
    family: str,
    *,
    root: str | os.PathLike | None = None,
    directory: str | os.PathLike | None = None,
    expected_contract: WindowContract | None = DEFAULT_CONTRACT,
) -> WindowSet:
    """Load a windowed partition for ``grid``/``family`` (or from an explicit ``directory``).

    ``expected_contract`` defaults to the frozen v1.0 spec; pass ``None`` to skip contract
    gating (e.g. when inspecting non-standard windows). The X array is memory-mapped read-only.
    """
    g = get_grid(grid)
    directory = Path(directory) if directory is not None else windows_dir(g, family, root)
    files = find_window_files(directory, topology=g.topology, contract=expected_contract)

    manifest = json.loads(files["manifest_path"].read_text())
    meta = json.loads(files["meta_path"].read_text())
    _verify_manifest(manifest, expected_contract, g.topology)

    # feature_hash consistency (contract C-MAN-3): the manifest hash must match the meta's
    # feature_names, catching a mangled/misaligned channel list before any windows are read.
    fhash = manifest.get("feature_hash")
    fnames = meta.get("feature_names", [])
    if fhash and fnames:
        h = hashlib.sha256()
        for nm in fnames:
            h.update(nm.encode("utf-8"))
            h.update(b"\n")
        if h.hexdigest() != fhash:
            raise ValueError(
                f"feature_hash mismatch for {files['x_path'].name}: manifest hash != "
                "sha256(meta.feature_names) (channel list is inconsistent with the manifest)"
            )

    n, length, nf = (manifest["shape"][k] for k in ("n", "l", "f"))
    X = np.memmap(files["x_path"], dtype=np.float32, mode="r").reshape((n, length, nf))

    y = pd.read_parquet(files["y_path"])
    assert_windows_y_schema(y)
    if len(y) != n:
        raise ValueError(f"X has {n} windows but y has {len(y)} rows")

    contract = WindowContract(
        window_length=manifest["config_fingerprint"]["window_length"],
        step_length_seconds=manifest["config_fingerprint"]["step_length_seconds"],
    )
    return WindowSet(
        X=X,
        y=y.reset_index(drop=True),
        feature_names=meta.get("feature_names", []),
        feature_groups=meta.get("feature_groups", {}),
        contract=contract,
        topology=g.topology,
        x_path=files["x_path"],
        channels=meta.get("channels"),
        view=(meta.get("extra") or {}).get("view"),
        provenance={
            "grid": g.key,
            "family": family,
            "x_path": str(files["x_path"]),
            "shape": manifest.get("shape"),
            "feature_hash": manifest.get("feature_hash"),
            "config_fingerprint": manifest.get("config_fingerprint"),
            "manifest_sha256": hashlib.sha256(files["manifest_path"].read_bytes()).hexdigest(),
        },
    )
