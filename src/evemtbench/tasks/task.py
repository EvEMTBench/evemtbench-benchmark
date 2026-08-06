"""Task assembly: splits + windows + observability + label derivation -> Task/TaskView.

``load_task`` is the primary user entry point. It reads the committed split for a
(protocol, grid), loads the relevant grids' windows, restricts each split to its episodes
(by ``sample_id``), applies the task's input setting (observability), and derives labels.

A split is a list of ``(family, grid, sim_idx)`` triples. Under the v0.1 ``held_out`` protocol
every triple shares the same target grid: train/val/test come from its ``adapt_grid`` family and
the extra held-out test from its ``benchmark`` family. With LOCAL observability every grid
yields a uniform 6-channel view.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..contract import CONTRACT_SHA256
from ..data.roots import GridSpec, get_grid
from ..data.windows import WindowSet, load_windows
from ..splits.loader import read_held_out_splits, read_splits, split_dir
from ..splits.protocols import HELD_OUT_SPLIT_NAMES, get_protocol
from ..version import SPLITS_VERSION
from . import line_view
from .labels.event_detection import derive_event_present_series
from .labels.event_present import derive_fault_active_series
from .labels.event_state import derive_event_state_series
from .labels.event_type import derive_event_type_series
from .labels.fault_category import derive_fault_category_series
from .labels.fault_class import derive_fault_class_series
from .labels.fault_grounded import derive_fault_grounded_series
from .labels.fault_location import derive_fault_location_series
from .labels.fault_origin import derive_fault_origin_series
from .labels.fault_phase import derive_fault_phase_series
from .labels.switch_detection import derive_switch_present_series
from .labels.switch_type import derive_switch_type_series
from .registry import get_task
from .taxonomy import Function, TaskDefinition
from .valid_samples import valid_sample_mask

_META_COLS = [
    "sample_id",
    "window_idx",
    "event_type",
    "event_family",
    "y_fault_present",
    "y_fault_line",
    "y_fault_location",
]


@dataclass
class _ViewSource:
    """One (windowset, masked rows) feeding a view; features are read+projected on demand."""

    ws: WindowSet
    rows: np.ndarray  # int row indices into ws.X / ws.y
    fault_lines: np.ndarray  # y_fault_line for those rows (per-row local/line projection)
    td: TaskDefinition
    grid: str
    # For the portable double_line view only: per-row terminal-order (0=EndA near, 1=EndB near),
    # canonicalising [near, far] by bus id to match the reference oracle. ``None`` otherwise.
    near_end: np.ndarray | None = None


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Write a small JSON sidecar via tmp + ``os.replace`` so a concurrent reader never sees a
    partial file. Best-effort: an unwritable cache dir must not kill training (the payload is a
    performance optimisation), but a *corrupt* sidecar is never produced."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.json.tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        os.replace(tmp, path)
    except OSError:
        pass


def _verify_cache_sidecar(path: Path, side: Path, ident: dict | None, expected_bytes: int) -> None:
    """Validate an existing projection-cache entry before it is trusted.

    Raises on a size or identity mismatch. A missing sidecar is tolerated (entries built before
    sidecars existed, and staged copies where only the ``.f32`` was transferred) but the size check
    still applies — that is the one check that catches a truncated NFS/rsync write, which numpy's
    memmap will happily map when the file is long enough and silently accept when it is not short.
    """
    actual = path.stat().st_size
    if actual != expected_bytes:
        raise RuntimeError(
            f"projection cache {path.name} is {actual} bytes, expected {expected_bytes} "
            f"— truncated or foreign entry; delete it and rebuild (hpc/build_cache.sh)"
        )
    if not side.is_file():
        return
    try:
        rec = json.loads(side.read_text())
    except (OSError, json.JSONDecodeError):
        return  # unreadable sidecar: the size check above still ran, and the name encodes identity
    if ident is not None and rec.get("identity") not in (None, ident):
        got = rec.get("identity") or {}
        differing = sorted(k for k in set(ident) | set(got) if got.get(k) != ident.get(k))
        raise RuntimeError(
            f"projection cache {path.name} identity mismatch: the entry was built for different "
            f"input data (differing keys: {differing}). Two dataset builds are colliding in one "
            f"cache root — do not overwrite; use a version-scoped cache root or clear it."
        )


def _transform_key(transform) -> str | None:
    """Cross-process-stable identity of a channel-stats transform, or ``None`` if it has none.

    The key lands in a content-addressed *filename*, so it must be reproducible in another process:
    ``id()``/``repr()`` are unusable. A plain function's ``__name__`` qualifies, and returning the
    historical spellings (``"none"`` for no transform, otherwise the function name) keeps every
    already-persisted stats file valid.

    ``None`` for anything without a stable name — a ``functools.partial`` (no ``__name__`` at all,
    which the previous ``getattr(transform, "__name__", "none")`` silently aliased to the *no
    transform* key) and a lambda (every lambda is ``"<lambda>"``, so two different ones collided).
    Callers treat ``None`` as "do not cache", which is slower but cannot serve wrong statistics.
    """
    if transform is None:
        return "none"
    name = getattr(transform, "__name__", None)
    if not name or name == "<lambda>":
        return None
    return name


def _stats_dirs() -> list[Path]:
    """Where persisted channel-stats live, probe order: node-local scratch first, then the
    version-scoped persistent root. Both are explicit-env-gated so library users / tests with
    neither var set stay hermetic (memo-only, no disk I/O). The persistent-root gate deliberately
    requires ``$EVEMTBENCH_PROJECTION_CACHE_ROOT`` (not the data-root fallback) so dev machines
    never write into the dataset tree."""
    dirs: list[Path] = []
    local = os.environ.get("EVEMTBENCH_PROJECTION_CACHE")
    if local:
        dirs.append(Path(local))
    if os.environ.get("EVEMTBENCH_PROJECTION_CACHE_ROOT"):
        from ..data.staging import persistent_cache_root

        dirs.append(persistent_cache_root())
    return dirs


def _load_stats(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load persisted (mean, std) or None on ANY problem (missing/corrupt/wrong shape) — a bad
    file on the shared FS must fall back to compute + rewrite, never crash a training job."""
    try:
        with np.load(path) as z:
            mean, std = z["mean"], z["std"]
        if mean.dtype != np.float32 or std.dtype != np.float32 or mean.shape != std.shape:
            return None
        if mean.ndim != 3 or mean.shape[:2] != (1, 1):
            return None
        return mean, std
    except Exception:  # noqa: BLE001 — deliberately broad: any unreadable file means "recompute"
        return None


def _save_stats_atomic(path: Path, mean: np.ndarray, std: np.ndarray) -> None:
    """Atomic npz write (tmp + os.replace, mirroring ``materialize_cache``). Best-effort: raising
    is the caller's choice — here we let OSErrors propagate and the caller swallows them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, mean=mean, std=std)
    os.replace(tmp, path)


class TaskView:
    """One split of a task. Labels + metadata are materialized (small); **features are streamed**
    from the memmap on demand (read + observability-projected per batch), so a wide ``global`` view
    on a big grid is never held as one dense ``(n, l, c)`` array. ``.features`` still materializes
    the full array for small views / tests — do not call it on the big global views."""

    def __init__(self, name, sources, labels, metadata, *, dense=None):
        self.name = name
        self._sources = sources  # list[_ViewSource]
        self.labels = np.asarray(labels)
        self.metadata = metadata.reset_index(drop=True)
        self._dense = dense  # optional pre-built array (empty views / future multi-grid concat)
        self._cache = None  # optional projection-cache memmap (materialize_cache); read path below
        # flat index: global position -> (source id, local offset within that source's rows)
        if sources:
            self._src_id = np.concatenate([np.full(len(s.rows), i) for i, s in enumerate(sources)])
            self._local = np.concatenate([np.arange(len(s.rows)) for s in sources])
        else:
            self._src_id = np.empty(0, int)
            self._local = np.empty(0, int)

    def __len__(self) -> int:
        return int(self._dense.shape[0]) if self._dense is not None else int(self.labels.shape[0])

    @property
    def sample_ids(self) -> np.ndarray:
        return self.metadata["sample_id"].to_numpy()

    @property
    def feature_shape(self) -> tuple[int, int, int]:
        """(n, l, c_obs) without materializing features."""
        if self._dense is not None:
            return tuple(self._dense.shape)
        s = self._sources[0]
        obs = s.td.observability.value
        if line_view.is_double_line_view(s.ws.feature_names, s.ws.view):
            c = line_view.double_line_channels(obs)
        else:
            oracle = line_view.build_oracle(s.grid, s.ws.feature_names, s.ws.channels)
            c = (
                oracle.n_line_cubicles * line_view.CHANNELS_PER_CUBICLE
                if obs == "global"
                else (6 if obs == "local" else 12)
            )
        return (len(self), int(s.ws.X.shape[1]), int(c))

    def _project_positions(self, positions: np.ndarray) -> np.ndarray:
        """Read the full on-disk windows at the given global positions and project them to the
        observability view, preserving their order. This reads ALL on-disk channels then slices to
        the view (6/12/420) — the expensive step ``materialize_cache`` precomputes away."""
        out = None
        for sid, src in enumerate(self._sources):
            m = self._src_id[positions] == sid
            if not m.any():
                continue
            local = self._local[positions[m]]
            xb = np.asarray(src.ws.X[src.rows[local]])
            proj = _observe(
                xb,
                src.fault_lines[local],
                src.td,
                src.grid,
                src.ws.feature_names,
                src.ws.channels,
                view=src.ws.view,
                near_end=(None if src.near_end is None else src.near_end[local]),
            )
            if out is None:
                out = np.empty((len(positions), proj.shape[1], proj.shape[2]), np.float32)
            out[m] = proj
        if out is None:  # empty
            _, length, c = self.feature_shape
            return np.empty((0, length, c), np.float32)
        return out

    def _read_positions(self, positions: np.ndarray) -> np.ndarray:
        """Windows at the given global positions, preserving order. Reads from the projection cache
        (only the view's channels) when materialized — bit-identical to ``_project_positions``,
        which fills it — otherwise projects live."""
        if self._cache is not None:
            return np.ascontiguousarray(self._cache[positions], dtype=np.float32)
        return self._project_positions(positions)

    def cache_identity(self) -> dict | None:
        """The full set of inputs the cached bytes are a function of, or ``None`` if uncacheable.

        This is the cache's *identity*, and it has to name the window CONTENT, not just its shape.
        The earlier key was (view name, grid set, observability, shape, sample_id vector), which two
        different builds of the dataset can share: sample ids are per-episode counters, so a
        regenerated corpus with the same episodes and geometry produced the same key for different
        waveforms. Since the cache is persistent and rsync-shared into ``$CROOT``
        (``hpc/build_cache.sh``), a stale entry would then be reused silently — the worst failure
        mode available, because nothing downstream can detect it.

        Added: ``window_idx`` (with ``sample_id`` it identifies the window, not just the episode),
        each source's ``manifest_sha256`` + ``feature_hash`` + family (the input bytes and channel
        layout), the window-contract sha, and ``near_end`` where it exists (it reorders the
        double_line terminals, so it changes the projected content). ``x_path`` is deliberately
        excluded: it differs between the vault master and a ``$TMPDIR`` staged copy of the same
        bytes, and keying on it would defeat staging.
        """
        if self._dense is not None or len(self) == 0:
            return None
        n, length, c = self.feature_shape
        srcs = []
        for s in self._sources:
            prov = s.ws.provenance or {}
            near = s.near_end
            srcs.append(
                {
                    "grid": s.grid,
                    "family": prov.get("family"),
                    "manifest_sha256": prov.get("manifest_sha256"),
                    "feature_hash": prov.get("feature_hash"),
                    "layout": s.ws.view,
                    "near_end": (
                        hashlib.sha256(
                            np.ascontiguousarray(near, dtype=np.int8).tobytes()
                        ).hexdigest()[:16]
                        if near is not None
                        else None
                    ),
                }
            )
        return {
            "view": self.name,
            "grids": "-".join(sorted({s.grid for s in self._sources})),
            "observability": self._sources[0].td.observability.value,
            "shape": [n, length, c],
            "contract_sha256": CONTRACT_SHA256,
            "rows_sha256": hashlib.sha256(
                np.ascontiguousarray(self.metadata["sample_id"].to_numpy()).tobytes()
                + b"|"
                + np.ascontiguousarray(self.metadata["window_idx"].to_numpy()).tobytes()
            ).hexdigest(),
            "sources": sorted(srcs, key=lambda d: (d["grid"], d["family"] or "")),
        }

    def cache_filename(self) -> str | None:
        """Content-addressed projection-cache filename for this view, or ``None`` if uncacheable
        (empty/dense). Stable across jobs — identical content yields the identical name — so a
        *persistent* cache dir deduplicates across jobs/seeds/baselines, and across tasks that share
        an observability+split (same key). Single source of truth for the cache identity, reused by
        :mod:`evemtbench.data.staging`.

        The hash covers :meth:`cache_identity` in full; the human-readable prefix is unchanged so
        cache directories stay browsable. NOTE: widening the identity changes every filename, so the
        persistent cache must be rebuilt (``hpc/build_cache.sh``) — that is the intended cost of no
        longer being able to collide with a different dataset build."""
        ident = self.cache_identity()
        if ident is None:
            return None
        n, length, c = ident["shape"]
        key = hashlib.sha1(
            json.dumps(ident, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        return (
            f"proj_{self.name}_{ident['grids']}_{ident['observability']}_{n}x{length}x{c}_{key}.f32"
        )

    def stats_filename(self, transform_key: str, batch_size: int = 8192) -> str | None:
        """Content-addressed filename for persisted channel-stats, or ``None`` if the view is
        uncacheable (empty/dense — same rule as :meth:`cache_filename`). Reuses the projection
        cache's content key verbatim (single source of identity truth) plus the transform's memo
        key and the accumulation batch size — the float64 reduction is chunk-partitioned, so stats
        from a different ``batch_size`` are not guaranteed bit-identical and must not collide."""
        fn = self.cache_filename()
        if fn is None:
            return None
        return f"stats_{fn[len('proj_') : -len('.f32')]}_{transform_key}_b{batch_size}.npz"

    def materialize_cache(self, cache_dir: str | os.PathLike | None) -> None:
        """Project every window ONCE into a memmap holding only the view's channels, then read
        batches from it thereafter. This removes the dominant GPU-starving cost: reading all on-disk
        channels (e.g. 2520) and re-projecting every batch, every epoch, every seed. No-op without
        ``cache_dir`` or for empty/dense views. The cache is content-addressed (see
        :meth:`cache_filename`), shared across seeds/baselines, and bit-identical to the live
        projection.

        If ``$EVEMTBENCH_CACHE_READONLY=1`` a missing cache **raises** instead of being built — GPU
        training jobs set this so they never spend GPU time building the cache from the shared
        filesystem. That one-time build belongs in the CPU prep job (``hpc/build_cache.sh``).

        Every entry carries a ``.json`` sidecar holding its :meth:`cache_identity` and expected byte
        count, re-checked here before the memmap is accepted. The filename hash already encodes the
        identity, so the sidecar is a belt-and-braces check against the cases a name cannot cover: a
        truncated or partially-written file (numpy only raises when the file is *shorter* than the
        requested shape — a longer or same-length-wrong-content file maps silently), and a hash
        collision or a hand-copied entry. A sidecar mismatch raises rather than silently rebuilding,
        because on a shared cache root it means two different datasets are colliding and the right
        response is to stop, not to overwrite someone else's bytes."""
        if not cache_dir or self._cache is not None or self._dense is not None or len(self) == 0:
            return
        n, length, c = self.feature_shape
        cdir = Path(cache_dir)
        fname = self.cache_filename()
        assert fname is not None  # non-None by the guard above (non-empty, non-dense view)
        path = cdir / fname
        side = path.with_suffix(".json")
        expected_bytes = n * length * c * 4  # float32
        ident = self.cache_identity()
        if not path.exists():  # build once; concurrent baselines reuse it (atomic rename below)
            if os.environ.get("EVEMTBENCH_CACHE_READONLY") == "1":
                raise RuntimeError(
                    f"projection cache missing ({path.name}) and EVEMTBENCH_CACHE_READONLY=1 — "
                    "build it first with hpc/build_cache.sh (no GPU-held cache builds)."
                )
            cdir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            mm = np.memmap(tmp, dtype=np.float32, mode="w+", shape=(n, length, c))
            for start in range(0, n, _OBSERVE_CHUNK):
                pos = np.arange(start, min(start + _OBSERVE_CHUNK, n))
                mm[pos] = self._project_positions(pos)
            mm.flush()
            del mm
            # Sidecar first, then the payload: the payload's existence is what gates reuse, so
            # publishing it last means a reader never sees data without its identity.
            _write_json_atomic(side, {"identity": ident, "bytes": expected_bytes})
            os.replace(tmp, path)
        else:
            _verify_cache_sidecar(path, side, ident, expected_bytes)
        self._cache = np.memmap(path, dtype=np.float32, mode="r", shape=(n, length, c))

    def iter_batches(self, batch_size: int, *, order: np.ndarray | None = None):
        """Yield ``(x_batch (b,l,c) float32, y_batch)`` streamed from the memmap.

        Pass a permuted ``order`` (length len(self)) for shuffled training batches."""
        n = len(self)
        order = np.arange(n) if order is None else order
        for i in range(0, n, batch_size):
            pos = order[i : i + batch_size]
            yield self._read_positions(pos), self.labels[pos]

    def channel_stats(
        self, batch_size: int = 8192, transform=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-channel (mean, std) over windows+time in one streaming pass — shape (1,1,c).

        If ``transform`` is given it is applied to each raw batch before accumulating stats
        (e.g. ``add_zero_sequence`` to compute stats over the augmented channel set).

        Stats are **seed-independent**, so the result is memoized on the view (keyed by the
        transform *and* ``batch_size``, matching the on-disk identity — see :meth:`stats_filename`)
        — the runner reuses one Task across all seeds, so this full read+project pass runs once per
        cell instead of once per ``fit()``. The numpy reduction is unchanged, so the memoized value
        is bit-identical to recomputing it (keeps results comparable across cells). A transform with
        no stable cross-process name (lambda / ``functools.partial``) is neither memoized nor
        persisted; see :func:`_transform_key`.

        The memo additionally persists to disk (2026-07-18, GPU audit follow-up: this pass cost up
        to 635 s per cell on ieee39 wide views, recomputed every cell and chain link). The file is
        content-addressed like the projection cache (see :meth:`stats_filename`) and probed
        node-local then persistent root (:func:`_stats_dirs`); the first computation writes back
        best-effort. Deliberate: ``EVEMTBENCH_CACHE_READONLY=1`` does NOT block this ~10 KB write —
        that guard exists to keep GPU jobs from *building the projection cache*; the stats compute
        cannot be avoided anyway, so persisting it makes every later cell/link load instead of
        re-pay. Deterministic single-threaded reduction + atomic replace ⇒ concurrent writers
        produce identical bytes; loaded values are bit-identical to a fresh compute.
        """
        cache = self.__dict__.setdefault("_stats_cache", {})
        # The memo key must carry EVERYTHING the value depends on, which is exactly what the on-disk
        # filename already carried: the transform AND batch_size (the float64 reduction is
        # chunk-partitioned, so a different batch_size is not guaranteed bit-identical). Keying on
        # the transform alone returned the first batch_size's stats for every later call.
        tkey = _transform_key(transform)
        key = (tkey, batch_size)
        if tkey is not None and key in cache:
            return cache[key]
        fname = (
            self.stats_filename(tkey, batch_size)
            if tkey is not None and hasattr(self, "stats_filename")
            else None
        )
        if fname is not None:
            for d in _stats_dirs():
                loaded = _load_stats(d / fname)
                if loaded is not None:
                    cache[key] = loaded
                    return cache[key]
        csum = csq = None
        cnt = 0
        for x, _ in self.iter_batches(batch_size):
            if transform is not None:
                x = transform(x)
            s = x.sum(axis=(0, 1), dtype=np.float64)
            q = (x.astype(np.float64) ** 2).sum(axis=(0, 1))
            csum = s if csum is None else csum + s
            csq = q if csq is None else csq + q
            cnt += x.shape[0] * x.shape[1]
        assert csum is not None and csq is not None  # a non-empty view yields >=1 batch
        mean = (csum / cnt).astype(np.float32)
        std = np.sqrt(np.maximum(csq / cnt - (csum / cnt) ** 2, 0)).astype(np.float32) + 1e-6
        result = (mean[None, None, :], std[None, None, :])
        if tkey is None:
            # No stable identity for this transform ⇒ neither memoize nor persist. Recomputing is
            # slower but correct; a wrong cache hit is not.
            return result
        cache[key] = result
        if fname is not None:
            for d in _stats_dirs():  # best-effort: a full/unwritable FS never kills training
                try:
                    _save_stats_atomic(d / fname, cache[key][0], cache[key][1])
                except OSError:
                    pass
        return cache[key]

    @property
    def features(self) -> np.ndarray:
        """Materialize the full (n,l,c) array (small views / tests only — streams via chunks)."""
        if self._dense is not None:
            return self._dense
        parts = [
            _observe_chunked(s.ws, s.rows, s.fault_lines, s.td, s.grid, s.near_end)
            for s in self._sources
        ]
        return np.concatenate(parts, axis=0) if parts else np.empty((0, 0, 0), np.float32)


@dataclass
class Task:
    definition: TaskDefinition
    protocol: str
    grid: str
    version: str
    train: TaskView
    val: TaskView
    test: TaskView  # in-distribution held-out (adapt_grid family)
    benchmark: TaskView  # cross-distribution held-out (benchmark family)
    # Input-data provenance for the run record: sha256 of the split files actually read plus the
    # manifest fingerprints of the windowed partitions actually loaded (see load_task).
    provenance: dict = field(default_factory=dict)

    def view(self, name: str) -> TaskView:
        return {
            "train": self.train,
            "val": self.val,
            "test": self.test,
            "benchmark": self.benchmark,
        }[name]


def load_arrays(
    task: Task, split: str = "train", *, max_bytes: int = 2 * 1024**3
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Materialize one split as ``(X, y, metadata)`` — the convenient BYO entry point.

    ``X`` is the dense ``(n, l, c)`` float32 feature array of the task's observability view,
    ``y`` the label vector, ``metadata`` the per-window frame (``sample_id``/``window_idx`` keys
    for :func:`~evemtbench.evaluation.evaluate`). Guarded: a view whose dense array would exceed
    ``max_bytes`` (default 2 GB) **raises loudly** instead of silently swallowing RAM — the big
    ``global`` views are meant to be streamed with ``task.view(split).iter_batches(batch_size)``
    (see ``examples/byo_model.py``). Raise ``max_bytes`` explicitly if you truly have the memory.
    """
    view = task.view(split)
    n, length, c = view.feature_shape
    need = n * length * c * 4  # float32
    if need > max_bytes:
        raise MemoryError(
            f"{task.definition.task_id}/{split} would materialize {need / 1e9:.1f} GB "
            f"({n}x{length}x{c} float32) > max_bytes={max_bytes / 1e9:.1f} GB. Stream it with "
            f"task.view({split!r}).iter_batches(batch_size), or pass a larger max_bytes."
        )
    return view.features, view.labels, view.metadata


def _observe(
    x_rows: np.ndarray,
    fault_lines: np.ndarray,
    td: TaskDefinition,
    grid: str,
    feature_names: list[str],
    channels: list[dict] | None = None,
    *,
    view: str | None = None,
    near_end: np.ndarray | None = None,
) -> np.ndarray:
    """Project masked window rows to the task's observability view.

    ``x_rows`` is (b, l, f) — already restricted to the kept (masked) windows.

    For the portable ``double_line`` view (multi_grid corpus) the window is already the 2-terminal
    line, so LOCAL/LINE are a bus-canonicalised slice (``near_end``) — no oracle. Otherwise project
    to the grid's main-line universe (from ``feature_names``), then slice per the observability
    oracle: GLOBAL -> all line cubicles; LINE -> the faulted line's 2 terminals (12 ch); LOCAL ->
    the near terminal (6 ch). LOCAL/LINE select per row using the episode-constant ``fault_lines``;
    all masked rows have a defined ``y_fault_line`` (the valid-sample rules guarantee it).
    """
    if line_view.is_double_line_view(feature_names, view):
        return line_view.project_double_line(x_rows, td.observability.value, near_end)
    oracle = line_view.build_oracle(grid, feature_names, channels)
    xp = x_rows[:, :, list(oracle.line_channels)]  # (b, l, n_line_cubicles*6)
    obs = td.observability.value
    if obs == "global":
        return np.ascontiguousarray(xp, dtype=np.float32)
    width = line_view.CHANNELS_PER_CUBICLE * (1 if obs == "local" else 2)
    out = np.zeros((xp.shape[0], xp.shape[1], width), dtype=np.float32)
    fl = np.asarray(fault_lines, dtype=object)
    for val in pd.unique(fl):
        cols = oracle.observation_columns(str(val), obs)
        idx = np.flatnonzero(fl == val)
        out[idx] = xp[idx][:, :, cols]
    return out


_OBSERVE_CHUNK = 8192  # rows per memmap read+project pass (bounds the peak transient)


def _observe_chunked(
    ws: WindowSet,
    rows: np.ndarray,
    fault_lines: np.ndarray,
    td: TaskDefinition,
    grid: str,
    near_end: np.ndarray | None = None,
) -> np.ndarray:
    """Read the kept windows from the memmap and project to the observability view in chunks.

    Reading ``ws.X[rows]`` for all kept rows at once materialises every on-disk channel
    (e.g. 270 for the MV grid) before the projection drops most of them — the dominant memory
    peak on the global view. Chunking caps that transient to one chunk's worth.
    """
    if len(rows) == 0:
        # derive the per-row output width by projecting an empty batch
        empty = np.zeros((0, ws.X.shape[1], ws.X.shape[2]), dtype=np.float32)
        return _observe(
            empty,
            np.asarray([], dtype=object),
            td,
            grid,
            ws.feature_names,
            ws.channels,
            view=ws.view,
            near_end=(None if near_end is None else near_end[:0]),
        )
    parts = []
    for start in range(0, len(rows), _OBSERVE_CHUNK):
        sl = slice(start, start + _OBSERVE_CHUNK)
        x_rows = np.asarray(ws.X[rows[sl]])  # only this chunk's full-channel windows in RAM
        parts.append(
            _observe(
                x_rows,
                fault_lines[sl],
                td,
                grid,
                ws.feature_names,
                ws.channels,
                view=ws.view,
                near_end=(None if near_end is None else near_end[sl]),
            ).astype(np.float32, copy=False)
        )
    return np.concatenate(parts, axis=0)


def add_zero_sequence(x: np.ndarray) -> np.ndarray:
    """Append per-cubicle zero-sequence channels I₀, U₀ to ``x`` of shape (..., c).

    Each group of 6 channels is [I_L1, I_L2, I_L3, U_L1, U_L2, U_L3]. Two channels are
    appended per cubicle: I₀ = mean(I_L1..L3), U₀ = mean(U_L1..L3) — the classical relay
    inputs used for ground-fault (HIF) detection. Returns (..., c + 2*(c//6)).
    """
    assert x.shape[-1] % 6 == 0, f"channel count must be multiple of 6; got {x.shape[-1]}"
    parts = [x]
    for i in range(x.shape[-1] // 6):
        b = i * 6
        parts.append(x[..., b : b + 3].mean(-1, keepdims=True))  # I₀
        parts.append(x[..., b + 3 : b + 6].mean(-1, keepdims=True))  # U₀
    return np.concatenate(parts, axis=-1)


def normalize_per_window(x: np.ndarray) -> np.ndarray:
    """Per-window, per-channel z-score: standardize each window ``(b, l, c)`` by its OWN mean/std
    over the time axis. Instance normalization — leakage-free and voltage/scale invariant, so a
    model pretrained at one voltage level transfers to another without target statistics. This is
    the default input transform for the cross-grid transfer protocols; see ``docs/protocol.md``."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return ((x - mu) / (sd + 1e-6)).astype(np.float32, copy=False)


def build_input_transform(
    zero_seq: bool = False, normalize: str = "none"
) -> Callable[[np.ndarray], np.ndarray] | None:
    """Compose the per-batch input transforms a baseline applies before standardization.

    Order: zero-sequence augmentation first (physical derived channels), then per-window z-score.
    ``normalize`` ∈ {``none``, ``per_window``}. Returns ``None`` when no transform is needed. The
    composed callable carries a stable ``__name__`` so :meth:`TaskView.channel_stats` memoization
    keys correctly (distinct transforms never collide)."""
    fns: list[Callable[[np.ndarray], np.ndarray]] = []
    if zero_seq:
        fns.append(add_zero_sequence)
    if normalize == "per_window":
        fns.append(normalize_per_window)
    elif normalize not in ("none", ""):
        raise ValueError(f"unknown normalize {normalize!r}; expected 'none' or 'per_window'")
    if not fns:
        return None
    if len(fns) == 1:
        return fns[0]

    def composed(x: np.ndarray) -> np.ndarray:
        for f in fns:
            x = f(x)
        return x

    composed.__name__ = "+".join(f.__name__ for f in fns)
    return composed


_LABEL_DERIVERS = {
    Function.FAULT_DETECTION: derive_fault_active_series,
    Function.FAULT_CLASS: derive_fault_class_series,
    Function.FAULT_LOCATION: derive_fault_location_series,
    Function.EVENT_STATE: derive_event_state_series,
    Function.EVENT_DETECTION: derive_event_present_series,
    Function.SWITCH_DETECTION: derive_switch_present_series,
    Function.EVENT_CLASSIFICATION: derive_event_type_series,
    Function.SWITCH_TYPE: derive_switch_type_series,
    Function.FAULT_GROUNDED: derive_fault_grounded_series,
    Function.FAULT_PHASE: derive_fault_phase_series,
    Function.FAULT_CATEGORY: derive_fault_category_series,
    Function.FAULT_ORIGIN: derive_fault_origin_series,
}


def _derive_labels(ws: WindowSet, td: TaskDefinition) -> np.ndarray:
    deriver = _LABEL_DERIVERS.get(td.function)
    if deriver is None:
        raise NotImplementedError(f"label derivation for {td.function} not implemented yet")
    return deriver(ws.y)


class _WindowsCache:
    """Loads each (grid, family) windowed partition once per ``load_task`` call.

    ``resolver`` maps a grid key to its :class:`GridSpec` (lets fixtures use toy grids that are
    not in the global registry). ``directory`` is only meaningful for a single-grid load.
    """

    def __init__(self, root, directory, resolver: dict[str, GridSpec]):
        self._root = root
        self._directory = directory
        self._resolver = resolver
        self._cache: dict[tuple[str, str], WindowSet] = {}

    def get(self, grid: str, family: str) -> WindowSet:
        key = (grid, family)
        if key not in self._cache:
            gspec = self._resolver.get(grid) or get_grid(grid)
            self._cache[key] = load_windows(
                gspec, family, root=self._root, directory=self._directory
            )
        return self._cache[key]


def _double_line_near_end(y: pd.DataFrame, rows: np.ndarray) -> np.ndarray | None:
    """Per-row terminal order for the double_line view: 0 if EndA is the smaller-bus terminal, 1 if
    EndB is. Canonicalises ``[near, far]`` to bus-sorted order (matching ``line_view._termkey``) so
    the projected channels align with the reference grids. Uses ``bus_a``/``bus_b`` (EndA<->bus_a,
    EndB<->bus_b by producer convention); returns ``None`` (keep on-disk EndA=near order) if the bus
    columns are absent or non-numeric."""
    if "bus_a" not in y.columns or "bus_b" not in y.columns:
        return None
    ba = pd.to_numeric(pd.Series(y["bus_a"].to_numpy()[rows]), errors="coerce").to_numpy()
    bb = pd.to_numeric(pd.Series(y["bus_b"].to_numpy()[rows]), errors="coerce").to_numpy()
    near_is_b = (bb < ba) & np.isfinite(ba) & np.isfinite(bb)
    return near_is_b.astype(np.int8)


def _source_channels(s: _ViewSource) -> int:
    """Projected channel count for one source: double_line slice (6/12) vs. main-line oracle."""
    obs = s.td.observability.value
    if line_view.is_double_line_view(s.ws.feature_names, s.ws.view):
        return line_view.double_line_channels(obs)
    oracle = line_view.build_oracle(s.grid, s.ws.feature_names, s.ws.channels)
    return (
        oracle.n_line_cubicles * line_view.CHANNELS_PER_CUBICLE
        if obs == "global"
        else (6 if obs == "local" else 12)
    )


def _build_view(name, triples, td, cache: _WindowsCache) -> TaskView:
    """Build a TaskView from ``(family, grid, sim_idx)`` triples, grouping by (family, grid)."""
    if not triples:
        # Empty partition (e.g. a grid whose benchmark family isn't windowed yet): an empty
        # view the runner skips.
        empty_md = pd.DataFrame({c: [] for c in _META_COLS + ["grid"]})
        return TaskView(name, [], np.empty((0,)), empty_md, dense=np.empty((0, 0, 0), np.float32))

    by_group: dict[tuple[str, str], set[int]] = {}
    for family, grid, sim_idx in triples:
        by_group.setdefault((family, grid), set()).add(sim_idx)

    sources, label_parts, meta_parts = [], [], []
    for (family, grid), sids in by_group.items():
        ws = cache.get(grid, family)
        # episode membership AND task-specific valid-sample rule (e.g. fault-active only)
        mask = np.isin(ws.y["sample_id"].to_numpy(), list(sids)) & valid_sample_mask(ws.y, td)
        rows = np.flatnonzero(mask)
        fault_lines = ws.y["y_fault_line"].to_numpy()[rows]
        # Portable double_line corpus: canonicalise the two terminals by bus id so channel k means
        # the same quantity as the reference-grid oracle's [near, far] ordering.
        near_end = (
            _double_line_near_end(ws.y, rows)
            if line_view.is_double_line_view(ws.feature_names, ws.view)
            else None
        )
        # Source-backed (lazy): features are read + projected from the memmap per batch when the
        # baseline streams the view, so the wide global view is never one dense in-RAM array.
        sources.append(
            _ViewSource(
                ws=ws, rows=rows, fault_lines=fault_lines, td=td, grid=grid, near_end=near_end
            )
        )
        label_parts.append(_derive_labels(ws, td)[rows])
        keep_cols = [c for c in _META_COLS if c in ws.y.columns]
        md = ws.y.loc[mask, keep_cols].copy()
        md["grid"] = grid
        meta_parts.append(md)

    # All sources must share the projected width (single-grid for held_out; multi-grid for
    # transfer/cross-topology must agree by construction). Cheap to check via a per-source count.
    widths = {_source_channels(s) for s in sources}
    if len(widths) > 1:
        raise ValueError(
            f"projected width differs across sources for observability {td.observability.value} "
            f"(grids {[s.grid for s in sources]}); cross-grid views require matching widths."
        )
    labels = np.concatenate(label_parts, axis=0)
    metadata = pd.concat(meta_parts, ignore_index=True)
    return TaskView(name, sources, labels, metadata)


# Default pre-training grid for the multi_grid protocols when no grid is given (the first — and,
# for now, only — windowed grid of the multi_grid corpus).
_DEFAULT_PRETRAIN_GRID = "template_20kv"


def _protocol_triples(
    protocol: str, grid_key: str, *, version: str, splits_root
) -> dict[str, list[tuple[str, str, int]]]:
    """Assemble the (family, grid, sim_idx) triples per split for a protocol.

    * ``held_out`` / ``transfer_finetune`` — the target grid's committed held_out split
      (train/val = adapt_grid to fit/fine-tune on; test/benchmark = held-out target families).
    * ``transfer_zeroshot`` — same held_out split but with EMPTY train/val (no target training;
      the model arrives pretrained), keeping the target ``test``/``benchmark`` eval views.
    * ``multi_grid_pretrain`` — the pre-training grid's committed multi_grid split (in-corpus
      train/val/test; empty benchmark).
    """
    if protocol in ("held_out", "transfer_finetune"):
        return read_held_out_splits(grid_key, version=version, splits_root=splits_root)
    if protocol == "transfer_zeroshot":
        s = read_held_out_splits(grid_key, version=version, splits_root=splits_root)
        return {"train": [], "val": [], "test": s["test"], "benchmark": s["benchmark"]}
    if protocol == "multi_grid_pretrain":
        return read_splits(protocol, grid_key, version=version, splits_root=splits_root)
    raise ValueError(f"protocol {protocol!r} has no split builder in load_task")


def _splits_provenance(protocol: str, grid_key: str, *, version: str, splits_root) -> dict:
    """sha256 of each committed split file this load reads — artifact-backed input provenance.

    The transfer protocols reuse the target grid's held_out split; multi_grid_pretrain reads its
    own dir (mirrors :func:`_protocol_triples`). Files are KB-sized, so hashing is negligible."""
    split_proto = (
        "held_out"
        if protocol in ("held_out", "transfer_finetune", "transfer_zeroshot")
        else protocol
    )
    d = split_dir(split_proto, grid_key, version=version, splits_root=splits_root)
    files = {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(d.glob("*.txt")) + sorted(d.glob("meta.json"))
        if f.is_file()
    }
    return {"protocol_dir": split_proto, "dir": str(d), "version": version, "files": files}


def _attach_aux(view: TaskView, grid: str, family: str, which: tuple[str, ...]) -> None:
    """Best-effort join of aux per-episode settings into ``view.metadata`` (metadata only — never a
    model feature, so no leakage). Silently skips if the aux overlay is absent (the default no-aux
    path). Currently supports ``"resistance"`` (enables the ``resistance_bin`` stratum)."""
    if len(view) == 0:
        return
    from ..data import aux_labels

    if "resistance" in which:
        try:
            settings = aux_labels.load_settings(grid, family)
        except (FileNotFoundError, RuntimeError):
            return  # no aux overlay / no aux root -> leave metadata untouched
        rcol = aux_labels.fault_resistance_column(settings)
        if rcol is not None:
            view.metadata = aux_labels.join_to_metadata(
                view.metadata, grid, family, [rcol], rename={rcol: "resistance"}
            )


def load_task(
    task_id: str,
    protocol: str,
    *,
    grid: str | GridSpec | None = None,
    version: str = SPLITS_VERSION,
    root: str | os.PathLike | None = None,
    splits_root: str | os.PathLike | None = None,
    directory: str | os.PathLike | None = None,
    extra_grids: dict[str, GridSpec] | None = None,
    attach_aux: tuple[str, ...] = (),
) -> Task:
    """Load a benchmark task under a protocol. ``grid`` is the target grid (defaults to the
    task's ``default_grid``). ``root``/``splits_root``/``directory`` are test/layout overrides.
    ``extra_grids`` seeds the grid resolver with non-registry grids (custom grids / fixtures).
    ``attach_aux`` optionally joins aux settings columns into each view's *metadata* (e.g.
    ``("resistance",)`` for resistance-binned stratification) when the aux overlay exists; it never
    affects features and silently no-ops if the overlay is absent."""
    td = get_task(task_id)
    get_protocol(protocol)
    if protocol not in td.protocols:
        raise ValueError(f"task {task_id!r} does not support protocol {protocol!r}")
    if grid is None and protocol == "multi_grid_pretrain":
        grid = _DEFAULT_PRETRAIN_GRID
    gspec = get_grid(grid if grid is not None else td.default_grid)

    triples = _protocol_triples(protocol, gspec.key, version=version, splits_root=splits_root)

    resolver = {gspec.key: gspec}  # lets fixtures use toy grids absent from the registry
    if extra_grids:
        resolver.update(extra_grids)
    cache = _WindowsCache(root, directory, resolver)
    views = {name: _build_view(name, triples[name], td, cache) for name in HELD_OUT_SPLIT_NAMES}
    if attach_aux:
        for name, view in views.items():
            # each triple is (family, grid, sim_idx); a held_out split is single-family
            family = triples[name][0][0] if triples[name] else None
            if family:
                _attach_aux(view, gspec.key, family, attach_aux)
    provenance = {
        "splits": _splits_provenance(protocol, gspec.key, version=version, splits_root=splits_root),
        # exactly the partitions this load actually memory-mapped (transfer_zeroshot loads no
        # train/val family, and that is what gets recorded)
        "partitions": sorted(
            (ws.provenance for ws in cache._cache.values() if ws.provenance),
            key=lambda pr: (pr["grid"], pr["family"]),
        ),
    }
    return Task(
        definition=td,
        protocol=protocol,
        grid=gspec.key,
        version=version,
        train=views["train"],
        val=views["val"],
        test=views["test"],
        benchmark=views["benchmark"],
        provenance=provenance,
    )
