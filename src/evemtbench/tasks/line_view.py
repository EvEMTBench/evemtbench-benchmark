"""Main-line channel projection + the faulted-element oracle (LOCAL/LINE/GLOBAL settings).

The grids write all cubicle measurements on disk (each cubicle = 6 channels
``[Isec L1,L2,L3, Usec L1,L2,L3]``). The benchmark observes only the **main lines** — the
line bays, i.e. every cubicle measuring a transmission/distribution *line* — dropping
transformer / load / IBR / generator cubicles. That line universe is the **GLOBAL** view; its
width is grid-dependent (variable topology is the point of the multi-grid benchmark).

Within that universe the three observability settings form a nested oracle centred on the
faulted element (known per episode from ``y_fault_line``, which is episode-constant):

* ``LOCAL``  — the near terminal of the faulted line (1 cubicle, 6 ch).
* ``LINE``   — both terminals of the faulted line (2 cubicles, 12 ch) = the differential pair.
* ``GLOBAL`` — every main-line cubicle.

The whole oracle is **derived from the windows ``feature_names``** (no external topology file):
a line cubicle is named ``Bus_<bus>_Line_<lineid>`` and a line's two terminals are the two
cubicles that share a ``lineid`` at different buses — exactly the current-differential pair. A
bus fault (``MainBus*``) has no single line; it maps to the line bays physically at that bus
(the relays electrically adjacent to the fault) — a documented, revisable convention.

Projected ("proj") indices number the kept main-line cubicles 0..K-1 in on-disk order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

CHANNELS_PER_CUBICLE = 6  # [Isec L1,L2,L3, Usec L1,L2,L3]

# The portable multi-grid pre-training view: a fixed 12-channel EndA/EndB layout
# ``[EndA: Isec L1,L2,L3, Usec L1,L2,L3][EndB: ...]`` (contract ``extra.view``). Line/grid
# identity lives in ``y`` (graph_id, line_id, bus_a, bus_b), not on the channel axis — so the
# topology oracle below cannot (and need not) parse it. It is already the LINE view; LOCAL is one
# terminal. GLOBAL is undefined (no full topology in a single window).
DOUBLE_LINE_VIEW = "double_line"

# feature name = "<cubicle_key>_<Isec|Usec>_<L1|L2|L3>_<A|V>"; the cubicle_key is everything
# before the measurement token (``_Isec_`` / ``_Usec_``), e.g. "Bus_<bus>_Line_<lineid>".
_LINE_KEY = re.compile(r"^Bus_(?P<bus>.+?)_Line_(?P<lineid>.+)$")
_TRAILING_INT = re.compile(r"(\d+)$")


def _cubicle_key(feature_name: str) -> str:
    """Cubicle identity = feature name up to its measurement token (robust to trailing units)."""
    for tok in ("_Isec_", "_Usec_"):
        i = feature_name.find(tok)
        if i != -1:
            return feature_name[:i]
    return feature_name


def _bus_norm(bus: str) -> str:
    """Normalise a bus token to its trailing integer when present (``MainBus1``/``1`` -> ``1``)."""
    m = _TRAILING_INT.search(bus)
    return m.group(1) if m else bus


def _cubicle_keys(feature_names: list[str]) -> list[str]:
    """One key per contiguous 6-channel cubicle block, in on-disk order."""
    if len(feature_names) % CHANNELS_PER_CUBICLE != 0:
        raise ValueError(
            f"feature count {len(feature_names)} is not a multiple of {CHANNELS_PER_CUBICLE}"
        )
    keys = []
    for c in range(len(feature_names) // CHANNELS_PER_CUBICLE):
        keys.append(_cubicle_key(feature_names[c * CHANNELS_PER_CUBICLE]))
    return keys


@dataclass(frozen=True)
class GridOracle:
    """Derived observability oracle for one grid (cached by grid + feature_names)."""

    grid: str
    line_channels: tuple[int, ...]  # on-disk channel indices of the main-line universe
    n_line_cubicles: int
    # faulted element label -> (near_proj, far_proj) cubicle pair in projected space
    _pairs: dict[str, tuple[int, int]]

    def observation_columns(self, fault_line: str, observability: str) -> list[int]:
        """Projected-space column indices to select for one row.

        ``observability`` ∈ {local, line, global} (case-insensitive). LOCAL/LINE require a
        known ``fault_line`` (the per-episode oracle); GLOBAL ignores it.
        """
        obs = observability.lower()
        if obs == "global":
            return list(range(self.n_line_cubicles * CHANNELS_PER_CUBICLE))
        pair = self._pairs.get(fault_line)
        if pair is None:
            raise KeyError(
                f"unknown y_fault_line {fault_line!r} for grid {self.grid!r}; "
                f"known: {sorted(self._pairs)}"
            )
        near, far = pair
        if obs == "local":
            return _cubicle_cols(near)
        if obs == "line":
            return _cubicle_cols(near) + _cubicle_cols(far)
        raise ValueError(f"unknown observability {observability!r}; expected local|line|global")


def _cubicle_cols(proj_cub: int) -> list[int]:
    """Projected-space column indices for one cubicle's 6 channels."""
    start = proj_cub * CHANNELS_PER_CUBICLE
    return list(range(start, start + CHANNELS_PER_CUBICLE))


_CACHE: dict[tuple[str, tuple[str, ...]], GridOracle] = {}


def build_oracle(
    grid: str,
    feature_names: list[str],
    channels: list[dict] | None = None,
) -> GridOracle:
    """Build (and cache) the main-line projection + faulted-element oracle.

    Prefers the structured ``channels`` descriptor (contract C-META-2) when available — the
    robust, grammar-independent path — and falls back to parsing ``feature_names`` with the
    ``Bus_<bus>_Line_<id>`` regex when a dataset predates the descriptor. Both paths feed the
    identical assembly, so they yield equivalent oracles.
    """
    cache_key = (grid, tuple(feature_names))
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    specs = (
        _line_specs_from_channels(channels) if channels else _line_specs_from_names(feature_names)
    )
    if not specs:
        raise ValueError(
            f"no line-bay cubicles found for grid {grid!r} (via "
            f"{'channels descriptor' if channels else 'feature_names'}); "
            "cannot build the main-line observability oracle"
        )
    oracle = _assemble_oracle(grid, specs)
    _CACHE[cache_key] = oracle
    return oracle


# One line-bay cubicle: (cubicle_index, bus, lineid). Non-line cubicles are excluded.
def _line_specs_from_names(feature_names: list[str]) -> list[tuple[int, str, str]]:
    specs: list[tuple[int, str, str]] = []
    for c, key in enumerate(_cubicle_keys(feature_names)):
        m = _LINE_KEY.match(key)
        if m is None:
            continue  # transformer/load/IBR/... -> dropped from the main-line universe
        specs.append((c, _bus_norm(m.group("bus")), m.group("lineid")))
    return specs


def _line_specs_from_channels(channels: list[dict]) -> list[tuple[int, str, str]]:
    specs: list[tuple[int, str, str]] = []
    n_cub = len(channels) // CHANNELS_PER_CUBICLE
    for c in range(n_cub):
        head = channels[c * CHANNELS_PER_CUBICLE]
        if head.get("element_type") == "Line":
            specs.append((c, _bus_norm(str(head.get("bus"))), str(head.get("element_id"))))
    return specs


def _assemble_oracle(grid: str, specs: list[tuple[int, str, str]]) -> GridOracle:
    """Assemble the oracle from line-bay cubicle specs (shared by both discovery paths)."""
    line_channels: list[int] = []
    line_terminals: dict[str, list[tuple[str, int]]] = {}  # lineid -> [(bus, proj)]
    bus_lines: dict[str, list[int]] = {}  # bus -> [proj] (line bays at that bus)
    for proj, (c, bus, lineid) in enumerate(specs):
        line_channels.extend(range(c * CHANNELS_PER_CUBICLE, (c + 1) * CHANNELS_PER_CUBICLE))
        line_terminals.setdefault(lineid, []).append((bus, proj))
        bus_lines.setdefault(bus, []).append(proj)

    pairs: dict[str, tuple[int, int]] = {}

    def _termkey(t: tuple[str, int]) -> tuple[int, int]:
        bus, proj = t
        return (int(bus) if str(bus).isdigit() else 0, proj)

    for lineid, terms in line_terminals.items():
        ordered = [p for _, p in sorted(terms, key=_termkey)]
        near = ordered[0]
        far = ordered[1] if len(ordered) > 1 else ordered[0]
        pairs[f"MainLn{lineid}"] = (near, far)
    for bus, projs in bus_lines.items():
        near = projs[0]
        far = projs[1] if len(projs) > 1 else projs[0]
        pairs[f"MainBus{bus}"] = (near, far)

    return GridOracle(
        grid=grid,
        line_channels=tuple(line_channels),
        n_line_cubicles=len(specs),
        _pairs=pairs,
    )


# --- portable double-line view (multi_grid pre-training corpus) --------------------------------
_END_KEY = re.compile(r"^End[AB]_")


def is_double_line_view(feature_names: list[str], view: str | None = None) -> bool:
    """True for the portable 12-channel EndA/EndB layout — by explicit ``view`` tag (from the
    windows' ``meta.extra.view``) or, as a fallback, by ``End[AB]_`` feature names."""
    if view == DOUBLE_LINE_VIEW:
        return True
    return bool(feature_names) and all(_END_KEY.match(n) for n in feature_names)


def double_line_channels(observability: str) -> int:
    """Projected channel count for the double-line view: LOCAL=6 (one terminal), LINE=12 (both).
    GLOBAL is undefined (a portable 2-terminal view carries no full topology)."""
    obs = observability.lower()
    if obs == "local":
        return CHANNELS_PER_CUBICLE
    if obs == "line":
        return 2 * CHANNELS_PER_CUBICLE
    raise ValueError(
        f"observability {observability!r} is not available on the double_line view "
        "(portable 2-terminal view has no GLOBAL); use 'local' or 'line'."
    )


def project_double_line(
    x_rows: np.ndarray, observability: str, near_end: np.ndarray | None = None
) -> np.ndarray:
    """Project a batch of double-line windows ``(b, l, 12)`` to LOCAL (6) or LINE (12).

    Channels are two 6-channel terminal blocks ``[EndA][EndB]``. To align with the reference-grid
    oracle — whose LINE view orders terminals ``[smaller-bus, larger-bus]`` and whose LOCAL view is
    the smaller-bus terminal (``line_view._termkey``) — pass ``near_end`` (per-row: 0 if EndA is the
    smaller-bus terminal, 1 if EndB is). Rows are then canonicalised to ``[near, far]`` so channel
    *k* means the same physical quantity in the pre-training and reference-grid views. ``near_end``
    of ``None`` keeps the on-disk EndA=near order (used only when bus ids are unavailable).
    """
    obs = observability.lower()
    if obs == "global":
        raise ValueError("double_line view has no GLOBAL observability; use 'local' or 'line'.")
    if x_rows.ndim != 3 or x_rows.shape[2] != 2 * CHANNELS_PER_CUBICLE:
        raise ValueError(
            f"double_line view expects (b, l, 12) windows; got shape {tuple(x_rows.shape)}"
        )
    end_a = x_rows[:, :, :CHANNELS_PER_CUBICLE]
    end_b = x_rows[:, :, CHANNELS_PER_CUBICLE:]
    if near_end is None:
        near, far = end_a, end_b
    else:
        pick_b = np.asarray(near_end).astype(bool)[:, None, None]  # (b,1,1): True => EndB is near
        near = np.where(pick_b, end_b, end_a)
        far = np.where(pick_b, end_a, end_b)
    if obs == "local":
        return np.ascontiguousarray(near, dtype=np.float32)
    return np.ascontiguousarray(np.concatenate([near, far], axis=-1), dtype=np.float32)
