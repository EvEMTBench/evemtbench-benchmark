"""Build the results-audit site — audit/*.html, a reviewer-facing provenance dashboard.

Renders hpc/audit_manifest.py (claims/pillars/gates/prose — no numbers) plus the committed
artifacts (outputs/**/result.json + run_record.json, splits, contract, lockfile) into a static,
dependency-free multi-page site: index, claim ledger, coverage matrix, verification-gate code
tour, environment, reproduce, provenance, and per-task / per-grid / per-baseline drill-downs.
Modeled on the UnderstandingVCS results audit (docs/results_audit/).

``build_audit_pages`` is pure (no I/O beyond reads, no timestamps) so the freshness gate
(hpc/check_results_fresh.py) can byte-compare the committed pages against a rebuild. Only
``main()`` writes files and stamps the single volatile "updated <ts> · git <sha>" line.

    PYTHONPATH=src python hpc/make_audit.py            # outputs/ -> audit/*.html
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import string
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO / "src", _REPO / "hpc"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import audit_manifest as MAN  # noqa: E402  (sibling hpc module)
import verify_results_env as VRE  # noqa: E402  (sibling hpc module; lock parser + package list)
from evemtbench.contract import CONTRACT_SHA256, EXPECTED_CONTRACT_VERSION  # noqa: E402
from evemtbench.coverage import (  # noqa: E402
    ALL_BASELINES,
    DEEP_BASELINES,
    LEARNED_BASELINES,
    PRETRAIN_GRID,
    PRETRAIN_GRIDS,
    REFERENCE_GRIDS,
    TRANSFER_PROTOCOLS,
    Cell,
    _arg_line,
    check_floor,
    expected_test_sets,
    iter_cells,
    protocol_grid_applies,
    scan,
)
from evemtbench.data.roots import GRIDS  # noqa: E402
from evemtbench.evaluation.metrics import METRIC_DIRECTION  # noqa: E402
from evemtbench.evaluation.significance import bootstrap_ci, ci95_halfwidth  # noqa: E402
from evemtbench.leaderboard import (  # noqa: E402
    BASE_CSS,
    DOCS_BASE,
    FAVICON,
    SORT_JS,
    Row,
    _fmt_params,
    _fmt_secs,
    _grid_sort_key,
    _grid_spec_label,
    scan_results,
)
from evemtbench.splits.loader import read_held_out_splits  # noqa: E402
from evemtbench.tasks.registry import get_task, list_tasks  # noqa: E402
from evemtbench.version import (  # noqa: E402
    BENCHMARK_VERSION,
    CURRENT_RESULTS_VERSIONS,
    MULTI_GRID_SPLITS_VERSION,
    SPLITS_VERSION,
    results_version,
)

PROTOCOLS = ("held_out", "multi_grid_pretrain", "transfer_zeroshot", "transfer_finetune")

# Representative Section-A cell per paper function: (short key, task id). double_line/held_out.
HEADLINE_CELLS = (
    ("fd", "fault_detection_local"),
    ("fc", "fault_classification_local"),
    ("fl", "fault_location_local"),
    ("ed", "event_detection_global"),
    ("ec", "event_classification_global"),
)
REP_GRID = "double_line"

# Every placeholder the manifest may use. compute_values() emits ALL of these (defaulting to
# "—" where outputs are absent), and test_audit_manifest asserts manifest placeholders ⊆ this.
VALUE_KEYS = tuple(
    [
        f"{k}_{s}"
        for k, _ in HEADLINE_CELLS
        for s in ("best_test", "best_bench", "best_name", "floor", "fit")
    ]
    + [
        "n_cells",
        "n_done",
        "n_pending",
        "n_failed",
        "n_all_cells",
        "n_all_done",
        "n_all_pending",
        "n_all_failed",
        "n_transfer_cells",
        "n_transfer_done",
        "n_pretrain_cells",
        "n_pretrain_done",
        "floor_n_groups",
        "floor_n_checked",
        "floor_n_violations",
        "floor_n_skipped",
        "n_run_records",
        "n_attested",
        "n_git_shas",
        "n_lockfile_shas",
        "n_hosts",
        "contract_version",
        "contract_sha",
        "n_tasks",
        "n_grids",
        "n_baselines",
    ]
)


# ---------------------------------------------------------------------------------------------
# Collectors (read-only)
# ---------------------------------------------------------------------------------------------
@dataclass
class CoverageView:
    version: str
    by_protocol: dict[str, dict[str, list[Cell]]]  # protocol -> {done|pending|failed: [Cell]}
    floor: dict  # check_floor() report

    def counts(self, protocol: str) -> tuple[int, int, int, int]:
        g = self.by_protocol.get(protocol, {})
        d, p, f = len(g.get("done", [])), len(g.get("pending", [])), len(g.get("failed", []))
        return d + p + f, d, p, f

    def all_cells(self) -> list[tuple[Cell, str]]:
        out = []
        for proto in PROTOCOLS:
            for status, cells in sorted(self.by_protocol.get(proto, {}).items()):
                out += [(c, status) for c in cells]
        return sorted(out, key=lambda cs: (cs[0].task, cs[0].protocol, cs[0].grid, cs[0].baseline))


@dataclass
class RunRecordStats:
    """Aggregated per-run provenance — the audit is a production reader of run_record.json."""

    n_records: int = 0
    n_attested: int = 0
    git_shas: list[str] = field(default_factory=list)
    lockfile_shas: list[str] = field(default_factory=list)
    pythons: list[str] = field(default_factory=list)
    torchs: list[str] = field(default_factory=list)
    hosts: list[str] = field(default_factory=list)  # populated by records that carry .host
    # Agreement between each record's independently captured package versions and the captured
    # results environment (env/results-v<version>.lock). Computed here rather than read from a
    # sidecar so the page stays a pure function of outputs/ + the repo, as the freshness gate needs.
    env_agree: int = 0
    env_disagree: int = 0
    # Per-cell provenance for the traceability index/links: cell key
    # "<task>/<protocol>/<grid>/<baseline_tag>" -> compact record fields (git_sha, wandb, args,
    # normalize, attestation, host, data, checkpoints, compute).
    by_cell: dict[str, dict] = field(default_factory=dict)


def collect_coverage(
    outputs: str | Path = "outputs", *, version: str | None = None
) -> CoverageView:
    out = Path(outputs)
    by_protocol = {p: scan(out, iter_cells(protocols=(p,)), version=version) for p in PROTOCOLS}
    floor = check_floor(out, iter_cells(protocols=("held_out",)), version=version)
    return CoverageView(version=version or BENCHMARK_VERSION, by_protocol=by_protocol, floor=floor)


def collect_run_records(
    outputs: str | Path = "outputs", *, version: str | None = None
) -> RunRecordStats:
    st = RunRecordStats()
    git_shas, lock_shas, pythons, torchs, hosts = set(), set(), set(), set(), set()
    # ``version=None`` (default) iterates every CURRENT results tree and keeps a record only if
    # its protocol is current IN that tree (results_version(protocol) == tree version) — the
    # superseded pretrain/transfer results still on disk under the older tree must not feed the
    # audit. An explicit ``version`` reads that single tree unfiltered (historical inspection).
    for tree in (version,) if version is not None else CURRENT_RESULTS_VERSIONS:
        root = Path(outputs) / tree
        # Reuse the verifier's parser + package list; duplicating them here is how the audit and
        # the check would drift apart. The env lock is per results tree (env/results-v<tree>.lock).
        _renv = _REPO / "env" / f"results-v{tree}.lock"
        _pins = VRE.parse_lock(_renv) if _renv.is_file() else {}
        for p in sorted(root.rglob("run_record.json")) if root.is_dir() else []:
            # Cell key relative to ITS tree root (<task>/<protocol>/<grid>/<baseline_tag>[...]):
            # the current trees' cells are disjoint by protocol, so keys never collide.
            try:
                cell_key = str(p.parent.relative_to(root))
            except ValueError:
                cell_key = None
            if version is None:
                parts = (cell_key or "").split("/")
                if len(parts) < 2 or results_version(parts[1]) != tree:
                    continue
            try:
                rec = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            st.n_records += 1
            att = rec.get("attestation") or {}
            if att.get("held_out_untouched"):
                st.n_attested += 1
            if rec.get("git_sha"):
                git_shas.add(rec["git_sha"])
            if rec.get("lockfile_sha256"):
                lock_shas.add(rec["lockfile_sha256"])
            if rec.get("python"):
                pythons.add(rec["python"])
            if rec.get("torch"):
                torchs.add(str(rec["torch"]))
            if _pins:
                got = rec.get("versions") or {}
                ok = all(got.get(pkg) == _pins.get(pkg) for pkg in VRE._SCORING_PKGS)
                rt = rec.get("torch")
                if rt is not None and _pins.get("torch") is not None and rt != _pins["torch"]:
                    ok = False
                if ok:
                    st.env_agree += 1
                else:
                    st.env_disagree += 1
            host = (rec.get("host") or {}).get("hostname")
            if host:
                # same shortening rule as _neutral_paths — FQDNs stay out of committed pages
                hosts.add(_HOST_SUFFIX.sub("", host))
            if cell_key is not None:
                st.by_cell[cell_key] = {
                    "git_sha": rec.get("git_sha"),
                    "wandb": rec.get("wandb"),
                    "args": rec.get("args"),
                    "normalize": rec.get("normalize"),
                    "attestation": rec.get("attestation"),
                    "host": rec.get("host"),
                    "data": rec.get("data"),
                    "checkpoints": rec.get("checkpoints"),
                    "compute": rec.get("compute"),
                }
    st.git_shas = sorted(git_shas)
    st.lockfile_shas = sorted(lock_shas)
    st.pythons = sorted(pythons)
    st.torchs = sorted(torchs)
    st.hosts = sorted(hosts)
    return st


# ---------------------------------------------------------------------------------------------
# Provenance index (B1) — the committed compact per-cell record that makes offline links resolve
# ---------------------------------------------------------------------------------------------
# outputs/ (incl. result.json / train.log / checkpoints) is git-ignored, so a repo-only audit site
# cannot link to it directly. This index is the COMMITTED bridge: one JSON per version, keyed by
# the cell path "<task>/<protocol>/<grid>/<baseline_tag>", carrying — from the (git-ignored)
# run_record.json + result.json — the config, provenance shas, per-test-set metric values, the
# exact reproduce command, the W&B run url, and the relative paths of every artifact. It is
# byte-deterministic (no timestamps copied; the run's git_sha is fixed at run time, not HEAD) so
# the freshness gate can byte-compare it like every other rendered view.


def _reproduce_cmd(task: str, protocol: str, grid: str, cfg: dict) -> str:
    """The verbatim runner invocation that (re)produces a cell, rebuilt from its recorded args
    (REFORMS 2e: an exact reproduction command for every number). ``PYTHONPATH=src`` prefix; the
    on-disk baseline tag's ``+zero_seq`` maps back to the ``--zero-seq`` flag on the raw baseline."""
    parts = [
        f"--task {task}",
        f"--protocol {protocol}",
        f"--grid {grid}",
        f"--baseline {cfg.get('baseline')}",
    ]
    if cfg.get("zero_seq"):
        parts.append("--zero-seq")
    seeds = cfg.get("seeds") or []
    if seeds:
        parts.append("--seeds " + " ".join(str(s) for s in seeds))
    if cfg.get("epochs") is not None:
        parts.append(f"--epochs {cfg['epochs']}")
    if str(protocol).startswith("transfer"):
        if cfg.get("pretrain_grid"):
            parts.append(f"--pretrain-grid {cfg['pretrain_grid']}")
        if cfg.get("pretrained_from"):
            parts.append(f"--pretrained-from {cfg['pretrained_from']}")
    return "PYTHONPATH=src python -m evemtbench.baselines.runner " + " ".join(parts)


def _metrics_by_cell(rows: list[Row], version: str | None) -> dict[str, dict]:
    """cell key -> {test_set: {metric: {mean, std, ci95}}, "_n": {test_set: seeds}} from Rows.

    Filtered to ``version`` (``None`` = each row's per-protocol CURRENT version,
    ``results_version(r.protocol)``): cell keys carry no version segment, so without this a
    leftover result.json from a superseded version (``scan_results`` reads every version present)
    could cross-attach its metrics to a same-keyed cell — the provenance index must carry only
    current numbers."""
    out: dict[str, dict] = {}
    for r in rows:
        if r.version != (results_version(r.protocol) if version is None else version):
            continue
        ck = f"{r.task}/{r.protocol}/{r.grid}/{r.baseline}"
        if r.protocol in TRANSFER_PROTOCOLS and r.source and r.source != PRETRAIN_GRID:
            ck += f"/pretrain-{r.source}"  # match the tokened run_record path (by_cell key)
        entry = out.setdefault(ck, {"_n": {}})
        entry[r.test_set] = {
            m: {k: v.get(k) for k in ("mean", "std", "ci95")} for m, v in r.metrics.items()
        }
        entry["_n"][r.test_set] = r.seeds
    return out


# run_record.json values are recorded on the compute node and carry machine-absolute paths and
# host FQDNs. The committed index is public (repro pack / public release), so every copied value
# is rewritten machine-neutral here — generically (filesystem-derived prefixes + shape rules, no
# environment variables), so a rebuild is byte-identical regardless of the builder's shell env.
# The /home rule is scoped to the NHR mount layout (/home/<mount>/<group>/<user>) on purpose: an
# unscoped 3-component rule would silently swallow real path components on ordinary
# /home/<user>/... layouts. An unmatched /home path survives into the index and is then caught by
# the release leak scans — fail closed, never mangled.
_HOME_PREFIX = re.compile(r"/home/(?:hpc|atuin|vault|woody)/[^/\s\"']+/[^/\s\"']+")
# node-local staging/scratch dirs embed the cluster username in the directory name itself
_TMP_PREFIX = re.compile(r"/(?:tmp|scratch|anvme|dev/shm|var/tmp)/[^/\s\"']+")
# The ONE hostname-shortening rule (also used by collect_run_records): strip the cluster domain
# so committed views carry only the node name. Extend the suffix list when a new cluster joins.
_HOST_SUFFIX = re.compile(r"\.(?:nhr|rrze)\.fau\.de\b")


def _neutral_prefixes() -> list[tuple[str, str]]:
    prefixes = [(str(_REPO) + "/", "")]
    out_resolved = (_REPO / "outputs").resolve()
    if out_resolved != _REPO / "outputs":
        prefixes.append((str(out_resolved) + "/", "outputs/"))
    return prefixes


_PREFIXES = _neutral_prefixes()


def _neutral_paths(value):
    """Recursively rewrite strings in a provenance value to be machine-neutral.

    Repo-absolute paths become repo-relative; paths under the resolved ``outputs`` tree become
    ``outputs/...`` (the symlink target differs from the repo path); NHR-shaped ``/home/...``
    prefixes become ``~``; node-local staging roots become ``$TMPDIR``; host FQDNs are shortened
    to the node name."""
    if isinstance(value, str):
        for prefix, repl in _PREFIXES:
            value = value.replace(prefix, repl)
        value = _HOME_PREFIX.sub("~", value)
        value = _TMP_PREFIX.sub("$TMPDIR", value)
        return _HOST_SUFFIX.sub("", value)
    if isinstance(value, dict):
        return {k: _neutral_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_neutral_paths(v) for v in value]
    return value


def provenance_cells(
    rows: list[Row], records: RunRecordStats, *, version: str | None = None
) -> dict[str, dict]:
    """The per-cell provenance map (Python dict, keyed by "<task>/<protocol>/<grid>/<tag>").

    Joins the run_record projection (``records.by_cell``) with the scanned metric values
    (``rows``) and derives the artifact paths, source-code paths and reproduce command. Shared by
    the committed JSON index (``build_provenance_index``) and the inline per-number trace links.
    All values are machine-neutral (``_neutral_paths``) — the index is a public artifact."""
    mbc = _metrics_by_cell(rows, version)
    cells: dict[str, dict] = {}
    for ck, rec in sorted(records.by_cell.items()):
        parts = ck.split("/")
        source = None
        if len(parts) == 5 and parts[4].startswith("pretrain-"):
            # tokened transfer cell: <task>/<proto>/<grid>/<baseline>/pretrain-<source>
            task, protocol, grid, baseline_tag, src_seg = parts
            source = src_seg.replace("pretrain-", "")
        elif len(parts) == 4:
            task, protocol, grid, baseline_tag = parts
        else:  # defensive: only well-formed cell paths
            continue
        cfg = dict(rec.get("args") or {})
        baseline_arg = cfg.get("baseline") or baseline_tag.split("+")[0]
        # version=None => the cell's artifacts live in its protocol's CURRENT tree (held_out
        # cells point into the frozen older tree, pretrain/transfer into the current one).
        base = f"outputs/{results_version(protocol) if version is None else version}/{ck}"
        cell_metrics = {k: v for k, v in mbc.get(ck, {}).items() if k != "_n"}
        n_by_ts = mbc.get(ck, {}).get("_n", {})
        ckpts = rec.get("checkpoints") or {}
        wandb = rec.get("wandb") or {}
        cells[ck] = {
            "task": task,
            "protocol": protocol,
            "grid": grid,
            "baseline": baseline_tag,
            "source": source,
            "baseline_arg": baseline_arg,
            "zero_seq": bool(cfg.get("zero_seq")),
            "git_sha": rec.get("git_sha"),
            "config": {
                **cfg,
                "normalize": rec.get("normalize"),
                "attestation": rec.get("attestation"),
            },
            "data": rec.get("data"),
            "compute": rec.get("compute"),
            "host": rec.get("host"),
            "checkpoints": ckpts,
            "wandb": {
                "enabled": bool(wandb.get("enabled")),
                "mode": wandb.get("mode"),
                "entity": wandb.get("entity"),
                "project": wandb.get("project"),
                "run_id": wandb.get("run_id"),
                "run_url": wandb.get("run_url"),
            },
            "metrics": cell_metrics,
            "n_seeds": n_by_ts,
            "paths": {
                "run_record": f"{base}/run_record.json",
                "train_log": f"{base}/train.log",
                "result": {ts: f"{base}/{ts}/result.json" for ts in sorted(cell_metrics)},
                "checkpoints": [
                    f"{base}/{v['path']}"
                    for _, v in sorted(ckpts.items())
                    if isinstance(v, dict) and v.get("path")
                ],
            },
            "code": {
                "training": f"src/evemtbench/baselines/{baseline_arg}.py",
                "eval": "src/evemtbench/evaluation/evaluate.py",
                "metrics": "src/evemtbench/evaluation/metrics.py",
                "runner": "src/evemtbench/baselines/runner.py",
            },
            "reproduce": _reproduce_cmd(task, protocol, grid, cfg),
        }
    return _neutral_paths(cells)


def build_provenance_index(
    rows: list[Row],
    records: RunRecordStats,
    *,
    version: str | None = None,
    cells: dict[str, dict] | None = None,
) -> dict[str, str]:
    """{"provenance/<version>/index.json": <deterministic json>} — the committed provenance store.

    ``version=None`` (default) emits the ONE current index at ``provenance/<BENCHMARK_VERSION>/``,
    covering every CURRENT cell — including held_out cells whose artifact paths point into the
    frozen older ``outputs/`` tree. Older ``provenance/<v>/`` indices are frozen history.

    Pass ``cells`` (a ``provenance_cells`` result) to share the one expensive per-cell join with
    ``build_audit_pages`` — ``fresh_views`` does; standalone callers may omit it."""
    if cells is None:
        cells = provenance_cells(rows, records, version=version)
    v = BENCHMARK_VERSION if version is None else version
    doc = {"version": v, "n_cells": len(cells), "cells": cells}
    return {f"provenance/{v}/index.json": json.dumps(doc, indent=2, sort_keys=True) + "\n"}


# ---------------------------------------------------------------------------------------------
# The placeholder engine
# ---------------------------------------------------------------------------------------------
class _Strict(dict):
    def __missing__(self, key: str) -> str:  # an unresolvable {placeholder} fails the build
        raise KeyError(f"audit manifest placeholder {{{key}}} is not in make_audit.VALUE_KEYS")


def _mci(row: Row | None, metric: str, *, with_n: bool = False) -> str:
    """A metric as ``mean ± ci95``; with ``with_n`` also the seed count as ``(n=…)`` (never a bare
    point estimate — checklist 7a/7b: central tendency AND variation AND the exact # of runs).
    ``with_n`` emits an HTML span, so use it only where the result is inserted as raw HTML (the
    drill-down tables / paper-numbers page), never in an ``_esc``-wrapped ledger value."""
    if row is None:
        return "—"
    m = row.metrics.get(metric) or {}
    mean, ci = m.get("mean"), m.get("ci95")
    if mean is None:
        return "—"
    base = f"{mean:.3f} ± {ci:.3f}" if ci is not None else f"{mean:.3f}"
    if with_n and getattr(row, "seeds", None):
        base += f" <span class=dim>(n={row.seeds})</span>"
    return base


def _pick(rows: list[Row], **want) -> list[Row]:
    return [r for r in rows if all(getattr(r, k) == v for k, v in want.items())]


def _headline_of(task: str) -> str:
    return get_task(task).metrics[0]


def compute_values(rows: list[Row], cov: CoverageView, records: RunRecordStats) -> dict[str, str]:
    """Fill every VALUE_KEYS entry from the scanned artifacts; '—' where nothing is measured."""
    v: dict[str, str] = {k: "—" for k in VALUE_KEYS}

    # coverage counts
    t, d, p, f = cov.counts("held_out")
    v.update(n_cells=str(t), n_done=str(d), n_pending=str(p), n_failed=str(f))
    totals = [cov.counts(proto) for proto in PROTOCOLS]
    v.update(
        n_all_cells=str(sum(x[0] for x in totals)),
        n_all_done=str(sum(x[1] for x in totals)),
        n_all_pending=str(sum(x[2] for x in totals)),
        n_all_failed=str(sum(x[3] for x in totals)),
    )
    zt, zd, _, _ = cov.counts("transfer_zeroshot")
    ft, fd_, _, _ = cov.counts("transfer_finetune")
    pt, pd, _, _ = cov.counts("multi_grid_pretrain")
    v.update(
        n_transfer_cells=str(zt + ft),
        n_transfer_done=str(zd + fd_),
        n_pretrain_cells=str(pt),
        n_pretrain_done=str(pd),
    )

    # floor report
    v.update(
        floor_n_groups=str(cov.floor.get("n_groups", 0)),
        floor_n_checked=str(cov.floor.get("n_checked", 0)),
        floor_n_violations=str(len(cov.floor.get("violations", []))),
        floor_n_skipped=str(len(cov.floor.get("skipped", []))),
    )

    # per-function headline bests (representative cell, best learned baseline on test)
    for key, task in HEADLINE_CELLS:
        headline = _headline_of(task)
        direction = METRIC_DIRECTION.get(headline, 1)
        cand = [
            r
            for r in _pick(rows, task=task, grid=REP_GRID, protocol="held_out", test_set="test")
            if r.baseline in LEARNED_BASELINES and r.mean(headline) is not None
        ]
        if cand:
            best = max(cand, key=lambda r: direction * r.mean(headline))
            bench = _pick(
                rows,
                task=task,
                grid=REP_GRID,
                protocol="held_out",
                test_set="benchmark",
                baseline=best.baseline,
            )
            v[f"{key}_best_test"] = _mci(best, headline)
            v[f"{key}_best_bench"] = _mci(bench[0] if bench else None, headline)
            v[f"{key}_best_name"] = best.baseline
            v[f"{key}_fit"] = _fmt_secs(best.fit_seconds)
        maj = _pick(
            rows,
            task=task,
            grid=REP_GRID,
            protocol="held_out",
            test_set="test",
            baseline="majority",
        )
        if maj and maj[0].mean(headline) is not None:
            v[f"{key}_floor"] = f"{maj[0].mean(headline):.3f}"

    # provenance / static
    v.update(
        n_run_records=str(records.n_records),
        n_attested=str(records.n_attested),
        n_git_shas=str(len(records.git_shas)),
        n_lockfile_shas=str(len(records.lockfile_shas)),
        n_hosts=str(len(records.hosts)),
        contract_version=EXPECTED_CONTRACT_VERSION,
        contract_sha=CONTRACT_SHA256[:12] + "…",
        n_tasks=str(len(list_tasks())),
        n_grids=str(len(REFERENCE_GRIDS)),
        n_baselines=str(len(ALL_BASELINES)),
    )
    return v


def _fill(template: str, values: dict[str, str]) -> str:
    return template.format_map(_Strict(values))


def manifest_placeholders() -> set[str]:
    """Every {placeholder} used in the formatted Claim fields (for the integrity test)."""
    names: set[str] = set()
    for c in MAN.CLAIMS:
        for text in (c.claim, c.value, c.detail, c.runtime, c.command):
            names |= {fn for _, fn, _, _ in string.Formatter().parse(text) if fn}
    return names


# ---------------------------------------------------------------------------------------------
# HTML helpers / chrome
# ---------------------------------------------------------------------------------------------
AUDIT_CSS = (
    BASE_CSS
    + """
    .pillars{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px;margin:12px 0}
    .pillar{border:1px solid #e5e5e5;border-left:3px solid #1890ff;border-radius:6px;padding:10px 12px;background:#fafcff}
    .pillar b{display:block;margin-bottom:4px}
    .pillar,.gate p,.ledgerdetail{font-size:13px}
    .badge{font-size:11px;font-weight:600;padding:2px 8px;border-radius:10px;border:1px solid;white-space:nowrap}
    .badge.measured{color:#389e0d;border-color:#389e0d55;background:#f6ffed}
    .badge.enforced{color:#096dd9;border-color:#096dd955;background:#e6f7ff}
    .badge.inprogress{color:#ad6800;border-color:#ad680055;background:#fffbe6}
    .badge.deferred{color:#8c8c8c;border-color:#8c8c8c55;background:#fafafa}
    .badge.note{color:#722ed1;border-color:#722ed155;background:#f9f0ff}
    .claimrow{border:1px solid #e5e5e5;border-radius:8px;margin:10px 0;background:#fff}
    .claimhead{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;padding:10px 14px;border-bottom:1px solid #f0f0f0}
    .claimhead .c{font-weight:600}.claimhead .val{color:#096dd9}
    .claimbody{padding:8px 14px}
    dl.meta{display:grid;grid-template-columns:max-content 1fr;gap:3px 14px;margin:6px 0;font-size:13px}
    dl.meta dt{color:#888}dl.meta dd{margin:0}
    .strip{margin:2px 0;line-height:1.1;white-space:nowrap}
    .strip .c{display:inline-block;width:13px;height:13px;border-radius:3px;margin:1px;vertical-align:middle;cursor:help}
    .c.done{background:#52c41a}.c.pending{background:#faad14}.c.failed{background:#f5222d}
    .c.na{background:#d9d9d9}
    .striplabel{display:inline-block;min-width:230px;font-size:12px}
    .gate{border:1px solid #e5e5e5;border-radius:8px;padding:12px 16px;margin:12px 0;background:#fafcff}
    .gate h3{margin:0 0 6px}
    .cmd{background:#f6f6f6;border-radius:4px;padding:1px 6px;font-size:12px}
    pre.block{background:#f6f6f6;padding:10px 12px;border-radius:6px;overflow-x:auto;font-size:12.5px}
    .cardnav{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin:14px 0}
    .cardnav a{display:block;border:1px solid #e5e5e5;border-radius:8px;padding:12px 14px;text-decoration:none;color:inherit;background:#fafcff}
    .cardnav a:hover{border-color:#1890ff}
    .cardnav b{color:#096dd9}
    .counts{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}
    .counts .k{border:1px solid #e5e5e5;border-radius:8px;padding:8px 14px;background:#fafcff}
    .counts .k b{font-size:19px;display:block}
    .aboutbox{border-left:3px solid #1890ff;padding:6px 12px;background:#fafcff;margin:10px 0;font-size:13.5px}
    .hrow{display:flex;align-items:center;gap:6px;font-size:11px;line-height:1.7}
    .hlab{min-width:190px;color:#555}.hn{color:#888}
    .hbar{display:inline-block;height:9px;background:#69c0ff;border-radius:2px}
    @media (prefers-color-scheme:dark){
      .pillar,.gate,.cardnav a,.counts .k,.aboutbox,.claimrow{background:#1b1e22;border-color:#2a2f36}
      .claimhead{border-color:#2a2f36}
      .cmd,pre.block{background:#1e2227}
      .c.na{background:#3a3f45}
      .badge.measured{background:#1c2a17}.badge.enforced{background:#111f2c}
      .badge.inprogress{background:#2b2410}.badge.deferred{background:#1e2227}.badge.note{background:#221833}
    }
    """
)

_BADGE_CLASS = {
    "measured": "measured",
    "enforced": "enforced",
    "deferred": "deferred",
    "note": "note",
    "in progress": "inprogress",
}

ASSETS_MANIFEST = _REPO / "audit" / "assets" / "manifest.json"


def _assets() -> dict:
    """The committed figure manifest (audit/assets/manifest.json), or empty dicts when no assets
    have been rendered yet. A committed repo file — pages built from it stay deterministic and
    the freshness gate ignores audit/assets/ itself (figures are inputs, not derived views)."""
    try:
        d = json.loads(ASSETS_MANIFEST.read_text())
    except (OSError, json.JSONDecodeError):
        d = {}
    return {"waveforms": d.get("waveforms", {}), "topologies": d.get("topologies", {})}


AUDIT_PAGES = [
    ("index.html", "overview"),
    ("ledger.html", "claim ledger"),
    ("paper_numbers.html", "paper numbers"),
    ("coverage.html", "coverage"),
    ("gates.html", "gates"),
    ("environment.html", "environment"),
    ("reproduce.html", "reproduce"),
    ("provenance.html", "provenance"),
    ("robustness.html", "robustness"),
    ("ground_truth.html", "ground truth"),
    ("progress.html", "burnup"),  # informational, built by make_burnup.py (ungated)
]


def _esc(s: object) -> str:
    return html.escape(str(s))


def _src(path: str, label: str | None = None, line: int | None = None) -> str:
    """A source link into the repo blob UI (works on the GitLab mirror of main)."""
    if not path:
        return _esc(label or "—")
    anchor = f"#L{line}" if line else ""
    return (
        f'<a href="{_esc(DOCS_BASE)}/{_esc(path)}{anchor}"><code>{_esc(label or path)}</code></a>'
    )


def _line_of(relpath: str, snippet: str) -> int:
    """1-based line of an exact source snippet — derived at build time, never hand-written.
    Raises if the snippet is missing or ambiguous so a moved anchor fails the build loudly."""
    text = (_REPO / relpath).read_text()
    if text.count(snippet) != 1:
        raise ValueError(f"gate anchor not found exactly once in {relpath}: {snippet!r}")
    return text[: text.index(snippet)].count("\n") + 1


# Metric definitions (B4): every reported number links to the exact code that defines its metric.
# metric -> (one-line definition, an EXACT unique def-snippet in evaluation/metrics.py). Lines are
# derived at build time via _line_of (loud failure if a dispatcher moves — same discipline as gates).
METRIC_DEFS: dict[str, tuple[str, str]] = {
    "balanced_accuracy": (
        "mean of per-class recall — the majority class cannot buy a good score",
        "def compute_multiclass_metrics(",
    ),
    "macro_f1": (
        "unweighted mean of per-class F1 (zero_division=0)",
        "def compute_multiclass_metrics(",
    ),
    "accuracy": ("fraction of exact-match predictions", "def compute_multiclass_metrics("),
    "missed_fault_rate": (
        "FN / (TP+FN) — fraction of true faults the model missed",
        "def compute_binary_metrics(",
    ),
    "missed_event_rate": ("FN-rate among positives (missed events)", "def compute_binary_metrics("),
    "false_alarm_rate": (
        "FP / (FP+TN) — fraction of non-faults flagged",
        "def compute_binary_metrics(",
    ),
    "false_event_trip_rate": (
        "FP-rate among negatives (tripping on a benign event)",
        "def compute_binary_metrics(",
    ),
    "mae": (
        "mean absolute error, in the grid's own distance units",
        "def compute_regression_metrics(",
    ),
    "rmse": ("root mean squared error", "def compute_regression_metrics("),
    "median_abs_error": ("median of absolute errors", "def compute_regression_metrics("),
    "r2": ("coefficient of determination 1 − SS_res/SS_tot", "def compute_regression_metrics("),
}
_METRICS_SRC = "src/evemtbench/evaluation/metrics.py"


def _metric_link(metric: str) -> str:
    """Inline link from a metric name to the exact function that computes it (+ its definition)."""
    d = METRIC_DEFS.get(metric)
    if not d:
        return f"<code>{_esc(metric)}</code>"
    _defn, snip = d
    try:
        line = _line_of(_METRICS_SRC, snip)
        return _src(_METRICS_SRC, metric, line=line)
    except (ValueError, OSError):
        return _src(_METRICS_SRC, metric)


def _metric_glossary(metrics: tuple[str, ...] | list[str]) -> str:
    """A small definition table: each metric -> one-line definition + its source function, so a
    reviewer reads what a number means and clicks straight to the code that computes it."""
    seen: list[str] = []
    for m in metrics:
        if m in METRIC_DEFS and m not in seen:
            seen.append(m)
    if not seen:
        return ""
    body = "".join(
        f"<tr><td>{_metric_link(m)}</td><td>{_esc(METRIC_DEFS[m][0])}</td>"
        f"<td>{_src('src/evemtbench/evaluation/evaluate.py', 'evaluate()')}</td></tr>"
        for m in seen
    )
    return (
        "<h2>Metric definitions</h2><p class=v>headline &amp; reported metrics for this view — the "
        "metric set is a frozen contract; each links to the exact function that computes it</p>"
        "<div class=tablewrap><table><thead><tr><th>metric</th><th>definition</th>"
        f"<th>entry</th></tr></thead><tbody>{body}</tbody></table></div>"
    )


def _trace_strip(prov_cell: dict | None, *, version: str | None = None) -> str:
    """The five per-number traceability links (+ W&B), resolved for the repo-only site:
    (1) output file, (2) aggregation/eval code, (3) training log, (4) training code, (5) config.
    ``prov_cell`` is one entry of ``provenance_cells``; ``None`` when no run_record exists yet
    (pending cell / empty outputs) — the code links still resolve, the artifact links say pending.
    ``version=None`` links the ONE current index (BENCHMARK_VERSION); per-cell artifact paths
    inside it are already per-protocol (see ``provenance_cells``)."""
    idx = f"audit/provenance/{BENCHMARK_VERSION if version is None else version}/index.json"
    if not prov_cell:
        return (
            "<span class=dim>trace:</span> "
            + _src("src/evemtbench/evaluation/evaluate.py", "eval-code")
            + " · "
            + _src("src/evemtbench/baselines/runner.py", "train-code")
            + " · <span class=dim>output / log / config pending (cell not run yet)</span>"
        )
    baseline_arg = prov_cell.get("baseline_arg") or "runner"
    wb = prov_cell.get("wandb") or {}
    wburl = wb.get("run_url") if wb.get("enabled") else None
    log = (
        f'<a href="{_esc(wburl)}">log</a>'
        if wburl
        else '<span class=dim title="cell-addressable train.log under outputs/ — its path + '
        'the W&amp;B run are recorded in the provenance index">log</span>'
    )
    links = [
        _src(idx, "output"),  # 1 committed provenance -> the result.json path + the value itself
        _src("src/evemtbench/evaluation/evaluate.py", "eval-code"),  # 2 aggregation / eval code
        log,  # 3 training log (the W&B run is the accessible online copy; path in the index)
        _src(f"src/evemtbench/baselines/{baseline_arg}.py", "train-code"),  # 4 training code
        _src(idx, "config"),  # 5 full config / args / attestation
    ]
    if wburl:
        links.append(f'<a href="{_esc(wburl)}">W&amp;B</a>')
    return "<span class=dim>trace:</span> " + " · ".join(links)


def _nav(active: str) -> str:
    items = ['<span style="font-weight:600">EvEMTBench · results audit</span>']
    for href, label in AUDIT_PAGES:
        style = ' style="font-weight:600;text-decoration:underline"' if href == active else ""
        items.append(f'<a href="{href}"{style}>{_esc(label)}</a>')
    items.append('<a href="../leaderboard.html">leaderboard</a>')
    items.append(f'<a href="{_esc(DOCS_BASE)}/README.md">repo</a>')
    return "<nav class=nav>" + " · ".join(items) + "</nav>"


def _page(active: str, title: str, body: str, *, generated_at: str | None) -> str:
    updated = f"<div class=v>updated {_esc(generated_at)}</div>" if generated_at else ""
    footer = (
        "<footer>EvEMTBench results audit · every number on this site is read from committed "
        "artifacts at build time · regenerate: <code>PYTHONPATH=src python hpc/make_audit.py"
        "</code> · gate: "
        + _src("hpc/check_results_fresh.py")
        + f' · <a href="{_esc(DOCS_BASE)}/README.md">repository</a></footer>'
    )
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        '<meta name=description content="EvEMTBench results audit — trace every claim to the '
        'script, command, artifact and verifying gate that produced it.">'
        f'<link rel=icon href="{FAVICON}">'
        f"<title>{_esc(title)}</title><style>{AUDIT_CSS}</style></head><body>"
        f"<header>{_nav(active)}<h1>{_esc(title)}</h1>"
        f"<div class=v>benchmark version {_esc(BENCHMARK_VERSION)} · held_out results "
        f"v{_esc(results_version('held_out'))} · splits v{_esc(SPLITS_VERSION)}/held_out, "
        f"v{_esc(MULTI_GRID_SPLITS_VERSION)}/multi_grid</div>"
        f"{updated}</header>{body}{footer}<script>{SORT_JS}</script></body></html>"
    )


def _verified_by(entries: tuple[str, ...]) -> str:
    parts = []
    gate_ids = {g.gate_id for g in MAN.GATES}
    for e in entries:
        if e in gate_ids:
            parts.append(f'<a href="gates.html#{_esc(e)}">{_esc(e)} gate</a>')
        else:
            parts.append(_src(e.split("::")[0], e))
    return " · ".join(parts) or "—"


def _resolve_status(claim: MAN.Claim, cov: CoverageView) -> str:
    """status='auto' -> measured / 'in progress (N pending[, M failed])' from the scan."""
    if claim.status != "auto":
        return claim.status
    sel = claim.cells or {}

    def _match(c: Cell) -> bool:
        return all(getattr(c, k) in v.split("|") for k, v in sel.items())

    pending = failed = 0
    for cell, status in cov.all_cells():
        if _match(cell):
            pending += status == "pending"
            failed += status == "failed"
    if pending == failed == 0:
        return "measured"
    return f"in progress ({pending} pending{f', {failed} failed' if failed else ''})"


# ---------------------------------------------------------------------------------------------
# Result-table helper (shared by task / grid / baseline drill-down pages)
# ---------------------------------------------------------------------------------------------
def _results_table(rows: list[Row], *, caption: str, prov: dict | None = None) -> str:
    """A sortable held_out results table: one row per (task, grid, baseline) with headline
    test/benchmark values (mean ± ci95, n), params, fit time, and — when a provenance map is
    supplied — a per-row trace strip to the five artifacts + W&B. ``rows`` is pre-filtered by the
    caller."""
    # Current-version at the rendering point too (not only via build_audit_pages): a shown number
    # must be a current number for its protocol so its trace strip resolves against the current
    # provenance index. See build_audit_pages for the rationale.
    held = [
        r for r in rows if r.protocol == "held_out" and r.version == results_version("held_out")
    ]
    keys = sorted(
        {(r.task, r.grid, r.baseline) for r in held},
        key=lambda k: (
            k[0],
            _grid_sort_key(k[1]),
            ALL_BASELINES.index(k[2]) if k[2] in ALL_BASELINES else 99,
        ),
    )
    if not keys:
        return "<p>No results on disk for this slice yet — see the coverage page.</p>"
    body = []
    for task, grid, baseline in keys:
        headline = _headline_of(task) if task in list_tasks() else "balanced_accuracy"
        arrow = "↓" if METRIC_DIRECTION.get(headline, 1) < 0 else "↑"
        test = next(
            (
                r
                for r in held
                if (r.task, r.grid, r.baseline, r.test_set) == (task, grid, baseline, "test")
            ),
            None,
        )
        bench = next(
            (
                r
                for r in held
                if (r.task, r.grid, r.baseline, r.test_set) == (task, grid, baseline, "benchmark")
            ),
            None,
        )
        some = test or bench
        trace = ""
        if prov is not None:
            cell = prov.get(f"{task}/held_out/{grid}/{baseline}")
            trace = f'<td class=dim style="font-size:11px">{_trace_strip(cell)}</td>'
        body.append(
            f'<tr><td><a href="t_{_esc(task)}.html">{_esc(task)}</a></td>'
            f'<td><a href="g_{_esc(grid)}.html">{_esc(grid)}</a></td>'
            f'<td class=base><a href="b_{_esc(baseline)}.html">{_esc(baseline)}</a></td>'
            f"<td>{_esc(headline)} {arrow}</td><td>{_mci(test, headline, with_n=True)}</td>"
            f"<td>{_mci(bench, headline, with_n=True)}</td>"
            f"<td class=dim>{_fmt_params(some.n_params if some else None)}</td>"
            f"<td class=dim>{_fmt_secs(some.fit_seconds if some else None)}</td>{trace}</tr>"
        )
    trace_head = "<th>trace</th>" if prov is not None else ""
    return (
        f"<div class=tablewrap><table class=sortable><caption>{_esc(caption)}</caption>"
        "<thead><tr><th>task</th><th>grid</th><th>baseline</th><th>headline</th>"
        f"<th>test</th><th>benchmark</th><th>params</th><th>fit/seed</th>{trace_head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


# ---------------------------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------------------------
def _how_to_read() -> str:
    rows = [
        ("Value", "the measured number, read from the committed result file at build time"),
        ("Script", "link to the exact source that produces it"),
        ("Command", "the verbatim invocation to reproduce it"),
        ("Artifact", "the output (result.json / rendered file) it writes"),
        ("Runtime &amp; Hardware", "wall-clock and the machine it ran on"),
        ("Verified by", "the test / gate that guards it"),
        (
            "Status",
            '<span class="badge measured">measured</span> vs '
            '<span class="badge inprogress">in progress</span> vs '
            '<span class="badge deferred">deferred</span> — a plan is never dressed up as a '
            "result",
        ),
    ]
    body = "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in rows)
    return (
        "<h2>How to read a claim</h2><div class=tablewrap><table>"
        f"<thead><tr><th>Field</th><th>What it gives the reviewer</th></tr></thead>{body}</table></div>"
    )


def build_index(values: dict[str, str], cov: CoverageView) -> str:
    pillars = "".join(
        f"<div class=pillar><b>{_esc(p['title'])}</b>{_esc(p['text'])}<br>"
        + (f'<a href="gates.html#{_esc(p["gate_id"])}">code tour →</a> · ' if p["gate_id"] else "")
        + _src(p["source"])
        + "</div>"
        for p in MAN.PILLARS
    )
    counts = (
        "<div class=counts>"
        f"<div class=k><b>{values['n_done']}</b>of {values['n_cells']} held_out cells done</div>"
        f"<div class=k><b>{values['n_all_done']}</b>of {values['n_all_cells']} cells across all protocols</div>"
        f"<div class=k><b>{values['n_run_records']}</b>provenance records</div>"
        f"<div class=k><b>{values['n_tasks']}</b>tasks × <b style='display:inline;font-size:19px'>{values['n_baselines']}</b> baselines</div>"
        "</div>"
    )
    cards = (
        "<div class=cardnav>"
        '<a href="ledger.html"><b>Claim ledger →</b><br>every paper claim resolved to script, '
        "command, artifact, runtime, hardware and verifying gate</a>"
        '<a href="paper_numbers.html"><b>Paper numbers →</b><br>every number the manuscript reports, '
        "traced to output file, eval code, training log, training code, config and W&amp;B run</a>"
        '<a href="coverage.html"><b>Coverage matrix →</b><br>every cell of the results matrix: '
        "done / pending / failed / N-A-by-rule, with its reproduce command</a>"
        '<a href="gates.html"><b>Verification gates →</b><br>a guided code tour of the five '
        "gates that guard the numbers</a>"
        '<a href="environment.html"><b>Environment →</b><br>hardware, pinned software, seeds, '
        "and input-data hashes</a>"
        '<a href="reproduce.html"><b>Reproduce →</b><br>cheapest gate first: from a 5-second '
        "self-check to regenerating any single cell</a>"
        '<a href="provenance.html"><b>Provenance →</b><br>tests, CI, hooks and the change '
        "log</a></div>"
    )
    docs = " · ".join(_src(p, label) for p, label in MAN.DOC_LINKS)
    body = (
        "<p>EvEMTBench is a protocol-driven ML benchmark for power-system protection on EMT "
        "waveform windows. Labels are exact by construction — every window's label derives from "
        "the simulation script that injected the event — so the honest risks are <i>leakage</i>, "
        "<i>held-out contamination</i>, <i>stale published tables</i> and <i>trivial-task "
        "artefacts</i>. Each has a mechanical guard:</p>"
        f"<div class=pillars>{pillars}</div>"
        f"{counts}{cards}{_how_to_read()}"
        f"<p class=v>Reference docs: {docs}</p>"
    )
    return body


def build_ledger(values: dict[str, str], cov: CoverageView) -> str:
    rows_html = []
    for c in MAN.CLAIMS:
        status = _resolve_status(c, cov)
        badge_cls = _BADGE_CLASS.get(
            status, "inprogress" if status.startswith("in progress") else "note"
        )
        meta = [
            ("Script", _src(c.script)),
            ("Command", f"<code class=cmd>{_esc(_fill(c.command, values))}</code>"),
            (
                "Artifact",
                # a row may name several artifacts, " · "-separated
                " · ".join(
                    f"<code>{_esc(a)}</code>" if a.startswith("outputs/") else _src(a)
                    for a in c.artifact.split(" · ")
                ),
            ),
            ("Runtime", _esc(_fill(c.runtime, values))),
            ("Hardware", _esc(c.hardware)),
            ("Verified by", _verified_by(c.verified_by)),
        ]
        meta_html = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in meta)
        rows_html.append(
            "<div class=claimrow>"
            f"<div class=claimhead><span class=c>{_esc(_fill(c.claim, values))}</span>"
            f"<span class=val>{_esc(_fill(c.value, values))}</span>"
            f'<span class="badge {badge_cls}">{_esc(status)}</span></div>'
            f"<div class=claimbody><p class=ledgerdetail>{_esc(_fill(c.detail, values))}</p>"
            f"<dl class=meta>{meta_html}</dl></div></div>"
        )
    return (
        "<p>One row per claim of the paper. Every number below is a placeholder filled from the "
        "committed artifacts at build time — the manifest source contains no digits (see "
        + _src("hpc/audit_manifest.py")
        + "). Rows whose cells are still being swept show "
        '<span class="badge inprogress">in progress</span> with live counts; planned work is '
        '<span class="badge deferred">deferred</span>, never a result.</p>' + "".join(rows_html)
    )


def build_coverage(cov: CoverageView) -> str:
    na_note = "".join(
        f"<li><b>{_esc(k)}</b> — {_esc(t)}</li>" for k, t in sorted(MAN.NA_REASONS.items())
    )
    legend = (
        '<p class=v>legend: <span class="c done"></span> done · <span class="c pending"></span> '
        'pending · <span class="c failed"></span> failed · <span class="c na"></span> N/A by '
        "rule (hover any square for the cell and its reproduce command)</p>"
    )
    sections = []
    status_of = {(c.task, c.protocol, c.grid, c.baseline, c.source): s for c, s in cov.all_cells()}
    for proto in PROTOCOLS:
        total, d, p, f = cov.counts(proto)
        grids = [g for g in sorted(GRIDS, key=_grid_sort_key) if protocol_grid_applies(proto, g)]
        strips = []
        for task in sorted(list_tasks(), key=lambda t: (get_task(t).tier or 9, t)):
            td = get_task(task)
            if proto not in td.protocols:
                continue
            squares = []
            is_transfer = proto in TRANSFER_PROTOCOLS
            for grid in grids:
                for baseline in ALL_BASELINES:
                    # transfer fans out over pre-training sources (deep baselines only); other
                    # protocols / classical baselines stay a single (source=None) square.
                    srcs = (
                        PRETRAIN_GRIDS if (is_transfer and baseline in DEEP_BASELINES) else (None,)
                    )
                    for source in srcs:
                        st = status_of.get((task, proto, grid, baseline, source))
                        if st is None:  # N/A by applicability rule
                            rule = (
                                "transfer_deep_only"
                                if proto.startswith("transfer")
                                else "threshold_scope"
                            )
                            title = f"{task} / {proto} / {grid} / {baseline} — N/A: {MAN.NA_REASONS[rule]}"
                            squares.append(f'<span class="c na" title="{_esc(title)}"></span>')
                        else:
                            cell = Cell(task, proto, grid, baseline, source=source)
                            cmd = f"python -m evemtbench.baselines.runner {_arg_line(cell)}"
                            src_note = f" / src={source}" if source else ""
                            title = (
                                f"{task} / {proto} / {grid} / {baseline}{src_note} — {st} · {cmd}"
                            )
                            squares.append(f'<span class="c {st}" title="{_esc(title)}"></span>')
            strips.append(
                f'<div class=strip><span class=striplabel><a href="t_{_esc(task)}.html">'
                f"{_esc(task)}</a></span>{''.join(squares)}</div>"
            )
        grid_note = " · ".join(f'<a href="g_{_esc(g)}.html">{_esc(g)}</a>' for g in grids)
        sections.append(
            f"<h2>{_esc(proto)} <span class=dim>· {d} done / {p} pending / {f} failed of "
            f"{total}</span></h2><p class=v>columns per task strip: grid (voltage order: "
            f"{grid_note}) × baseline ({' · '.join(ALL_BASELINES)})</p>" + "".join(strips)
        )
    return (
        "<p>A <b>cell</b> is one (task × protocol × grid × baseline) runner invocation; it is "
        "<b>done</b> only when every expected test-set result.json is valid. Three non-green "
        "states are kept distinct and none is hidden: <b>pending</b> (queued sweep work), "
        "<b>failed</b> (partial artifacts), and <b>N/A</b> (excluded by a stated applicability "
        "rule, not missing).</p>"
        f"<ul>{na_note}</ul>{legend}"
        + "".join(sections)
        + "<p>The same matrix is committed as <code>coverage_snapshot.json</code> with the exact "
        "runner argument line per cell, and summarized by "
        "<code>python -m evemtbench.coverage --summary</code>.</p>"
    )


def build_gates(cov: CoverageView) -> str:
    fl = cov.floor
    floor_line = (
        f"Current report: {fl.get('n_checked', 0)} comparisons across {fl.get('n_groups', 0)} "
        f"groups, {len(fl.get('violations', []))} violations, {len(fl.get('skipped', []))} "
        "skipped (pending cells)."
    )
    cards = []
    for g in MAN.GATES:
        line = _line_of(g.file, g.anchor_snippet)
        extra = f"<p class=v>{_esc(floor_line)}</p>" if g.gate_id == "floor" else ""
        stack_rows = "".join(
            f"<tr><td>{_src(f, f'{f.rsplit(chr(47), 1)[-1]}:{_line_of(f, snip)}', line=_line_of(f, snip))}"
            f"<br><code>{_esc(snip.strip())}</code></td><td>{_esc(computes)}</td></tr>"
            for f, snip, computes in g.stack
        )
        stack = (
            "<div class=tablewrap><table><thead><tr><th>Step in the call stack</th>"
            f"<th>What is actually computed</th></tr></thead>{stack_rows}</table></div>"
            if g.stack
            else ""
        )
        cards.append(
            f'<div class=gate id="{_esc(g.gate_id)}"><h3>{_esc(g.title)}</h3>'
            f"<p><b>Asserts:</b> {_esc(g.asserts_what)}</p>"
            f"<p><b>Where:</b> {_src(g.file, f'{g.file}:{line}', line=line)} — "
            f"<code>{_esc(g.anchor_snippet.strip())}</code> <span class=dim>(all line numbers "
            "derived at build time)</span></p>"
            f"{stack}"
            f"<p><b>On failure:</b> {_esc(g.failure_looks_like)}</p>"
            f"<p><b>How to read it:</b> {_esc(g.how_to_read)}</p>{extra}"
            f"<p><b>Run it:</b> <code class=cmd>{_esc(g.run_command)}</code></p></div>"
        )
    return (
        "<p>A reviewer should be able to confirm what each gate asserts without reading the "
        "whole codebase. The two diagrams show where each gate sits in the pipeline; each card "
        "then walks the gate's real call stack — every step linked at the exact "
        "(build-time-derived) source line — down to the single assertion that matters.</p>"
        f"<h2>The data path</h2><pre class=block><code>{_esc(MAN.DATA_PIPELINE)}</code></pre>"
        f"<h2>The publishing path</h2><pre class=block><code>{_esc(MAN.PUBLISH_PIPELINE)}</code></pre>"
        "<h2>The six gates</h2>" + "".join(cards)
    )


def build_environment(values: dict[str, str], records: RunRecordStats) -> str:
    hw = "".join(
        f"<tr><td><b>{_esc(a)}</b></td><td>{_esc(b)}</td><td>{_esc(c)}</td></tr>"
        for a, b, c in MAN.ENVIRONMENT["hardware"]
    )
    lock = _REPO / "requirements.lock"
    lock_sha = hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else "—"
    # The held_out results keep their original lock; the pretrain/transfer re-scores get their
    # own env/results-v<BENCHMARK_VERSION>.lock — list it only once it is captured (is_file()).
    held_v = results_version("held_out")
    renv = _REPO / "env" / f"results-v{held_v}.lock"
    renv_sha = hashlib.sha256(renv.read_bytes()).hexdigest() if renv.exists() else "—"
    cur_lock_li = ""
    cur_renv = _REPO / "env" / f"results-v{BENCHMARK_VERSION}.lock"
    if BENCHMARK_VERSION != held_v and cur_renv.is_file():
        cur_sha = hashlib.sha256(cur_renv.read_bytes()).hexdigest()
        cur_lock_li = (
            f"<li><b>results environment (v{_esc(BENCHMARK_VERSION)} re-scores)</b> "
            f"{_src(f'env/results-v{BENCHMARK_VERSION}.lock')} — sha256 <code>{_esc(cur_sha)}"
            "</code>. The environment for the pretrain/transfer re-scores under the corrected "
            "multi_grid splits; the held_out results keep the lock above.</li>"
        )
    fmt_list = lambda xs: _esc(", ".join(xs)) if xs else "— (none recorded yet)"  # noqa: E731
    recs = "".join(
        f"<tr><td><b>{_esc(k)}</b></td><td>{v}</td></tr>"
        for k, v in [
            ("run records", _esc(values["n_run_records"])),
            ("with no-tuning attestation", _esc(values["n_attested"])),
            (
                "distinct git SHAs",
                fmt_list(records.git_shas[:8]) + (" …" if len(records.git_shas) > 8 else ""),
            ),
            ("distinct lockfile SHAs", fmt_list([s[:16] + "…" for s in records.lockfile_shas])),
            (
                "agree with the results-env capture",
                _esc(
                    f"{records.env_agree} of {records.env_agree + records.env_disagree}"
                    + (
                        f" — {records.env_disagree} DISAGREE"
                        if records.env_disagree
                        else " (every record's own captured versions match)"
                    )
                ),
            ),
            ("python versions", fmt_list(records.pythons)),
            ("torch versions", fmt_list(records.torchs)),
            ("recorded hosts", fmt_list(records.hosts)),
        ]
    )
    return (
        f"<h2>Hardware</h2><div class=tablewrap><table><thead><tr><th>Role</th><th>Machine</th>"
        f"<th>Used for</th></tr></thead>{hw}</table></div>"
        f"<h2>Determinism &amp; seeds</h2><p>{_esc(MAN.ENVIRONMENT['seed_policy'])}</p>"
        f"<p>{_esc(MAN.ENVIRONMENT['staging'])}</p>"
        "<h2>Pins</h2><ul>"
        f"<li><b>results environment</b> {_src(f'env/results-v{held_v}.lock')} — sha256 "
        f"<code>{_esc(renv_sha)}</code>. The conda prefix that actually produced these numbers "
        "(torch + CUDA stack included), captured by <code>hpc/capture_results_env.sh</code>. Every "
        "run record independently captured its own package versions at run time; agreement between "
        "the two is verified by <code>hpc/verify_results_env.py</code> — see the row below.</li>"
        f"{cur_lock_li}"
        f"<li>CI test-gate locks {_src('requirements-ci-py3.10.lock')} / "
        f"{_src('requirements-ci-py3.12.lock')} — hash-pinned, installed by CI with "
        "<code>--require-hashes</code>. Torch-free: they pin the <em>test gate</em>, not the "
        "results.</li>"
        f"<li>legacy lockfile {_src('requirements.lock')} — sha256 <code>{_esc(lock_sha)}</code>. "
        "The v1.0.0-era CI pin, recorded as <code>lockfile_sha256</code> in every run record and "
        "kept byte-identical for that reason. It describes neither the results environment (no "
        "torch) nor current CI.</li>"
        f"<li>window contract v{_esc(values['contract_version'])} — pinned sha256 "
        f"<code>{_esc(values['contract_sha'])}</code> ({_src('src/evemtbench/contract/contract.json')})</li>"
        f"<li>committed splits v{_esc(SPLITS_VERSION)} (held_out) — "
        f"{_src(f'src/evemtbench/splits/v{SPLITS_VERSION}')} · "
        f"v{_esc(MULTI_GRID_SPLITS_VERSION)} (multi_grid_pretrain, topology-grouped) — "
        f"{_src(f'src/evemtbench/splits/v{MULTI_GRID_SPLITS_VERSION}')}</li></ul>"
        "<h2>What the run records say</h2><p>Aggregated at build time from every "
        "<code>run_record.json</code> under outputs/ — the audit reads the records, it does not "
        f"assert them.</p><div class=tablewrap><table>{recs}</table></div>"
    )


def build_reproduce(values: dict[str, str]) -> str:
    block = """# 0 · seconds, DATALESS (pip install only): the scoring harness proves itself
PYTHONPATH=src python -m evemtbench.evaluation.cli --self-test

# 1 · seconds, read-only: coverage + the analytic floor self-check
PYTHONPATH=src python -m evemtbench.coverage --summary
PYTHONPATH=src python -m evemtbench.coverage --check-floor

# 2 · seconds: are the committed rendered views (incl. this site) byte-fresh?
python hpc/check_results_fresh.py

# 3 · minutes (SLURM — never on the login node): the full test gate
srun --cpus-per-task=4 --mem=8G --time=15:00 bash -c 'PYTHONPATH=src pytest'

# 4 · regenerate any single cell (its exact arg line is on the coverage page / snapshot)
PYTHONPATH=src python -m evemtbench.baselines.runner \\
    --task <task> --protocol <protocol> --grid <grid> --baseline <baseline> --seeds 0 1 2 3 4

# 5 · rebuild every rendered view from outputs/
PYTHONPATH=src python hpc/make_results.py && PYTHONPATH=src evemtbench-leaderboard \\
    && PYTHONPATH=src python hpc/make_audit.py"""
    rows = "".join(
        f"<tr><td>{_esc(_fill(c.claim, values))}</td>"
        f"<td><code class=cmd>{_esc(_fill(c.command, values))}</code></td>"
        f"<td><code>{_esc(c.artifact)}</code></td></tr>"
        for c in MAN.CLAIMS
        if c.command != "—"
    )
    return (
        "<p>Cheapest gate first — a reviewer can stop at any level of effort.</p>"
        f"<pre class=block><code>{_esc(block)}</code></pre>"
        "<h2>Per-claim commands</h2><p class=v>the exact invocation behind every ledger row</p>"
        "<div class=tablewrap><table><thead><tr><th>Claim</th><th>Command</th><th>Artifact</th>"
        f"</tr></thead>{rows}</table></div>"
    )


def build_provenance() -> str:
    test_files = sorted((_REPO / "tests").rglob("test_*.py"))
    n_funcs = sum(f.read_text().count("def test_") for f in test_files)
    ci_rows = [
        (
            ".github/workflows/ci.yml",
            "ruff + mypy + pytest (torch-free) on 3.10/3.12; the "
            "leakage audit re-runs as a named step so a leaky split fails legibly",
        ),
        (
            ".github/workflows/nightly.yml",
            "the same gate on fresh runners nightly — catches dependency/runner drift",
        ),
    ]
    ci = "".join(f"<tr><td>{_src(p)}</td><td>{_esc(d)}</td></tr>" for p, d in ci_rows)
    hooks = [
        (
            ".githooks/pre-commit",
            "runs the freshness gate when rendered views are staged — a "
            "stale RESULTS.md / leaderboard.html / audit page cannot be committed",
        ),
        (
            ".pre-commit-config.yaml",
            "ruff lint+format, mypy, hygiene hooks, and the same "
            "results-fresh gate for pre-commit users",
        ),
    ]
    hk = "".join(f"<tr><td>{_src(p)}</td><td>{_esc(d)}</td></tr>" for p, d in hooks)
    logs = [
        ("CHANGELOG.md", "versioned change log"),
    ]
    lg = "".join(f"<tr><td>{_src(p)}</td><td>{_esc(d)}</td></tr>" for p, d in logs)
    return (
        f"<h2>Test suite</h2><p><b>{n_funcs}</b> test functions across <b>{len(test_files)}</b> "
        "files (counted from source at build time) — unit, integration (incl. leakage, splits "
        "audit, freshness, contract conformance, per-baseline runners), e2e smoke, and "
        "data-integrity checks that run only where the real data lives.</p>"
        f"<h2>Continuous integration</h2><div class=tablewrap><table>{ci}</table></div>"
        f"<h2>Hooks</h2><div class=tablewrap><table>{hk}</table></div>"
        f"<h2>Change log</h2><div class=tablewrap><table>{lg}</table></div>"
        "<p>Progress itself is an append-only ledger: the sweep loop commits a snapshot "
        "(<code>chore(results): sweep progress snapshot — … done … pending …</code>) with the "
        "coverage summary embedded in every message, so the history of the matrix is in "
        "<code>git log</code>.</p>"
    )


def _example_window(fn: str) -> str:
    """The task's example-window figure (committed asset) with its provenance caption, or an
    honest pending note when the asset has not been rendered for this function yet."""
    entry = _assets()["waveforms"].get(fn)
    if not entry:
        return (
            "<h2>Example window</h2><p class=v>figure pending — render with "
            "<code>python hpc/make_audit_assets.py</code> (needs the data root)</p>"
        )
    cap = (
        f"event <code>{_esc(entry['event_type'])}</code> · grid {_esc(entry['grid'])} "
        f"({_esc(entry['family'])}/{_esc(entry['split'])} split) · sample "
        f"{_esc(entry['sample_id'])}, window #{entry['window_index']}, cubicle "
        f"{entry['cubicle']} (max-RMS) · 50 ms / 480 samples @ 9.6 kHz · top: 3-phase current, "
        "bottom: 3-phase voltage · figure provenance in "
    )
    return (
        f'<h2>Example window</h2><img src="assets/{_esc(entry["figure"])}" '
        'alt="example waveform window" loading="lazy">'
        f"<p class=v>{cap}{_src('audit/assets/manifest.json', 'assets/manifest.json')}</p>"
    )


def build_task_page(task: str, rows: list[Row], cov: CoverageView, prov: dict | None = None) -> str:
    td = get_task(task)
    fn, view = td.function.value, td.observability.value
    about = MAN.FUNCTION_ABOUT.get(fn, td.description)
    view_about = MAN.VIEW_ABOUT.get(view, "")
    headline = _headline_of(task)
    arrow = "↓ lower is better" if METRIC_DIRECTION.get(headline, 1) < 0 else "↑ higher is better"
    metrics = " · ".join(f"<b>{_esc(m)}</b>" if m == headline else _esc(m) for m in td.metrics)
    pend = sum(1 for c, s in cov.all_cells() if c.task == task and s != "done")
    status = (
        '<span class="badge measured">all cells measured</span>'
        if pend == 0
        else f'<span class="badge inprogress">{pend} cells pending/failed</span>'
    )
    classes = f" · {td.num_classes} classes" if td.num_classes else " · regression"
    return (
        f"<p class=v>tier {td.tier} · {_esc(fn)} · {_esc(view)} view{classes} · {status}</p>"
        f"<div class=aboutbox><b>What this task is.</b> {_esc(about)} {_esc(view_about)}</div>"
        f"<h2>How it's scored</h2><p>Headline metric <b>{_esc(headline)}</b> ({arrow}); full "
        f"metric set: {metrics}. Metrics are a frozen contract "
        f"({_src('src/evemtbench/evaluation/metrics.py')}); label derivation: "
        f"{_src('src/evemtbench/tasks/labels')}.</p>"
        + _example_window(fn)
        + "<h2>Results (held_out)</h2>"
        + _results_table(
            [r for r in rows if r.task == task],
            caption=f"{task} — every grid × baseline; click headers to sort",
            prov=prov,
        )
        + _metric_glossary(td.metrics)
        + "<h2>Reproduce one cell</h2><pre class=block><code>PYTHONPATH=src python -m "
        f"evemtbench.baselines.runner --task {_esc(task)} --protocol held_out \\\n"
        "    --grid &lt;grid&gt; --baseline &lt;baseline&gt; --seeds 0 1 2 3 4</code></pre>"
    )


def _topology_figure(grid: str) -> str:
    """The grid's topology diagram (rendered from the aux graph), or an honest pending note."""
    entry = _assets()["topologies"].get(grid)
    if not entry:
        return (
            "<h2>Topology</h2><p class=v>diagram pending — the aux topology graph for this "
            "grid is not extracted yet (<code>sbatch hpc/extract_aux.sh</code>, then "
            "<code>python hpc/make_audit_assets.py</code>)</p>"
        )
    return (
        f'<h2>Topology</h2><img src="assets/{_esc(entry["figure"])}" alt="grid topology" '
        'loading="lazy">'
        f"<p class=v>{entry['n_nodes']} buses/elements · {entry['n_edges']} lines/transformers/"
        f"switches · layout computed from {_esc(entry['source'])} · provenance in "
        f"{_src('audit/assets/manifest.json', 'assets/manifest.json')}</p>"
    )


def build_grid_page(grid: str, rows: list[Row], cov: CoverageView, prov: dict | None = None) -> str:
    spec = _grid_spec_label(grid)
    try:
        splits = read_held_out_splits(grid)
        split_line = " · ".join(f"{k}: {len(v)}" for k, v in sorted(splits.items()))
    except Exception:
        split_line = "no held_out split (pretrain corpus)"
    bench = "test + benchmark" if "benchmark" in expected_test_sets(grid) else "test only"
    d = sum(1 for c, s in cov.all_cells() if c.grid == grid and s == "done")
    p = sum(1 for c, s in cov.all_cells() if c.grid == grid and s == "pending")
    f = sum(1 for c, s in cov.all_cells() if c.grid == grid and s == "failed")
    # The ieee39 adapt_grid refresh landed in the v1.0.0 re-simulation, so every grid is final
    # for this version and no per-grid caveat is shown (kept as a hook for any future note).
    caveat = ""
    return (
        f"<p class=v>{_esc(spec)} · expected test sets: {_esc(bench)} · cells: {d} done / {p} "
        f"pending / {f} failed</p>{caveat}"
        f"<p><b>Committed held_out split sizes</b> (episode-level, family-pure): "
        f"{_esc(split_line)} — files: {_src(f'src/evemtbench/splits/v{SPLITS_VERSION}/held_out/{grid}') if grid in REFERENCE_GRIDS else '—'}</p>"
        + _topology_figure(grid)
        + "<h2>Results on this grid (held_out)</h2>"
        + _results_table(
            [r for r in rows if r.grid == grid],
            caption=f"{grid} — every task × baseline; click headers to sort",
            prov=prov,
        )
    )


def _detex_num(s: str) -> str:
    """Render a ``paper_lib._fmt_mean_ci`` cell (LaTeX) as readable HTML, preserving every digit so
    the value traces verbatim (``0.907 $\\pm$ 0.011`` -> ``0.907 ± 0.011``; a clipped interval
    ``0.848$\\,{\\scriptscriptstyle[0.626,\\,1.000]}$`` -> ``0.848 [0.626, 1.000]``)."""
    s = s.replace("$\\,{\\scriptscriptstyle[", " [").replace(",\\,", ", ").replace("]}$", "]")
    return s.replace("$\\pm$", "±").replace("$", "").replace("\\,", "").strip()


def _derived_metrics_block(baseline: str, brows: list[Row]) -> str:
    """Protection-grade rates (FD/ED dependability = missed, security = false-alarm) and per-class
    F1 (FC) for THIS baseline — the secondary metrics the Supplementary (Package C) tables select.
    Rendered from the same held_out result.json cells with paper_lib's 3-dp formatting, so every SI
    number resolves on the dashboard (enforced by tests/integration/test_paper_numbers_covered.py)."""
    from paper_lib import _BINARY_RATE_KEYS, _FC_F1, _fmt_mean_ci  # noqa: PLC0415

    def _val(r: Row, metric: str) -> str | None:
        m = r.mean(metric)
        if m is None:
            return None
        return _detex_num(_fmt_mean_ci(m, (r.metrics.get(metric) or {}).get("ci95"), metric))

    ho = [r for r in brows if r.protocol == "held_out"]
    items: list[str] = []
    for stem, (mk, fk) in _BINARY_RATE_KEYS.items():
        for r in sorted(
            (x for x in ho if x.task.startswith(f"{stem}_")),
            key=lambda r: (r.task, _grid_sort_key(r.grid), r.test_set),
        ):
            for label, key in (("missed", mk), ("false-alarm", fk)):
                v = _val(r, key)
                if v is not None:
                    items.append(
                        f"{_esc(r.task)} · {_esc(r.grid)} · {r.test_set} · {label}: <b>{v}</b>"
                    )
    for r in sorted(
        (x for x in ho if x.task == "fault_classification_line" and x.test_set == "test"),
        key=lambda r: _grid_sort_key(r.grid),
    ):
        for key, lab in _FC_F1:
            v = _val(r, key)
            if v is not None:
                items.append(f"FC line · {_esc(r.grid)} · {_esc(lab)} F<sub>1</sub>: <b>{v}</b>")
    if not items:
        return ""
    return (
        "<h2>Protection-grade &amp; per-class metrics</h2>"
        "<p class=v>Dependability/security rates and per-class F<sub>1</sub> for this baseline — the "
        "same held_out result.json cells the Supplementary (Package C) tables select, traced here so "
        "every reported SI number resolves on the dashboard.</p>"
        "<ul class=trace><li>" + "</li><li>".join(items) + "</li></ul>"
    )


def build_baseline_page(baseline: str, rows: list[Row], prov: dict | None = None) -> str:
    about, ref = MAN.BASELINE_ABOUT[baseline]
    impl = f"src/evemtbench/baselines/{baseline}.py"
    scope = ""
    if baseline == "threshold":
        scope = f"<p class=v>Scope: {_esc(MAN.NA_REASONS['threshold_scope'])}.</p>"
    elif baseline in ("majority", "random_forest"):
        scope = f"<p class=v>Note: {_esc(MAN.NA_REASONS['transfer_deep_only'])}.</p>"
    brows = [r for r in rows if r.baseline == baseline]
    return (
        f"<div class=aboutbox><b>What it does.</b> {_esc(about)}</div>"
        f"<p><b>Reference:</b> {_esc(ref)} · <b>Implementation:</b> {_src(impl)}</p>{scope}"
        "<h2>Results (held_out)</h2>"
        + _results_table(
            brows,
            caption=f"{baseline} — every task × grid; click headers to sort",
            prov=prov,
        )
        + _derived_metrics_block(baseline, brows)
    )


def collect_headline_seeds(outputs: str | Path = "outputs", *, version: str | None = None) -> dict:
    """task -> grid -> baseline -> per-seed headline values (held_out / test set), read straight
    from the committed result.json files (the leaderboard Row drops per_seed). Headline cells are
    held_out, so ``version=None`` (default) resolves to that protocol's CURRENT results tree."""
    data: dict = {}
    root = Path(outputs) / (results_version("held_out") if version is None else version)
    if not root.is_dir():
        return data
    for p in sorted(root.glob("*/held_out/*/*/test/result.json")):
        try:
            doc = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        task, grid, baseline = doc.get("task"), doc.get("grid"), doc.get("baseline")
        if not task or task not in list_tasks() or baseline not in LEARNED_BASELINES:
            continue
        headline = _headline_of(task)
        vals = [(r.get("metrics") or {}).get(headline) for r in doc.get("per_seed") or []]
        vals = [v for v in vals if v is not None]
        if vals:
            data.setdefault(task, {}).setdefault(grid, {})[baseline] = vals
    return data


def build_robustness(seeds: dict) -> str:
    """audit/robustness.html — the UnderstandingVCS uncertainty.py idiom: the same committed
    numbers, never re-measured, re-aggregated under defensible alternative schemes. Deterministic
    (sorted iteration + seeded bootstrap), so the freshness gate covers it."""
    if not seeds:
        return "<p>No held_out results on disk yet — this page fills as the sweep completes.</p>"

    sections = []
    n_stable = n_scored = 0
    for task in sorted(seeds, key=lambda t: (get_task(t).tier or 9, t)):
        headline = _headline_of(task)
        direction = METRIC_DIRECTION.get(headline, 1)
        grids = sorted(seeds[task], key=_grid_sort_key)
        baselines = sorted({b for g in seeds[task].values() for b in g})
        rows = []
        best = {}
        for baseline in baselines:
            grid_means = [
                sum(seeds[task][g][baseline]) / len(seeds[task][g][baseline])
                for g in grids
                if baseline in seeds[task][g]
            ]
            if not grid_means:
                continue
            mean_of_grids = sum(grid_means) / len(grid_means)
            med = sorted(grid_means)[len(grid_means) // 2]
            logo = (
                [
                    sum(x for j, x in enumerate(grid_means) if j != i) / (len(grid_means) - 1)
                    for i in range(len(grid_means))
                ]
                if len(grid_means) > 1
                else grid_means
            )
            pooled = [
                v for g in grids if baseline in seeds[task][g] for v in seeds[task][g][baseline]
            ]
            bci = bootstrap_ci(pooled)
            tci = ci95_halfwidth(pooled)
            best[baseline] = {
                "mean": mean_of_grids,
                "median": med,
                "logo_min": min(logo),
                "logo_max": max(logo),
            }
            rows.append(
                f"<tr><td class=base>{_esc(baseline)}</td><td>{mean_of_grids:.3f}"
                + (f" ± {tci:.3f}" if tci is not None else "")
                + f"</td><td>{med:.3f}</td><td>{min(logo):.3f} … {max(logo):.3f}</td>"
                + (f"<td>[{bci[0]:.3f}, {bci[1]:.3f}]</td>" if bci else "<td>—</td>")
                + f"<td class=dim>{len(grid_means)} grids · {len(pooled)} seed-values</td></tr>"
            )
        if not best:
            continue
        n_scored += 1

        def _argbest(key, best=best, direction=direction):
            return max(best, key=lambda bl: direction * best[bl][key])

        winners = {_argbest("mean"), _argbest("median"), _argbest("logo_min"), _argbest("logo_max")}
        stable = len(winners) == 1
        n_stable += stable
        verdict = (
            f'<span class="badge measured">rank-stable: {_esc(next(iter(winners)))} wins under '
            "every scheme</span>"
            if stable
            else f'<span class="badge inprogress">rank-sensitive: winners {_esc(sorted(winners))}'
            "</span>"
        )
        arrow = "↓" if direction < 0 else "↑"
        sections.append(
            f'<div class=gate id="{_esc(task)}"><h3><a href="t_{_esc(task)}.html">{_esc(task)}'
            f"</a> <span class=dim>· {_esc(headline)} {arrow} (test set)</span> {verdict}</h3>"
            "<div class=tablewrap><table><thead><tr><th>baseline</th>"
            "<th>mean over grids ± t-CI</th><th>median over grids</th>"
            "<th>leave-one-grid-out range</th><th>seed-bootstrap 95% CI</th><th>n</th>"
            f"</tr></thead>{''.join(rows)}</table></div></div>"
        )

    return (
        "<p><b>The same committed numbers, never re-measured</b> — every value below is "
        "re-aggregated from the per-seed vectors already in <code>result.json</code>, under "
        "defensible alternative schemes (equal-weight-by-grid mean, median, leave-one-grid-out, "
        "seeded percentile bootstrap). A headline that survives all of them is robust to the "
        "aggregation choice; a rank-sensitive one is flagged, not hidden.</p>"
        f"<div class=counts><div class=k><b>{n_stable}</b>of {n_scored} scored task headlines "
        "rank-stable under every scheme</div></div>" + "".join(sections)
    )


def build_ground_truth(values: dict[str, str], cov: CoverageView) -> str:
    """The lay explainer (UnderstandingVCS ground_truth.html analogue): where the labels come
    from, the three dataset families, how results are graded, and enriched per-grid cards."""
    assets = _assets()

    fam_cards = (
        "<div class=pillars>"
        "<div class=pillar><b>adapt_grid — the in-distribution family</b>"
        "Simulated at each grid's nominal operating point. Split 70/15/15 into train / val / "
        "test at the <i>episode</i> level: every window of one simulated event stays in one "
        "partition, so a model can never see a different slice of an evaluation episode during "
        "training.</div>"
        "<div class=pillar><b>benchmark — the never-trained family</b>"
        "A second simulation campaign at a <i>shifted operating point</i>. It is scored for "
        "every cell but never fitted or model-selected on — every run record carries a "
        "no-tuning attestation. Generalisation here is the benchmark's headline claim.</div>"
        "<div class=pillar><b>multi_grid — the pre-training corpus</b>"
        "Many synthetic 20 kV topologies in a portable 12-channel line view; the source domain "
        "for the transfer protocols (pretrain → zero-shot / fine-tune on unseen grids).</div>"
        "</div>"
    )
    grading = (
        "<div class=pillars>"
        "<div class=pillar><b>Balanced accuracy / macro-F1</b>Classification headline: every "
        "class weighs equally, so the majority class cannot buy a good score — and the "
        "majority baseline must land exactly on the analytic 1/k floor, which the floor gate "
        "checks over the whole matrix.</div>"
        "<div class=pillar><b>Missed-fault &amp; false-alarm rate</b>Protection-specific "
        "asymmetry for binary tasks: a missed fault (FN) and a needless trip (FP) have very "
        "different real-world costs, so both are always reported next to the headline.</div>"
        "<div class=pillar><b>MAE / RMSE / R²</b>Fault-localisation regression, in the grid's "
        "own distance units; the majority analogue (train-mean predictor) is the floor.</div>"
        "</div>"
    )

    # per-grid cards: spec, families, class balance (committed splits meta), topology thumb,
    # coverage badge — the UnderstandingVCS per-game card grid, grid-shaped.
    cards = []
    for grid in sorted(GRIDS, key=_grid_sort_key):
        held = grid in REFERENCE_GRIDS
        meta_path = (
            _REPO
            / "src"
            / "evemtbench"
            / "splits"
            / f"v{SPLITS_VERSION}"
            / "held_out"
            / grid
            / "meta.json"
        )
        hist_html = ""
        if meta_path.is_file():
            hists = json.loads(meta_path.read_text()).get("class_histograms", {})
            test_h = hists.get("test", {})
            if test_h:
                mx = max(test_h.values())
                bars = "".join(
                    f"<div class=hrow><span class=hlab>{_esc(c)}</span>"
                    f'<span class=hbar style="width:{max(4, int(120 * n / mx))}px"></span>'
                    f"<span class=hn>{n}</span></div>"
                    for c, n in sorted(test_h.items())
                )
                hist_html = f"<details><summary>test-split class balance</summary>{bars}</details>"
        topo = assets["topologies"].get(grid)
        thumb = (
            f'<img src="assets/{_esc(topo["figure"])}" alt="topology" loading="lazy">'
            if topo
            else "<p class=v>topology diagram pending aux extraction</p>"
        )
        d = sum(1 for c, s in cov.all_cells() if c.grid == grid and s == "done")
        p_ = sum(1 for c, s in cov.all_cells() if c.grid == grid and s != "done")
        role = "reference grid (held_out + transfer)" if held else "pre-training corpus"
        cards.append(
            f'<div class=gate><h3><a href="g_{_esc(grid)}.html">{_esc(grid)}</a> '
            f"<span class=dim>· {_esc(_grid_spec_label(grid))}</span></h3>"
            f"<p class=v>{role} · cells: {d} done / {p_} pending</p>{thumb}{hist_html}</div>"
        )

    return (
        "<p><b>Where the ground truth comes from.</b> Every window's label derives from the "
        "PowerFactory EMT simulation script that <i>injected</i> the event — the fault type, "
        "location, phases or switching action are simulation inputs, not annotations. Labels "
        "are therefore exact by construction: there is no annotation noise and no external "
        "label source to verify. What CAN go wrong — and what the "
        '<a href="gates.html">six gates</a> exist to catch — is leakage between partitions, '
        "contamination of the held-out sets, published tables drifting from artifacts, and "
        "class-imbalance artefacts.</p>"
        "<h2>The three dataset families</h2>" + fam_cards + "<h2>How results are graded</h2>"
        "<p class=v>the metric set is a frozen contract "
        f"({_src('src/evemtbench/evaluation/metrics.py')}); changing it is a major version "
        "bump — results across versions are never comparable</p>"
        + grading
        + f"<h2>The grids</h2><p class=v>{values['n_grids']} reference grids spanning "
        "20–345 kV plus the pre-training corpus; class balances read from the committed "
        f"split metadata ({_src(f'src/evemtbench/splits/v{SPLITS_VERSION}')})</p>"
        f'<div class=cardnav style="grid-template-columns:repeat(auto-fit,minmax(340px,1fr))">'
        + "".join(cards)
        + "</div>"
    )


def build_paper_numbers(
    rows: list[Row], prov: dict | None, *, version: str = BENCHMARK_VERSION
) -> str:
    """The paper-numbers page (B3): every number the manuscript reports, traced to the dashboard.

    The paper's result numbers are 100% mechanical — each cell of ``generated/tab_res_<fn>.tex`` is
    the best learned baseline for (task, grid, observability view, held_out test set), selected by
    ``make_paper_tables.best_cell_choice`` from the SAME ``result.json`` the dashboard reads. This
    page renders that identical selection, so the number a reviewer reads in the paper is provably
    the number here — and every populated number carries its five-artifact trace + W&B run.
    ``tests/integration/test_paper_numbers_covered.py`` enforces the paper<->dashboard agreement."""
    from make_paper_tables import GRID_ROWS, OBS, PAPER_FUNCTIONS, TEST_SETS, best_cell_choice
    from make_results import load_rows

    per_grid = {g: load_rows(g) for g, _ in GRID_ROWS}
    _TS = {"test": "in-distribution (test)", "benchmark": "never-trained (benchmark)"}
    n_numbers = 0
    sections = []
    for key, name, stem, metric, _label in PAPER_FUNCTIONS:
        arrow = "↓" if METRIC_DIRECTION.get(metric, 1) < 0 else "↑"
        head_cells = "".join(
            f"<th>{_esc(obs)}<br><span class=hn>{_esc(_TS[ts])}</span></th>"
            for ts in TEST_SETS
            for obs in OBS
        )
        body_rows = []
        for grid, _ in GRID_ROWS:
            tds = []
            for ts in TEST_SETS:
                for obs in OBS:
                    task_id = f"{stem}_{obs}"
                    choice = best_cell_choice(per_grid[grid], task_id, ts, metric)
                    if not choice:
                        tds.append("<td class=dim>—</td>")
                        continue
                    n_numbers += 1
                    base = choice["baseline"]
                    mean, ci, n = choice["mean"], choice["ci"], choice["n_seeds"]
                    val = f"{mean:.3f} ± {ci:.3f}" if ci is not None else f"{mean:.3f}"
                    cell_key = f"{task_id}/held_out/{grid}/{base}"
                    prov_cell = (prov or {}).get(cell_key)
                    repro = (prov_cell or {}).get("reproduce", "")
                    tds.append(
                        f"<td>{val} <span class=dim>(n={n})</span><br>"
                        f'<span class=dim style="font-size:10px" title="{_esc(repro)}">'
                        f"[{_esc(base)}]</span><br>"
                        f'<span style="font-size:10px">{_trace_strip(prov_cell)}</span></td>'
                    )
            body_rows.append(
                f'<tr><td><a href="g_{_esc(grid)}.html">{_esc(grid)}</a></td>{"".join(tds)}</tr>'
            )
        # Link the PUBLIC results bundle, not the private paper/manuscript/ tree (never published).
        frag = f"results/{version}/tables/tab_res_{key}.tex"
        disp = re.sub(r"\\ac\{(\w+)\}", lambda m: m.group(1).upper(), name)  # \ac{fd} -> FD
        sections.append(
            f"<h2>{_esc(disp)} <span class=dim>· headline {_metric_link(metric)} {arrow}</span></h2>"
            f"<p class=v>paper table <code>tab:res_{_esc(key)}</code> · table {_src(frag)} · "
            "best learned baseline per grid × view × test set</p>"
            "<div class=tablewrap><table><thead><tr><th>grid</th>"
            f"{head_cells}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
        )
    empty_note = (
        f"<div class=note>No paper numbers are wired yet — the v{_esc(BENCHMARK_VERSION)} sweep "
        "has not run, so every "
        "generated table cell is <code>\\res</code> and every cell below is “—”. The mapping is "
        "already fixed: once the sweep populates <code>outputs/</code>, each cell becomes a traced "
        "number here and in the manuscript simultaneously (same selection, same "
        "<code>result.json</code>).</div>"
        if n_numbers == 0
        else ""
    )
    metrics_used = tuple(dict.fromkeys(m for _, _, _, m, _ in PAPER_FUNCTIONS))
    return (
        "<p>Every result number in the manuscript lives <b>only</b> in auto-generated tables, "
        "published in the committed results bundle "
        f"<code>results/{_esc(version)}/tables/*.tex</code> ({_src('hpc/make_results_bundle.py')}; "
        f"built from the same code as {_src('hpc/make_paper_tables.py')} + "
        f"{_src('hpc/make_supplement_tables.py')}) — never hand-typed in prose, and reproducible "
        "from the aggregated data in the same bundle "
        f"(<code>results/{_esc(version)}/aggregated_results.csv</code>). Each is the best learned "
        "baseline for one (task, grid, observability "
        "view, <code>held_out</code> test set), chosen by the same code and the same "
        "<code>result.json</code> the dashboard reads. So <b>every paper number is on this page</b>, "
        "and every number carries the five links a reviewer needs — <b>output file</b>, "
        "<b>aggregation/eval code</b>, <b>training log</b>, <b>training code</b>, <b>config</b> — plus "
        "its <b>W&amp;B</b> run and (on hover of the winning baseline) its exact reproduce command. "
        "The committed backing store is the provenance index "
        f"({_src(f'audit/provenance/{version}/index.json', 'index.json')}); the paper↔dashboard "
        "agreement is a test gate.</p>"
        f"{empty_note}"
        f"<p class=v>headline result tables ({n_numbers} traced numbers). Full per-baseline numbers "
        "are on the per-baseline pages (linked from each grid/task page); the Supplementary tables "
        "draw from the same cells.</p>" + "".join(sections) + _metric_glossary(metrics_used)
    )


# ---------------------------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------------------------
def build_audit_pages(
    rows: list[Row],
    cov: CoverageView,
    records: RunRecordStats,
    *,
    headline_seeds: dict | None = None,
    generated_at: str | None = None,
    prov_cells: dict[str, dict] | None = None,
) -> dict[str, str]:
    """{relative path under audit/: html} for every page. Pure — no I/O writes, no clock.

    Pass ``prov_cells`` (a ``provenance_cells`` result) to share the one expensive per-cell join
    with ``build_provenance_index`` — ``fresh_views`` does; standalone callers may omit it."""
    # The audit is the provenance record of the CURRENT results. scan_results returns rows for
    # EVERY version present under outputs/ (superseded prior-version data may still be on disk), so
    # keep only each protocol's current rows (results_version(protocol) — held_out stays at its
    # frozen version, pretrain/transfer at BENCHMARK_VERSION) — otherwise a superseded number would
    # be published on the audit site with no matching provenance/trace, contradicting the
    # traceability guarantee. Mirrors make_results / the coverage scan / the provenance index /
    # paper_numbers.
    rows = [r for r in rows if r.version == results_version(r.protocol)]
    values = compute_values(rows, cov, records)
    prov = prov_cells if prov_cells is not None else provenance_cells(rows, records)
    headline_seeds = headline_seeds or {}
    pages: dict[str, str] = {}

    def add(name: str, title: str, body: str, active: str | None = None) -> None:
        pages[name] = _page(active or name, title, body, generated_at=generated_at)

    add("index.html", "EvEMTBench — results audit", build_index(values, cov))
    add("ledger.html", "Claim ledger", build_ledger(values, cov))
    add("paper_numbers.html", "Paper numbers — traced", build_paper_numbers(rows, prov))
    add("coverage.html", "Coverage matrix", build_coverage(cov))
    add("gates.html", "Verification gates — code tour", build_gates(cov))
    add("environment.html", "Environment & determinism", build_environment(values, records))
    add("reproduce.html", "Reproduce", build_reproduce(values))
    add("provenance.html", "Provenance of the process", build_provenance())
    add("ground_truth.html", "Ground truth & grading", build_ground_truth(values, cov))
    add(
        "robustness.html", "Robustness — alternative aggregations", build_robustness(headline_seeds)
    )
    for task in list_tasks():
        add(
            f"t_{task}.html",
            f"Task — {task}",
            build_task_page(task, rows, cov, prov),
            "coverage.html",
        )
    for grid in sorted(GRIDS, key=_grid_sort_key):
        add(
            f"g_{grid}.html",
            f"Grid — {grid}",
            build_grid_page(grid, rows, cov, prov),
            "coverage.html",
        )
    for baseline in ALL_BASELINES:
        add(
            f"b_{baseline}.html",
            f"Baseline — {baseline}",
            build_baseline_page(baseline, rows, prov),
            "coverage.html",
        )
    return pages


def fresh_pages(
    outputs: str | Path = "outputs", *, generated_at: str | None = None
) -> dict[str, str]:
    """One-call entry point for the freshness gate and main() — the HTML site pages only."""
    out = Path(outputs)
    rows = scan_results(out) if out.is_dir() else []
    cov = collect_coverage(out)
    records = collect_run_records(out)
    seeds = collect_headline_seeds(out)
    return build_audit_pages(rows, cov, records, headline_seeds=seeds, generated_at=generated_at)


def fresh_provenance(outputs: str | Path = "outputs") -> dict[str, str]:
    """The committed provenance index as {rel-path-under-audit: json}. A SEPARATE gated view (not
    an HTML page): the offline backing store that makes every per-number trace link resolve in the
    repo-only site. Deterministic (no timestamp), so the freshness gate byte-compares it like the
    other committed views (RESULTS.md / coverage_snapshot.json)."""
    out = Path(outputs)
    rows = scan_results(out) if out.is_dir() else []
    records = collect_run_records(out)
    return build_provenance_index(rows, records)


def fresh_views(
    outputs: str | Path = "outputs", *, generated_at: str | None = None
) -> tuple[dict[str, str], dict[str, str]]:
    """(pages, provenance-index) from ONE outputs scan and ONE per-cell provenance join.

    ``fresh_pages`` + ``fresh_provenance`` each rescan outputs/ and rebuild the (large,
    machine-neutralized) provenance cells; callers that need both — main() and the freshness
    gate, i.e. every hourly loop cycle and every gated commit — use this instead and pay once."""
    out = Path(outputs)
    rows = scan_results(out) if out.is_dir() else []
    cov = collect_coverage(out)
    records = collect_run_records(out)
    seeds = collect_headline_seeds(out)
    cells = provenance_cells(rows, records)
    pages = build_audit_pages(
        rows, cov, records, headline_seeds=seeds, generated_at=generated_at, prov_cells=cells
    )
    prov = build_provenance_index(rows, records, cells=cells)
    return pages, prov


def _git_short_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=_REPO)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the results-audit site (audit/*.html).")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--out", default="audit", help="output directory")
    args = ap.parse_args(argv)
    from datetime import datetime, timezone

    stamp = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · git {_git_short_sha()}"
    pages, prov = fresh_views(args.outputs, generated_at=stamp)  # one scan feeds both views
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, content in sorted({**pages, **prov}.items()):
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)  # provenance/<version>/index.json is nested
        dest.write_text(content)
    print(f"wrote {len(pages)} pages + {len(prov)} provenance file(s) to {out}/ (updated {stamp})")


if __name__ == "__main__":
    main()
