"""Internal visual leaderboard — aggregate ``result.json`` artifacts into a static HTML page.

Scans an ``outputs/`` tree for the ``result.json`` files written by ``evemtbench-run`` and
renders a self-contained, sortable HTML leaderboard (one section per task, rows = baseline ×
protocol, a bar for the headline metric). No server, no external assets — open the file in a
browser. Intended for internal tracking, not as part of the frozen benchmark contract.

    evemtbench-leaderboard --outputs ./outputs --out leaderboard.html
"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .evaluation.metrics import METRIC_DIRECTION
from .evaluation.significance import ci95_halfwidth
from .tasks.registry import get_task, list_tasks

# Registry size — the per-grid coverage denominator (was hard-coded to 9).
_N_TASKS = len(list_tasks())

# Docs live as Markdown in the repo; link them in the public GitHub blob UI.
DOCS_BASE = "https://github.com/EvEMTBench/evemtbench-benchmark/blob/main"

# Per-task headline metric: the first present wins (classification → regression).
HEADLINE_PRIORITY = ("balanced_accuracy", "mae")
# Headline metrics that are absolute in [0,1] and higher-is-better (bar = the value itself).
# Others (e.g. mae, rmse, r2) get a section-relative bar oriented by METRIC_DIRECTION.
_ABSOLUTE_01 = {"balanced_accuracy", "accuracy", "macro_f1"}


@dataclass
class Row:
    task: str
    protocol: str
    grid: str
    baseline: str
    test_set: str
    version: str
    n_test: int
    seeds: int
    metrics: dict[str, dict[str, float]]  # metric -> {mean, std, ci95}
    n_params: int | None = None  # trainable params (capacity, reported not equalized)
    fit_seconds: float | None = None  # mean per-seed training wall-time (compute reporting)
    source: str | None = None  # transfer pre-training source grid (None = non-transfer / 20 kV)

    def mean(self, metric: str) -> float | None:
        m = self.metrics.get(metric)
        return None if m is None else m.get("mean")


def _fmt_params(n: int | None) -> str:
    """Humanize a trainable-parameter count (1.6M / 52k / —)."""
    if not n:
        return "—"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.0f}k"
    return str(n)


def _fmt_secs(s: float | None) -> str:
    """Humanize a wall-time in seconds (1.1h / 3.2m / 12s / —)."""
    if not s:
        return "—"
    if s >= 3600:
        return f"{s / 3600:.1f}h"
    if s >= 60:
        return f"{s / 60:.1f}m"
    return f"{s:.0f}s"


def scan_results(outputs_dir: str | Path) -> list[Row]:
    """Parse every ``result.json`` under ``outputs_dir`` into a Row (keyed by its own fields,
    so it is robust to directory layout)."""
    rows: list[Row] = []
    for p in sorted(Path(outputs_dir).rglob("result.json")):
        try:
            doc = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        agg = doc.get("aggregate", {})
        metrics = {k: v for k, v in agg.items() if isinstance(v, dict)}
        # 95% CI half-width per metric: prefer the runner-stored ``ci95``; derive it from the
        # persisted per-seed vector for artifacts written before ci95 existed (so confidence
        # intervals render for every done cell without re-running anything).
        per_seed = doc.get("per_seed") or []
        for k, v in metrics.items():
            if "ci95" not in v:
                vals = [(r.get("metrics") or {}).get(k) for r in per_seed]
                metrics[k] = {**v, "ci95": ci95_halfwidth(vals) if vals else None}
        # Per-class F1 (``f1_<class>``) is only defined for classes that occur in this eval set.
        # A class absent here (e.g. HIF on a non-MV grid) has no support, so its F1 is n/a — not a
        # genuine 0.0. The event_type breakdown lists exactly the classes present; flag the rest so
        # the board shows "n/a" instead of a misleading "0.000". (Guard older artifacts that stored
        # absent-class F1 as 0.0, as well as new ones that already store it as null.)
        present = set((doc.get("breakdowns") or {}).get("event_type", {}).keys())
        if present:
            for k, v in metrics.items():
                if k.startswith("f1_") and k[3:] not in present:
                    metrics[k] = {
                        **v,
                        "mean": None,
                        "std": None,
                        "ci95": None,
                        "n_defined": 0,
                        "absent": True,
                    }
        rows.append(
            Row(
                task=doc.get("task", "?"),
                protocol=doc.get("protocol", "?"),
                grid=doc.get("grid", "?"),
                baseline=doc.get("baseline", "?"),
                test_set=doc.get("test_set", "test"),
                version=doc.get("benchmark_version", "?"),
                n_test=int(doc.get("n_test", 0)),
                seeds=len(doc.get("seeds", []) or []),
                metrics=metrics,
                n_params=doc.get("n_params"),
                fit_seconds=(doc.get("compute") or {}).get("fit_seconds_mean"),
                source=doc.get("pretrain_grid"),
            )
        )
    return rows


def _headline_metric(trows: list[Row]) -> str:
    """The ranking/bar metric for a task section: first HEADLINE_PRIORITY metric present, else any
    present metric (deterministic) so every task section always has a headline."""
    present = set().union(*[set(r.metrics) for r in trows]) if trows else set()
    for m in HEADLINE_PRIORITY:
        if m in present:
            return m
    return next(iter(sorted(present)), "balanced_accuracy")


def _fmt(row: Row, metric: str) -> str:
    m = row.metrics.get(metric)
    # Three distinct states: a real value, "n/a" (class has no samples on this grid/test set — an
    # undefined metric, NOT zero skill), and "—" (metric simply not recorded for this run).
    if m and m.get("absent"):
        return (
            '<span class=na title="no samples of this class on this grid/test set — '
            'metric undefined, not zero performance">n/a</span>'
        )
    if not m or m.get("mean") is None:
        return "—"
    mean = m["mean"]
    ci = m.get("ci95")
    if ci is not None:
        # mean stays the leading token: the column sorter parseFloats the first number in a cell.
        return (
            f'<span title="95% CI half-width (Student t across seeds)">{mean:.3f} ± {ci:.3f}</span>'
        )
    std = m.get("std")
    return f"{mean:.3f} ± {std:.3f}" if std else f"{mean:.3f}"


def _grid_meta(key: str) -> dict | None:
    """Topology spec (level/voltage/frequency) from the grid registry; None for unknown keys
    (e.g. test fixtures), so the board degrades gracefully."""
    try:
        from .data.roots import get_grid

        g = get_grid(key)
        return {"level": g.level, "voltage_kv": g.voltage_kv, "frequency_hz": g.frequency_hz}
    except Exception:
        return None


def _grid_sort_key(key: str):
    """Order grids by voltage (MV → HV → EHV); unknown grids sort last, alphabetically."""
    m = _grid_meta(key)
    return (0, m["voltage_kv"], key) if m else (1, 0.0, key)


def _grid_spec_label(key: str) -> str:
    m = _grid_meta(key)
    return f"{m['level']} · {m['voltage_kv']:g} kV · {m['frequency_hz']} Hz" if m else "—"


def _coverage_panel(rows: list[Row]) -> str:
    """A prominent per-grid coverage panel (all four topologies first-class): one card per grid in
    voltage order with run count, tasks covered, and benchmark-set availability."""
    if not rows:
        return ""
    grids = sorted({r.grid for r in rows}, key=_grid_sort_key)
    cards = []
    for g in grids:
        grows = [r for r in rows if r.grid == g]
        n_tasks = len({r.task for r in grows})
        has_bench = any(r.test_set == "benchmark" for r in grows)
        bench = (
            "<span class=ok>✓ benchmark</span>"
            if has_bench
            else "<span class=dim>fine-tune only</span>"
        )
        cards.append(
            f"<div class=gridcard><div class=gridkey>{html.escape(g)}</div>"
            f"<div class=gridspec>{_grid_spec_label(g)}</div>"
            f"<div class=gridstat>{len(grows)} runs · {n_tasks}/{_N_TASKS} tasks</div>"
            f"<div class=gridstat>{bench}</div></div>"
        )
    return (
        "<section class=coverage><h2>Grids "
        "<span class=dim>· four reference topologies (MV · HV · EHV)</span></h2>"
        f"<div class=gridcards>{''.join(cards)}</div></section>"
    )


def _comparison_section(rows: list[Row]) -> str:
    """Per-topology comparison: one matrix per task (rows = baseline, columns = the grids in
    voltage order, cells = the task's headline metric), so the *same task across all grids* is
    legible at a glance. A test/benchmark toggle switches the held-out set shown."""
    grids = sorted({r.grid for r in rows}, key=_grid_sort_key)
    test_sets = [ts for ts in ("test", "benchmark") if any(r.test_set == ts for r in rows)]
    if len(grids) < 2 or not test_sets:
        return ""  # nothing meaningful to compare across topologies
    tasks = sorted({r.task for r in rows})
    blocks = []
    for ts in test_sets:
        task_tables = []
        for task in tasks:
            trows = [r for r in rows if r.task == task]
            headline = _headline_metric(trows)
            if not headline:
                continue
            protos = [r.protocol for r in trows]
            primary = "held_out" if "held_out" in protos else Counter(protos).most_common(1)[0][0]
            direction = METRIC_DIRECTION.get(headline, 1)
            idx = {
                (r.baseline, r.grid): r for r in trows if r.protocol == primary and r.test_set == ts
            }
            if not idx:
                continue
            baselines = sorted({b for b, _ in idx})

            def _best(b, *, headline=headline, idx=idx):
                vals = [
                    idx[(b, g)].mean(headline)
                    for g in grids
                    if (b, g) in idx and idx[(b, g)].mean(headline) is not None
                ]
                return max(vals) if vals else None

            baselines.sort(key=lambda b: (_best(b) is None, -direction * (_best(b) or 0)))
            head = (
                "<tr><th>baseline</th>"
                + "".join(
                    f"<th>{html.escape(g)}<br><span class=gridspec>{_grid_spec_label(g)}</span></th>"
                    for g in grids
                )
                + "</tr>"
            )
            body = []
            for b in baselines:
                cells = "".join(
                    f"<td>{_fmt(idx[(b, g)], headline) if (b, g) in idx else '—'}</td>"
                    for g in grids
                )
                body.append(f"<tr><td class=base>{html.escape(b)}</td>{cells}</tr>")
            task_tables.append(
                f"<div class=cmptask><h3>{html.escape(task)} "
                f"<span class=dim>· {html.escape(headline)} · {html.escape(primary)}</span></h3>"
                f"<div class=tablewrap><table class=sortable>"
                f"<caption>{html.escape(task)} across topologies ({ts})</caption>"
                f"<thead>{head}</thead><tbody>{''.join(body)}</tbody></table></div></div>"
            )
        disp = "" if ts == test_sets[0] else "display:none"
        blocks.append(
            f'<div class=cmpset data-cmpts="{ts}" style="{disp}">{"".join(task_tables)}</div>'
        )
    toggle = ""
    if len(test_sets) > 1:
        opts = "".join(f'<option value="{ts}">{ts}</option>' for ts in test_sets)
        toggle = f"<label>test set <select id=cmpts>{opts}</select></label>"
    return (
        "<section class=compare><h2>Topology comparison "
        "<span class=dim>· same task across all grids</span></h2>"
        f"<div class=cmpctl>{toggle}</div>{''.join(blocks)}</section>"
    )


_TRANSFER_PROTO_LABELS = [("transfer_zeroshot", "zero-shot"), ("transfer_finetune", "fine-tune")]
_DEFAULT_SOURCE = (
    "template_20kv"  # legacy transfer results (no pretrain_grid field) sourced from 20 kV
)


def _source_kv(src: str | None) -> float:
    """Voltage (kV) of a transfer pre-training SOURCE grid; None (legacy) => 20 kV."""
    s = (src or _DEFAULT_SOURCE).replace("template_", "").replace("kv", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _source_label(src: str | None) -> str:
    return f"{_source_kv(src):g} kV"


def _transfer_section(rows: list[Row]) -> str:
    """Cross-grid transfer (OOD): per task, zero-shot vs fine-tune from the multi_grid pre-training,
    across the reference grids in **voltage order** (each annotated same-V / cross-V vs the 20 kV
    source), on the held-out ``benchmark`` set, best baseline per cell. Empty if no transfer runs."""
    trows = [
        r
        for r in rows
        if r.protocol in ("transfer_zeroshot", "transfer_finetune") and r.test_set == "benchmark"
    ]
    if not trows:
        return ""
    grids = sorted({r.grid for r in trows}, key=_grid_sort_key)
    task_tables = []
    for task in sorted({r.task for r in trows}):
        task_rows = [r for r in trows if r.task == task]
        headline = _headline_metric(task_rows)
        if not headline:
            continue
        direction = METRIC_DIRECTION.get(headline, 1)

        srcs = sorted({r.source or _DEFAULT_SOURCE for r in task_rows}, key=_source_kv)

        def _best(proto, src, g, *, headline=headline, direction=direction, task_rows=task_rows):
            cand = [
                r
                for r in task_rows
                if r.protocol == proto
                and (r.source or _DEFAULT_SOURCE) == src
                and r.grid == g
                and r.mean(headline) is not None
            ]
            return max(cand, key=lambda r: direction * r.mean(headline)) if cand else None

        head = (
            "<tr><th>protocol · source</th>"
            + "".join(
                f"<th>{html.escape(g)}<br><span class=gridspec>{_grid_spec_label(g)}</span></th>"
                for g in grids
            )
            + "</tr>"
        )
        body = []
        for proto, label in _TRANSFER_PROTO_LABELS:
            for src in srcs:  # source axis: one row per (protocol, pre-training source)
                cells = "".join(
                    f"<td>{_fmt(_best(proto, src, g), headline) if _best(proto, src, g) else '—'}</td>"
                    for g in grids
                )
                body.append(f"<tr><td class=base>{label} · {_source_label(src)}</td>{cells}</tr>")
        task_tables.append(
            f"<div class=cmptask><h3>{html.escape(task)} "
            f"<span class=dim>· {html.escape(headline)} · best baseline · benchmark</span></h3>"
            f"<div class=tablewrap><table class=sortable>"
            f"<caption>{html.escape(task)} — cross-grid transfer (benchmark)</caption>"
            f"<thead>{head}</thead><tbody>{''.join(body)}</tbody></table></div></div>"
        )
    if not task_tables:
        return ""
    return (
        "<section class=compare><h2>Cross-grid transfer (OOD) "
        "<span class=dim>· pretrain on the multi_grid corpus (20/110/345 kV sources) → "
        "zero-shot / fine-tune on unseen grids (voltage-ordered)</span></h2>"
        f"{''.join(task_tables)}</section>"
    )


# --- shared page chrome (also imported by hpc/make_audit.py — keep content byte-stable) ---
BASE_CSS = """
    body{font:14px/1.5 system-ui,sans-serif;margin:24px;color:#1a1a1a;background:#fff}
    h1{margin:0 0 4px} .v{color:#666;font-size:13px;margin:0 0 4px}
    header{border-bottom:1px solid #e5e5e5;padding-bottom:10px;margin-bottom:12px}
    .nav{font-size:13px;margin-bottom:8px} .nav a{color:#1890ff;text-decoration:none}
    .nav a:hover{text-decoration:underline}
    .note{background:#fff7e6;border:1px solid #ffd591;padding:8px 12px;border-radius:6px;margin:12px 0;font-size:13px}
    details.legend,details.cite{margin:12px 0;font-size:13px} details summary{cursor:pointer;font-weight:600}
    details.legend ul{margin:6px 0 0 18px} details.cite pre{background:#f6f6f6;padding:8px;border-radius:6px;overflow:auto;font-size:12px}
    h2{margin:24px 0 8px;font-size:16px} h3{margin:14px 0 4px;font-size:14px}
    .tablewrap{overflow-x:auto;-webkit-overflow-scrolling:touch;margin-bottom:8px}
    table{border-collapse:collapse;width:100%}
    caption{caption-side:top;text-align:left;color:#888;font-size:12px;padding:2px 0}
    th,td{border:1px solid #e5e5e5;padding:6px 9px;text-align:left;white-space:nowrap}
    th{background:#fafafa;font-weight:600;cursor:pointer;user-select:none}
    thead th{position:sticky;top:0;z-index:1}
    td.base{font-weight:600} td.dim{color:#888;text-align:right}
    .na{color:#8a8a8a;font-style:italic;cursor:help}
    td.bg{min-width:160px}
    .bar{position:relative;background:#f0f0f0;border-radius:3px;height:18px}
    .bar span{position:absolute;left:0;top:0;bottom:0;background:linear-gradient(90deg,#69c0ff,#1890ff);border-radius:3px}
    .bar em{position:relative;padding-left:6px;font-style:normal;font-size:12px;line-height:18px}
    tr:hover td{background:#f6fbff}
    .vfilter{margin:12px 0;font-size:13px;display:flex;flex-wrap:wrap;gap:14px;align-items:center}
    .vfilter select,.cmpctl select{font:13px system-ui;padding:2px 4px;margin-left:4px}
    #fcount{margin-left:4px;color:#888}
    .gridcards{display:flex;flex-wrap:wrap;gap:10px}
    .gridcard{border:1px solid #e5e5e5;border-radius:8px;padding:10px 14px;min-width:150px;background:#fafcff}
    .gridkey{font-weight:600} .gridspec{color:#888;font-size:12px;font-weight:400}
    .gridstat{font-size:12px;color:#555} .ok{color:#389e0d}
    .compare{margin-top:8px} .cmpctl{font-size:13px;margin:4px 0}
    .cmptask{margin-bottom:6px}
    .cta{display:inline-block;background:#1890ff;color:#fff;padding:6px 12px;border-radius:6px;
    text-decoration:none;font-size:13px;margin:8px 0}.cta:hover{background:#096dd9}
    footer{border-top:1px solid #e5e5e5;margin-top:24px;padding-top:8px;color:#888;font-size:12px}
    footer a{color:#1890ff;text-decoration:none}
    @media (prefers-color-scheme:dark){
      body{background:#15171a;color:#e6e6e6}
      header,footer{border-color:#2a2f36}
      th{background:#1e2227}thead th{background:#1e2227} th,td{border-color:#2a2f36}
      .note{background:#2b2410;border-color:#5c4a12}
      .bar{background:#262b31} tr:hover td{background:#1b2733}
      .nav a,footer a{color:#69c0ff} .na{color:#8a8a8a} td.dim,.v,caption,.gridspec{color:#9aa0a6}
      .gridcard{background:#1b1e22;border-color:#2a2f36} .gridstat{color:#b0b4b9} .ok{color:#73d13d}
      details.cite pre{background:#1e2227}
    }
    """

SORT_JS = """
    // Click-to-sort, tbody-scoped (header in thead). Toggles ▲/▼ + aria-sort on the active column.
    document.querySelectorAll('table.sortable').forEach(t=>{
      const heads=t.querySelectorAll('thead th'); const body=t.tBodies[0];
      if(!heads.length||!body) return;
      heads.forEach((th,i)=>th.addEventListener('click',()=>{
        const asc=th.dataset.asc=th.dataset.asc==='1'?'':'1';
        const num=v=>{const m=parseFloat((v||'').replace('±',' ').trim());return isNaN(m)?-1:m};
        heads.forEach(h=>{if(h!==th){h.removeAttribute('aria-sort');h.textContent=h.textContent.replace(/[ ▲▼]+$/,'');}});
        th.textContent=th.textContent.replace(/[ ▲▼]+$/,'')+(asc?' ▲':' ▼');
        th.setAttribute('aria-sort',asc?'ascending':'descending');
        [...body.rows].sort((a,b)=>{const x=num(a.cells[i].innerText),y=num(b.cells[i].innerText);return asc?x-y:y-x})
          .forEach(r=>body.appendChild(r));
      }));
    });
    // Topology-comparison test-set toggle (test ↔ benchmark): show one cmpset at a time.
    const cts=document.getElementById('cmpts');
    if(cts) cts.addEventListener('change',()=>{
      document.querySelectorAll('.cmpset').forEach(s=>{s.style.display=s.dataset.cmpts===cts.value?'':'none';});
    });
    // Combined filters on the detail tables: version / topology / test-set / model / protocol.
    const FILTERS=[['fv','v'],['fgrid','grid'],['fts','ts'],['fbase','base'],['fproto','proto']];
    const val=id=>{const e=document.getElementById(id);return e?e.value:'__all__';};
    function applyFilters(){
      let shown=0;
      document.querySelectorAll('tr[data-v]').forEach(r=>{
        const ok=FILTERS.every(([id,key])=>{const v=val(id);return v==='__all__'||r.dataset[key]===v;});
        r.style.display=ok?'':'none'; if(ok)shown++;
      });
      document.querySelectorAll('section.task').forEach(s=>{
        const vis=[...s.querySelectorAll('tr[data-v]')].some(r=>r.style.display!=='none');
        s.style.display=vis?'':'none';
      });
      const c=document.getElementById('fcount'); if(c)c.textContent=`(${shown} rows shown)`;
    }
    FILTERS.forEach(([id])=>{const e=document.getElementById(id);if(e)e.addEventListener('change',applyFilters);});
    applyFilters();
    """

# Inline-SVG favicon as a data URI: keeps the page asset-free (no network fetch).
FAVICON = (
    "data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E"
    "%3Crect width='16' height='16' rx='3' fill='%231890ff'/%3E"
    "%3Cpath d='M8 2 L3 9 h4 l-1 5 6-8 h-4 z' fill='white'/%3E%3C/svg%3E"
)


def nav_html(docs_base: str = DOCS_BASE, extra: tuple[tuple[str, str], ...] = ()) -> str:
    """The shared docs nav bar; ``extra`` appends (href, label) links (e.g. the audit site)."""

    def _doc(slug: str, label: str) -> str:
        return f'<a href="{html.escape(docs_base)}/docs/{slug}">{html.escape(label)}</a>'

    items = [
        f'<a href="{html.escape(docs_base)}/README.md">overview</a>',
        _doc("protocol.md", "protocol"),
        _doc("taxonomy.md", "taxonomy"),
        _doc("grids.md", "grids"),
        _doc("metrics.md", "metrics"),
        _doc("baselines.md", "baselines"),
        _doc("submission.md", "submit"),
    ]
    items += [f'<a href="{html.escape(href)}">{html.escape(label)}</a>' for href, label in extra]
    return "<nav class=nav>" + " · ".join(items) + "</nav>"


def render_html(
    rows: list[Row],
    *,
    version: str = "",
    title: str = "EvEMTBench leaderboard",
    generated_at: str | None = None,
    docs_base: str = DOCS_BASE,
    citation: str = "",
) -> str:
    """Render a self-contained HTML leaderboard string.

    ``generated_at`` (a preformatted string; ``None`` => omitted) keeps the function pure — the
    caller stamps the time so output stays deterministic for tests. ``docs_base`` is the blob-UI
    base for doc links; ``citation`` is an optional BibTeX block."""

    def _tier(t: str) -> int:
        try:
            tier = get_task(t).tier
        except KeyError:  # a task no longer in the registry (e.g. archived outputs)
            return 99
        return tier if tier is not None else 99

    _TIER_LABEL = {1: "Tier 1 — easy anchors", 2: "Tier 2 — medium", 3: "Tier 3 — hard frontier"}
    tasks = sorted({r.task for r in rows}, key=lambda t: (_tier(t), t))
    sections = []
    _cur_tier: object = None
    for task in tasks:
        _t = _tier(task)
        if _t != _cur_tier:
            _cur_tier = _t
            sections.append(f'<h2 class="tier">{_TIER_LABEL.get(_t, "Untiered")}</h2>')
        trows = [r for r in rows if r.task == task]
        headline = _headline_metric(trows)
        direction = METRIC_DIRECTION.get(headline, 1)  # +1 higher-better, -1 lower-better
        # metric columns = union of metrics seen for this task, headline first
        metric_cols: list[str] = []
        for r in trows:
            for k in r.metrics:
                if k not in metric_cols:
                    metric_cols.append(k)
        metric_cols.sort(key=lambda k: (k != headline, k))
        # rank by headline, oriented by direction (best first); missing values sink to the bottom
        trows.sort(key=lambda r: (r.mean(headline) is None, -direction * (r.mean(headline) or 0)))

        # bar scaling for non-[0,1] headline metrics: section min-max, oriented by direction.
        hvals = [v for r in trows if (v := r.mean(headline)) is not None]
        lo, hi = (min(hvals), max(hvals)) if hvals else (0.0, 1.0)

        def _bar(hv: float | None, *, headline=headline, lo=lo, hi=hi, direction=direction) -> str:
            if hv is None:
                return ""
            if headline in _ABSOLUTE_01:
                frac = max(0.0, min(1.0, hv))
            elif hi > lo:
                frac = (hv - lo) / (hi - lo)
                if direction < 0:  # lower-is-better → invert so the best is the fullest bar
                    frac = 1.0 - frac
            else:
                frac = 1.0
            return (
                f'<div class="bar"><span style="width:{frac * 100:.1f}%"></span>'
                f"<em>{hv:.3f}</em></div>"
            )

        body = []
        for r in trows:
            cells = "".join(f"<td>{_fmt(r, m)}</td>" for m in metric_cols)
            src_cell = (
                _source_label(r.source)
                if r.protocol in ("transfer_zeroshot", "transfer_finetune")
                else "—"
            )
            body.append(
                f'<tr data-v="{html.escape(r.version)}" data-grid="{html.escape(r.grid)}"'
                f' data-ts="{html.escape(r.test_set)}" data-base="{html.escape(r.baseline)}"'
                f' data-proto="{html.escape(r.protocol)}" data-source="{html.escape(r.source or "")}">'
                f"<td class=base>{html.escape(r.baseline)}</td>"
                f"<td>{html.escape(r.protocol)}</td>"
                f"<td class=dim>{html.escape(src_cell)}</td>"
                f"<td>{html.escape(r.grid)}</td>"
                f"<td>{html.escape(r.test_set)}</td>"
                f"<td class=dim>{html.escape(r.version)}</td>"
                f"<td class=bg>{_bar(r.mean(headline))}</td>{cells}"
                f"<td class=dim>{r.seeds or '—'}</td><td class=dim>{r.n_test or '—'}</td>"
                f"<td class=dim>{_fmt_params(r.n_params)}</td>"
                f"<td class=dim>{_fmt_secs(r.fit_seconds)}</td></tr>"
            )
        head = (
            "<tr><th>baseline</th><th>protocol</th><th>source</th><th>grid</th><th>test_set</th>"
            "<th>version</th>"
            f"<th>{html.escape(headline or 'headline')}</th>"
            + "".join(f"<th>{html.escape(m)}</th>" for m in metric_cols)
            + "<th>seeds</th><th>n_test</th><th>params</th><th>train/seed</th></tr>"
        )
        sections.append(
            f"<section class=task><h2>{html.escape(task)}</h2>"
            f"<div class=tablewrap><table class=sortable>"
            f"<caption>{html.escape(task)} — ranked by {html.escape(headline or 'headline')}</caption>"
            f"<thead>{head}</thead><tbody>{''.join(body)}</tbody></table></div></section>"
        )

    # Filter bar: version (default latest) + grid / test_set / baseline / protocol (default all).
    versions = sorted({r.version for r in rows}, reverse=True)
    latest = versions[0] if versions else ""
    vopts = (
        (
            f'<option value="{html.escape(latest)}">latest ({html.escape(latest)})</option>'
            if latest
            else ""
        )
        + "".join(
            f'<option value="{html.escape(v)}">{html.escape(v)}</option>'
            for v in versions
            if v != latest
        )
        + '<option value="__all__">all versions</option>'
    )

    def _sel(sid: str, label: str, values) -> str:
        opts = '<option value="__all__">all</option>' + "".join(
            f'<option value="{html.escape(v)}">{html.escape(v)}</option>' for v in sorted(values)
        )
        return f'<label>{html.escape(label)} <select id="{sid}">{opts}</select></label>'

    grids = {r.grid for r in rows}
    test_sets = {r.test_set for r in rows}
    baselines = {r.baseline for r in rows}
    protocols = {r.protocol for r in rows}
    vfilter = (
        "<div class=vfilter>"
        f"<label>version <select id=fv>{vopts}</select></label>"
        + _sel("fgrid", "topology", grids)
        + _sel("fts", "test set", test_sets)
        + _sel("fbase", "model", baselines)
        + (_sel("fproto", "protocol", protocols) if len(protocols) > 1 else "")
        + "<span id=fcount class=dim></span></div>"
    )

    note = (
        "Internal leaderboard — conditional results (task × protocol × input setting × metric). "
        "Aggregate metrics can hide failure modes; read protocol and per-class/grid breakdowns. "
        "Per-class F1: “n/a” = no samples of that class on this grid/test set (undefined, e.g. HIF "
        "is MV-only) — distinct from a genuine 0.000 (class present, model scored zero)."
    )
    css = BASE_CSS
    js = SORT_JS
    # --- page chrome (head/nav/legend/cta/footer) ---
    favicon = FAVICON

    nav = nav_html(docs_base, extra=(("audit/index.html", "audit"),))
    cta = f'<a class=cta href="{html.escape(docs_base)}/docs/submission.md">How to submit results →</a>'
    legend = (
        "<details class=legend><summary>Legend &amp; metric conventions</summary><ul>"
        "<li><b>headline</b> — balanced-accuracy (classification) or MAE (regression); rows are "
        "ranked by it and the bar is oriented so a fuller bar is better.</li>"
        "<li><span class=na>n/a</span> — the class has no samples on this grid/test set "
        "(metric undefined, e.g. HIF is MV-only). <b>Not</b> zero skill.</li>"
        "<li><b>0.000</b> — class present, model scored zero.</li>"
        "<li><b>± x.xxx</b> — 95% CI half-width over seeds (Student t, ddof=1); "
        "single-seed cells show the bare mean.</li>"
        "<li><b>—</b> — metric not recorded for this run.</li></ul></details>"
    )
    updated = f"<div class=v>updated {html.escape(generated_at)}</div>" if generated_at else ""
    cite = (
        f"<details class=cite><summary>Cite this benchmark</summary><pre>{html.escape(citation)}</pre>"
        "</details>"
        if citation
        else ""
    )
    footer = (
        "<footer>EvEMTBench · MIT · "
        f'<a href="{html.escape(docs_base)}/README.md">repository</a> · internal leaderboard'
        f"{cite}</footer>"
    )
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        '<meta name=description content="EvEMTBench leaderboard — a protocol-driven ML benchmark '
        'for power-system protection across four reference grids (MV/HV/EHV) on EMT waveforms.">'
        f'<link rel=icon href="{favicon}">'
        f"<title>{html.escape(title)}</title><style>{css}</style></head><body>"
        f"<header>{nav}<h1>{html.escape(title)}</h1>"
        f"<div class=v>benchmark version {html.escape(version or 'n/a')} · {len(rows)} runs total</div>"
        f"{updated}{cta}</header>"
        f"{_coverage_panel(rows)}"
        f"<div class=note>{html.escape(note)}</div>"
        f"{legend}"
        f"{_comparison_section(rows)}"
        f"{_transfer_section(rows)}"
        f"{vfilter if rows else ''}"
        f"{''.join(sections) or '<p>No result.json files found.</p>'}"
        f"{footer}"
        f"<script>{js}</script></body></html>"
    )


def _bibtex_from_cff(path: Path | None = None) -> str:
    """Build a BibTeX software entry from CITATION.cff so the page citation never drifts from it."""
    path = path or Path(__file__).resolve().parents[2] / "CITATION.cff"
    try:
        import yaml

        d = yaml.safe_load(path.read_text())
    except Exception:
        return ""
    authors = (
        " and ".join(
            f"{a.get('family-names', '')}, {a.get('given-names', '')}".strip(", ")
            for a in d.get("authors", [])
        )
        or "EvEMTBench"
    )
    fields = {
        "title": d.get("title", "EvEMTBench"),
        "author": authors,
        "version": d.get("version", ""),
        "url": d.get("repository-code", ""),
    }
    lines = ",\n".join(f"  {k:7s} = {{{v}}}" for k, v in fields.items() if v)
    return "@software{evemtbench,\n" + lines + "\n}"


def main(argv: list[str] | None = None) -> None:
    from datetime import datetime, timezone

    from .version import BENCHMARK_VERSION

    p = argparse.ArgumentParser(description="Render an internal HTML leaderboard from outputs/.")
    p.add_argument("--outputs", default="outputs", help="directory tree of result.json files")
    p.add_argument("--out", default="leaderboard.html", help="output HTML path")
    args = p.parse_args(argv)

    rows = scan_results(args.outputs)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html_str = render_html(
        rows, version=BENCHMARK_VERSION, generated_at=generated_at, citation=_bibtex_from_cff()
    )
    Path(args.out).write_text(html_str)
    print(f"leaderboard: {len(rows)} runs -> {args.out}")


if __name__ == "__main__":
    main()
