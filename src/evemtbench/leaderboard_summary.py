"""User-facing leaderboard: one ranked table per view, protocol and evaluation set.

The full per-task, per-grid page (``leaderboard.render_html``) stays the complete record for the
paper and reviewers. This page answers the user's question instead: *which model, and by how
much?* The design follows the survey in ``docs/internal/leaderboard_design_study.md``:

* **Skill score** (fev-bench, WeatherBench 2): for every cell, the model's error divided by the
  error of a fixed trivial reference (``majority``: the most frequent train class, or the train
  mean for regression), clipped to [0.01, 100]. The skill over a set of cells is
  ``1 - geometric mean of those ratios``: 0 = no better than the reference, 1 = perfect. It does
  not move when other entries are added.
* **Error** per cell is ``1 - balanced_accuracy`` for classification and ``mae`` for regression,
  the headline metrics of the full page.
* **Uncertainty**: a 95% bootstrap interval of the skill over cells, and a rank range (LMArena):
  the best and worst rank once only *paired* differences (same resampled cells for both models)
  that are significant at 95% count. Unpaired intervals overlap almost always, because the
  spread between cells dwarfs the differences between models.
* **Win rate** (fev-bench): the share of shared cells on which a model beats each other ranked
  entry, averaged over those entries. It changes as entries are added and is shown second.
* **Partial entries** (Matbench): only entries covering every cell of a view are ranked. Partial
  entries are listed below with their skill on their own cells and the best ranked entry's skill
  on the same cells.

Transfer protocols have no ``majority`` runs; their reference is the ``held_out`` majority of the
same task, target grid and evaluation set, used only when the evaluation set has the same size.
"""

from __future__ import annotations

import html
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cache

import numpy as np

from .evaluation.metrics import METRIC_DIRECTION
from .leaderboard import (
    _ABSOLUTE_01,
    _DEFAULT_SOURCE,
    _TIER_CSS,
    DOCS_BASE,
    FAVICON,
    HEADLINE_PRIORITY,
    Row,
    _is_current,
    _source_label,
    nav_html,
)
from .tasks import get_task

REFERENCE = "majority"
RATIO_CLIP = (0.01, 100.0)
N_BOOT = 1000
BOOT_SEED = 0  # fixed: the page must be byte-identical across rebuilds (freshness gate)

VIEWS = (("line", "Line"), ("local", "Local"), ("global", "Global"))
PROTOCOLS = (
    ("held_out", "Unseen grid"),
    ("transfer_zeroshot", "Zero-shot transfer"),
    ("transfer_finetune", "Fine-tune transfer"),
)
TEST_SETS = (("benchmark", "Benchmark set"), ("test", "Test set"))

FAMILY_LABELS = {
    "fault_detection": "Fault detection",
    "fault_class": "Fault classification",
    "fault_location": "Fault location",
    "fault_category": "Fault category",
    "fault_phase": "Faulted phases",
    "fault_grounded": "Grounded fault",
    "fault_origin": "Fault origin",
    "event_detection": "Event detection",
    "event_classification": "Event classification",
    "event_state": "Event state",
    "switch_detection": "Switching detection",
    "switch_type": "Switching type",
}
_FAMILY_ORDER = list(FAMILY_LABELS)


def headline(row: Row) -> tuple[str, float] | None:
    """The cell's headline metric and its error (lower is better), or None if not scorable.

    The metric is the full page's ``HEADLINE_PRIORITY`` choice. A lower-is-better metric is its
    own error; a higher-is-better metric bounded in [0, 1] (``_ABSOLUTE_01``) has error
    ``1 - value``. A missing or non-finite value (NaN/inf in a result.json) is not scored."""
    for metric in HEADLINE_PRIORITY:
        v = row.mean(metric)
        if v is None:
            continue
        if not math.isfinite(v):
            return None
        if METRIC_DIRECTION.get(metric, 1) < 0:
            return metric, v
        if metric in _ABSOLUTE_01:
            return metric, 1.0 - v
        return None  # higher-is-better but unbounded: no error scale to compare against
    return None


def _higher_is_better(metric: str) -> bool:
    return METRIC_DIRECTION.get(metric, 1) > 0


@cache
def _task_meta(task: str) -> tuple[str, str] | None:
    """(view, family) of a registered task, looked up once per task."""
    try:
        t = get_task(task)
    except KeyError:
        return None
    return t.observability.value, t.function.value


def _family_sort_key(family: str) -> tuple[int, str]:
    """Known families in FAMILY_LABELS order; a family added later sorts after them by name."""
    order = _FAMILY_ORDER.index(family) if family in FAMILY_LABELS else len(_FAMILY_ORDER)
    return order, family


def model_ids(rows: list[Row]) -> dict[int, str]:
    """A unique display id per row's model, keyed by ``id(row)``.

    In-house runs keep their baseline name. An outside submission whose name equals an in-house
    baseline or the reference gets `` (submitted)`` appended, so it can never overwrite those rows,
    replace the reference, or put its tier badge on an in-house model."""
    inhouse = {r.baseline for r in rows if r.tier is None} | {REFERENCE}
    return {
        id(r): r.baseline if r.tier is None or r.baseline not in inhouse
        else f"{r.baseline} (submitted)"
        for r in rows
    }  # fmt: skip


def _ratio(err: float, ref: float) -> float | None:
    if not (math.isfinite(err) and math.isfinite(ref)) or ref <= 0:
        return None
    return min(max(err / ref, RATIO_CLIP[0]), RATIO_CLIP[1])


def skill(ratios) -> float | None:
    """``1 - geometric mean`` of error ratios (None for an empty set)."""
    r = np.asarray(list(ratios), dtype=float)
    if r.size == 0:
        return None
    return float(1.0 - np.exp(np.log(r).mean()))


def skill_ci(ratios, *, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float] | None:
    """95% percentile bootstrap interval of :func:`skill`, resampling cells."""
    r = np.log(np.asarray(list(ratios), dtype=float))
    if r.size < 2:
        return None
    idx = np.random.default_rng(seed).integers(0, r.size, size=(n_boot, r.size))
    boot = 1.0 - np.exp(r[idx].mean(axis=1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


@dataclass
class _Cell:
    key: tuple  # (task, grid, source)
    family: str
    label: str  # grid, or "grid ← source" for transfer
    metric: str
    ref: float  # reference error
    ref_value: float | None  # reference raw metric value (shown in the drill-down)
    errors: dict[str, float] = field(default_factory=dict)  # model -> error
    shown: dict[str, tuple[float | None, float | None]] = field(
        default_factory=dict
    )  # model -> raw mean, ci95


def _cells(
    rows: list[Row], protocol: str, test_set: str, view: str, ids: dict[int, str] | None = None
) -> tuple[list[_Cell], int]:
    """The scorable cells of one table, and how many runs were left out because their evaluation
    set differs in size from the reference's (transfer cells borrow the held_out reference)."""
    ids = ids if ids is not None else model_ids(rows)
    ref_rows = {
        (r.task, r.grid): r
        for r in rows
        if r.protocol == "held_out"
        and r.test_set == test_set
        and r.baseline == REFERENCE
        and r.tier is None
    }
    cells: dict[tuple, _Cell] = {}
    mismatched = 0
    for r in rows:
        if (r.protocol, r.test_set) != (protocol, test_set):
            continue
        meta = _task_meta(r.task)
        if meta is None or meta[0] != view or (r.baseline == REFERENCE and r.tier is None):
            continue
        h = headline(r)
        ref_row = ref_rows.get((r.task, r.grid))
        if h is None or ref_row is None:
            continue
        if ref_row.n_test != r.n_test:
            mismatched += 1
            continue
        ref_h = headline(ref_row)
        if ref_h is None or ref_h[0] != h[0] or not ref_h[1] > 0:
            continue
        source = (r.source or _DEFAULT_SOURCE) if protocol.startswith("transfer") else None
        key = (r.task, r.grid, source)
        if key not in cells:
            label = r.grid if source is None else f"{r.grid} ← {_source_label(source)}"
            cells[key] = _Cell(key, meta[1], label, h[0], ref_h[1], ref_row.mean(h[0]))
        c = cells[key]
        m = ids[id(r)]
        c.errors[m] = h[1]
        c.shown[m] = (r.mean(h[0]), r.metrics.get(h[0], {}).get("ci95"))
    return list(cells.values()), mismatched


def _ratios(cells: list[_Cell], model: str) -> list[float]:
    """The model's clipped error ratios on the cells it covers. Cells hold finite errors and a
    positive reference only, so no covered cell is dropped here; the filter keeps the type exact."""
    return [
        x for c in cells if model in c.errors if (x := _ratio(c.errors[model], c.ref)) is not None
    ]


def paired_better(
    models: list[str], cells: list[_Cell], *, n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> set[tuple[str, str]]:
    """Pairs ``(a, b)`` where ``a`` has the higher skill with 95% confidence.

    Paired bootstrap over cells (fev-bench): every model is evaluated on the SAME resampled cells,
    so the cell-to-cell spread that all models share cancels out of the difference. ``a`` beats
    ``b`` when the 97.5th percentile of ``mean log-ratio(a) - mean log-ratio(b)`` is below 0."""
    if len(models) < 2 or len(cells) < 2:
        return set()
    logs = {m: np.log(_ratios(cells, m)) for m in models}  # full entries: one ratio per cell
    idx = np.random.default_rng(seed).integers(0, len(cells), size=(n_boot, len(cells)))
    means = {m: v[idx].mean(axis=1) for m, v in logs.items()}
    out = set()
    for a in models:
        for b in models:
            if a != b and np.percentile(means[a] - means[b], 97.5) < 0:
                out.add((a, b))
    return out


def _round(x: float | None, n: int = 3) -> float | None:
    return None if x is None else round(x, n)


def build_table(
    rows: list[Row], protocol: str, test_set: str, view: str, ids: dict[int, str] | None = None
) -> dict | None:
    """The summary for one (protocol, evaluation set, view): ranked, partial and drill-down data."""
    ids = ids if ids is not None else model_ids(rows)
    cells, mismatched = _cells(rows, protocol, test_set, view, ids)
    if not cells:
        return None
    tiers = {ids[id(r)]: r.tier for r in rows if r.tier}
    models = sorted({m for c in cells for m in c.errors})
    families = sorted({c.family for c in cells}, key=_family_sort_key)
    n = len(cells)
    covered = {m: [c for c in cells if m in c.errors] for m in models}
    full = [m for m in models if len(covered[m]) == n]

    def _entry(m: str, cs: list[_Cell]) -> dict:
        rs = _ratios(cs, m)
        ci = skill_ci(rs)
        fams = {}
        for f in families:
            fr = _ratios([c for c in cs if c.family == f], m)
            fams[f] = _round(skill(fr))
        return {
            "model": m,
            "tier": tiers.get(m),
            "skill": _round(skill(rs)),
            "ci": [_round(ci[0]), _round(ci[1])] if ci else None,
            "families": fams,
            "cells": len(cs),
        }

    ranked = [_entry(m, cells) for m in full]
    # win rate among ranked entries, over cells both cover (ties count half)
    for e in ranked:
        rates = []
        for o in full:
            if o == e["model"]:
                continue
            a, b = e["model"], o
            wins = [
                1.0 if c.errors[a] < c.errors[b] else 0.5 if c.errors[a] == c.errors[b] else 0.0
                for c in cells
            ]
            rates.append(sum(wins) / len(wins))
        e["win_rate"] = _round(sum(rates) / len(rates)) if rates else None
    ranked.sort(key=lambda e: (-(e["skill"] if e["skill"] is not None else -1e9), e["model"]))
    better = paired_better([e["model"] for e in ranked], cells)
    for i, e in enumerate(ranked):
        m = e["model"]
        n_above = sum(1 for o in ranked if (o["model"], m) in better)
        n_below = sum(1 for o in ranked if (m, o["model"]) in better)
        e["rank"], e["rank_range"] = i + 1, [1 + n_above, len(ranked) - n_below]

    partial = []
    for m in models:
        if m in full:
            continue
        cs = covered[m]
        e = _entry(m, cs)
        same = [(skill(_ratios(cs, f)), f) for f in full]
        same = [(s, f) for s, f in same if s is not None]
        if same:
            s, f = max(same)
            e["best_ranked_same_cells"] = {"model": f, "skill": _round(s)}
        partial.append(e)
    partial.sort(key=lambda e: (-(e["skill"] if e["skill"] is not None else -1e9), e["model"]))

    detail = {}
    for f in families:
        fc = sorted((c for c in cells if c.family == f), key=lambda c: (c.key[0], c.label))
        detail[f] = {
            "metric": fc[0].metric,
            "higher_is_better": _higher_is_better(fc[0].metric),
            "columns": [
                f"{c.label}" + ("" if len({x.key[0] for x in fc}) == 1 else f" · {c.key[0]}")
                for c in fc
            ],  # fmt: skip
            "reference": [_round(c.ref_value) for c in fc],
            "values": {
                m: [
                    [_round(c.shown[m][0]), _round(c.shown[m][1])] if m in c.shown else None
                    for c in fc
                ]
                for m in models
            },
        }
    return {
        "n_cells": n,
        "n_mismatched": mismatched,
        "families": families,
        "ranked": ranked,
        "partial": partial,
        "detail": detail,
    }


def build_summary(rows: list[Row]) -> dict:
    """Every (protocol, evaluation set, view) table, keyed ``protocol|test_set|view``."""
    cur = [r for r in rows if _is_current(r)]
    ids = model_ids(cur)
    refs: dict[str, list[Row]] = defaultdict(list)  # held_out reference rows per evaluation set
    buckets: dict[tuple[str, str, str], list[Row]] = defaultdict(list)
    for r in cur:
        meta = _task_meta(r.task)
        if meta is None:
            continue
        buckets[(r.protocol, r.test_set, meta[0])].append(r)
        if r.protocol == "held_out" and r.baseline == REFERENCE and r.tier is None:
            refs[r.test_set].append(r)
    tables = {}
    for p, _ in PROTOCOLS:
        for t, _ in TEST_SETS:
            for v, _ in VIEWS:
                extra = [] if p == "held_out" else refs[t]
                tab = build_table(buckets.get((p, t, v), []) + extra, p, t, v, ids)
                if tab is not None:
                    tables[f"{p}|{t}|{v}"] = tab
    return {
        "tables": tables,
        "views": dict(VIEWS),
        "protocols": dict(PROTOCOLS),
        "test_sets": dict(TEST_SETS),
        "families": FAMILY_LABELS,
        "default": "held_out|benchmark|line",
    }


CSS = """
:root{--fg:#1a1a1a;--muted:#666;--line:#e5e5e5;--soft:#f6f7f9;--accent:#1677ff;--accent-bg:#e8f1ff;--bg:#fff}
@media (prefers-color-scheme:dark){:root{--fg:#e6e6e6;--muted:#9aa0a6;--line:#2a2f36;--soft:#1c2025;
--accent:#69b1ff;--accent-bg:#14263d;--bg:#15171a}}
body{font:14px/1.55 system-ui,sans-serif;margin:0;color:var(--fg);background:var(--bg)}
main{max-width:1100px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:22px;margin:4px 0 2px}.v{color:var(--muted);font-size:13px}
.nav{font-size:13px}.nav a,a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.lead{margin:10px 0 16px;max-width:780px}
.ctl{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.seg button{font:13px system-ui;border:0;background:none;color:var(--muted);padding:6px 12px;cursor:pointer}
.seg button[aria-pressed=true]{background:var(--accent-bg);color:var(--accent);font-weight:600}
.bar{display:inline-block;height:7px;border-radius:4px;background:var(--accent);vertical-align:middle;margin-right:8px}
.ci{color:var(--muted);font-size:12px}
h2{font-size:16px;margin:28px 0 6px}
.note{color:var(--muted);font-size:13px;max-width:820px}
.best{font-weight:700}
.links{display:flex;flex-wrap:wrap;gap:10px;margin:22px 0}
.links a{border:1px solid var(--line);border-radius:8px;padding:8px 14px;background:var(--soft)}
button.dl{font:13px system-ui;border:1px solid var(--line);border-radius:8px;background:var(--soft);
color:var(--fg);padding:5px 10px;cursor:pointer}
.tabulator{font-size:13px;background:transparent;border:0;margin:4px 0}
.tabulator .tabulator-header,.tabulator .tabulator-header .tabulator-col{background:var(--bg);color:var(--muted);border-color:var(--line)}
.tabulator .tabulator-tableholder{background:transparent}
.tabulator-row{background:var(--bg);color:var(--fg)}.tabulator-row.tabulator-row-even{background:var(--soft)}
.tabulator-row .tabulator-cell{border-color:var(--line)}.tabulator-row .tabulator-cell.tabulator-frozen{background:inherit}
.tabulator-row.tabulator-selectable:hover{background:var(--accent-bg)}.tabulator-row.refrow{color:var(--muted)}
#chart,#chart-partial{width:100%;margin:6px 0 12px}
.famsel{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:20px 0 4px;color:var(--muted);font-size:13px}
footer{border-top:1px solid var(--line);margin-top:28px;padding-top:8px;color:var(--muted);font-size:12px}
"""

JS = r"""
// Page state is ONE object {view, protocol, test_set, family}. It lives in the URL hash
// (#view=line&protocol=held_out&test_set=benchmark&family=fault_class), so every view has a
// shareable link and the back button works. normalize() is the only transition rule: it maps any
// requested state to a valid one (a combination with results, a family present in it). When the
// requested combination has no results it keeps the keys in `keep` (what the user just clicked)
// and changes as few of the others as possible, view first, then protocol, then evaluation set.
const D=JSON.parse(document.getElementById('data').textContent);
const KEYS=['view','protocol','test_set','family'];
const tkey=s=>`${s.protocol}|${s.test_set}|${s.view}`;
function normalize(s,keep=[]){
 const [p,t,v]=D.default.split('|'),dflt={protocol:p,test_set:t,view:v};
 s={...dflt,...s};
 for(const k of ['view','protocol','test_set'])if(!(s[k]in(k==='view'?D.views:k==='protocol'?D.protocols:D.test_sets)))s[k]=dflt[k];
 const c=Object.keys(D.tables).map(k=>k.split('|')).map(([protocol,test_set,view])=>({protocol,test_set,view}));
 if(!c.length)return{...s,family:null};
 if(!D.tables[tkey(s)]){
  const w={view:4,protocol:2,test_set:1},score=x=>Object.keys(w).reduce((a,k)=>a+(x[k]===s[k]?w[k]:0),0);
  const pool=c.filter(x=>keep.every(k=>!(k in w)||x[k]===s[k]));
  s={...s,...(pool.length?pool:c).reduce((a,x)=>score(x)>score(a)?x:a)}}
 if(s.family&&!D.tables[tkey(s)].detail[s.family])s.family=null;
 return s}
const read=()=>normalize(Object.fromEntries([...new URLSearchParams(location.hash.slice(1))].filter(([k])=>KEYS.includes(k))));
function go(patch){const s=normalize({...st,...patch},Object.keys(patch));
 const h='#'+new URLSearchParams(Object.entries(s).filter(([,v])=>v)).toString();
 if(h!==location.hash)history.pushState(null,'',h);st=s;render()}
let st=read();
window.addEventListener('popstate',()=>{st=read();render()});
const $=s=>document.querySelector(s);
const f3=x=>x==null?'—':x.toFixed(3), pct=x=>x==null?'—':Math.round(100*x)+'%';
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const badge=t=>t?` <span class="tier ${t==='reproduced'?'repro':'self'}">${t}</span>`:'';
function seg(id,opts,k,patch=v=>({[k]:v,family:null})){const el=$(id);el.innerHTML='';for(const[v,l]of Object.entries(opts)){
 const b=document.createElement('button');b.textContent=l;b.setAttribute('aria-pressed',st[k]===v);
 const probe={...st,[k]:v};b.title=D.tables[tkey(probe)]?'':'No results for this combination; the closest one is shown';
 b.onclick=()=>go(patch(v));el.appendChild(b)}}
// Rendering: Tabulator tables and a Vega-Lite interval chart, rebuilt from `st` on every
// transition. clear() disposes the previous widgets so re-renders do not leak.
let widgets=[],charts=[];
function clear(){widgets.forEach(t=>t.destroy());widgets=[];charts.forEach(p=>p.then(r=>r.finalize(),()=>{}));charts=[]}
const skillFmt=c=>{const v=c.getValue();return v==null?'—':`<span class=bar style="width:${Math.max(0,v)*70}px"></span>${v.toFixed(3)}`};
const numFmt=c=>f3(c.getValue());
const ivFmt=c=>{const d=c.getRow().getData();return d.lo==null?'':`<span class=ci>${f3(d.lo)} – ${f3(d.hi)}</span>`};
const modelFmt=c=>{const d=c.getRow().getData();return esc(d.model)+badge(d.tier)};
const muteRef=row=>{if(row.getData().ref)row.getElement().classList.add('refrow')};
const TOPTS={layout:'fitDataStretch',columnHeaderVertAlign:'bottom'};
function rowOf(e){const r={model:e.model,tier:e.tier,skill:e.skill,lo:e.ci?e.ci[0]:null,hi:e.ci?e.ci[1]:null,
 win:e.win_rate,cells:e.cells,rank:e.rank,
 rr:e.rank_range?(e.rank_range[0]===e.rank_range[1]?`${e.rank_range[0]}`:`${e.rank_range[0]}–${e.rank_range[1]}`):'',
 best:e.best_ranked_same_cells?`${e.best_ranked_same_cells.model} ${f3(e.best_ranked_same_cells.skill)}`:'—',
 bm:e.best_ranked_same_cells?e.best_ranked_same_cells.model:null,bs:e.best_ranked_same_cells?e.best_ranked_same_cells.skill:null};
 for(const[f,v]of Object.entries(e.families))r['f_'+f]=v;return r}
const famCols=T=>T.families.map(f=>({title:D.families[f]||f,field:'f_'+f,hozAlign:'right',sorter:'number',formatter:numFmt}));
// Charts are optional: a failure to draw one never blocks the tables (they are built first).
const X={type:'quantitative',title:'Skill score (0 = reference, 1 = perfect), 95% interval'};
function themed(){const dark=matchMedia('(prefers-color-scheme: dark)').matches,fg=dark?'#e6e6e6':'#1a1a1a',g=dark?'#2a2f36':'#e5e5e5';
 return{view:{stroke:null},axis:{labelColor:fg,titleColor:fg,gridColor:g,domainColor:g,tickColor:g},legend:{labelColor:fg,titleColor:fg}}}
function embed(id,spec){
 try{const p=vegaEmbed(id,{width:'container',height:{step:24},background:'transparent',config:themed(),...spec},{actions:false,renderer:'svg'});
  charts.push(p);p.catch(()=>{const el=$(id);if(el)el.textContent=''})}
 catch(err){const el=$(id);if(el)el.textContent=''}}
function chart(T){
 // Ranked entries only: their skills are on the same cells, so the axis compares like with like.
 const pts=T.ranked.map(rowOf).filter(d=>d.skill!=null);
 if(pts.length)embed('#chart',{data:{values:pts},encoding:{y:{field:'model',type:'nominal',sort:{field:'skill',order:'descending'},title:null}},
  layer:[{mark:{type:'rule',strokeWidth:2,color:'#1677ff'},encoding:{x:{field:'lo',...X},x2:{field:'hi'}}},
   {mark:{type:'point',filled:true,size:80,color:'#1677ff'},encoding:{x:{field:'skill',...X},tooltip:[{field:'model',title:'Model'},
    {field:'skill',title:'Skill',format:'.3f'},{field:'lo',title:'95% low',format:'.3f'},{field:'hi',title:'95% high',format:'.3f'}]}}]});
 // Partial entries: each next to the best ranked entry on ITS cells, never on the ranked axis.
 const pp=T.partial.map(rowOf).filter(d=>d.skill!=null).flatMap(d=>[
  {model:d.model,who:'This entry',skill:d.skill,lo:d.lo,hi:d.hi,cells:d.cells},
  ...(d.bs!=null?[{model:d.model,who:'Best ranked entry, same cells',skill:d.bs,name:d.bm,cells:d.cells}]:[])]);
 if(pp.length)embed('#chart-partial',{data:{values:pp},encoding:{y:{field:'model',type:'nominal',title:null},
   color:{field:'who',type:'nominal',title:null,scale:{domain:['This entry','Best ranked entry, same cells'],range:['#d48806','#8c8c8c']},legend:{orient:'bottom'}}},
  layer:[{mark:{type:'rule',strokeWidth:2},encoding:{x:{field:'lo',...X},x2:{field:'hi'}}},
   {mark:{type:'point',filled:true,size:80},encoding:{x:{field:'skill',...X},shape:{field:'who',type:'nominal',legend:null},
    tooltip:[{field:'model',title:'Entry'},{field:'who',title:'Point'},{field:'name',title:'Ranked model'},{field:'skill',title:'Skill',format:'.3f'},{field:'cells',title:'Cells'}]}}]})}
function render(){
 seg('#s-view',D.views,'view');seg('#s-proto',D.protocols,'protocol');seg('#s-set',D.test_sets,'test_set');
 clear();const T=D.tables[tkey(st)];const out=$('#board');
 if(!T){out.innerHTML='<p class=note>No results to rank yet.</p>';$('#detail').innerHTML='';return}
 out.innerHTML=`<div id=chart></div><div id=t-ranked></div>`+
  `<p class=note>${T.ranked.length} ranked entries on ${T.n_cells} cells (task × grid${st.protocol.startsWith('transfer')?' × source':''}). Entries share a rank range unless a paired comparison on the same cells separates them at 95%.${T.n_mismatched?` ${T.n_mismatched} runs are left out because their evaluation set differs in size from the reference.`:''} Click a column heading to sort.</p>`+
  (T.partial.length?`<h2>Partial entries</h2><p class=note>These cover only some cells, so they are not ranked with the entries above. Each is compared with the best ranked entry on the same cells.</p><div id=chart-partial></div><div id=t-partial></div>`:'')+
  `<div class=famsel><span>Per-grid detail:</span><div class=seg id=s-fam role=group aria-label="Task family"></div></div>`;
 const ref={model:'majority (reference)',skill:0,rr:'—',ref:true};T.families.forEach(f=>ref['f_'+f]=0);
 widgets.push(new Tabulator('#t-ranked',{...TOPTS,data:[...T.ranked.map(rowOf),ref],rowFormatter:muteRef,columns:[
  {title:'Rank',field:'rank',sorter:'number',frozen:true,formatter:c=>c.getRow().getData().rr,headerTooltip:'Best and worst rank once only paired differences significant at 95% count'},
  {title:'Model',field:'model',frozen:true,formatter:modelFmt},
  {title:'Skill score',field:'skill',sorter:'number',hozAlign:'right',formatter:skillFmt},
  {title:'95% interval',field:'lo',headerSort:false,hozAlign:'right',formatter:ivFmt},
  {title:'Win rate',field:'win',sorter:'number',hozAlign:'right',formatter:c=>pct(c.getValue())},
  ...famCols(T)]}));
 if(T.partial.length)widgets.push(new Tabulator('#t-partial',{...TOPTS,data:T.partial.map(rowOf),columns:[
  {title:'Model',field:'model',frozen:true,formatter:modelFmt},
  {title:'Cells',field:'cells',sorter:'number',hozAlign:'right',formatter:c=>`${c.getValue()}/${T.n_cells}`},
  {title:'Skill on its cells',field:'skill',sorter:'number',hozAlign:'right',formatter:skillFmt},
  {title:'95% interval',field:'lo',headerSort:false,hozAlign:'right',formatter:ivFmt},
  {title:'Best ranked entry on the same cells',field:'best',headerSort:false,formatter:c=>esc(c.getValue())},
  ...famCols(T)]}));
 seg('#s-fam',Object.fromEntries(T.families.map(f=>[f,D.families[f]||f])),'family',f=>({family:st.family===f?null:f}));
 drill();chart(T)}
function drill(){const T=D.tables[tkey(st)],el=$('#detail'),fam=st.family;
 if(!fam){el.innerHTML='<p class=note>Pick a task family above to see it per grid.</p>';return}
 const d=T.detail[fam],hi=d.higher_is_better,better=(a,b)=>hi?a>b:a<b;
 const mname={balanced_accuracy:'Balanced accuracy',mae:'Mean absolute error'}[d.metric]||d.metric;
 const best=d.columns.map((_,i)=>{let bv=null;for(const v of Object.values(d.values)){const x=v[i]&&v[i][0];if(x!=null&&(bv==null||better(x,bv)))bv=x}return bv});
 const rows=[...T.ranked,...T.partial].map(e=>e.model).filter(m=>d.values[m]).map(m=>{const r={model:m};d.values[m].forEach((x,i)=>{r['c'+i]=x?x[0]:null;r['e'+i]=x?x[1]:null});return r});
 const ref={model:'majority (reference)',ref:true};d.reference.forEach((x,i)=>ref['c'+i]=x);
 el.innerHTML=`<h2>${esc(D.families[fam]||fam)} per grid</h2><p class=note>${esc(mname)} (${hi?'higher':'lower'} is better), mean ± 95% interval over seeds. Best per column in bold. <button class=dl id=csv>Download CSV</button></p><div id=t-detail></div>`;
 const t=new Tabulator('#t-detail',{...TOPTS,data:[...rows,ref],rowFormatter:muteRef,columns:[
  {title:'Model',field:'model',frozen:true,formatter:c=>esc(c.getValue())},
  ...d.columns.map((c,i)=>({title:c,field:'c'+i,sorter:'number',hozAlign:'right',formatter:cell=>{
   const v=cell.getValue(),r=cell.getRow().getData(),e=r['e'+i];if(v==null)return'—';
   return `<span class="${v===best[i]&&!r.ref?'best':''}">${f3(v)}</span>${e!=null?` <span class=ci>± ${f3(e)}</span>`:''}`}}))]});
 widgets.push(t);
 $('#csv').onclick=()=>t.download('csv',`evemtbench_${tkey(st).replace(/\|/g,'_')}_${fam}.csv`)}
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>render());
render();
"""


VENDOR_DIR = "_vendor"
# Inline order. vega-embed's UMD bundle reads the ``vega`` and ``vegaLite`` globals when it loads,
# so they must come first; VENDOR.json key order is NOT relied on.
VENDOR_LOAD_ORDER = (
    "tabulator_simple.min.css",
    "tabulator.min.js",
    "vega.min.js",
    "vega-lite.min.js",
    "vega-embed.min.js",
)


@cache
def vendored_assets() -> tuple[str, str, str]:
    """(CSS, <script> tags, licence notices) of the vendored page libraries.

    The libraries ship with the package (``_vendor/``, pinned in ``VENDOR.json`` with sha256) and
    are inlined, so the page stays one self-contained, offline, byte-deterministic file. The
    licence texts go into an HTML comment at the end of the page (MIT / BSD-3-Clause require the
    notice to travel with redistributed copies)."""
    from importlib.resources import files

    root = files("evemtbench").joinpath(VENDOR_DIR)
    manifest = json.loads(root.joinpath("VENDOR.json").read_text())
    if set(manifest) != set(VENDOR_LOAD_ORDER):
        raise ValueError(f"VENDOR.json and VENDOR_LOAD_ORDER disagree: {sorted(manifest)}")
    css, js, notices = [], [], []
    for name in VENDOR_LOAD_ORDER:
        meta = manifest[name]
        text = root.joinpath(name).read_text(encoding="utf-8")
        if name.endswith(".css"):
            css.append(text)
        else:
            js.append(f"<script>{text}</script>")
        notice = root.joinpath(meta["license_file"]).read_text(encoding="utf-8")
        entry = f"{meta['package']} {meta['version']} ({meta['license']})\n{notice.strip()}"
        if entry not in notices:
            notices.append(entry)
    body = "\n\n".join(notices)
    if "--" in body:  # the notices are embedded verbatim in an HTML comment, where "--" is invalid
        raise ValueError("a vendored licence notice contains '--'; embed it another way")
    return "".join(css), "".join(js), body


def render_summary_html(
    rows: list[Row],
    *,
    version: str = "",
    generated_at: str | None = None,
    docs_base: str = DOCS_BASE,
    full_href: str = "leaderboard_full.html",
) -> str:
    """The user-facing leaderboard page (self-contained; data embedded as JSON)."""
    lib_css, lib_js, notices = vendored_assets()
    data = json.dumps(build_summary(rows), separators=(",", ":"), sort_keys=True)
    data = data.replace("</", "<\\/")
    nav = nav_html(docs_base, extra=((full_href, "full results"), ("audit/index.html", "audit")))
    updated = f"<div class=v>updated {html.escape(generated_at)}</div>" if generated_at else ""
    d = html.escape(docs_base)
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        '<meta name=description content="EvEMTBench leaderboard: which model detects, classifies '
        'and locates power-system faults best on unseen grids.">'
        f'<link rel=icon href="{FAVICON}"><title>EvEMTBench leaderboard</title>'
        f"<style>{lib_css}{CSS}{_TIER_CSS}</style></head><body><main>"
        f"{nav}<h1>EvEMTBench leaderboard</h1>"
        f"<div class=v>benchmark v{html.escape(version or 'n/a')}</div>{updated}"
        "<p class=lead>Models ranked by <b>skill score</b>: 0 means no better than predicting the "
        "majority class (or the mean), 1 means perfect. It combines every task and grid in the "
        "selected view as one minus the geometric mean of the model's error relative to that "
        "reference, so adding new entries does not change existing scores. "
        f'<a href="{d}/docs/metrics.md">How scores are computed</a>.</p>'
        '<div class=ctl><div class=seg id=s-view role=group aria-label="Input view"></div>'
        '<div class=seg id=s-proto role=group aria-label="Protocol"></div>'
        '<div class=seg id=s-set role=group aria-label="Evaluation set"></div></div>'
        "<div id=board></div><div id=detail></div>"
        "<div class=links>"
        f'<a href="{html.escape(full_href)}">Full results: every task, grid and metric</a>'
        f'<a href="{d}/docs/submission.md">Submit your model</a>'
        f'<a href="{d}/results">Download raw results</a></div>'
        "<footer>EvEMTBench · MIT · "
        f'<a href="{d}/README.md">repository</a> · '
        f'<a href="{d}/docs/known_issues.md">known issues</a></footer>'
        f'</main><script type="application/json" id=data>{data}</script>'
        f"{lib_js}<script>{JS}</script>"
        f"<!--\n{notices}\n--></body></html>"
    )
