# The "results audit" dashboard — concept and general recipe

*Extracted from Prof. Maier's [UnderstandingVCS results audit](https://akmaier.github.io/UnderstandingVCS/)
([repo](https://github.com/akmaier/UnderstandingVCS), site source in `docs/`). Two companion
documents — a case study of that repo's implementation and the design mapping onto this repo —
are internal design notes and live in the development repository.*

---

## 1. What a results audit is

A **results audit** is a static, dependency-free website (or document set) that lets a skeptical
reviewer — the PI, an advisor, an area chair — verify **every claim in a paper down to the code
that produced it**, without trusting the authors and without re-running everything. Its tagline on
UnderstandingVCS is literally:

> *"Audit every result, down to the script that made it."*

The purpose is pre-submission verification: before a paper goes out, the PI clicks through every
headline number and can answer, for each one, *where did this come from, can I reproduce it, and
what guards it against being wrong?* It is **not** a marketing page and not a leaderboard — it is
an evidence ledger with a navigation layer.

The core object is the **provenance chain**. Every claimed result must resolve, in one place, to:

```
claim → measured value → producing script → exact command → output artifact
      → runtime → hardware → verifying gate → status (measured | deferred)
```

If any link of that chain is missing for a claim, the audit makes that visible instead of hiding it.

## 2. The ten principles

These are the design rules the UnderstandingVCS site is built on, stated generally.

### P1 — The claim ledger
Enumerate every claim the paper makes as a **row** in a machine-readable manifest. One row = one
auditable result. Each row carries the full provenance chain above, plus a free-text `detail`
(what exactly was measured, at what precision/scope) and an optional `note` ("where this comes
from" caveats). The paper's ledger *is* the audit; every other page is derived from or linked to it.

### P2 — Single source of truth for numbers (no hand-transcription)
Numbers are **never typed into the site by hand**. A small generator script reads the *committed
result files* (JSON/CSV/markdown emitted by the experiment code) and produces one central store
(e.g. `site_data.json`). Every page reads numbers only from that store, so **no two pages can ever
show a conflicting value**. Prose that mentions numbers uses `{placeholder}` template slots filled
at build time from the store. When the underlying results change, one regeneration updates every
page, figure caption and blurb consistently.

### P3 — Honesty statuses: measured vs deferred, scored vs excluded
Every ledger row carries a status badge. `measured` = the number exists in a committed artifact.
`deferred` = planned/optional work that was *not* run — listed anyway, "for honesty", so a plan is
never dressed up as a result. The same discipline applies to the unit-of-analysis level: items
excluded from scoring (games, grids, cells…) are shown **dimmed with the explicit exclusion
reason**, never silently dropped. The counts strip states both numbers ("54 carry a label, 42 in
the scored battery, 12 excluded").

### P4 — An external oracle ("why you can trust this")
The strongest defence against *"the AI/author hallucinated it"* is agreement with something the
project did **not** produce: an independent reference implementation, externally published labels,
a second independently-written implementation that agrees bit-for-bit, or a mathematical
self-check. The audit's front page leads with this — a "Why you can trust this" panel with 3–5
**pillars**, each one sentence + a link (external oracle, cross-check, verifying harness, open
development log).

### P5 — "Verified by": every claim names its guard
Each ledger row names the **test or gate that would catch it being wrong** (a conformance test, a
self-check assertion inside the analysis script, a CI job, a hash table). A dedicated
**conformance / code-tour page** then walks the verification harness itself: for each gate, a
table of the actual call stack — *step in the code* (with a line-anchored source link) vs *what is
actually computed* — so the reviewer can confirm what a test asserts without reading the whole
codebase.

### P6 — Environment, determinism, and input provenance
One page records **where every number was produced**: hardware (machine per result class),
pinned software versions (with a pointer to the pin file: lockfile, `Project.toml`,
`pyproject.toml`), seeds and the determinism story (what is seeded, what makes reruns bit-exact),
and **input data provenance** (how inputs are obtained and hash-verified, e.g. SHA-256 tables, when
the inputs themselves cannot be redistributed).

### P7 — A reproduce page: verbatim commands, cheapest gate first
A page that collects, automatically from the ledger, the **exact invocation behind every claim**
and the artifact it writes. Ordered so the reviewer can start with a ~20-second smoke gate and
escalate to the multi-hour full suites. Includes the command to rebuild the audit site itself.

### P8 — Provenance of the *process*: tests, CI, open dev log
A page listing the test suites (with counts **grepped from source at build time**, not quoted from
docs — and an honest note when different roll-ups disagree), the CI workflows (fast PR gate +
scheduled heavy suite), and the open development log (running bug/fix/dead-end log, per-phase
status ledger). The reviewer sees that verification is continuous machinery, not a one-off claim.

### P9 — Separate computation from plotting
Figure scripts that *only draw* committed numbers are listed as **separate ledger rows** with an
`inputs:` list naming the upstream scripts/artifacts that actually computed the data — "listed
separately so the figure script is not mistaken for the source of the numbers." A reviewer
auditing a figure must land on the computation, not the matplotlib call.

### P10 — Build integrity: deterministic, stdlib-only, commit-stamped
The site builder is boring on purpose: pure standard library, no framework, no external requests,
plain HTML + one hand-written CSS file, hosted as static files (GitHub Pages from `/docs`, with
`.nojekyll`). Rebuilding is one or two commands documented in the footer of every page, and the
footer stamps **the exact git commit the site was built from**, linking it. The audit itself is
reproducible.

## 3. Reference architecture

The minimal pipeline, independent of domain:

```
experiment runners ──▶ committed result artifacts        (out/*.json, results_*.md, *.csv)
                              │
                              ▼
                    aggregate analysis script             (leaderboard.py … itself with self-checks)
                              │  writes
                              ▼
                    committed aggregate artifacts         (leaderboard.json, bootstrap.json)
                              │
                              ▼
                    store generator (gen_site_data.py) ──▶ site_data.json   ← THE single number source
                                                              │
        manifest.py (claims, env, provenance — no numbers) ──┤
                                                              ▼
                                            build_pages.py ──▶ docs/*.html  (static, commit-stamped)
                                            build_assets.py ─▶ docs/assets/ (imgs/gifs/videos from PDFs/MP4s)
```

Key separation of concerns:

| Piece | Contains | Must NOT contain |
|---|---|---|
| `manifest.py` | claim text, script/command/artifact paths, env & provenance metadata, qualitative prose with `{placeholders}` | any result number |
| store generator | code that *reads committed artifacts* and re-derives per-item numbers **using the same loader code the paper's aggregation uses** | hand-entered values |
| store (`site_data.json`) | every number the site shows, plus meta (counts, item lists, "generated_by", sources) | prose |
| page builder | rendering only: ledger rows, tables, badges, source links | computation of results |

Two subtle but important tricks:

- **Reuse the analysis loaders.** The store generator *imports the aggregation script's own
  loader functions* so per-item numbers on the site are computed by exactly the same code path as
  the paper's aggregate — not a reimplementation that could drift.
- **Line-anchored source links, verified.** Deep links into the code (`file#L123`) are only
  credible if the line numbers are checked against the source at build/review time; keep a snippet
  to re-verify them when files move, and link against `main` (not a pinned SHA) so links survive
  file moves while the footer records the exact build commit.

## 4. Page inventory (the standard set)

| Page | Role | Answers |
|---|---|---|
| `index.html` | Hero + oracle panel + paper cards + **"How to read a claim"** legend table | Why trust this at all? How do I use the site? |
| one page per paper | The **claim ledger** (all rows), figure gallery, video gallery | Where does every number in paper *N* come from? |
| `conformance.html` | Guided **code tour** of each verification harness (call-stack tables, line links) | What do the tests actually assert? |
| ground-truth / dataset page | Lay explainer of what "ground truth" means here + **per-item cards** (badges: label origin, scored/excluded + reason, regime) + counts strip | What can results even be scored against, item by item? |
| methods / systems page | Catalogue of every method/baseline: reference, implementation script, record file | What was run, per method? |
| per-item detail pages | One page per method and per game/grid: lay "what it does", "how it's scored" (numbers templated from store), example record, **"in the audit" box**, sortable per-item results table | Drill-down for any cell of the results matrix |
| `provenance.html` | Test counts (grepped at build), harnesses, tooling, CI, dev logs | Is the verification machinery real and continuous? |
| `environment.html` | Hardware, pinned software, determinism/seeds, input hash table | Where/how was it run; can bytes be reproduced? |
| `reproduce.html` | Conformance gates (cheap→expensive) + auto-collected per-claim command table + site rebuild | What do I type? |

The "How to read a claim" legend from the index page is worth copying verbatim — it defines the
reviewer contract:

| Field | What it gives the reviewer |
|---|---|
| Value | the measured number, read from the committed result file |
| Script | link to the exact source that produces it |
| Command | the verbatim invocation to reproduce it |
| Artifact | the committed output (results table / JSON / figure) it wrote |
| Runtime & Hardware | wall-clock and the machine it ran on |
| Verified by | the test / oracle gate that guards it |
| Status | `measured` vs `deferred` — we never dress a plan up as a result |

## 5. The harness beneath the dashboard

The dashboard is only the **reporting layer**. It is credible because of the machinery it reports
on, which has to exist first:

1. **Verification harnesses** — named test suites that compare the system against the oracle
   (in UnderstandingVCS: PXC1/PXC2/PXC-S/PXC4 replay-and-diff harnesses). Each has a stable name,
   a one-line "what it asserts", and a source path — that naming is what makes `verified_by:` on a
   ledger row meaningful.
2. **Committed result records** — every experiment writes a small, committed artifact (JSON with
   `metric_name`, `value`, `game/state`, seed, flags like `bit_exact_rerun`). No result lives only
   in a terminal scrollback or a plot.
3. **Aggregators with embedded self-checks** — the leaderboard script asserts known-answer rows on
   every run (e.g. "the oracle-copy method must score F=1.000, the uniform baseline must be at
   floor"). Sanity anchors ride along with every rebuild.
4. **Two-tier CI** — a fast gate on every push/PR, and a scheduled heavy workflow (nightly +
   manual) for the slow suites, with a documented reason for the split.
5. **An open development log** — an append-only bug/fix/ruled-out-hypothesis log and a per-phase
   status ledger, so the process itself is inspectable ("769 commits with the prompt→code chain;
   a 7,525-line running bug-fix log" is itself one of the trust pillars).
6. **Operating rules** — the repo's agent/contributor rules encode the audit culture:
   *"Verify, don't claim — report numbers you actually measured; if tests fail, say so with the
   output."* Commit history doubles as the developer log.

## 6. Step-by-step recipe

1. **Enumerate the claims.** Go through the paper draft; every number, table and figure becomes a
   ledger row. Include the ones you *didn't* run (`status: deferred`) and the methods that don't
   apply (an explicit "N/A audit" row beats silence).
2. **Commit the artifacts.** Make sure every claimed number exists in a committed machine-readable
   file, produced by a script in the repo. If a number currently only exists in a figure or a
   notebook, fix that first — this step usually finds real gaps.
3. **Name the gates.** For each row, identify the test that guards it. If none exists, either
   write one or record the guard honestly as e.g. "raw JSON env block" (self-describing artifact).
4. **Write the store generator** reading the committed artifacts (reusing analysis loaders), and
   emit `site_data.json` with a `meta` block (counts, item list, `generated_by`, `source`).
5. **Write the manifest** — claims, environment, provenance, prose with placeholders. Rule: if you
   catch yourself typing a digit that appears in the paper, stop and template it.
6. **Write the builder** — stdlib-only, renders manifest+store to static pages; stamp `git rev-parse
   HEAD` in the footer; document the rebuild commands.
7. **Add the drill-downs** — per-item cards and detail pages, each "in the audit" box reading the
   store.
8. **Write the code tour** for the harnesses — call-stack tables with verified line anchors.
9. **Host it** — GitHub/GitLab Pages from `/docs` (add `.nojekyll`), or commit the HTML into the
   repo if Pages is unavailable.
10. **Wire it into the workflow** — regenerating the store + pages after results change must be
    routine (ideally enforced: a freshness gate that fails the commit when rendered pages don't
    match a rebuild).

## 7. Acceptance checklist (the reviewer test)

The audit is done when a reviewer who has never seen the repo can, in **at most two clicks from
the front page**, answer each of these for any number in the paper:

- [ ] Which committed file does this number live in, and which script wrote it?
- [ ] What is the verbatim command to regenerate it, and roughly how long / on what hardware?
- [ ] Which test or oracle would have caught it being wrong?
- [ ] What seed / environment / input hashes does it depend on?
- [ ] What was excluded or deferred, and what is the stated reason?
- [ ] Is any number on the site potentially inconsistent with another page? (Must be *impossible
      by construction* — single store.)
- [ ] Which git commit was this audit built from?
