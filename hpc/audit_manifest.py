"""Manifest for the results-audit site — claims, pillars, gates, prose. NO numbers.

Every result number shown on the audit site is a ``{placeholder}`` filled at build time by
``make_audit.compute_values()`` from the committed artifacts (outputs/, splits, lockfile). If you
catch yourself typing a digit that appears in the paper, stop and template it. Rendered by
``hpc/make_audit.py``; integrity-checked by ``tests/integration/test_audit_manifest.py`` (paths
exist, placeholders resolve, gate anchors match source, paper anchors occur verbatim in main.tex,
every Contributions bullet is backed by a claim row).

Modeled on the UnderstandingVCS results audit (see docs/results_audit/): each claim resolves to
claim -> value -> script -> command -> artifact -> runtime -> hardware -> verified-by -> status.
"""

from __future__ import annotations

from dataclasses import dataclass

from evemtbench.version import MULTI_GRID_SPLITS_VERSION, SPLITS_VERSION

# Claim statuses. "auto" resolves against the coverage scan at build time: "measured" when every
# cell selected by ``cells`` is done, else "in progress (N pending, M failed)". We never dress a
# plan up as a result: planned work is "deferred", honest caveats are "note".
STATUSES = ("measured", "in progress", "enforced", "deferred", "note", "auto")

RUNNER = "PYTHONPATH=src python -m evemtbench.baselines.runner"


@dataclass(frozen=True)
class Claim:
    claim: str  # one-sentence claim ({placeholders} allowed)
    value: str  # value template, e.g. "{fd_best_test} test / {fd_best_bench} benchmark"
    detail: str  # 2-3 sentences of context ({placeholders} allowed)
    script: str  # repo-relative producing script ("" only for deferred/note rows)
    command: str  # verbatim reproduce command, or "—"
    artifact: str  # committed path, or an outputs/<...> pattern for runtime artifacts
    runtime: str  # wall-clock template
    hardware: str  # where it ran
    verified_by: tuple[str, ...]  # gate ids (GATES) and/or test node ids "path::test_name"
    status: str  # one of STATUSES
    cells: dict | None = None  # coverage selector for status="auto": subset of
    #   {task, protocol, grid, baseline}; missing key = wildcard
    paper_anchor: str = ""  # verbatim (whitespace-normalized) substring of main.tex
    contribution: int | None = None  # which Contributions \item (1-5) this row backs


@dataclass(frozen=True)
class Gate:
    gate_id: str
    title: str
    asserts_what: str  # lay one-liner
    file: str  # repo-relative source
    anchor_snippet: str  # EXACT source substring; the line anchor is derived at build
    #   time by make_audit._line_of() — never a hand-written line number
    failure_looks_like: str
    how_to_read: str  # the single-assertion-that-matters paragraph
    run_command: str
    # The guided call stack (UnderstandingVCS conformance-tour idiom): (file, exact unique
    # snippet, what-is-actually-computed). Line anchors are derived at build time; a moved or
    # duplicated snippet fails the build and the manifest-integrity test.
    stack: tuple[tuple[str, str, str], ...] = ()


# ---------------------------------------------------------------------------------------------
# The "why you can trust this" pillars (audit/index.html hero panel).
# (title, one sentence, gate_id or "" for none, source path shown as the link)
# ---------------------------------------------------------------------------------------------
PILLARS: list[dict] = [
    dict(
        title="Frozen, audited splits",
        text=(
            "Episode-disjoint, family-pure train/val/test/benchmark splits are committed to the "
            "repo and re-audited by named tests on every CI run — no training window shares an "
            f"episode with an evaluation window (held_out v{SPLITS_VERSION}; multi_grid pretrain "
            f"v{MULTI_GRID_SPLITS_VERSION}, topology-grouped)."
        ),
        gate_id="leakage",
        source="src/evemtbench/splits",  # both committed trees: v1.0 held_out + v1.1 multi_grid
    ),
    dict(
        title="Analytic floor self-check",
        text=(
            "The majority baseline must land on the closed-form 1/k chance floor and no learned "
            "baseline may fall below it — a known-answer check over the whole results matrix, "
            "re-runnable in seconds."
        ),
        gate_id="floor",
        source="src/evemtbench/coverage.py",
    ),
    dict(
        title="Never-trained benchmark family + attestation",
        text=(
            "The benchmark family is scored but never fitted or model-selected on; every run "
            "commits a machine-readable no-tuning attestation alongside git SHA, lockfile hash "
            "and seeds in its run record."
        ),
        gate_id="",
        source="docs/determinism.md",
    ),
    dict(
        title="Byte-fresh rendered views",
        text=(
            "RESULTS.md, leaderboard.html and this audit site are pure functions of outputs/ — a "
            "pre-commit gate fails any commit where the committed pages differ from a fresh "
            "rebuild, so a stale number cannot be published."
        ),
        gate_id="freshness",
        source="hpc/check_results_fresh.py",
    ),
]

# ---------------------------------------------------------------------------------------------
# Verification gates (audit/gates.html code tour). Anchor snippets are exact source substrings;
# a manifest test asserts each occurs exactly once in its file.
# ---------------------------------------------------------------------------------------------
GATES: list[Gate] = [
    Gate(
        gate_id="leakage",
        title="Leakage gate — episode-disjoint, family-pure committed splits",
        asserts_what=(
            "No episode contributes windows to more than one partition, and every partition "
            "draws from its declared dataset family only."
        ),
        file="tests/integration/test_leakage.py",
        anchor_snippet="def test_committed_splits_are_disjoint_and_family_pure(grid):",
        failure_looks_like=(
            "pytest fails naming the grid and the episode id that appears in two partitions "
            "(or the window whose family is wrong)."
        ),
        how_to_read=(
            "The assertion that matters is assert_disjoint_triples() over the committed split "
            "files of each grid — the exact files every run reads. CI re-runs this test by name "
            "on every push so a leaky split can never merge silently."
        ),
        run_command="pytest tests/integration/test_leakage.py -q",
        stack=(
            (
                "tests/integration/test_leakage.py",
                "def test_committed_splits_are_disjoint_and_family_pure(grid):",
                "Entry point, parametrized over every registered grid; data-free — it reads "
                "only the committed, packaged split files.",
            ),
            (
                "src/evemtbench/splits/loader.py",
                "path = split_dir(protocol, grid, version=version, splits_root=splits_root)"
                ' / f"{split}.txt"',
                "Resolves the concrete committed artifact: "
                "splits/v<ver>/held_out/<grid>/<split>.txt.",
            ),
            (
                "src/evemtbench/splits/loader.py",
                "for line in path.read_text().split():",
                "The actual read — each line is one (family, grid, sim_idx) episode triple; "
                "these files are authoritative, never recomputed at run time.",
            ),
            (
                "src/evemtbench/splits/leakage.py",
                'raise LeakageError(f"{t} appears in both {seen[t]!r} and {name!r} partitions")',
                "THE core check: raises the moment any episode appears in two partitions "
                "(single dict scan over all four partitions).",
            ),
            (
                "tests/integration/test_leakage.py",
                'assert all(fam == "adapt_grid" for fam, _g, _i in triples[name])',
                "Family purity: train/val/test may draw only from the adapt_grid family "
                "(benchmark-family purity asserted separately just below).",
            ),
        ),
    ),
    Gate(
        gate_id="splits_audit",
        title="Splits audit — the validator is itself validated",
        asserts_what=(
            "The split auditor detects a planted duplicate episode and a missing held-out "
            "class — proving the leakage checks can actually catch what they claim to."
        ),
        file="tests/integration/test_splits_audit.py",
        anchor_snippet="def test_audit_flags_episode_leakage():",
        failure_looks_like=(
            "pytest fails because a deliberately corrupted split was NOT flagged — i.e. the "
            "audit machinery itself regressed."
        ),
        how_to_read=(
            "This is the mirror of the leakage gate: it corrupts a split on purpose and asserts "
            "the auditor raises. A green run means the leakage gate's 'pass' is meaningful, not "
            "vacuous."
        ),
        run_command="pytest tests/integration/test_splits_audit.py -q",
        stack=(
            (
                "tests/integration/test_splits_audit.py",
                "def test_audit_flags_episode_leakage():",
                "The mirror gate's entry: proves the auditor is non-vacuous by making it catch "
                "a deliberately planted leak.",
            ),
            (
                "tests/integration/test_splits_audit.py",
                '"val": [0, *range(14, 17)],',
                "Plants the leak — episode 0 is put in val while it already sits in train.",
            ),
            (
                "src/evemtbench/splits/audit.py",
                'not (sets["train"] & sets["val"])',
                "The disjointness computation (pairwise set intersections across partitions); "
                "the planted duplicate makes this intersection non-empty.",
            ),
            (
                "src/evemtbench/splits/audit.py",
                '"ok": disjoint and fractions_ok and coverage_ok,',
                "The report: ok only when disjointness, split fractions AND per-class held-out "
                "coverage all hold.",
            ),
            (
                "tests/integration/test_splits_audit.py",
                'assert not rep["disjoint"] and not rep["ok"]',
                "THE assertion: the planted leak must flip both flags — if this ever passes "
                "with a healthy report, the auditor itself regressed.",
            ),
        ),
    ),
    Gate(
        gate_id="floor",
        title="Analytic floor gate — majority ≈ 1/k, learned ≥ floor",
        asserts_what=(
            "Per (task × grid): the majority baseline's balanced accuracy equals the analytic "
            "1/k chance floor (tolerance 0.02), and no learned baseline scores below the floor "
            "on the test split — direction-aware, pending cells skipped (never a false fail)."
        ),
        file="src/evemtbench/coverage.py",
        anchor_snippet="def check_floor(",
        failure_looks_like=(
            "exit 1 listing each violating (task, grid, baseline) with its value vs the floor — "
            "e.g. a mis-mapped label set or an evaluation bug pushing majority off 1/k."
        ),
        how_to_read=(
            "A known-answer test over the whole results matrix: the floor is computable by hand "
            "(1/k for k classes), so agreement is evidence the metric pipeline, label "
            "derivations and aggregation are wired correctly end to end."
        ),
        run_command="PYTHONPATH=src python -m evemtbench.coverage --check-floor",
        stack=(
            (
                "src/evemtbench/coverage.py",
                "def check_floor(",
                "Entry: iterates the coverage cells grouped by (task, protocol, grid) on the "
                "in-distribution test set; pending/undefined cells are skipped, never failed.",
            ),
            (
                "src/evemtbench/coverage.py",
                "maj_doc = _load_result(cell_dir(out, maj_cell, version=version)"
                ' / test_set / "result.json")',
                "Reads the majority baseline's committed result.json for the group.",
            ),
            (
                "src/evemtbench/coverage.py",
                "floor = 1.0 / k",
                "The analytic chance floor — computable by hand: 1/k over the k classes "
                "present in the eval set.",
            ),
            (
                "src/evemtbench/coverage.py",
                "if abs(maj - floor) > tol:",
                "Plausibility: the majority baseline's balanced accuracy must sit within tol "
                "(0.02) of 1/k, else labels/vocabulary/eval alignment are broken.",
            ),
            (
                "src/evemtbench/coverage.py",
                "if direction * (val - maj) < -band:",
                "Dominance: no learned baseline may score below the majority floor — "
                "direction-aware, with a relative band for regression scales.",
            ),
        ),
    ),
    Gate(
        gate_id="freshness",
        title="Freshness gate — committed views byte-match a rebuild",
        asserts_what=(
            "RESULTS.md, leaderboard.html, audit/ and coverage_snapshot.json equal a fresh "
            "rebuild from outputs/ byte-for-byte (one volatile 'updated …' line normalized)."
        ),
        file="hpc/check_results_fresh.py",
        anchor_snippet="if _normalize(committed) != _normalize(fresh):",
        failure_looks_like=(
            "exit 1 with a unified diff of the stale file; the pre-commit hook blocks the "
            "commit until `--fix` regenerates it."
        ),
        how_to_read=(
            "The one comparison that matters is committed-vs-fresh equality per rendered file. "
            "Because the builders are pure functions of outputs/, every number on this site is "
            "mechanically traceable to a result.json — hand-editing a published number is "
            "impossible without the gate failing."
        ),
        run_command="python hpc/check_results_fresh.py",
        stack=(
            (
                "hpc/check_results_fresh.py",
                "def _fresh_leaderboard() -> str:",
                "Fresh-build calls: every view is rebuilt from outputs/ in-process, all with "
                "generated_at=None so the rebuild is deterministic.",
            ),
            (
                "hpc/make_audit.py",
                "def fresh_pages(",
                "Builder purity: the single audit-site entry point the gate and the writer "
                "share — one code path, so verify and regenerate can never diverge.",
            ),
            (
                "src/evemtbench/leaderboard.py",
                "def render_html(",
                "Same for the leaderboard: the caller stamps time, the renderer is pure.",
            ),
            (
                "hpc/check_results_fresh.py",
                "if _normalize(committed) != _normalize(fresh):",
                "THE comparison: committed vs fresh, byte-for-byte, after the single volatile "
                "'updated …' line is normalized away.",
            ),
            (
                "hpc/check_results_fresh.py",
                "if p.name not in audit_fresh and p.name not in _UNGATED",
                "Stray-page check: a committed audit page a rebuild no longer produces is "
                "stale too (progress.html is the one documented, history-derived exemption).",
            ),
        ),
    ),
    Gate(
        gate_id="integrity",
        title="Artifact integrity — the outputs corpus itself is checked",
        asserts_what=(
            "Every done cell carries the full seed set, a schema-valid provenance record with "
            "non-null git/lockfile pins, run_record.seeds == result.seeds, and existing "
            "checkpoints; --deep additionally verifies every predictions parquet's row count "
            "against n_test (truncated-write detection)."
        ),
        file="src/evemtbench/integrity.py",
        anchor_snippet="def check_outputs(",
        failure_looks_like=(
            "exit 1 listing each problem as '<cell>: <what is inconsistent>'; pending cells are "
            "skipped, never failed (coverage owns those)."
        ),
        how_to_read=(
            "CI tests validate the code; this gate validates the artifacts the sweep left "
            "behind, which only exist on the loop host. It also reports the distinct "
            "(git sha, lockfile hash) pairs across the corpus — the evidence behind the paper's "
            "'attributable to a frozen configuration' sentence. Run at every freeze."
        ),
        run_command="PYTHONPATH=src python -m evemtbench.integrity",
        stack=(
            (
                "src/evemtbench/integrity.py",
                "def check_outputs(",
                "Entry: iterates every applicable cell and aggregates per-cell problems plus "
                "the distinct (git sha, lockfile hash) provenance pairs.",
            ),
            (
                "src/evemtbench/integrity.py",
                'if cell_status(out, cell, version=version) != "done":',
                "Skip-pending: incomplete cells are counted as skipped, never failed — "
                "coverage owns those.",
            ),
            (
                "src/evemtbench/integrity.py",
                "if expected_seeds is not None and seeds != expected_seeds:",
                "Seed-set check: every done cell must carry the full production seed set "
                "[0..4] in each test set's result.json.",
            ),
            (
                "src/evemtbench/integrity.py",
                "for err in validator.iter_errors(record):",
                "run_record.json validates against the committed JSON Schema "
                "(baselines/run_record.schema.json).",
            ),
            (
                "src/evemtbench/integrity.py",
                'if n_rows != doc.get("n_test"):',
                "--deep: each predictions parquet's metadata row count must equal n_test — "
                "catches truncated NFS writes no code test can see.",
            ),
        ),
    ),
    Gate(
        gate_id="contract_pin",
        title="Contract pin — SHA-256-pinned window contract and inputs",
        asserts_what=(
            "The vendored window contract matches its pinned SHA-256 and version; every run "
            "record pins the dependency lockfile hash; every windowed input carries a sidecar "
            "manifest whose feature hash is verified at load time."
        ),
        file="src/evemtbench/contract/__init__.py",
        anchor_snippet="def verify_pin() -> list[str]:",
        failure_looks_like=(
            "verify_pin() returns a non-empty problem list (version or SHA mismatch) and the "
            "contract-conformance tests fail."
        ),
        how_to_read=(
            "Three pins close the input side: contract SHA (window geometry can't drift), the "
            "per-file feature hash checked when windows are memory-mapped (channel order can't "
            "drift), and the environment. Read the environment pin precisely: a lockfile SHA in a "
            "run record proves only which file sat in the repo, NOT which interpreter ran — so the "
            "real check is that each record's independently captured numpy/pandas/scikit-learn/"
            "scipy/torch versions agree with the captured results environment "
            "(env/results-v<version>.lock, verified by hpc/verify_results_env.py). Two independent "
            "captures agreeing is the guarantee; the hash alone is a pointer."
        ),
        run_command="pytest tests/integration/test_contract_conformance.py -q",
        stack=(
            (
                "src/evemtbench/contract/__init__.py",
                "def verify_pin() -> list[str]:",
                "Offline pin entry — returns a list of problems (empty = OK).",
            ),
            (
                "src/evemtbench/contract/__init__.py",
                "if actual != CONTRACT_SHA256:",
                "The SHA compare: live sha256(contract.json) must equal the pinned constant — "
                "the window-format contract cannot drift unreviewed.",
            ),
            (
                "src/evemtbench/data/windows.py",
                'if manifest.get("status") != "complete":',
                "Load-time manifest gate: status, topology and window geometry are all "
                "verified before a single byte of waveform data is mapped.",
            ),
            (
                "src/evemtbench/data/windows.py",
                "if h.hexdigest() != fhash:",
                "Feature-hash consistency: sha256 over the channel-name list must equal the "
                "producer manifest's feature_hash — channel order cannot silently shift.",
            ),
            (
                "src/evemtbench/baselines/runner.py",
                "def _results_env_hash(version: str) -> str | None:",
                "The environment pin: sha256(env/results-v<version>.lock) — the captured conda "
                "prefix incl. torch + CUDA — recorded into every new run_record.json. The older "
                "lockfile_sha256 pins the CI dependency set, which contains no torch and so never "
                "described the results environment.",
            ),
        ),
    ),
]

# The two pipelines the gates guard, as plain diagrams (gates.html header). Real module names.
DATA_PIPELINE = f"""\
PowerFactory EMT simulation            (producer; labels exact by construction)
   |  writes X_<topology>.raw + y_<topology>.parquet + *.raw.manifest.json (feature_hash)
   v
data/windows.load_windows      -- _verify_manifest: status/topology/geometry  [contract_pin]
   |                              sha256(feature_names) == manifest.feature_hash
   v
splits/loader.read_held_out_splits -> committed v{SPLITS_VERSION}/held_out/<grid>/*.txt   [leakage]
   |  episode-disjoint, family-pure                                          [splits_audit]
   |  (multi_grid_pretrain: v{MULTI_GRID_SPLITS_VERSION}/multi_grid_pretrain/, topology-grouped)
   v
baselines/runner  (fit on train, select on val; benchmark family predict-only)
   v
outputs/<ver>/<task>/<protocol>/<grid>/<baseline>/<test_set>/result.json
        + predictions_seed*.parquet + run_record.json                        [integrity]"""

PUBLISH_PIPELINE = """\
outputs/<ver>/**/result.json + run_record.json
   |-- make_results.build_results        -> RESULTS.md
   |-- leaderboard.render_html           -> leaderboard.html                 [floor]
   |-- make_audit.fresh_pages            -> audit/*.html
   '-- coverage.render_snapshot          -> coverage_snapshot.json
   v
committed views  ->  hpc/check_results_fresh.py (byte-compare)               [freshness]
   v
.githooks/pre-commit blocks any drifting commit; CI re-runs the same gates"""


# ---------------------------------------------------------------------------------------------
# Why a matrix cell can be N/A by construction (grey in audit/coverage.html), keyed by rule id.
# These mirror coverage.baseline_applies / protocol_grid_applies — N/A is a stated rule, never a
# silent gap, and N/A cells are excluded from all denominators by construction.
# ---------------------------------------------------------------------------------------------
NA_REASONS: dict[str, str] = {
    "threshold_scope": (
        "threshold is a single-terminal overcurrent relay rule — it applies only to the binary "
        "local fault-detection task (coverage.baseline_applies)"
    ),
    "transfer_deep_only": (
        "transfer protocols need transferable weights, so they apply only to the deep baselines "
        "mlp/gru/cnn/resnet (coverage.baseline_applies)"
    ),
    "pretrain_grid_only": (
        "multi_grid_pretrain runs only on the template_20kv/template_110kv corpora; held_out and "
        "the transfer protocols run only on the four reference grids "
        "(coverage.protocol_grid_applies)"
    ),
}

# ---------------------------------------------------------------------------------------------
# Environment (audit/environment.html). Static facts; per-run numbers come from run records.
# ---------------------------------------------------------------------------------------------
ENVIRONMENT: dict = {
    "hardware": [
        (
            "GPU sweeps",
            "NHR@FAU Alex — 1× NVIDIA A40 per job (SLURM a40 partition)",
            "deep baselines (mlp/gru/cnn/resnet), 24 h wall per grid sweep (hpc/sweep.sh)",
        ),
        (
            "Classical baselines",
            "NHR@FAU Alex — A40 MIG slice",
            "majority/threshold/random_forest sweeps auto-routed by hpc/submit.sh",
        ),
        (
            "Login node",
            "cluster frontend (no compute)",
            "submission, git, log reads, deterministic re-rendering only",
        ),
    ],
    "seed_policy": (
        "Every cell aggregates seeds 0–4 (5 seeds); results report mean ± 95% CI half-width "
        "(Student t, ddof=1). What is and is not bit-reproducible is documented in "
        "docs/determinism.md."
    ),
    "staging": (
        "Window memmaps are staged to node-local $TMPDIR before training; sidecar manifests "
        "(feature hash, shape, contract fingerprint) travel with the data and are re-verified "
        "at load time."
    ),
}

# Trust docs the audit links instead of duplicating. (repo path, label)
DOC_LINKS: list[tuple[str, str]] = [
    ("docs/protocol.md", "protocol — what is scored and how"),
    ("docs/determinism.md", "determinism — what is and is not reproducible"),
    ("docs/submission.md", "submission — artifact schemas & repro requirements"),
    ("docs/metrics.md", "metrics — the frozen metric contract"),
    ("docs/grids.md", "grids — the four reference topologies"),
]

# ---------------------------------------------------------------------------------------------
# THE CLAIM LEDGER (audit/ledger.html).
# Section A — headline results (status="auto" resolves from the coverage scan).
# Section B — protocol / infrastructure claims (mostly "enforced").
# Section C — deferred / honesty rows.
# ---------------------------------------------------------------------------------------------


def _headline(fn_key: str, task: str, label: str, metric_phrase: str) -> Claim:
    """One Section-A headline row per paper function (representative cell: double_line)."""
    k = fn_key
    return Claim(
        claim=f"{label}: learnable in-distribution and scored on the never-trained "
        f"benchmark family",
        value=(
            f"{{{k}_best_test}} test / {{{k}_best_bench}} benchmark ({metric_phrase}; best "
            f"learned baseline {{{k}_best_name}}; floor {{{k}_floor}})"
        ),
        detail=(
            f"Representative cell: {task} on double_line under the held_out protocol, 5 seeds, "
            "mean ± 95% CI. Trained on the adapt_grid train split, model-selected on val; the "
            "benchmark family was never fitted or tuned on (attestation in every run record). "
            "Every grid × view combination is in the coverage matrix and the per-task page."
        ),
        script="src/evemtbench/baselines/runner.py",
        command=(
            f"{RUNNER} --task {task} --protocol held_out --grid double_line "
            f"--baseline {{{k}_best_name}} --seeds 0 1 2 3 4"
        ),
        artifact=f"outputs/<version>/{task}/held_out/double_line/<baseline>/*/result.json",
        runtime=f"~{{{k}_fit}} fit per seed (compute.fit_seconds_mean)",
        hardware="NHR@FAU Alex, 1× A40 · seeds 0–4",
        verified_by=("floor", "leakage"),
        status="auto",
        cells={"task": task, "protocol": "held_out"},
    )


CLAIMS: list[Claim] = [
    # ---- Section A — headline results -------------------------------------------------------
    _headline("fd", "fault_detection_local", "Fault detection (FD, binary)", "balanced accuracy"),
    _headline(
        "fc",
        "fault_classification_local",
        "Fault-type classification (FC, 9-class)",
        "balanced accuracy",
    ),
    _headline(
        "fl", "fault_location_local", "Fault localisation (FL, regression)", "MAE, lower better"
    ),
    _headline("ed", "event_detection_global", "Event detection (ED, binary)", "balanced accuracy"),
    _headline(
        "ec",
        "event_classification_global",
        "Event classification (EC, 23-class)",
        "balanced accuracy",
    ),
    Claim(
        claim=(
            "Cross-grid transfer from the multi_grid pretrain corpora (zero-shot vs fine-tune) "
            f"under the corrected topology-grouped source splits (v{MULTI_GRID_SPLITS_VERSION})"
        ),
        value="{n_transfer_done} of {n_transfer_cells} transfer cells measured "
        "(+ {n_pretrain_done}/{n_pretrain_cells} pretrain cells)",
        detail=(
            "Deep baselines pretrained on the multi_grid template corpus are evaluated zero-shot "
            "and fine-tuned on the four reference grids. Transfer protocols are defined only for "
            "baselines with transferable weights — classical baselines are N/A by rule, not "
            "missing."
        ),
        script="hpc/sweep.sh",
        command=(
            f"{RUNNER} --task fault_detection_local --protocol transfer_finetune "
            "--grid testgrid_110kv --baseline resnet --seeds 0 1 2 3 4"
        ),
        artifact="outputs/<version>/*/transfer_*/*/*/*/result.json",
        runtime="≤ 24 h wall per grid sweep job",
        hardware="NHR@FAU Alex, 1× A40",
        verified_by=("tests/integration/test_transfer_protocols.py", "floor"),
        status="auto",
        cells={"protocol": "transfer_zeroshot|transfer_finetune"},
    ),
    # ---- Section B — protocol / infrastructure claims ---------------------------------------
    Claim(
        claim="Protocol-driven benchmark: every result is conditional evidence, never one score",
        value="every value keyed by (task × observability × grid × protocol × test set × metric)",
        detail=(
            "The dataset (waveforms, labels, metadata) is separated from the protocol that fixes "
            "how it may be used. RESULTS.md, the leaderboard and the paper tables all render "
            "per-cell values; no aggregate across tasks or grids exists anywhere in the "
            "pipeline."
        ),
        script="hpc/make_results.py",
        command="PYTHONPATH=src python hpc/make_results.py",
        artifact="RESULTS.md",
        runtime="seconds (pure read of outputs/)",
        hardware="any",
        verified_by=("freshness",),
        status="enforced",
        paper_anchor="no aggregate score is defined",
        contribution=1,
    ),
    Claim(
        claim="Five tasks with frozen label derivations; one scoring rule per window",
        value="{n_tasks} registered task ids: 12 objective functions × nested observability",
        detail=(
            "ED, EC, FD, FC, FL (plus fault-attribute and switching tasks) each carry a frozen "
            "label derivation and valid-sample mask, so a window is scored by exactly one rule "
            "per task. Changing a label derivation is a major-version event."
        ),
        script="src/evemtbench/tasks/registry.py",
        command='PYTHONPATH=src python -c "from evemtbench.tasks.registry import '
        'BENCHMARK_TASKS; print(len(BENCHMARK_TASKS))"',
        artifact="src/evemtbench/tasks/labels/",
        runtime="—",
        hardware="—",
        verified_by=(
            "tests/integration/test_registry_coherence.py",
            "tests/integration/test_new_tasks_load.py",
        ),
        status="enforced",
        paper_anchor="frozen target derivation and valid-sample mask",
        contribution=2,
    ),
    Claim(
        claim="Three nested observability views on four reference grids spanning 20–345 kV",
        value="local (6 ch) ⊂ line (12 ch) ⊂ global (per-grid width) × {n_grids} grids",
        detail=(
            "Single-terminal local, two-terminal line-differential and grid-wide global views "
            "are evaluated on double_line (110 kV), cigre_mv (20 kV), testgrid_110kv (110 kV) "
            "and ieee39 (345 kV, 60 Hz), plus the template_20kv + template_110kv pretrain corpora."
        ),
        script="src/evemtbench/data/roots.py",
        command='PYTHONPATH=src python -c "from evemtbench.data.roots import GRIDS; '
        'print(list(GRIDS))"',
        artifact="src/evemtbench/data/roots.py",
        runtime="—",
        hardware="—",
        verified_by=("tests/integration/test_line_view.py", "leakage"),
        status="enforced",
        paper_anchor="three nested views",
        contribution=3,
    ),
    Claim(
        claim="Episode-disjoint, family-pure 70/15/15 committed splits with leakage controls",
        value="committed split files for {n_grids} grids × 4 partitions, re-audited on every CI "
        "run",
        detail=(
            "Splits are stratified at episode level and committed as plain text, so every run "
            "reads exactly the same partitions. The leakage tests re-derive and re-audit them; "
            "the splits auditor is itself tested to catch planted leaks."
        ),
        script="src/evemtbench/splits/loader.py",
        command="pytest tests/integration/test_leakage.py tests/integration/test_splits_audit.py"
        " -q",
        # both committed trees, " · "-separated (the ledger renders each as its own link)
        artifact=f"src/evemtbench/splits/v{SPLITS_VERSION}/"
        f" · src/evemtbench/splits/v{MULTI_GRID_SPLITS_VERSION}/",
        runtime="seconds",
        hardware="any (CI re-runs it)",
        verified_by=("leakage", "splits_audit"),
        status="enforced",
        paper_anchor="Episodes rather than windows are the unit of partitioning",
        contribution=1,
    ),
    Claim(
        claim="Transparent, beatable baseline suite from trivial floor to deep learning",
        value="{n_baselines} baselines: majority · threshold · random_forest · mlp · gru · cnn "
        "· resnet",
        detail=(
            "The majority floor anchors chance level analytically; the overcurrent threshold "
            "represents conventional protection practice; RF and four deep architectures span "
            "the learned spectrum. Splits, leakage checks and evaluation scripts are released "
            "with the repo."
        ),
        script="src/evemtbench/baselines/",
        command=f"{RUNNER} --help",
        artifact="src/evemtbench/baselines/",
        runtime="—",
        hardware="—",
        verified_by=("floor", "tests/integration/test_runner_cli.py"),
        status="enforced",
        paper_anchor=(
            "The reference suite spans trivial, conventional, feature-based, "
            "and deep-learning approaches"
        ),
        contribution=3,
    ),
    Claim(
        claim="The benchmark family is scored but never fitted or model-selected on",
        value="{n_attested} of {n_run_records} run records carry the no-tuning attestation",
        detail=(
            "Every run record commits fit_on / model_selection_on / held_out_untouched plus a "
            "plain-language statement. The audit page is a production reader of these records: "
            "the count on the left is computed from them at build time."
        ),
        script="src/evemtbench/baselines/runner.py",
        command="jq .attestation outputs/<version>/<task>/<protocol>/<grid>/<baseline>/"
        "run_record.json",
        artifact="outputs/<version>/*/*/*/*/run_record.json",
        runtime="—",
        hardware="—",
        verified_by=("tests/integration/test_runner_cli.py",),
        status="measured",
    ),
    Claim(
        claim="Analytic floor self-check over the whole matrix",
        value="{floor_n_checked} comparisons across {floor_n_groups} (task × grid) groups: "
        "{floor_n_violations} violations, {floor_n_skipped} skipped (pending)",
        detail=(
            "The majority baseline must land on the closed-form 1/k floor and no learned "
            "baseline may fall below it on the test split. A violation would indicate a broken "
            "label mapping or metric wiring — the benchmark's known-answer test."
        ),
        script="src/evemtbench/coverage.py",
        command="PYTHONPATH=src python -m evemtbench.coverage --check-floor",
        artifact="outputs/<version>/ (read-only check)",
        runtime="seconds",
        hardware="any machine with outputs/",
        verified_by=("floor",),
        status="measured",
    ),
    Claim(
        claim="Rendered views are byte-fresh: RESULTS.md, leaderboard.html and this audit site",
        value="pre-commit gate + CI-tested builders; one normalized volatile line",
        detail=(
            "All rendered views are pure functions of outputs/. The pre-commit hook runs the "
            "freshness gate, so a commit with stale numbers is rejected; this row is "
            "self-referential — the page you are reading is covered by the gate it describes."
        ),
        script="hpc/check_results_fresh.py",
        command="python hpc/check_results_fresh.py",
        artifact="RESULTS.md · leaderboard.html · audit/ · coverage_snapshot.json",
        runtime="seconds",
        hardware="loop host (skips where outputs/ is absent)",
        verified_by=("freshness", "tests/integration/test_check_results_fresh.py"),
        status="enforced",
    ),
    Claim(
        claim="Frozen window contract: 50 ms windows, 5 ms stride, 480 samples @ 9.6 kHz",
        value="contract v{contract_version}, SHA-256 {contract_sha}",
        detail=(
            "The window geometry and channel order are vendored with a pinned SHA-256; each "
            "windowed input file carries a sidecar manifest whose feature hash is verified at "
            "load time. Changing the contract is a major-version event."
        ),
        script="src/evemtbench/contract/__init__.py",
        command="pytest tests/integration/test_contract_conformance.py -q",
        artifact="src/evemtbench/contract/contract.json",
        runtime="seconds",
        hardware="any",
        verified_by=("contract_pin", "hpc/check_contract_sync.py"),
        status="enforced",
        paper_anchor="windows of 480 samples",
    ),
    Claim(
        claim="Every run emits a machine-readable provenance record",
        value="{n_run_records} run records · {n_git_shas} distinct git SHAs · "
        "{n_lockfile_shas} lockfile hash(es) · {n_hosts} recorded hosts",
        detail=(
            "run_record.json per cell: git commit, dependency-lockfile SHA-256, library "
            "versions, seeds, per-seed compute, checkpoint SHA-256s and the no-tuning "
            "attestation; newer records add the executing host, SLURM job id and input-data "
            "hashes. The counts here are read from the records at build time."
        ),
        script="src/evemtbench/baselines/runner.py",
        command="jq '.git_sha,.lockfile_sha256,.host' outputs/<version>/<task>/<protocol>/"
        "<grid>/<baseline>/run_record.json",
        artifact="outputs/<version>/*/*/*/*/run_record.json",
        runtime="written at the end of each cell run",
        hardware="the machine that ran the cell (self-recorded)",
        verified_by=("integrity", "tests/integration/test_checkpoints_telemetry.py"),
        status="measured",
        paper_anchor="Each run records the benchmark version, code commit",
    ),
    Claim(
        claim="Paper result tables are machine-generated from result.json (a live gated view)",
        value="5 main-text fragments + 15 supplementary tables, all rendered from outputs/",
        detail=(
            "hpc/make_paper_tables.py and hpc/make_supplement_tables.py render every numeric "
            "table cell from result.json files; the committed fragments are a live gated view "
            "(byte-checked and --fix-regenerated like RESULTS.md), with five-state markers "
            "(\\pending / \\notapp / \\undefm / \\oos) standing in for any cell without a result. "
            "The public results bundle (results/<version>/tables/) mirrors the same tables from "
            "the same builders, byte-identically."
        ),
        script="hpc/make_paper_tables.py",
        command="PYTHONPATH=src python hpc/make_paper_tables.py",
        artifact="results/",
        runtime="seconds",
        hardware="any",
        verified_by=("tests/integration/test_paper_tables.py", "freshness"),
        status="enforced",
        paper_anchor=(
            "Full results for all candidate baselines are retained in the Supplementary Information"
        ),
    ),
    Claim(
        claim="Coverage: {n_done} of {n_cells} held_out cells measured",
        value="{n_done} done · {n_pending} pending · {n_failed} failed (held_out); "
        "{n_all_done}/{n_all_cells} across all four protocols",
        detail=(
            "A cell is one (task × protocol × grid × baseline) runner invocation; it counts as "
            "done only when every expected test-set result.json is valid. The full matrix with "
            "per-cell status and the exact reproduce command is on the coverage page and in the "
            "committed coverage_snapshot.json."
        ),
        script="src/evemtbench/coverage.py",
        command="PYTHONPATH=src python -m evemtbench.coverage --summary",
        artifact="coverage_snapshot.json",
        runtime="seconds",
        hardware="any machine with outputs/",
        verified_by=("freshness", "tests/unit/test_coverage.py"),
        status="auto",
        cells={"protocol": "held_out"},
    ),
    Claim(
        claim="The scoring harness is self-testing — dataless controls + a per-run oracle check",
        value="12 ceiling/floor/alignment controls (pip-install only) + a positive control "
        "before every persisted result",
        detail=(
            "evemtbench-evaluate --self-test proves the evaluator scores known inputs correctly "
            "with no data: perfect predictions hit the metric ceiling, constant predictions land "
            "exactly on the analytic 1/k floor, malformed prediction frames raise. Every sweep "
            "run additionally feeds the TRUE labels through the same evaluate() path and refuses "
            "to write results unless they score the ceiling (run_record.harness_control)."
        ),
        script="src/evemtbench/evaluation/selftest.py",
        command="evemtbench-evaluate --self-test",
        artifact="outputs/<version>/*/*/*/*/run_record.json",
        runtime="~2 s (self-test) · one extra evaluate per test set (runner control)",
        hardware="any (self-test needs no data root)",
        verified_by=(
            "tests/unit/test_eval_selftest.py",
            "tests/integration/test_runner_cli.py::test_broken_harness_aborts_the_cell",
        ),
        status="enforced",
    ),
    Claim(
        claim="Headline results survive alternative aggregation",
        value="per-task re-aggregation under 4 schemes + seeded bootstrap CIs — see the "
        "robustness page",
        detail=(
            "The same committed per-seed numbers, never re-measured, re-aggregated under "
            "equal-weight-by-grid mean, median, leave-one-grid-out and a seeded percentile "
            "bootstrap; tasks whose best baseline changes under any scheme are flagged "
            "rank-sensitive on audit/robustness.html rather than hidden."
        ),
        script="hpc/make_audit.py",
        command="PYTHONPATH=src python hpc/make_audit.py",
        artifact="audit/robustness.html",
        runtime="seconds (pure read of outputs/)",
        hardware="any machine with outputs/",
        verified_by=("freshness", "tests/unit/test_significance.py"),
        status="measured",
    ),
    # ---- Section C — deferred / honesty rows -------------------------------------------------
    Claim(
        claim="Topology-aware GNN baseline",
        value="deferred",
        detail=(
            "A graph baseline that consumes the grid topology is planned but was not run for "
            "this version — listed for honesty, not dressed up as a result. Design notes: "
            "docs/design/cross_topology.md."
        ),
        script="",
        command="—",
        artifact="—",
        runtime="—",
        hardware="—",
        verified_by=(),
        status="deferred",
        paper_anchor="models that use topology and grid parameters explicitly",
        contribution=None,
    ),
    Claim(
        claim="Cross-topology protocol for topology-agnostic models",
        value="deferred",
        detail=(
            "Training on one set of grids and evaluating on an unseen topology is designed but "
            "not part of this version's scored matrix. Design notes: "
            "docs/design/cross_topology.md."
        ),
        script="",
        command="—",
        artifact="—",
        runtime="—",
        hardware="—",
        verified_by=(),
        status="deferred",
        paper_anchor="topology-held-out evaluation",
        contribution=None,
    ),
    Claim(
        claim="Open-challenge tracks: robustness, selectivity, cross-cluster, pretrain-transfer",
        value="designed, not built",
        detail=(
            "Four extension tracks are specified as design documents only "
            "(docs/design/robustness.md, selectivity.md, pretrain_transfer.md; the "
            "cross-cluster federation design lives in the development repo). No numbers exist "
            "and none are claimed."
        ),
        script="",
        command="—",
        artifact="docs/design/",
        runtime="—",
        hardware="—",
        verified_by=(),
        status="deferred",
    ),
    Claim(
        claim="ieee39 adapt_grid dataset — final",
        value="final",
        detail=(
            "The ieee39 adapt_grid dataset refresh landed in the v1.0.0 re-simulation; all "
            "grids' results (ieee39 included) are final for this version."
        ),
        script="",
        command="—",
        artifact="—",
        runtime="—",
        hardware="—",
        verified_by=(),
        status="note",
    ),
    Claim(
        claim="First reference evaluation, released as an extensible reproducible foundation",
        value="baselines + splits + leakage checks + evaluation scripts, versioned and gated",
        detail=(
            "The reference results are rendered from committed result.json by the gated builders "
            "and traced on the audit dashboard; the released splits, leakage checks, baseline "
            "implementations, and evaluation scripts let others reproduce them and extend the "
            "benchmark with new methods, grids, and a future challenge set."
        ),
        script="hpc/make_results_bundle.py",
        command="PYTHONPATH=src python hpc/make_results_bundle.py",
        artifact="results/",
        runtime="seconds",
        hardware="any",
        verified_by=(
            "tests/integration/test_results_bundle.py",
            "tests/integration/test_paper_numbers_covered.py",
        ),
        status="enforced",
        paper_anchor=(
            "the benchmark implementation, reference baselines, and evaluation resources released"
        ),
        contribution=3,
    ),
]

# ---------------------------------------------------------------------------------------------
# Lay prose (the UnderstandingVCS "What it does?" idiom).
# FUNCTION_ABOUT is keyed by the registry's objective-function value; task pages compose
# FUNCTION_ABOUT[fn] + VIEW_ABOUT[view]. BASELINE_ABOUT is keyed by baseline key.
# ---------------------------------------------------------------------------------------------
FUNCTION_ABOUT: dict[str, str] = {
    "fault_detection": (
        "Decide, from one 50 ms waveform window, whether a short-circuit fault is present. "
        "This is the core protection decision: a relay that misses a fault leaves it burning "
        "(missed-fault rate), one that trips on a healthy window disconnects customers for "
        "nothing (false-alarm rate). Both error types are reported alongside balanced accuracy "
        "because their costs are asymmetric in practice."
    ),
    "fault_class": (
        "Given a faulted window, name the fault type from the nine-class vocabulary (which "
        "phases are involved and whether ground participates). Fault typing drives which poles "
        "a breaker trips and how the event is analysed afterwards. The class vocabulary is "
        "frozen for the benchmark's major version."
    ),
    "fault_location": (
        "Estimate how far along the line the fault sits, as a continuous distance regression. "
        "Crews are dispatched by this number; errors are reported as MAE in the grid's own "
        "distance units, lower is better, with the median absolute error as a robustness "
        "companion."
    ),
    "event_state": (
        "Classify the operating state of the window — steady state, fault, or switching "
        "transient. This is the coarse triage that upstream automation uses to route a window "
        "to the right specialist model."
    ),
    "event_detection": (
        "Decide whether anything at all is happening in the window (any event vs steady "
        "state). The easiest anchor task: a floor for what 'trivially learnable' looks like on "
        "this data."
    ),
    "switch_detection": (
        "Decide whether the window contains a switching action (breaker or disconnector "
        "operating). Switching transients look superficially like faults — this task isolates "
        "that confusion pair."
    ),
    "switch_type": (
        "Given a switching window, name what kind of switching action it is. Distinguishing "
        "planned topology changes from protection trips matters for post-event analysis."
    ),
    "event_classification": (
        "The hard frontier: one 23-class decision naming the exact event in the window — fault "
        "types, switching kinds, and steady state in one vocabulary. High class count and "
        "imbalance make the analytic floor very low; macro-F1 accompanies balanced accuracy."
    ),
    "fault_grounded": (
        "Given a faulted window, decide whether the fault involves ground. Ground faults "
        "dominate real statistics and their detection interacts with the grid's earthing "
        "concept, which differs across the four reference grids."
    ),
    "fault_phase": (
        "Given a faulted window, name which phase (or phase pair) is affected. Phase selection "
        "is what lets single-pole tripping keep the other two phases in service."
    ),
    "fault_category": (
        "Group the fault into its coarse category (e.g. single-phase-to-ground vs multi-phase) "
        "— the level of typing many protection schemes actually act on, between binary "
        "detection and the full nine-class vocabulary."
    ),
    "fault_origin": (
        "Decide whether the fault lies on the observed line or elsewhere in the grid — the "
        "selectivity question. A relay must trip for its own zone and stay quiet for faults it "
        "merely sees from a distance."
    ),
}

VIEW_ABOUT: dict[str, str] = {
    "local": (
        "Local view: one measurement cubicle — three currents and three voltages (6 channels), "
        "what a single protection relay physically sees."
    ),
    "line": (
        "Line view: the cubicles at both ends of one line (12 channels), the information basis "
        "of a line-differential scheme."
    ),
    "global": (
        "Global view: every cubicle in the grid at once (width differs per grid), the "
        "wide-area upper bound on observability."
    ),
}

# (about, reference) per baseline key.
BASELINE_ABOUT: dict[str, tuple[str, str]] = {
    "majority": (
        "Predicts the most frequent training class for every window (or the training-mean for "
        "regression). It contains zero information about the input and exists as the analytic "
        "chance floor: its balanced accuracy must equal 1/k, which the floor gate checks. Any "
        "model worth reporting must clear it.",
        "trivial floor (analytic)",
    ),
    "threshold": (
        "A conventional overcurrent relay rule: trip when the current magnitude exceeds a "
        "threshold fitted on the training split. It represents a century of protection practice "
        "in one parameter and is the domain-classical reference the ML baselines must beat — "
        "defined only for the binary local fault-detection task, exactly like the physical "
        "device it mimics.",
        "conventional protection practice (ANSI 50/51 style)",
    ),
    "random_forest": (
        "A random forest on flattened window features — the strong classical-ML reference: "
        "no architecture tuning, no GPU, surprisingly hard to beat on tabularised waveform "
        "statistics. Trains in minutes on CPU.",
        "Breiman, Machine Learning 2001",
    ),
    "mlp": (
        "A multi-layer perceptron on the flattened window. The simplest learned neural "
        "baseline: no temporal inductive bias, so it measures how far raw capacity gets "
        "without architecture. Its parameter count grows with the global view width, which is "
        "why it is the largest model on ieee39.",
        "Rumelhart et al., Nature 1986",
    ),
    "gru": (
        "A gated recurrent network reading the window sample by sample — the sequential "
        "inductive bias: waveforms as time series with state.",
        "Cho et al., EMNLP 2014",
    ),
    "cnn": (
        "A 1-D convolutional network over the time axis — the local-pattern inductive bias: "
        "transients as translation-invariant wave shapes.",
        "LeCun et al., Proc. IEEE 1998 (1-D adaptation)",
    ),
    "resnet": (
        "A 1-D residual network, the strongest deep baseline in the suite: stacked "
        "convolutional blocks with skip connections, the standard strong time-series "
        "classification architecture.",
        "He et al., CVPR 2016; Wang et al., IJCNN 2017 (time-series 1-D form)",
    ),
}
