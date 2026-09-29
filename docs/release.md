# Release checklist — from frozen results to a citable artifact

[docs/versioning.md](versioning.md) defines *what* a version means (the frozen benchmark
contract); this page defines *how* a release ships. Nothing here is optional at a real release —
each step is either mechanically gated already or produces the artifact the next step needs.
Version consistency across `pyproject.toml` / `CITATION.cff` / `BENCHMARK_VERSION` / `CHANGELOG`
is enforced by `tests/unit/test_version_consistency.py`, so drift fails CI long before a release.

## Pre-freeze (any time before the tag)

1. **Coverage complete or accounted for** — `python -m evemtbench.coverage --summary`; any
   pending/failed cell that will stay out of the release needs a note in the audit ledger
   (`hpc/audit_manifest.py`, status `deferred`/`note`).
2. **Floor gate green** — `PYTHONPATH=src python -m evemtbench.coverage --check-floor`
   (violations are findings; resolve or document them in the paper before freezing).
3. **Deep artifact integrity** — `PYTHONPATH=src python -m evemtbench.integrity --deep` must
   exit 0 over the full corpus (checks every predictions parquet against `n_test`; run on the
   loop host, minutes over NFS).

## Freeze (one commit, on the loop host)

4. **Bump out of dev**: `pyproject.toml` `version`, `src/evemtbench/version.py`
   `BENCHMARK_VERSION` (drop the `-dev` suffix), `CITATION.cff` `version` + `date-released` —
   the consistency test forces all three to move together.
5. **Finalize `CHANGELOG.md`** for the version (the consistency test requires the section).
6. **Snapshot the live tables**: the paper tables are already a live gated view (populated from
   `outputs/`, committed by the publisher each cycle — see the paper-agent notes in the development
   repository, which are not part of the public tree). For a submission snapshot ensure the sweep is complete (or accounted for), then
   `PYTHONPATH=src python hpc/check_results_fresh.py --fix` so `generated/*.tex` match the frozen
   corpus, then rebuild **both** data figures from the matplotlib env (`$PAPERFIGS_PY`, via `srun`):
   `hpc/paper_artifacts/build_fig_synthesis.py` and `build_fig_transfer_overview.py`. Each writes
   `figures/<stem>.{pdf,svg,png}` (manuscript vector / editable vector / ≥300 dpi preview);
   `pytest tests/integration/test_paper_figure_values.py` then asserts every plotted mean and CI
   still equals its `generated/*.tex` cell.
7. **Freeze by tag, not marker**: the annotated tag below IS the freeze — it pins the exact commit
   whose `generated/*.tex` are the reported numbers. Optionally pause the autocommit publisher so
   the tagged fragments don't keep moving.

## Tag & publish

8. **Annotated tag**: `git tag -a v<X.Y.Z> -m "EvEMTBench <X.Y.Z> — frozen benchmark contract"`
   on the freeze commit; push with `git push origin v<X.Y.Z>`.
   **The public export is a separate history, so this tag does not reach it.** After pushing the
   export (step 9), tag the corresponding sync commit in `dist/evemtbench-benchmark-public` as well
   and push that tag — the GitHub Release, and therefore the Zenodo DOI, is minted from *that* tag:
   ```bash
   git -C dist/evemtbench-benchmark-public tag -a v<X.Y.Z> -m "EvEMTBench <X.Y.Z> — frozen benchmark contract"
   git -C dist/evemtbench-benchmark-public push origin v<X.Y.Z>
   ```
9. **Public GitHub export**: run `bash hpc/make_public_release.sh` — it stages a clean
   `git archive` export of HEAD, prunes the private/operator-only paths (manuscript, internal
   docs, agent config, ops daemons), verifies the public tree is self-consistent, **refuses to
   write anything if the leak scan finds forbidden strings** (home/cluster paths, account names,
   keys, tokens, private-forge URLs), and emits a ready-to-push single-commit git repo under
   `dist/`. Push it to `github.com/EvEMTBench/evemtbench-benchmark` (fresh history — the private
   development history is never pushed; **later releases must land as new commits on top of the
   public `main`, never a rewritten history**). Then create the **GitHub Release** with the
   **repro pack** attached: `bash hpc/make_repro_pack.sh` builds
   `dist/evemtbench_repro_pack_<version>.zip` + `SHA256SUMS` (reviewer-facing artifacts + live
   gate transcripts, leak-scan-gated the same way) — everything a reviewer needs without cluster
   access.

   **First-time GitHub/org setup (one-off, operator):**
   - Create the org `EvEMTBench` (free plan). The org name is frozen the moment it is printed
     anywhere — org renames do NOT redirect (repo renames/transfers do).
   - Create the repo `evemtbench-benchmark` **empty** (no auto-README/license, or the export
     push conflicts).
   - Add a **second org owner** (bus factor), require **2FA** for members *before* inviting
     anyone, set the org base member permission to **Read**.
   - After the first push: protect `main` (forbid force-pushes; require the CI check for PRs),
     confirm the first Actions run is green (the README badge goes live with it), and optionally
     enable **Pages** (main / root) so `leaderboard.html` and `audit/` are browsable URLs.
     NB GitHub disables the scheduled `nightly.yml` after 60 days of inactivity — re-enable is
     one click.
10. **Zenodo DOI**: log into Zenodo **with GitHub** and flip the integration toggle for the repo
    *before* creating the GitHub Release — the release then mints the versioned + concept DOI
    automatically. On the fresh record: add the DFG funding block (project 535389056, matching
    the dataset-generation record) and check the author list. Paste the concept DOI into
    `CITATION.cff` (`doi:`) and the README, update the manuscript's Code-availability DOI, and
    update the dataset cross-reference if its DOI landed too.
11. **Tell the audit**: the ledger's coverage/headline rows flip to `measured` automatically;
    walk `audit/index.html` once end-to-end (the professor test: any paper number → its
    `result.json` → its command → its gate → its run record, in ≤ 2 clicks).

## After

12. Reopen development: bump `pyproject`/`BENCHMARK_VERSION` to the next `-dev`, resume the
    autocommit publisher if you paused it for the tag, and start the next `CHANGELOG` section.
