# EvEMTBench — build orchestration for the manuscript + Supplementary Information artifacts.
#
# Artifact rule (see paper/manuscript/generated/README.md):
#
#     one scientific table/figure  ->  exactly one Python generator  ->  exactly one .tex/.pdf
#
# This Makefile is the ONLY orchestrator: it invokes the individual per-artifact generators under
# hpc/paper_artifacts/. No multi-output Python script writes scientific artifacts any more.
#
#   make supp-artifacts         regenerate every Supplementary fragment
#   make check-supp-artifacts   verify every committed fragment is fresh (writes nothing)
#   make supp-figures           rebuild the SI data figures (needs matplotlib)
#   make supplementary          regenerate + compile supplementary.pdf
#   make check-supplement       run the scientific consistency checker
#
# Three SI fragments read the released dataset rather than only outputs/ (EC support, ED negative
# composition, current-scale diagnostic). They are grouped under the *-dataset targets and, on the
# cluster, MUST run inside a SLURM allocation — never on the shared login node:
#
#   srun --gres=gpu:a100small:1 -p a100mig --cpus-per-task=8 --time=1:00:00 -A <acct> \
#       make supp-artifacts-dataset

PY ?= python
#: A python carrying matplotlib — the SI figure builders only. Set in hpc/clusters/local.env.
PAPERFIGS_PY ?= $(PY)
PYTHONPATH ?= src
export PYTHONPATH

ARTIFACTS := hpc/paper_artifacts
GEN       := paper/manuscript/generated
MANUSCRIPT:= paper/manuscript

# --- artifact groups -------------------------------------------------------------------------
# Dataset-backed SI fragments (need $EVEMT_ROOT; run under SLURM). Declared first so the
# repository-only group can filter them out.
DATASET_BUILDERS := $(ARTIFACTS)/build_supp_tab_ec_support.py \
                    $(ARTIFACTS)/build_supp_tab_ed_negative_support.py \
                    $(ARTIFACTS)/build_supp_tab_current_shift.py
# Held-out headline tables (also \input by the SI).
RES_BUILDERS   := $(wildcard $(ARTIFACTS)/build_tab_res_*.py)
# Full per-baseline supplement tables + the repository-only SI evidence tables.
SUPP_BUILDERS  := $(filter-out $(DATASET_BUILDERS),$(wildcard $(ARTIFACTS)/build_supp_tab_*.py))
# Cross-grid transfer tables.
XFER_BUILDERS  := $(wildcard $(ARTIFACTS)/build_tab_transfer_*.py)
# Static specification tables (authored content, no data inputs).
SPEC_BUILDERS  := $(ARTIFACTS)/build_tab_task_spec.py $(ARTIFACTS)/build_tab_task_suite.py \
                  $(ARTIFACTS)/build_tab_observability.py $(ARTIFACTS)/build_tab_baselines.py \
                  $(ARTIFACTS)/build_tab_dataset_families.py \
                  $(ARTIFACTS)/build_tab_topology_overview.py

# SI data figures. These need matplotlib, which the sweep's PY does not carry, so they run under
# PAPERFIGS_PY and are NOT part of the *-artifacts targets or the commit-time freshness gate (that
# gate byte-compares .tex only). Rebuild them whenever outputs/ changes.
FIG_BUILDERS  := $(wildcard $(ARTIFACTS)/build_fig_supp_*.py)

REPO_BUILDERS := $(RES_BUILDERS) $(SUPP_BUILDERS) $(XFER_BUILDERS) $(SPEC_BUILDERS)
ALL_BUILDERS  := $(REPO_BUILDERS) $(DATASET_BUILDERS)

.PHONY: help supp-artifacts supp-artifacts-repo supp-artifacts-dataset paper-artifacts \
        supp-figures check-supp-artifacts check-supp-artifacts-dataset check-supplement \
        norm-control check-norm-control supplementary manuscript list-artifacts

help:
	@sed -n '1,24p' Makefile

## Regenerate every SI fragment that needs only committed repository artefacts.
supp-artifacts-repo:
	@for b in $(REPO_BUILDERS); do echo "-> $$b"; $(PY) $$b || exit 1; done

## Regenerate the three dataset-backed SI fragments (needs the mounted dataset; run under SLURM).
supp-artifacts-dataset:
	@for b in $(DATASET_BUILDERS); do echo "-> $$b"; $(PY) $$b || exit 1; done

supp-artifacts: supp-artifacts-repo supp-artifacts-dataset

## Alias: the manuscript fragments are the same artifact set.
paper-artifacts: supp-artifacts

## Verify the committed fragments match a fresh rebuild. Writes nothing; non-zero on staleness.
check-supp-artifacts:
	@for b in $(REPO_BUILDERS); do $(PY) $$b --check || exit 1; done

check-supp-artifacts-dataset:
	@for b in $(DATASET_BUILDERS); do $(PY) $$b --check || exit 1; done

## Regenerate the input-normalization control SI table. ANALYSIS-ONLY: needs the isolated control
## tree (held_out re-run with --normalize per_window), which is NOT in outputs/ and NOT in this
## repository, so this fragment is deliberately outside the commit-time freshness gate -- same
## arrangement as the dataset-backed fragments above. Pass the tree with ABL=<dir containing outputs/>.
norm-control:
	@test -n "$(ABL)" || { echo "usage: make norm-control ABL=<control tree>"; exit 2; }
	$(PY) hpc/paper_artifacts/build_supp_tab_norm_control.py --abl "$(ABL)"

check-norm-control:
	@test -n "$(ABL)" || { echo "usage: make check-norm-control ABL=<control tree>"; exit 2; }
	@$(PY) hpc/paper_artifacts/build_supp_tab_norm_control.py --abl "$(ABL)" --check

## Rebuild the SI data figures (needs matplotlib: PAPERFIGS_PY, and a SLURM allocation).
supp-figures:
	@for b in $(FIG_BUILDERS); do echo "-> $$b"; $(PAPERFIGS_PY) $$b || exit 1; done

## Scientific consistency checker (coverage, selection, leakage, main-text claims).
check-supplement:
	@$(PY) hpc/check_supplement_consistency.py

## Compile the Supplementary Information (twice, for cross-references).
supplementary:
	@$(MAKE) check-supp-artifacts
	@$(PY) hpc/build_supplement.py

manuscript:
	@$(PY) hpc/build_paper.py

list-artifacts:
	@for b in $(ALL_BUILDERS); do echo "$$b"; done
