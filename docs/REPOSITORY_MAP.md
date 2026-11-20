> **ARCHIVED HISTORICAL SNAPSHOT — NOT CURRENT STATUS.** This document
> records decisions and experiment availability at the time it was written.
> Newer B1–B8 100-query end-to-end runs, 132-query retrieval experiments,
> G1 stability analysis and benchmark v2 are indexed in
> [the canonical MTP results registry](../research_artifacts/unified/experiment_registry_v1.json)
> and [unified README](../readme.md). Older “pending” labels must not be
> interpreted as current results. No historical experimental number was altered.

# CDF-RAG-2 — Research repository map and safe cleanup

**Canonical working directory:** `~/Folder/CDF-RAG-2`  
**Active GitHub branch:** `research-v2`  
**Deadline:** November 18, 2026 M.Tech progress presentation.

## Five places to look first

1. `readme.md` — primary project README, commands and experiment status.
2. `evaluate_all_baselines.py` — single CLI for presentation preparation, validation, smoke tests, B1–B8 runs and reports.
3. `research_v2/pipelines/` — central B1–B8 registry, factory and per-model descriptions.
4. `presentation_20261118/` — generated local outputs (benchmark manifest, candidate corpus, configurations, one folder per run, metrics and figures); created by `prepare`.
5. `docs/PRESENTATION_SUPERVISOR_20261118.md` — slides outline, results, speaking notes, honest caveats and decisions requested.

## Existing code grouped by purpose

| Area | Canonical location | Notes |
| --- | --- | --- |
| Six new B1–B6 retrieval methods | `research_v2/presentation_baselines.py` | Graph snapshot or provisional source-summary corpus |
| Two original B7/B8 full pipelines | `agents/document_causal_retriever_no_rl.py`, `agents/document_causal_retriever_rl.py` | Keep untouched for reproducibility |
| Shared wrappers/adapters | `research_v2/legacy_no_rl.py`, `research_v2/legacy_rl.py` | Uniform output format |
| Common schema | `research_v2/schema.py`, `research_v2/base_pipeline.py` | Common results and timing fields |
| Unified 100-query orchestration | `research_v2/presentation_matrix.py` | One result JSONL per model per run |
| Graph provenance / G0 validation | `research_v2/graph_identity.py`, `research_v2/preflight.py` | Local G0 fingerprint check |
| Draft 100-query pilot | `data/benchmark/medical_causal_pilot100_draft_v1.jsonl` | No independently reviewed gold |
| Source-review scripts | `research_v2/*review*.py` | Preserve Phase 2 evidence workflow |
| Original G0 graph | `research_artifacts/graph-G0/` | 35 nodes and 37 edges; immutable |
| Past Phase 2 outputs | `research_artifacts/experiment-artifacts/phase2/` | Versioned, not medically adjudicated |
| Historic RL/No-RL evaluations | Root CSVs and `evaluate_*_rl.py`, `evaluate_*_no_rl.py` | Retain to support old results |
| Previous paper metadata | `docs/ORIGINAL_CDF_RAG_README.md` | Original research authorship and baseline context |

## Changes made for organization

- Added a genuine project-oriented `readme.md` (the upstream paper README was preserved under `docs/`).
- Added the `research_v2/pipelines` B1–B8 registry and updated the unified matrix runner to use it.
- Created dedicated presentation and historical-results folders, plus an archived six-model measured CSV.
- Moved the temporary exploratory `temp.py` to `research_artifacts/legacy22/scripts/temp_hallucination_report.py`; original repository history still exists.
- Deleted the obsolete root `NEW_README.md`, whose commands were superseded by the maintained README.
- Kept raw historical experiment CSVs in place to avoid breaking old scripts referencing hard-coded relative paths. **No scientific result was removed.**

## Do NOT delete without an explicit dependency/provenance audit

- Raw historical CSVs, PPO checkpoint, model configs, original code, G0, source-review tasks, graph exports, training scripts, notebooks.
- Files changed locally but not committed: `research_v2/remaining_review_workspace.py`, `reconcile_remaining70_audit.py`, `repair_editable94_review.py`. The GitHub integration cannot see those changes; **do not reset or clean**.
- Medical candidate or reviewer evidence. A candidate is not an expert-reviewed reference label.

## Publication requirements

Before pushing `presentation_20261118/` model output, inspect for private patient details, credentials, third-party copied full text and other sensitive data. Review generated answer text before publishing; aggregate metrics and manifest hashes are generally the safer default.

## Next architectural improvements after presentation

Move the document corpus to independently reviewed, licensed texts. Build G1 with provenance, relation categories and causal direction adjudication. Complete true multi-round feedback and stopping decisions. Separate the effect of RL from dynamic feedback. Re-run all architectures with common generator, matched evidence budget, controlled model backend, fixed question set and independently annotated metrics.
