# MTP Unified Implementation Plan, Provenance and Remaining Work

**Single canonical entry point:** [../readme.md](../readme.md)
**Machine-checkable result registry:** [../research_artifacts/unified/experiment_registry_v1.json](../research_artifacts/unified/experiment_registry_v1.json)

## The actual research chronology

1. **Pre-mid-semester 22-query feasibility:** original No-RL and RL (PPO)
   CDF-RAG on limited medical graph. Deterministic proxy groundedness
   0.6746975 vs 0.6747123; latency 39.576 vs 50.086 s. An alternative
   `gold`-named export is preserved separately and *not* independently
   physician-validated gold.
2. **100-query B1–B8 experiment:** selected end-to-end runs are all
   error-free in the November 16 matrix, even though older November 15
   B1–B6 attempts failed. Do not select the older failures to claim the
   current models still fail. B7 vs B8 evidence yields 49 and 65; mean
   latencies 28.95 and 48.53 s. Different retrieval implementations
   prevent attributing all change to PPO.
3. **132-query B1–B6 retrieval-only:** existing 100+32 drafts, 792
   model-query records, zero execution errors, sentence encoder.
   **The raw run folders are local only** until explicitly staged.
4. **G0/G1 expansion:** frozen G0 35/37, candidate G1 78/52 causal-format
   (37 inherited G0 links unverified; 15 new typed causes; eight risk
   factors and one association). Query-target drift caused B6 evidence
   coverage 41->35; stabilization recovered to 44 (B5 49->53->55).
5. **Changed-case review:** 31 selected model-query cases, 132 evidence
   occurrences, 58 unique assertions. Seventeen flagged cases cover
   12 unique questions. Not a general clinical accuracy estimate.
6. **132 question annotation v2:** AI source screening, 24 topical
   NIH/CDC/MedlinePlus sources, full task/answerability annotation,
   but zero independently medical-adjudicated answers/edges/paths.
   All 132 remain development data (used to design G1/resolver).

## What is authoritative versus archived

| Source | Current role |
|---|---|
| `presentation_20261118/legacy_results/rl_vs_no_rl_comparison.csv` | Original 22-query deterministic metrics |
| `evaluation_summary_*_gold.csv` | Separate exploratory legacy export; filename not proof of medical gold |
| `presentation_20261118/metrics/matrix_20261116T111224165031Z/pipeline_comparison.csv` | Selected end-to-end 100 B1-B8 matrix |
| `presentation_20261118/metrics/legacy_pair_20261116T115151459376Z_c68613/comparison.json` | Same-query B7/B8 operational difference |
| `presentation_20261118/runs/` | Archived individual 100-query attempts, including older failures |
| `research_artifacts/unified/experiment_registry_v1.json` | Current metadata index and explicit local-not-in-Git statuses |
| `data/benchmark/medical_causal_*_draft_v1.jsonl` | Frozen original 100 + 32 question content |
| `data/benchmark/v2/` | Annotated source-screened development benchmark, not gold |
| `docs/PRESENTATION_SUPERVISOR_20261118.md` | Historical dated slide snapshot (older B7/B8 "pending" now superseded) |
| `docs/ORIGINAL_CDF_RAG_README.md` | Upstream third-party paper results, NOT own experiments |

**Selection rule:** Use a result only with its dataset hash, run ID,
evidence source, execution mode, model version, metrics definition and
validation status. Different experiment conditions are different studies,
not contradictions that may be repaired by editing their numerical outputs.

## MTP phases — completion checklist

### Phase 1: Research foundation and reproducibility
- [x] Preserve old 22-query scripts, CSVs and PPO.
- [x] Standardized B1–B8 schema, runners, raw logs and G0 snapshot.
- [x] Explicit provenance registry and unified branch lineage.
- [ ] Freeze/import local final 132/G1 aggregate artifacts.

### Phase 2: Dataset and question/task definitions
- [x] Original 100+32 draft queries preserved, eight categories.
- [x] Candidate task, entity-role, answerability, relation-type annotations.
- [x] Source-screened 24-page catalog and CI structural checks.
- [ ] Two independent clinical reviewers and external adjudicator.
- [ ] Independently held-out benchmark; avoid source/family leakage.
- [ ] Claim-level gold evidence spans, causal edges/paths and approved answers.

### Phase 3: Causal graph and provenance
- [x] Frozen G0 snapshot and provenance.
- [x] Reviewed candidate G1 expansion with typed relationship ledger.
- [x] G0/G1 regression diagnosis and stabilized resolver.
- [ ] Independent verification of inherited G0 causal relationships.
- [ ] Independently reviewed G1 evidence and scalable graph deployment.

### Phase 4: Baseline models
- [x] Six standard retrieval baselines B1–B6.
- [x] Legacy B7/B8 adapters; all eight 100-query selected runs.
- [x] Six-model retrieval-only 132-query experiment (local).
- [ ] Fair common-evidence-corpus generator/LLM/hardware eight-model evaluation.
- [ ] Target and answer correctness scores against medically adjudicated references.

### Phase 5: Proposed adaptive CDF-RAG
- [x] Legacy PPO refinement; stable query resolver diagnostic prototype.
- [ ] Evidence sufficiency estimator with hard source/direction constraints.
- [ ] Multi-round evidence feedback / strategy switching.
- [ ] Calibrated answer/refine/abstain controller and early stopping.
- [ ] PPO training with *actual* retrieval feedback and cost-aware reward.
- [ ] Evidence-grounded generation and verified citations.

### Phase 6: Experimental matrix and optimization
- [x] 22-, 100-, 132-query scoped feasibility measurements.
- [x] Coverage-vs-latency and graph-expansion diagnostic comparisons.
- [ ] Ablations on rounds, PPO, gate, typed KG, caching and verification.
- [ ] P50/P95, token cost, memory, throughput, scaling and repeated-run CI.
- [ ] Matched B1-B8 plus proposed model, not pooled heterogeneous baselines.

### Phase 7: Evaluation and statistical validation
- [x] Selected changed-case annotation and technical integrity audit.
- [ ] Full independent evidence and medical answer review.
- [ ] Precision/Recall/Hit@k and hallucination accuracy on true gold.
- [ ] Statistical uncertainty, leakage audit and reproducibility freeze.

### Phase 8: Thesis and final presentation
- [x] Partial progress findings and supervisor-facing evidence.
- [ ] Integrate true clinically adjudicated results and adaptive-system ablations.
- [ ] Resolve validity limitations and author final thesis/viva.

## Importing verified local results safely

The following outputs were produced on Ubuntu but do not exist in the
GitHub tree. Do not recreate them from rounded aggregate values:

```
presentation_20261118/partial_submission_sprint/submission_20261118T124507167424Z_76df52/
presentation_20261118/submission_readiness/audit_20261118T125754420305Z_e4ed34/
presentation_20261118/g1_error_triage/triage_20261118T125808659756Z_35d31c/
presentation_20261118/submission_case_studies/case_studies_20261118T164251762850Z_14eb3d/
```

Use `python -m research_v2.unified_import_local --help` after pulling
the consolidated branch. It validates the original submission and audit
SHA before copying an allowlisted set of aggregate numerical results.
It does not overwrite input data or automatically push potentially
private/copyrighted model generations.

**Original files remain untouched.** If the user has local uncommitted code,
back it up, use `git status` and `git fetch`; do not force reset, clean or
discard work.

## Repository/branch reconciliation

The default branch was named `master`, not `main`.
The `research-v2` branch is a strict ancestor of
`g1-query-evidence-foundation`, which contributes the G1 and v2 dataset
changes. A single two-parent merge integrates those commits with the
original `master` history. Master-only `NEW_README.md` was superseded by
this guide; master-only `temp.py` has a documented legacy replacement,
and both remain recoverable from Git history. The archived upstream
paper README is preserved for attribution. No experimental CSV is edited
to force disagreement away.

See the canonical registry for exact SHA-locked source files and any
locally reported experimental results still awaiting Git archival.

## Import commands for a separate safe Git worktree

These commands should be run on your Ubuntu machine where the **original
locally audited experiments already exist**. Do not replace or reset the
original `~/Folder/CDF-RAG-2` directory.

```bash
cd ~/Folder/CDF-RAG-2
git status --short                      # inspect, do not discard local changes
git fetch origin master
git worktree add -b mtp-local-results ../CDF-RAG-2-unified origin/master

cd ../CDF-RAG-2-unified
source ../CDF-RAG-2/venv/bin/activate

python -m research_v2.unified_integrity --json
python -m research_v2.benchmark_v2_validation

python -m research_v2.unified_import_local \
  --source-repo ../CDF-RAG-2 \
  --submission-dir "presentation_20261118/partial_submission_sprint/submission_20261118T124507167424Z_76df52" \
  --audit-dir "presentation_20261118/submission_readiness/audit_20261118T125754420305Z_e4ed34" \
  --triage-dir "presentation_20261118/g1_error_triage/triage_20261118T125808659756Z_35d31c"
```

By default, the importer includes audited **aggregate** 22/100/132 and
G0/G1 tables, graphs and manifests, but excludes per-query question texts,
raw generated model answers and full reviewer case notes. The original
reviewed CSVs and G1 graph snapshots are not silently copied or approved.

For full 792 query-model retrieval records, first inspect them for
patient information, credentials and copyrighted passages, then rerun with
`--include-per-query`. For reviewer details, `--include-reviewed-examples`
and `--case-studies-dir` are separately opt-in.

After the importer succeeds:

```bash
git status --short
# Inspect the new research_artifacts/unified/imported_local/import_... directory
git add research_artifacts/unified/imported_local/
git diff --cached --stat
git diff --cached --check
git commit -m "data(mtp): archive reviewed aggregate results from audited local 132-query and G1 runs"
git push -u origin mtp-local-results
```

Open and review a PR from `mtp-local-results` to `master` before merging;
do not force-push or replace historical artifacts. This keeps unreviewed
medical outputs and third-party text out of the public repository unless
you explicitly inspect and approve their inclusion.

**Important:** A successful import makes the aggregate files available on
GitHub, but it does not certify medical accuracy or independently adjudicate
the 132-question benchmark. Full G1 graph artifacts and independent review
worksheets require a separate intentional, privacy-aware preservation step.

