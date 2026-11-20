# MTP fixed 132-query thesis: executable protocol and research constraints

**Canonical branch:** master after CI-approved fast-forward from \`mtp-thesis132\`.
**Dataset:** exactly 100 frozen pilot questions + 32 extension questions,
with source-screened v2 annotations. **No new questions will be added to the
thesis experiment cohort.** Original 22-query results remain a separate
historical feasibility study, never pooled into 132-question accuracy.

**Scientific boundary:** All 132 questions were used in prior G1 concept or
resolver development. Family-grouped prospective splits control *future*
fine-tuning and policy selection, but no split of the same 132 can honestly
be described as a previously unseen external test set. Current v2 AI source
screening has **zero independently medically approved gold answers**.
Software cannot generate its own independent physician adjudication.

## Research sources for the evaluation design

- [TRIPOD-LLM reporting guideline](https://www.nature.com/articles/s41591-024-03425-5):
  sources and leakage disclosures, reproducibility, clinically interpreted
  evaluation, model specifications and uncertainty.
- [scikit-learn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html):
  semantic-family separation to avoid evaluation leakage.
- [Stable Baselines3 RL evaluation](https://stable-baselines3.readthedocs.io/en/master/common/evaluation.html):
  separate and deterministic evaluation from policy training.
- NIH/NHLBI/NIDDK/NIMH/NINDS, CDC and MedlinePlus topical source
  reference catalog: \`data/benchmark/v2/source_catalog_screened_20261121.json\`.

## Command sequence: FIRST update from GitHub safely

The original Ubuntu folder \`~/Folder/CDF-RAG-2\` may contain uncommitted
research artifacts and G1 snapshots. Do not reset, clean or force-merge it.
Use a separate Git worktree for the completed MTP thesis pipeline.

\`\`\`bash
cd ~/Folder/CDF-RAG-2
git status --short                       # inspect your original local work
git fetch origin master
git worktree add -b thesis132-run ../CDF-RAG-2-thesis132 origin/master

cd ../CDF-RAG-2-thesis132
git pull --ff-only origin master
source ../CDF-RAG-2/venv/bin/activate

python -m research_v2.unified_integrity --json
python -m research_v2.benchmark_v2_validation
python -m unittest discover -s tests -p 'test_thesis132*_local.py'
\`\`\`

If the worktree or branch exists already, inspect \`git worktree list\` before
creating another. The checkout must contain the latest thesis132 source
modules. If tests fail, do not start full LLM batches; retain exact tracebacks.

For the legacy Neo4j/Ollama B7/B8 pipelines, privately copy the original
\`.env\` into the **new checkout only**, if appropriate; never commit it:

\`\`\`bash
cd ~/Folder/CDF-RAG-2-thesis132
if test -f ../CDF-RAG-2/.env && ! test -f .env; then
    cp ../CDF-RAG-2/.env .env
fi
\`\`\`

### Phase 0 — Freeze, inspect, and prepare the fixed benchmark

\`\`\`bash
python -m research_v2.thesis132 freeze
python -m research_v2.benchmark_v2_validation --prepare-review
python -m research_v2.benchmark_v2_export
\`\`\`

- Freezes the **exact** 132 query texts/IDs in a SHA-named, idempotent
  query-only file; no target/answer metadata reaches inference.
- Records five deterministic semantic-family folds (prospective
  within-benchmark cross-validation; historical exposure still disclosed).
- Produces **264 empty review slots** for two independent qualified reviewers.
  Reviewer fields and gold references remain *unset* until real review.
- The v2 answer rubrics are **candidates**, not independent gold.

### Phase 1 — Analyze the documented G1 target/evidence failures

The G1 candidate graph snapshot and 17-case triage file are **local-only**
and are deliberately not fabricated on GitHub.

In your **new worktree**, copy one previously frozen G1 candidate folder
from the original checkout after confirming its fingerprint matches the
G1 graph used in the November 17 paired/stabilized analysis.

\`\`\`bash
cd ~/Folder/CDF-RAG-2-thesis132
find ../CDF-RAG-2/research_artifacts -type f \
  -path '*/G1_CANDIDATE_*/manifest.json' -print

# Set this to the parent directory containing the actual manifest.json:
G1_SOURCE="/home/avneelsingh/Folder/CDF-RAG-2/PASTE_EXACT_G1_CANDIDATE_PATH"
test -f "$G1_SOURCE/manifest.json" || exit 1
sha256sum "$G1_SOURCE/manifest.json"

mkdir -p research_artifacts/local_g1_snapshots
cp -a "$G1_SOURCE" research_artifacts/local_g1_snapshots/
G1="research_artifacts/local_g1_snapshots/$(basename "$G1_SOURCE")"
test -f "$G1/typed_relationships.csv" || exit 1

# Copy your original 17-case triage into the new worktree without modifying
# the original; reviewer notes may contain sensitive text: do not git add.
TRIAGE_SRC="../CDF-RAG-2/presentation_20261118/g1_error_triage/triage_20261118T125808659756Z_35d31c"
test -f "$TRIAGE_SRC/priority_cases.csv" || exit 1
mkdir -p presentation_20261118/g1_error_triage
cp -a "$TRIAGE_SRC" presentation_20261118/g1_error_triage/
TRIAGE="presentation_20261118/g1_error_triage/$(basename "$TRIAGE_SRC")"

python -m research_v2.thesis132_g1 \
  --graph-dir "$G1" --triage-dir "$TRIAGE"
\`\`\`

This produces 264 model-query records (B5/B6 × 132) and a separately
joined 17-case review analysis. It does not use reviewer labels during
inference or assert a medically correct target. Report cases safely
withheld, source-reviewed candidates retained and previously returned
evidence lost; check regressions across **all** 132 questions. Merely
abstaining is not proof of an accurate medical answer.

**Exit gate:** no claim of fixing 8 wrong-target and 6 unsupported-gain
cases until the case IDs, graph SHA, reviewed evidence and actual outcomes
are inspected against the existing 31-case review.

### Phase 2 — Independent medical evidence and source adjudication

Open the worksheet generated by Phase 0. Two real qualified independent
reviewers should check source passages and annotate, per question:
correct entity roles, relation type and direction, supported answer,
abstain/clarify expectation, links in any multi-hop chain, provenance
and source spans. A separate adjudicator resolves disagreements.
Existing G1 "approved" source-review entries are not automatically
independently medically verified. Frozen G0 inherited links must be
checked before any expert-gold graph export.

\`\`\`bash
python -m research_v2.benchmark_v2_review_gate \
  --review-file PATH_TO_ACTUALLY_COMPLETED_DOUBLE_REVIEW_WORKSHEET.csv
\`\`\`

The gate will remain incomplete on the currently empty worksheet.
It checks structured declarations but cannot authenticate reviewers.
Supervisor verification is required. No automatically filled gold labels.

**Exit gate:** a supervisor-confirmed versioned source/edge/answer
adjudication with verified/uncertain/rejected statuses. No clinical
ground truth is claimed until this exists.

### Phase 3 — Independent reference quality within the 132 cohort

The user has chosen to evaluate **only these 132 questions**. Therefore
create gold references *for the same 132* by independent, blinded
reviewers. Freeze reviewers' evidence spans, reference answer, answerability
and causal paths **before** final model selection. Keep family groups
together for prospective five-fold model-development experiments.
Report the original G1 exposure and do **not** call these an external
unseen holdout. Do not add more questions or relabel development data as
unseen gold.

**Exit gate:** 132 separately adjudicated references. Independent
references and independent *unseen questions* are different properties.

### Phase 4 — Evidence-aware retrieval prototype on 132

G1 typed links are review-gated candidates; BM25 documents are
AI-assisted candidate summaries. Prototype B9 combines these without
allowing document-only retrieval to certify a causal answer. It records
evidence presence, abstention and whether the typed relation requirement
was met. It does **not** generate a medical answer yet.

\`\`\`bash
python -m research_v2.thesis132_adaptive \
  --graph-dir "$G1" --policy fixed1
\`\`\`

**Exit gate:** compare typed candidate yield, exact retrieval paths,
provenance, abstentions and per-query cost to B5/B6, with clinician
adjudication before claiming increased clinical relevance.

### Phase 5 — Rule-based adaptive stopping and ablation

\`\`\`bash
for policy in fixed1 fixed2 fixed3 adaptive; do
  python -m research_v2.thesis132_adaptive \
    --graph-dir "$G1" --policy "$policy"
done
\`\`\`

Use the **actual paths** printed by these runs, all with the same graph,
corpus and input hashes, for the paired ablation:

\`\`\`bash
python -m research_v2.thesis132_adaptive_compare \
  --run-dirs PATH_FIXED1 PATH_FIXED2 PATH_FIXED3 PATH_ADAPTIVE
\`\`\`

This is a **0–2 additional retrieval-round** prototype, not a
language-model query rewriting/feedback agent. It tests selective
retrieval and early stopping on fixed 132 questions. It does not
establish answer groundedness or medical accuracy.

**Exit gate:** measured candidate-evidence and latency differences
versus fixed retrieval budgets, plus review of possible regressions.

### Phase 6 — Improved PPO: NOT YET TRAINED

The original saved PPO checkpoint was trained with a small, simulated
retrieval environment and a reward without an explicit latency penalty.
It is preserved as historical **B8**, not promoted to a finished new PPO.

A new PPO agent must be trained on **real retrieval feedback** and
evaluated with prospective semantic-family folds and clinically adjudicated
references. The state needs current evidence sufficiency, uncertainty,
remaining budget and retrieval history; the actions must include answer,
retrieve/refine, and abstain. The cost-aware reward should trade validated
quality against latency and unsupported evidence. Compare against:
no refinement, unconditional refinement, fixed retrieval budgets and
rule-based adaptive stopping on the *same* source/corpus/LLM.

**This stage is blocked pending genuine gold references and a real,
implemented PPO training environment. No command is supplied that
would falsely suggest the new PPO model is already trained.**

### Phase 7 — B1–B8 full 132-query runs + B9 prototype analysis

1. Quick **retrieval-only smoke** for all supported B1–B6:
\`\`\`bash
python -m research_v2.thesis132 run \
  --models B1,B2,B3,B4,B5,B6 --mode retrieval_only \
  --encoder sentence --limit 3
\`\`\`

2. Full 132-query **retrieval-only** baseline matrix:
\`\`\`bash
python -m research_v2.thesis132 run \
  --models B1,B2,B3,B4,B5,B6 --mode retrieval_only \
  --encoder sentence
\`\`\`

3. Configure the same Ollama generator for B1–B6.
Confirm local Neo4j and Ollama services, G0 lockfile and PPO checkpoint
before starting B7/B8. Use the existing \`.env\` credentials privately:

\`\`\`bash
python -m research_v2.ollama_generation check
python -m research_v2.thesis132 run \
  --models all --mode end_to_end --encoder sentence \
  --allow-live --limit 2
\`\`\`

4. Full **all-eight end-to-end** 132-query evaluation, sequentially:
\`\`\`bash
python -m research_v2.thesis132 run \
  --models B1,B2,B3,B4,B5,B6,B7,B8 \
  --mode end_to_end --encoder sentence --allow-live
\`\`\`

The runner preserves the original B7/B8 live G0 Neo4j and PPO model,
enforces existing graph-drift preflights and writes new versioned runs.
It does **not** silently convert Neo4j errors to abstentions.

If execution is interrupted *between models*, resume the same frozen
session using the \`--session-dir\` path printed by the runner, the same
model selection, mode and encoder. If a model recorded errors, examine
its output and launch a **new** session instead of overwriting results.

5. Use the **full** completed session directory printed by the command;
do not accidentally analyze the \`--limit 2\` smoke run:

\`\`\`bash
SESSION="presentation_20261118/thesis132/runs/PASTE_FULL_132_BATCH_DIR"
python -m research_v2.thesis132 summarize --session-dir "$SESSION"
python -m research_v2.thesis132_analysis --session-dir "$SESSION"
\`\`\`

The analyst exports per-category and per-fold counts, observed evidence
yield, answers present, direct causes, chain depth, detected hallucination
*flags*, validator *flags*, P50/P95 latencies, QPS and B7/B8 paired
operational differences. It also generates semantic-group bootstrap
uncertainty; these are descriptive intervals, not external-generalization
accuracy confidence intervals.

**Don't equate** legacy detector flags with an independently evaluated
hallucination rate. Don't equate structural G0 edge F1 with medical accuracy.
No 132-query groundedness or medical F1 can be computed until actual gold
references and answer-claim judges have been verified.

### Phase 8 — Final thesis and scientific claims

After the runs and source-adjudicated answer scoring are finished:
- Audit the frozen code, dataset and graph SHA, hardware/encoder/LLM
  settings, precise run IDs and used source passages.
- Compare B1–B8 plus B9 and (if implemented) the improved PPO under
  matched generators, evidence source/budgets and hardware. Report
  differences in evidence sources as an explicit confound where they remain.
- Measure Precision@k, Recall@k, Hit@k, answer correctness and supported
  causal direction **only where gold references exist**.
- Run ablations for relation typing, provenance, sufficiency gate,
  early stop, retrieval rounds, caching and PPO once all are implemented.
- Report overall and category-level uncertainty, failures and limitations.
- Write thesis narrative anchored to verified outputs, seek supervisor
  feedback, and finalize only after new experiments actually complete.

## Clear status now

| Task | Status |
|---|---|
| Preserve original 22/100/132 histories | Completed |
| Freeze exact 132 and prospective family folds | Implemented, GitHub CI tested |
| New B1–B8 132 runner | Implemented, B1 1-query CI smoke passed; FULL runs need Ubuntu |
| G1 17-case comparative diagnostic | Implemented; needs local G1/triage files |
| Independent medical adjudication | **Not completed** |
| Source-screened candidate v2 | Completed (not medical gold) |
| B9 deterministic typed/BM25 multi-stage prototype | Implemented/tested; full run needs local G1 |
| Fixed vs adaptive B9 ablations | Implemented; results not yet run on G1 |
| New PPO trained on true retrieval feedback | **Not completed** |
| Matched 132-question E2E B1–B8 | **Not yet run** |
| Statistical analysis code | Implemented; needs completed local run |
| Final thesis/research acceptance | **Pending evidence and supervisor review** |
