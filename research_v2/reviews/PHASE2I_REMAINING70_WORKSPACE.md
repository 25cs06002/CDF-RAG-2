# Phase 2I-A — Review workspace for remaining 70 medical QA pilot questions

## Goal and provenance

Phase 2H's reviewed 30-question snapshot is frozen and **untouched**. This
module creates a **separate**, editable 70-question medical evidence review
workspace for pilot IDs PILOT_031 through PILOT_100. All source links are
*reading candidates*, not verified support.

Input:
- medical_causal_pilot100_draft_v1.jsonl (100 draft questions)
- source_candidates15_v2.jsonl (15 candidate topic pages)
- additional_source_candidates8_v1.jsonl (8 additional candidate pages)
- remaining70_source_map_v1.jsonl (70 unreviewed source mappings)

Expected initial counts:
- 70 question review tasks
- 89 question–source review tasks
- 30 blank link review tasks (two slots each for 15 multi-hop questions)
- 189 total pending review decisions
- 18 questions flagged for additional-source work
- ZERO independently verified gold labels or causal paths

The 30 link rows are **blank placeholders for reviewers to supply evidence**,
not generated biological relations. Some questions require more than two
links; this version supports *at most two* and does not establish path
completeness.

## Create review workspace

Run on Ubuntu from the repository root:

    cd ~/Folder/CDF-RAG-2
    source venv/bin/activate
    git switch research-v2
    git pull --ff-only origin research-v2
    python3 -m unittest discover -s tests -p 'test_remaining_review_workspace.py' -v

    OUT="$HOME/Folder/CDF-RAG-2-experiment-artifacts/phase2/human_review70_$(date -u +%Y%m%dT%H%M%SZ)"
    python3 -m research_v2.remaining_review_workspace --output-dir "$OUT"
    echo "REVIEW70_DIR=$OUT"
    ls -lh "$OUT"

Expected output directory contains:
- question_review.csv
- source_review.csv
- link_review.csv
- review_manifest.json

The module *refuses to overwrite* an existing workspace. Do not mix the new
workspace with human_review_20261113T135114Z.

## How to review source evidence (not AI confirmation)

1. In source_review.csv, each task links one candidate document to one query.
   Open the actual URL; inspect the correct version and passage; record whether
   the source supports the requested claim, partially supports it, does not
   support it, or needs more evidence. Use exact passage locators and the
   supported_claim_summary field. A related disease topic is not necessarily
   evidence for the question. Enter your actual reviewer ID/date and rationale.
2. In question_review.csv, decide if the question is answerable from *reviewed
   registered evidence*, needs more evidence, or should explicitly abstain.
   For answerable questions, include a source-grounded answer draft and only
   IDs of source rows reviewed as supports. Do not add arbitrary new source IDs:
   first add such sources in a **versioned source registry/mapping**, create a
   new workspace, and preserve older reviews.
3. For multi-hop PILOT_046–PILOT_060, inspect link_review.csv. Both slots
   begin with no assumed nodes. Enter source-backed cause/effect/relation,
   source IDs and passage evidence independently for each arrow. A two-link
   chain has matching intermediate nodes but **does not prove medical
   causation merely by passing the validator**. If only one link is supported,
   mark the question's candidate_path_status accordingly.
4. For comparative PILOT_061–PILOT_072, record both comparison topics and
   source IDs for each side. Some sources might cover both comparison sides;
   check each claim, not only document titles.
5. Risk-factor questions should explain why risk associations do not assert
   individual deterministic causation. Ambiguous individual symptoms and
   exact-prediction questions should abstain where an evidence-grounded
   diagnosis cannot be established. Unsupported exact-person predictions are
   not permitted to be labeled answerable.
6. The initial discovery map flags 18 questions needing additional sources,
   including every multi-hop question. If you mark one answerable based only
   on existing registered evidence, a scope_resolution_note is mandatory
   and an audit warning remains. This is a flag for later adjudication, NOT
   a claim that the gap has truly been resolved.

## Audit

    python3 -m research_v2.remaining_review_workspace --audit-dir "$OUT"

At the start this reports 189 pending decisions and zero gold.

Do **not** use --strict until every question/source/link task is reviewed.

    python3 -m research_v2.remaining_review_workspace --audit-dir "$OUT" --strict

The strict audit validates recorded review fields, fixed source provenance,
relation formatting, question-source consistency, and structural consistency
for any two-link path declared source-supported. It **does not** verify:
- authenticity of the recorded reviewer
- actual medical correctness or causal direction
- source scientific quality
- completeness of two-link representations for complex causal pathways
- clinical expert adjudication
- benchmark gold labels or final train/validation/test splits

## Review order

Use small topic-based batches instead of approving 189 tasks automatically.
First process indirect concept questions PILOT_031–PILOT_036 and their source
rows; then progress to the remaining indirect questions, multi-hop evidence,
comparisons, risk-factor, ambiguous, and unsupported categories. Preserve
every unresolved claim separately for later source discovery/adjudication.

The G0 graph, old 30-question human reviews, Phase 2H exports, performance
baselines, PPO and the eventual G1 medical graph remain unchanged.
