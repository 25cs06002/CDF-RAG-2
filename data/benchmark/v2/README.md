# CDF-RAG-2: 132-question source-screened benchmark v2

**Status: AI-assisted, web-source-screened development annotations.
NOT independently clinically adjudicated gold.**
Screened: 2026-11-21. This package preserves original v1 100+32 question
text and target labels and does not alter any historical 22-, 100-,
or 132-question experiment results.

## What is actually completed

- [x] Original 100+32 questions preserved verbatim and IDs unchanged.
- [x] 132 structured `task_objective` and `expected_response_type` labels.
- [x] Explicit `expected_answerability`, including 24 ambiguity/
      unsupported individual-inference queries.
- [x] 132 `focus_entities` and documented `target_role`, with fixes
      proposed in **separate fields**, not silently changing G0 targets.
- [x] 132 candidate response rubrics (AI draft; not clinical gold).
- [x] 132 required relation-semantics descriptions.
- [x] 80 semantic groups to help avoid paraphrase/split leakage.
- [x] 24 identified NIH/NHLBI/NIDDK/NIMH/NINDS/CDC/MedlinePlus
      web pages in a screened bibliographic catalog.
- [x] 122 questions with topical source references; 10 deliberately
      source-free insufficient-data negatives.
- [x] Automatic schema checks, frozen-input/Git-blob checks,
      source-reference integrity, label logic, anti-false-gold checks.
- [x] Two independent-review slots per question (264 unfilled slots)
      and a strict review-completeness gate.
- [x] Export adapter for 132 query-only development inputs, with no
      answer or target metadata exposed at inference.

## Not completed, and why it must remain unclaimed

- [ ] Independent clinician review of all 132 candidate rubrics.
- [ ] Specific source passage/evidence-span verification for each claim.
- [ ] Independently adjudicated gold answers, directed edges and paths.
- [ ] Full assessment of medical correctness and unsafe inference.
- [ ] Separate held-out questions that have not been used to create G1.
- [ ] External authentication of reviewer qualifications/independence
      and supervisor sign-off.

The model that created these annotations **cannot serve as its own
independent medical reviewer**. Bibliographic page-level relevance is not
gold claim evidence. No gold labels are manufactured.

## Main data files

- `medical_causal_132_source_screened_v2.jsonl`: 132 complete
  *operational* annotations with source IDs and AI candidate rubrics.
- `source_catalog_screened_20261121.json`: 24 vetted-domain topical
  source pages, claim-scope notes, screening levels and warnings.
- `manifest.json`: frozen input blob SHAs, category counts, status
  and explicit no-gold warnings.

### Important fields

- `query_id`, `query`, `query_type`, `medical_domain`: copied
  unchanged from v1.
- `original_target_concept`: immutable original draft target, not
  an expert-approved answer.
- `focus_entities`, `target_role`: revised semantic representation.
- `task_objective`, `required_relation_semantics`: distinguishes
  concept identification, direct evidence, multi-hop, comparative,
  risk-factor and insufficiency tasks.
- `expected_answerability`, `expected_response_type`: tentative
  task-policy expectation, **not clinically adjudicated correctness**.
- `response_rubric_candidate`: editable AI draft, never gold.
- `source_reference_ids`: page-level bibliographic leads.
  A source URL is *not* a checked passage for every statement.
- `semantic_group`: cluster for paraphrase/development-leakage control.
- `split=development_not_independent_test`: deliberately no test claim.
- `gold_answer`, `gold_entities`, `gold_causal_edges`,
  `gold_paths`, `gold_evidence_spans`: null until independent review.
- `annotation_review_status`: pending external clinician review.

## Important question annotations

- `PILOT_050`: original target was type 2 diabetes even though the
  question asks about vascular/kidney complications from hyperglycemia;
  v2 retains the original target for provenance and introduces a
  downstream endpoint role.
- `PILOT_058`: stress, sleep and depression are explicitly discussed
  without assuming a directed causal chain.
- `PILOT_061`, `PILOT_065`, `EXTQ_017`: comparisons include multiple
  named entities or subtypes rather than one inherited target.
- `EXTQ_021`: type 2 diabetes modifiable versus nonmodifiable risk,
  with an explicit risk-vs-direct-causation distinction.
- `PILOT_085`–`PILOT_100`, `EXTQ_025`–`EXTQ_032`:
  questions about exact individual diagnoses/predictions require
  uncertainty or abstention rather than forced retrieval-based answers.

## Local validation and optional review handoff

From repository root, with the existing virtual environment active:

```bash
python -m py_compile research_v2/benchmark_v2_validation.py \
  research_v2/benchmark_v2_review_gate.py \
  research_v2/benchmark_v2_export.py

python -m unittest discover -s tests \
  -p 'test_benchmark_v2*_local.py'

python -m research_v2.benchmark_v2_validation

# Generates an empty 264-slot CSV for TWO real independent reviewers
# per question; it does not fill their judgments.
python -m research_v2.benchmark_v2_validation --prepare-review

# After independent reviewers fill both slots with source passage
# locations, assessment and actual qualifications:
python -m research_v2.benchmark_v2_review_gate \
  --review-file PATH_TO_FILLED_WORKSHEET.csv

# Separate safe development-only query inputs for future experiments;
# never use them as held-out data:
python -m research_v2.benchmark_v2_export
```

The review gate verifies declared fields and counts but cannot authenticate
the existence, credentials or actual judgments of clinicians. An authorized
supervisor must verify real reviewers and adjudicate disagreements before
approving any `gold_v3` release.

## Recommended gold-standard release procedure (not yet executed)

1. Two real, independent qualified reviewers evaluate every question.
2. For each cited answer and causal edge, reviewers record exact source
   location/quotation scope and the relation direction and confidence.
3. Disagreements go to an independent adjudicator; unresolved claims
   remain uncertain/excluded.
4. Supervisor verifies credentials, independence, decisions and source
   licensing. Sign the immutable new version, not this v2 candidate.
5. Prepare a separate blind held-out benchmark from questions that did
   not influence G1, resolver tuning or PPO training.
6. Only after these conditions are met, publish expert-reviewed gold
   answers and compute medical groundedness/accuracy statistics.
7. Freeze dataset hash, reviewer approvals and experiment configuration.

## Evidence and validity warnings

- G0's inherited causal-format edges are not independent medical gold.
- 132-question development prompts influenced the G1 graph and resolver.
- The 132 questions contain templates and near paraphrases; source/document
  grouping and semantic-family splitting are required for unbiased tests.
- Query-type and difficulty are confounded in the initial draft.
- B1–B4 and B5–B8 did not share identical evidence corpora; cross-family
  evidence availability is not direct causal-accuracy evidence.
- Source pages may be updated; recorded citation URL and date must be
  retained, and claims should be independently checked from specific
  passages (not merely the retrieved snippet).
- An independent clinical gold release is a **different artifact** from
  this source-screened development dataset.
