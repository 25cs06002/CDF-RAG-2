# Phase 2: independently reviewable medical causal QA benchmark

This is the start of **Phase 2** in `MTP Plan.pdf`, not a completed
medical dataset. The initial target is a **100-question pilot**, followed by a
**650-question benchmark** after source review, annotation, and split checks.

## Study design (targets, not existing results)

| Primary query category | Planned final questions |
| --- | ---: |
| Direct factual | 100 |
| Paraphrased | 100 |
| Indirect concept | 100 |
| Multi-hop causal | 100 |
| Comparative | 75 |
| Risk-factor | 75 |
| Ambiguous | 50 |
| Unsupported / abstention | 50 |
| **Total** | **650** |

Final proposed partitions: **train 400**, **validation 100**, **test 150**.
Keep question families, concept families, and source-document families
together in one partition. The 100-question pilot is for refining annotation
quality and procedures; do not treat it as a held-out final test set.

## Annotation workflow

1. Write questions as `draft`. A draft has **no established gold truth**.
2. Find independently sourced evidence. Record each source's title, URL and
   type; distinguish `CAUSES`, `RISK_FACTOR_FOR`, `ASSOCIATED_WITH`,
   `PROTECTS_AGAINST` and `UNCERTAIN`. Do not relabel correlation as causality.
3. Attach source IDs to every annotated causal edge and path. Record
   answerability within the **defined evidence corpus**, rather than making
   blanket claims of medical certainty.
4. Have an identified reviewer inspect source support and causal direction.
   Use `source_reviewed` or `expert_reviewed` only when that actual review
   happened. These statuses are human attestations, **not** verified by code.
5. Assign `question_family` for paraphrases of one underlying question,
   `concept_family` for related topics, and `source_family` for the same
   source-document family. These labels are for leakage prevention, not models.
6. Assign train/validation/test partitions only after review. Run strict
   validation to reject drafts, missing partitions, duplicates and families
   spanning partitions. Split creation is a subsequent task.
7. Keep reference annotations separate from question-only inference exports.
   Do **not** give held-out gold answers, causal edges, or reviewer notes to
   PPO training, query refinement or the generation prompt.

## Data fields

Each JSONL record has `query_id`, `query`, `query_type`, `difficulty`,
`target_concept`, `answerable`, `gold_answer`, `gold_entities`,
`gold_causal_edges`, `gold_paths`, `evidence_sources`,
`abstention_reason`, `review_status`, `reviewer_id`, `split`,
`question_family`, `concept_family`, and `source_family`.

A relationship has `cause`, `effect`, `relation`, and
`evidence_source_ids`. A path has `nodes` and `evidence_source_ids`.
This schema deliberately does **not** preassign a `gold_hallucination`
boolean to a question: hallucination is judged for a **generated claim**,
independently of that model's own detector, in later evaluation.

The reviewer must check real sources and their licensing. This validator
checks metadata consistency, **not** whether medical claims are true or
whether a claimed reviewer actually conducted a review. Existing
G0 graph-derived regression labels must not be promoted to medical gold.

## CLI

```bash
python3 -m research_v2.medical_benchmark \
  --input research_v2/examples/medical_benchmark_draft_v1.jsonl

# For reviewed annotations that have completed split assignment:
python3 -m research_v2.medical_benchmark \
  --input path/to/reviewed_benchmark.jsonl --strict \
  --report path/to/audit_report.json
```

This phase does not yet add documents, train models, modify Neo4j,
create automatic partitions, or evaluate medical accuracy.

PubMedQA, MedQA or MedMCQA may be incorporated as **separate external QA
tracks** after inspecting license and task formats. Their existing answer
labels must not be treated as ready-made causal-edge/path gold.


## Phase 2A: 100-question unreviewed pilot

The repository now includes `data/benchmark/medical_causal_pilot100_draft_v1.jsonl`.
These questions are **AI-authored drafts for later human checking**, not a
validated medical benchmark and not gold labels. Do not report their contents
as established medical mechanisms. Every row has an explicit medical domain,
a target topic, question/concept family IDs, `review_status="draft"`,
`answerable=null`, and **no final split or evidence**.

The pilot distribution is deliberately stratified:

| Type | Draft questions |
| --- | ---: |
| Direct | 15 |
| Paraphrased | 15 |
| Indirect | 15 |
| Multi-hop | 15 |
| Comparative | 12 |
| Risk-factor | 12 |
| Ambiguous | 8 |
| Unsupported | 8 |
| **Total** | **100** |

The 15 direct prompts have corresponding paraphrased prompts in the same
`question_family`. Never put members of the same family in different
train/validation/test partitions. A second script validates the exact
distribution and exports only `query_id` and `query` for inference.

```bash
python3 -m research_v2.pilot_preparation \
  --input data/benchmark/medical_causal_pilot100_draft_v1.jsonl

# Optional: write query-only export and a provenance profile.
python3 -m research_v2.pilot_preparation \
  --input data/benchmark/medical_causal_pilot100_draft_v1.jsonl \
  --output-dir ../CDF-RAG-2-experiment-artifacts/phase2/pilot100_inputs_v1
```

The exporter never creates ground truth or verified medical judgments.
The pilot has not been split or source-reviewed. Its queries should be revised
against medical sources before any official evaluation. An export is useful
for schema testing, but **do not** treat the results from running these queries
against G0 as an accuracy or quality benchmark: most draft concepts are not
represented in the tiny G0 graph.

### Upcoming annotation milestone

Review each question for clarity, answerability, source support, and whether
its proposed intermediate causal links are valid. Record the source IDs and
medical relationship types; do not assume indirect or multi-hop causal paths
exist. Mark `source_reviewed` only after an actual review. Obtain expert
review of a sample when possible. Assign families and final splits only after
review. Then create **separate** gold and inference-only exports, and validate
that no held-out answers enter refinement or PPO training.
