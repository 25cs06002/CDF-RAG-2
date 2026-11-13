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
