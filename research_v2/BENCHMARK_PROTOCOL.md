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


## Phase 2B: candidate evidence-source reading queue

`data/benchmark/source_candidates15_v1.jsonl` contains 15 **candidate**
public-health references, one for each initial direct/paraphrased
question topic. These URLs were located on NIH, CDC, NCI, or MedlinePlus
websites; they are **not reviewed citations for individual gold claims**.
Candidate metadata says `awaiting_source_review`, `gold_annotation_status:
none`, and `verified_medical_gold: false`.

Run from the project root:

```bash
python3 -m research_v2.source_review_queue

OUT="../CDF-RAG-2-experiment-artifacts/phase2/source_review_$(date -u +%Y%m%dT%H%M%SZ)"
python3 -m research_v2.source_review_queue --output-dir "$OUT"
```

This creates a **30-question worklist** using 15 direct questions and
their 15 paraphrases. Each worklist row includes a matching source
candidate, its scope warnings, and tasks for a human reviewer.

**Critical scope distinctions**:
- Asthma pages about *attack triggers* do not establish *asthma onset causes*.
- Migraine *attack triggers* are not equivalent to migraine etiology.
- Epilepsy causes must be distinguished from individual seizure triggers.
- For stroke, separate ischemic versus hemorrhagic mechanisms.
- Risk-factor association alone must not automatically be encoded as
  a verified `CAUSES` relationship.
- Pneumonia and tuberculosis have different exposure, cause, and disease-
  progression questions; avoid conflating them.

The worklist includes candidate URLs, **not** claims that those URLs support
each question. Before annotation, open each source, verify its medical scope,
check the specific proposition and direction, and record the actual reviewer's
identity. Seek stronger primary research or guidelines for contested causal
steps, long multi-hop chains, or complex questions. At this phase the queue
only covers 30 questions; the other 70 have not received candidates.

The source-aware partition checker rejects reuse of the **same source URL**
in different splits, even under different source IDs or source-family labels.
Reviewed unanswerable questions without any source may leave
`source_family=null` but still require a documented abstention reason.
No automated validation can certify human clinical review.

**Next substantive milestone**: independently inspect the sources and create
a first *human-reviewed*, source-attributed annotation subset. Preserve the
drafts and candidate registry as separate immutable versions. The G0 graph
snapshot and existing legacy RL/No-RL code are unchanged.


## Phase 2C: first source-checked evidence proposals (not gold)

`data/benchmark/evidence_proposals3_v1.jsonl` contains **three topic-level
proposals covering six draft questions** and **six candidate relations**.
The three source pages were inspected on 2026-11-13:

1. `SRC001`: NHLBI, *Coronary Heart Disease Causes and Prevention*,
   https://www.nhlbi.nih.gov/health/coronary-heart-disease/causes
2. `SRC004`: NIDDK, *Risk Factors for Type 2 Diabetes*,
   https://www.niddk.nih.gov/health-information/diabetes/overview/risk-factors-type-2-diabetes
3. `SRC007`: NHLBI, *COPD Causes and Risk Factors*,
   https://www.nhlbi.nih.gov/health/copd/causes

These are **source-triaged candidate relations only**, not independently
reviewed reference labels. Every proposal has an explicit relationship
type, source heading, paraphrased support summary, question for human review,
and any unresolved scope issue. No evidence excerpt is reproduced.

Candidate counts: `CAUSES=2`, `RISK_FACTOR_FOR=4`.
No candidate was added to the G0 graph, promoted to medical gold,
assigned to a train/test partition, or used to calculate accuracy.

**Critical annotation distinctions**:

- `EVP001`: NHLBI discusses *coronary heart disease* and plaque buildup.
  The pilot uses *coronary artery disease*. A reviewer must resolve their
  exact scope and ontology mapping **before** treating the relationship
  as a gold edge for the target concept.
- `EVP002`: NIDDK presents overweight/obesity, inactivity, and family
  history as type 2 diabetes **risk factors**, not as deterministic,
  individually predictive causes.
- `EVP003`: NHLBI describes smoking as a cause of COPD and discusses
  pollutant exposures as risks. Review exposure duration, population,
  inherited susceptibility, and the evidence for each proposed edge.

### Generate the evidence-review pack

```bash
python3 -m research_v2.evidence_proposal_review

OUT="../CDF-RAG-2-experiment-artifacts/phase2/evidence_review_$(date -u +%Y%m%dT%H%M%SZ)"
python3 -m research_v2.evidence_proposal_review --output-dir "$OUT"
```

Generated files: `evidence_review_worklist.jsonl` and
`evidence_review_profile.json`. Neither contains verified medical gold.

**Next milestone**: inspect the source text independently, resolve the
ontology mismatches, review medical claims, and create a separate reviewed
annotation version. The remaining 24 source-linked direct/paraphrase
questions have not been source-triaged at this level, and the remaining
70 pilot questions still lack a candidate-source worklist.


## Phase 2D: complete initial source triage for 30 direct/paraphrase questions

The first candidate source triage set is now **15 topic-level work items,
covering all 30 direct/paraphrase questions**. Versioned inputs:

- `data/benchmark/source_candidates15_v2.jsonl`
- `data/benchmark/evidence_proposals15_v2.jsonl`

The version 1 files remain unchanged for reproducibility. The version 2
candidate-source registry replaces exactly two URLs that were inaccessible
during triage (NINDS epilepsy page and NIMH depression page) with available
National Library of Medicine MedlinePlus summaries. Both source variants
remain candidate references pending independent review.

### Outcome reporting (do not inflate evidence coverage)

The 15 source-checked topic proposals yielded:

- **14** topic proposals with one or more *candidate* relationships,
  covering **28** draft questions;
- **1** topic proposal with **no applicable onset evidence**, covering
  **2** draft questions (asthma);
- **24** candidate relation descriptions: 12 `CAUSES`,
  11 `RISK_FACTOR_FOR`, and 1 `ASSOCIATED_WITH`;
- **0** medically verified gold labels, **0** human-reviewed annotations,
  and **0** final train/validation/test assignments.

The asthma source is explicitly about triggers of an asthma *attack*,
which are not evidence that the trigger caused the underlying asthma disease
to develop. An `insufficient_for_question` proposal **must** have zero
candidate relationships and a specific reason explaining the evidence gap.
This is an annotation-data distinction, not a model abstention result.

Other qualifications requiring review:
- The stroke source distinguishes ischemic from hemorrhagic stroke; the
  candidate cause applies only to ischemic stroke.
- Heart failure subtypes HFrEF and HFpEF have different mechanisms.
- Weight gain, obesity, and factors associated with weight change are not
  interchangeable target labels.
- Migraine attack triggers must not be mislabeled as causes of migraine onset.
- TB exposure must be distinguished from progression to active disease
  after infection.
- Epilepsy and depression commonly have multifactorial or uncertain causes.
- Population-level risk factors are not proof of an individual's diagnosis.

### Generate the full preliminary review worklist

```bash
python3 -m research_v2.source_review_queue
python3 -m research_v2.evidence_proposal_review

OUT="../CDF-RAG-2-experiment-artifacts/phase2/evidence_review_15topics_$(date -u +%Y%m%dT%H%M%SZ)"
python3 -m research_v2.evidence_proposal_review --output-dir "$OUT"
```

The generated review pack **does not** become ground truth by passing
automated tests. A human reviewer must inspect the specific source passage,
population, causal relation and terminology, and assign genuine review
metadata. For unsupported source coverage, seek another source or record
the question as unresolved; never generate a causal edge merely to increase
the benchmark score.

Next substantive task: independent annotation of this 30-question subset
and evidence retrieval for the other 70 draft questions, especially indirect
and multi-hop questions. Once a *real* reviewed subset exists, build
source- and concept-grouped partitions, followed by evidence-backed G1 graph
snapshots. Do not change the old G0 graph or existing PPO model.


## Phase 2E: human annotation worksheets for the first 30 questions

To move from AI-authored candidate relationships toward **independently
reviewed medical gold**, create immutable-source, editable-decision files:

```bash
OUT="$HOME/Folder/CDF-RAG-2-experiment-artifacts/phase2/human_review_$(date -u +%Y%m%dT%H%M%SZ)"
python3 -m research_v2.reviewer_workspace --output-dir "$OUT"

# Before changing any decision fields, this should report 55 pending tasks:
python3 -m research_v2.reviewer_workspace --audit-dir "$OUT"
```

**Outputs**:
- `claim_review.csv`: 24 proposed relationships plus one
  **insufficient-source gap** (25 tasks). The draft claim, type,
  source, section, medical scope warning, and proposal ID are immutable.
- `question_review.csv`: one row for each of 30 direct/paraphrased
  questions, for human answerability and reference-answer review.
- `review_manifest.json`: SHA-256 hashes of the pilot,
  candidate-source registry, and proposal file.

Edit **only reviewer decision columns**. Keep all original-source fields
unchanged. If the candidate claim is wrong, use `revise` and fill in
corrected cause/effect/relation; never silently rewrite the AI-authored
candidate fields. A reviewed `accept` or `revise` decision must include
a source passage locator and reviewer rationale. A `reject` or
`needs_more_evidence` decision requires rationale but does not imply
there is a supported gold relationship.

For question rows, set answerability as `answerable`,
`unanswerable`, or `needs_more_evidence`. If `answerable`,
provide a reference answer and supporting registered source ID(s).
This first worksheet version restricts support IDs to the matching
candidate source; introducing additional sources requires a new,
versioned registry and review workbook. For medical questions,
`unanswerable` means not answerable from the defined evidence and
case details, not a universal claim that the condition has no explanation.

For all decided rows, fill in an **actual** `reviewer_id`,
`reviewed_date` as YYYY-MM-DD, and `review_note`. Do not put fabricated
reviewers into test or official annotations. If the source is
inadequate, record `needs_more_evidence` instead of inventing support.

```bash
# Reports pending / decided tasks and rejects edited source columns.
python3 -m research_v2.reviewer_workspace --audit-dir "$OUT"

# Fails until all rows have genuine, complete decisions.
python3 -m research_v2.reviewer_workspace --audit-dir "$OUT" --strict
```

This tool checks format, task coverage, immutable candidate metadata,
referenced source IDs and reviewer decision completeness. It does
**not** independently authenticate reviewers, judge medical evidence
quality, check live webpages, or create benchmark gold labels.
Even a completed `--strict` check does not authorize automatically
adding relationships to Neo4j or making clinical accuracy claims.

**Research next step:** review the 30 questions and candidate claims,
resolve the asthma source gap, then prepare a separately adjudicated
gold-label version after any domain-expert review required by the
study protocol. In parallel, locate evidence for the remaining 70
questions. Maintain the G0 graph and the original proposal datasets.
