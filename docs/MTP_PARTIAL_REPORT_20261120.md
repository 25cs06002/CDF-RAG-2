# CDF-RAG-2 MTP Partial Submission — Experimental Report Draft

**Prepared 18 November 2026; preliminary research prototype.**
All values are from the existing local 100+32 draft question experiments and
archived G0/G1 comparisons. This manuscript does not claim independently
verified medical accuracy, a final adaptive multi-round architecture, or
completion of the thesis.

## Abstract

Reliable domain-specific question answering requires retrieval of evidence
that is relevant, source-grounded and consistent with requested causal or
risk-factor relations. We investigate a causal retrieval-augmented generation
(RAG) research prototype and compare eight baseline configurations on a
frozen 100-question medical QA pilot. We also execute six retrieval
configurations across the original 100 and an additional 32 draft questions,
preserving 792 standardized model-query results. A frozen 35-node G0 causal
graph is expanded to a 78-node source-review-gated G1 candidate graph,
and B5/B6 retrieval performance is compared under frozen, expanded, and
stabilized query-resolver configurations. G1 legacy retrieval decreases B6
evidence yield from 41/132 to 35/132; a conservative stability-aware
resolver reaches 44/132. Manual review of 31 deliberately selected
changed-result cases produces 132 evidence judgments and identifies 17
cases requiring further investigation. The study demonstrates that
knowledge-graph expansion alone is insufficient to guarantee dependable
retrieval; query-target resolution and evidence provenance warrant explicit
controls. The proposed adaptive sufficiency-gated multi-round system
remains future work.

## 1. Problem statement and objectives

Classical retrieval can return textual passages without differentiating
causes, associated factors and clinical risk factors. Graph-based causal
retrieval makes relationships explicit but is sensitive to graph coverage,
entity mapping and source quality. Increased retrieval or refinement depth
can also increase latency.

The partial-stage objectives are to (i) create a reproducible baseline
comparison; (ii) expand and inspect a candidate causal knowledge graph with
typed, source-reviewed relations; (iii) identify how graph growth changes
retrieval coverage and concept resolution; (iv) distinguish evidence yield
from relevant, direction-correct evidence; and (v) motivate a future adaptive
evidence-sufficiency mechanism.

## 2. Literature positioning

MRD-RAG uses repeated retrieval/interaction to refine diagnostic reasoning;
GARMLE-G emphasizes retrieved guideline grounding; CDF-RAG motivates
causal retrieval, verification and feedback. These complement one another
but do not automatically resolve the empirical coverage–relevance–latency
tradeoff observed in our prototype. This draft's main contribution is the
measured retrieval foundation and failure analysis, not a completed new
end-to-end architecture.

## 3. Benchmark and experimental material

The experimental draft consists of 100 frozen pilot questions and 32
additional questions, covering direct, paraphrased, indirect, risk-factor,
ambiguous, unsupported and other query types. The questions and target
concept inventory influenced construction and review of G1; therefore the
132-question pool is not an independent held-out set.

B1–B4 retrieve a provisional AI-assisted source-summary corpus. B5/B6
operate on G0 graph-derived evidence. Original end-to-end B7/B8 use G0
through preserved No-RL and PPO-based implementations. These sources are
not identical, so returned-document counts are not an architecture-only
accuracy comparison.

## 4. Knowledge-graph construction and stabilization

Frozen G0 contains 35 nodes and 37 inherited unverified causal-format
edges. The G1 candidate reaches 78 nodes and 52 causal-format edges;
its typed relationship review includes 15 CAUSES, eight RISK_FACTOR_FOR
and one ASSOCIATED_WITH candidate relation. The inherited edges are not
thereby independently medically certified.

The G1 paired shadow experiment finds that concept matching and evidence
coverage can regress on adding nodes and relations. A conservative
stability-aware resolver preserves legacy anchors unless sufficiently
explicit new targets are available. It is evaluated separately from
the original B5/B6 configurations.

## 5. Experiment design and metrics

Eight baseline systems B1–B8 are represented by frozen 100-question
end-to-end archived runs with run identifiers, status counts and timings.
Six B1–B6 systems were also run in retrieval-only mode on the 100 and
32 cohorts, yielding 600+192=792 standardized model-query records,
zero execution errors, and evidence-presence, abstention and timing measures.

Status `ok`, returned evidence, generated answer presence, and correct
evidence are separate quantities. Retrieval-only latency excludes
generator time, initialization and other preflight cost. Archived B7/B8
end-to-end latencies must not be compared as equivalent costs.

## 6. Results

### 6.1 Fresh B1–B6 retrieval-only evaluation on 132 questions

| Model | Queries returning evidence | Mean retrieval latency (ms) |
|---|---:|---:|
| B1 Naive lexical | 132/132 | 0.245 |
| B2 BM25 | 132/132 | 0.408 |
| B3 Dense | 132/132 | 12.725 |
| B4 Hybrid | 132/132 | 13.918 |
| B5 Graph-neighborhood | 49/132 | 0.162 |
| B6 Directed causal KG | 41/132 | 0.146 |

This illustrates a clear contrast in evidence availability for the
chosen source summary corpus versus the small G0 causal graph.
It is not a measure of clinical answer correctness. Retrieval timing
differences should be interpreted under the recorded encoder and corpus
conditions.

### 6.2 Paired G0/G1 retrieval coverage

| Model | G0 | G1 legacy | G1 stabilized |
|---|---:|---:|---:|
| B5 | 49/132 | 53/132 | 55/132 |
| B6 | 41/132 | 35/132 | 44/132 |

The B6 regression from 41 to 35 after G1 expansion suggests a
concept-selection problem, not proof that added graph knowledge is
medically invalid. The stabilized configuration recovers to 44 but
remains a separate shadow algorithm. The gains are evidence yield, not
a statistical improvement in causal answer accuracy.

### 6.3 Historical No-RL and PPO contrast

On the archived 100-question experiment, B7 returned graph evidence for
49 questions and B8 for 65 questions; both selected runs were execution-
error-free. Mean end-to-end latencies were 28.95 s and 48.53 s.
A 20-question subset scored against structural labels derived from G0
and draft target mappings had edge micro-F1 of 0.872 and 0.971.
These structural labels are not independent medical gold and cannot
support a clinical causal-F1 claim. The earlier 22-query feasibility
comparison is maintained as a separate historical experiment.

### 6.4 Manually reviewed changed-case sample

Thirty-one model-query cases were reviewed, covering 20 distinct questions.
The 132 retrieved evidence occurrences reduce to 58 distinct
question-evidence assertions for review. Seventeen model-query cases
over 12 distinct questions were prioritized for investigation, involving
64 evidence occurrences. Eight were P0 wrong-target cases, six were P1
unsupported gains, and three were P2 uncertain target cases. Five cases
carried an uncertainty flag in the overlapping flag taxonomy.

The manual sample was deliberately selected because retrieval behavior
changed. Neither it nor source-check attestation establishes general
clinical accuracy. A reviewer-grounded P0 and P1 example is to be inserted
from `representative_case_studies.md` after reading the actual local
case details.

## 7. Preliminary contributions

1. Standardized, reproducible eight-baseline and six-model extended pilot
   pipelines with execution statuses and versioned provenance.
2. Source-review-gated expansion from frozen G0 to a typed G1 research
   graph without silently certifying inherited claims.
3. Empirical identification of graph-expansion-induced query-target drift,
   including a B6 retrieval regression.
4. Conservative stabilization and separate shadow evaluation.
5. Reviewer-grounded error analysis separating wrong targets, unsupported
   retrieval gains, evidence relevance and directional correctness.
6. Independent technical integrity audit verifying the final submitted
   tables against saved 792 query-model records and frozen inputs.

## 8. Threats to validity and limitations

- No independent medically adjudicated gold answer/path/edge reference.
- G0 inherited causal-format claims are medically unverified.
- The 132-question draft is not independent of graph design.
- Different evidence bases across model families.
- Retrieval coverage is not accuracy, groundedness or hallucination rate.
- The manual review covers deliberately changed model-query cases.
- Legacy end-to-end timings and clean retrieval-only measurements differ.
- No implemented and independently evaluated evidence-sufficiency
  multi-round feedback controller yet.

## 9. Planned scientific work

First, inspect and resolve wrong target and unsupported evidence examples
using source-checked subtype mappings and typed evidence sufficiency.
Evaluate modifications on separately adjudicated questions. Next,
add multi-round refinement with a bounded budget and early stopping.
Compare rule-based, no-RL and PPO-enhanced refinement under matched
datasets, evidence corpora and hardware. Report accuracy where supported,
latency percentiles, throughput and compute cost, with uncertainty estimates.

## Appendix: reproducibility and submission artifacts

- Archived 100-query B1–B8 results under `presentation_20261118/runs/`.
- 132-question B1–B6 retrieval results under
  `presentation_20261118/partial_submission_sprint/`.
- Combined CSV tables, manifest and figures within the assembled
  `submission_*` directory.
- `PASS_TECHNICAL_INTEGRITY` audit and `slide_factsheet.md`
  under `presentation_20261118/submission_readiness/`.
- Reviewed-error triage under `presentation_20261118/g1_error_triage/`.

This draft must be checked against the archived paper bibliographies,
local run hashes, selected examples and supervisor submission rules
before formal submission.
