# CDF-RAG-2 M.Tech Partial-Work Presentation — 20 November 2026

**Status:** Research prototype, preliminary experimental evaluation. Numbers
refer to *draft questions* and candidate evidence; these are not medically
validated accuracy results. Supervisor-facing content, 8 slides (~6–8 min).
Use the independently audited `slide_factsheet.md` for final numerical copy.

## Slide 1 — Title / research motivation

**Adaptive Multi-Round Retrieval-Augmented Generation Architecture for Reliable
Domain-Specific Question Answering**

- Reliable answers require relevant evidence and correct relationship types.
- Multi-round reasoning increases computational cost.
- Our immediate focus: explain and repair retrieval failures before adaptive
  feedback optimization.

**Speaker note:** State that this is the partial-stage research foundation,
not the finished adaptive multi-round architecture.

## Slide 2 — Problem, literature, and research question

- MRD-RAG motivates iterative query clarification.
- GARMLE-G motivates trustworthy source/guideline-grounded retrieval.
- CDF-RAG motivates causal retrieval, verification and feedback.
- **Gap being investigated:** improving evidence coverage can also introduce
  misleading concepts or irrelevant relationships.
- **Research question:** How can a graph-augmented RAG system detect when its
  selected query target or evidence is insufficient, while controlling cost?

**Speaker note:** The proposed sufficiency gate and optimized feedback
controller are planned, not yet proven.

## Slide 3 — Existing implementations and baselines

- B1 Naive lexical; B2 BM25; B3 Dense Vector; B4 Hybrid.
- B5 Graph-neighborhood; B6 Directed Causal KG.
- B7 original CDF-RAG without RL; B8 original CDF-RAG with PPO.
- Standardized run records, timing fields, abstention and errors.
- **Critical control:** B1–B4 retrieve provisional AI-assisted candidate-source
  summaries; B5–B8 use frozen G0 graph information.

**Speaker note:** A cross-family evidence yield comparison is descriptive,
not a matched-corpus clinical accuracy ranking.

## Slide 4 — Dataset and graph foundation

- 100 pilot questions + 32 additional draft questions.
- Query categories include direct, paraphrased, indirect/multi-hop,
  risk-factor, ambiguous and unsupported cases.
- Frozen G0: 35 concepts and 37 inherited unverified causal-format edges.
- G1 candidate: 78 concepts, 52 causal-format edges, plus 24 reviewed
  typed relationships (15 CAUSES, 8 RISK_FACTOR_FOR, 1 ASSOCIATED_WITH).
- Keep candidate source-review provenance; inherited G0 claims are not
  independently certified.

**Speaker note:** G1 is a review-gated research graph candidate, not a
clinically validated causal knowledge graph.

## Slide 5 — Reproducible baseline results

| Model | Evidence / 132 | Mean retrieval latency (ms) |
|---|---:|---:|
| B1 Naive | 132 | 0.245 |
| B2 BM25 | 132 | 0.408 |
| B3 Dense | 132 | 12.725 |
| B4 Hybrid | 132 | 13.918 |
| B5 Graph-neighborhood | 49 | 0.162 |
| B6 Directed causal KG | 41 | 0.146 |

- 792/792 retrieval-only B1–B6 model-query records; 0 execution errors.
- All 8 original 100-question end-to-end runs recovered with zero errors
  in the selected archives.
- B5/B6 legitimately abstain when the smaller graph cannot retrieve evidence.
- Retrieval latency excludes initialization and generation.

**Speaker note:** The result establishes corpus coverage and runtime
behavior, not evidence accuracy. Use local audited figures.

## Slide 6 — Main experimental contribution: G0 → G1 regression

| Model | Frozen G0 | G1 legacy | Stabilized G1 |
|---|---:|---:|---:|
| B5 | 49/132 | 53/132 | 55/132 |
| B6 | 41/132 | 35/132 | 44/132 |

- Expanding G1 changes concept selection.
- B6 loses six evidence-returning queries with the legacy resolver.
- Conservative stability-aware matching recovers coverage to 44/132.
- This is a retrieval coverage result, not a verified causal improvement.

**Speaker note:** The contribution is *identifying and measuring the
regression, developing stabilization, and validating its behavior*. Some
newly returned evidence is still incorrect or irrelevant.

## Slide 7 — Reviewed cases and root-cause investigation

- 31 selected changed model-query cases over 20 distinct reviewed questions.
- 132 reviewed evidence occurrences; 58 unique question-evidence assertions.
- 17 prioritized error cases span 12 distinct questions and 64 occurrences.
- P0 wrong target: 8; P1 unsupported gain: 6; P2 uncertain: 3.
- Two selection-reason groups in the flagged cases: subtype ambiguity (8),
  explicit connected new target (9).
- **Insert two concrete cases** from `representative_case_studies.md`:
  one P0 wrong target and one distinct P1 unsupported evidence gain.

**Speaker note:** Review flags overlap; priority classes partition the 17
cases. These labels are not medically adjudicated gold.

## Slide 8 — Costs, contributions, conclusion and next work

- Original B7 No-RL: evidence on 49/100, mean end-to-end latency 28.95 s.
- Original B8 PPO-RL: evidence on 65/100, mean latency 48.53 s.
- The 20 draft-target-mapped G0 structural reference showed micro-F1
  0.872 (B7) vs 0.971 (B8); **not clinical causal F1**.
- Key tradeoff: increased structural coverage comes with added latency.
- Completed contributions: reproducible eight-model baseline framework,
  G1 source/typed-graph candidate, controlled G0/G1 comparison,
  stabilization and reviewer-grounded regression diagnosis.
- Planned: source-aware target selection, evidence sufficiency, confidence-
  aware abstention, adaptive multi-round stopping, then held-out evaluation.

**Speaker note:** Ask the supervisor which subset of graph claims must be
independently adjudicated before the next formal experiment.

---

## Three research claims to separate in Q&A

1. **Established empirically:** G1 expansion changes retrieval coverage, and
   the older G1 resolver can regress relative to frozen G0.
2. **Established provisionally:** A stability-aware query resolver recovers
   selected coverage, but still has reviewed relevance/target errors.
3. **Hypothesis/future work:** A source/type-aware evidence sufficiency gate
   with adaptive rounds will improve the reliability–latency tradeoff.
   This has not yet been tested as a completed model.

## Three common questions

**Is 132/132 retrieval yield 100% medical accuracy?**
No. It only states that document evidence was returned. B1–B4 operate on
AI-assisted provisional source summaries, not independently judged gold.

**Does B8 PPO definitively beat B7?**
No. It has higher G0 structural agreement and retrieval yield on the selected
draft experiment, at a substantially greater observed end-to-end latency.
No independent medical gold or broad statistical generalization is established.

**What is your original contribution beyond rerunning known RAG models?**
The systematic graph-expansion regression analysis, evidence provenance and
typed relationship review, stability-aware target matching, and manually
audited error taxonomy that identifies a measurable adaptive retrieval need.

**Source artifacts:** `presentation_20261118/partial_submission_sprint/`,
`presentation_20261118/submission_readiness/`, and
`presentation_20261118/g1_error_triage/`.
