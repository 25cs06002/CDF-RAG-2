# M.Tech progress presentation — November 18, 2026

**Research title:** Adaptive Multi-Round RAG Architecture for Reliable Domain-Specific Question Answering  
**Underlying implementation:** CDF-RAG-2 (research-v2), medical causal QA pilot.  
**Target speech duration:** 5 minutes, 3 principal slides + 2 backup slides.  
**GitHub implementation:** [CDF-RAG-2/research-v2](https://github.com/25cs06002/CDF-RAG-2/tree/research-v2).

## Slide 1 — Research question and system design (about 1:10)

**Problem:** Single retrieval or indiscriminate multi-round refinement is not guaranteed to produce reliable answers. Additional reasoning can increase latency without increasing retrieved evidence. Medical causal queries require correct relationship direction and traceable source support.

**Research objective:** Evaluate standard document RAG, graph-driven retrieval and legacy CDF-RAG consistently, then design an *evidence-sufficiency-aware* controller that retrieves again only when useful.

Explain the proposed workflow:
`Question → Query understanding → BM25/Dense/Graph retrieval → Evidence assessment → [Stop / Refine + retrieve] → Grounded generation → Claim validation`.

**What is actually built now:** original PPO RL and No-RL legacy adapters; six additional presentation retrieval configurations; unified experimental interface and standardized result schema; benchmark export / preflight / corpus provenance tools.

**Not yet a completed contribution:** optimized multi-round controller and independently medically validated answer quality.

## Slide 2 — Implementation, benchmark, and fairness (about 1:20)

**Original experiment:** 22 related medical queries, PPO RL vs No-RL only.

**Expanded current pilot:** 100 unreviewed draft queries across 8 categories:
`direct 15, paraphrased 15, indirect 15, multi-hop 15, comparative 12, risk-factor 12, ambiguous 8, unsupported 8`.

**Eight configurations:**

| Model | Retrieval method | Evidence / execution |
| --- | --- | --- |
| B1 | Naive lexical overlap RAG | Provisional Phase 2 source summaries |
| B2 | BM25 RAG | Same source summaries |
| B3 | Dense RAG (LSA offline pilot; sentence embedding locally) | Same source summaries |
| B4 | Hybrid BM25 + dense RRF | Same source summaries |
| B5 | Graph-neighborhood RAG | G0 graph relationships |
| B6 | Directed causal KG-RAG | G0 graph relationships |
| B7 | Original CDF-RAG No-RL | Local Neo4j + Ollama — 100-query run pending |
| B8 | Original CDF-RAG PPO RL | Local Neo4j + Ollama + PPO — 100-query run pending |

**Reproducibility:** common 100 query IDs, SHA256 dataset identity, G0 graph snapshot, non-overwriting run folders, execution status and per-query latency logs, GitHub Actions tests.

**Important limitation:** candidate summaries are not the same evidence source as G0; this is a *pilot operational study*, not a fair common-corpus medical accuracy ranking. Independently adjudicated medical gold labels are not yet available.

## Slide 3 — Completed results and request for guidance (about 2:00)

### Newly measured: six 100-query retrieval-only baseline runs

**Date/source:** November 15, 2026 GitHub-hosted CPU, six baselines × 100 = **600 actual retrieval operations**. Original log and generated artifacts: [successful experiment run](https://github.com/25cs06002/CDF-RAG-2/actions/runs/43717632591). Versioned summary CSV: [retrieval_only_100_v2_multihop.csv](../research_artifacts/presentation_20261118/retrieval_only_100_v2_multihop.csv).

| Baseline | Questions | Returned evidence | Abstained | Errors | Mean retrieval execution (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 — Naive lexical | 100 | 100 | 0 | 0 | 0.165 |
| B2 — BM25 | 100 | 100 | 0 | 0 | 0.327 |
| B3 — LSA dense | 100 | 100 | 0 | 0 | 1.239 |
| B4 — Hybrid (BM25+LSA) | 100 | 100 | 0 | 0 | 1.875 |
| B5 — Graph neighborhood | 100 | 41 | 59 | 0 | 0.111 |
| B6 — Directed causal KG | 100 | 36 | 64 | 0 | 0.101 |

The directed causal KG baseline now preserves complete two-hop paths, averaging **0.26 path records per query** across all 100 pilot questions (when retrieval is possible). These are **evidence-return counts**, not correctness, citation faithfulness, hallucination rates, answer quality or end-to-end generation latency. The tiny G0 graph has 35 nodes and 37 edges, limiting coverage.

### Earlier 22-query PPO versus No-RL (not directly comparable with the new rows)

| Legacy configuration | Retrieved causal coverage proxy | Mean end-to-end latency |
| --- | ---: | ---: |
| No-RL | 1.00 | 39.58 s |
| PPO RL | 1.00 | 50.09 s |

Earlier PPO took approximately **26.6% longer** without increasing this retrieval-coverage proxy in the old 22-query cohort. It does not establish that PPO is generally worse; the old experiment was small and shallow.

### Preliminary research interpretation

1. A tiny, single-domain graph cannot adequately cover a broad medical benchmark.
2. Returning more candidate text does not demonstrate relevant, correct or causal evidence.
3. PPO/refinement needs to be assessed separately from retrieval feedback and stopping.
4. The thesis contribution should be **reliable evidence assessment with performance-aware adaptive retrieval**.

### Ask the supervisor for rectification and priorities

1. **Benchmark:** Is a 100-question operational pilot sufficient for the current review, or should we restrict quantitative quality claims to a source-reviewed G0/G1-covered subset first?
2. **Knowledge graph:** Should we prioritize a provenance-validated G1 graph before larger pipeline comparisons? What should count as adequate domain coverage?
3. **Fair baseline:** Is a common, licensed document corpus plus an appropriately derived graph required before claiming architecture superiority?
4. **Adaptive contribution:** Do you prefer an evidence-sufficiency/early-stop ablation first, or improving the PPO reward and training environment?
5. **Evaluation:** Which three primary metrics should be emphasized: retrieval recall/F1, claim-level faithfulness, causal path accuracy, end-to-end latency or throughput?
6. **Deliverables:** Are there specific changes, tables or experiments required for the next supervisor review?

**Decision request:** Approve a prioritized plan: (i) source-reviewed benchmark + G1, (ii) common-corpus B1–B8 end-to-end evaluation, (iii) early stopping and feedback ablations, (iv) significance testing and writeup.

## Backup slide A — Query distribution and reproducibility

100-question draft category counts: 15 / 15 / 15 / 15 / 12 / 12 / 8 / 8. All inference files contain query IDs and question text only. G0 snapshot: 35 nodes / 37 edges. Corpus: AI-assisted paraphrases of candidate public-health sources; provenance retained. Original 22-query CSVs preserved. Automated CI tests and 600 retrieval-only experiments succeeded.

**Threats to validity:** no independent medical gold, different corpora for document vs graph runs, no LLM answer generation in current six CI runs, small graph, single CI hardware environment, LSA differs from local SentenceTransformer, B7–B8 not yet rerun on 100 queries.

## Backup slide B — Timeline after presentation

- First: execute B7/B8 and B1–B6 end-to-end locally with consistent model settings; validate records and latency.
- Next: independently source-review a gold subset, improve graph relation types/provenance, expand graph coverage to G1.
- Then: controlled architecture comparisons, multi-hop and query-refinement variants, early-stop/feedback ablations.
- Finally: scalability, robustness, model comparisons, statistical tests and thesis writing.

**Caution on status language:** say *implemented*, *code-tested*, *600 retrieval-only queries executed*, and *100-query local end-to-end run pending*; do **not** say *eight end-to-end pipelines fully benchmarked* or *medical accuracy improved* until evidence supports it.

## Supervisor discussion script

“Earlier, my results mainly compared RL and No-RL on only 22 related queries, which was insufficient to demonstrate the value of the architecture. I have expanded the pilot to 100 queries and implemented six additional retrieval configurations so the study covers lexical, BM25, dense, hybrid, graph-neighborhood and causal-graph methods. Six retrieval-only baseline runs on all 100 questions are now completed and reproducible. The document methods returned some candidate evidence for all 100 questions, while the small original graph returned evidence for 41 and 36 queries in the two graph variants. These are coverage and system-behavior measurements, not verified medical accuracy. The original 22-query PPO result also showed about 26.6 percent more latency without a coverage gain, motivating performance-aware refinement and early stopping. The next step is the 100-query local end-to-end comparison including PPO, then source-reviewed reference labels and G1 expansion. I would appreciate your corrections on benchmark validity, graph scope and which optimization experiment to prioritize first.”

## Files for supervisor inspection

- [Model list and README](../readme.md)
- [Executable experiment driver](../evaluate_all_baselines.py)
- [Central B1–B8 model registry](../research_v2/pipelines/)
- [100-query dataset](../data/benchmark/medical_causal_pilot100_draft_v1.jsonl)
- [Confirmed six-run numerical CSV](../research_artifacts/presentation_20261118/retrieval_only_100_v2_multihop.csv)
- [CI proof](https://github.com/25cs06002/CDF-RAG-2/actions/runs/43717632591)
