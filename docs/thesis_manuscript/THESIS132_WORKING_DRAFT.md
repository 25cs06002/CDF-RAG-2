# Adaptive Multi-Round Retrieval-Augmented Generation Architecture for Reliable Domain-Specific Question Answering

**Working thesis manuscript — experimental chapters.** Source of truth: [frozen final experimental release](../../presentation_20261118/thesis132/final_thesis_results_132/release_20261125T212811325446Z/README.md). This draft does not claim independent clinical validation. Institutional formatting, author/supervisor details, and a verified literature bibliography must be completed before submission.

## Abstract

Retrieval-augmented generation (RAG) can improve the evidential basis of domain-specific question answering, but the retrieval of candidate evidence does not necessarily ensure relevant, complete or factually supported responses. Iterative retrieval and generation can also increase inference latency, particularly on resource-constrained systems. This work proposes a bounded adaptive multi-round RAG architecture for medical question answering that integrates query-dependent retrieval, a typed candidate causal graph, BM25 document fallback and evidence-constrained response construction. A rule-based controller schedules up to three retrieval rounds and uses a local language model to select relevant evidence identifiers. Structural checks enforce identifier membership and coverage of comparison targets; the selected source-attributed candidate text is rendered deterministically, with abstention or explicit partial-evidence qualification where appropriate. Eight baseline pipelines and eleven proposed-system configurations were evaluated on a fixed, development-exposed set of 132 medical questions, yielding 2,508 archived model-query executions. Under an AI-assisted evidence review protocol, the selected B9-v2 configuration produced 99 substantive responses, of which 95 met the strict supported-and-grounded criterion and 36 were rated complete. Its mean latency was 5.331 seconds and throughput was 0.1876 queries per second, compared with 49.344 seconds and 0.0203 queries per second for the historical PPO-based CDF-RAG configuration. The adaptive policy used 169 retrieval rounds compared with 303 for the fixed-three-round policy, although it did not reduce measured end-to-end latency in that comparison. Compact evidence identifiers further reduced observed inference latency at the cost of answer completeness. These results establish development-set quality–efficiency trade-offs, not clinical accuracy or independent generalization; external expert adjudication and held-out evaluation remain future work.

**Keywords:** retrieval-augmented generation, adaptive retrieval, causal knowledge graph, constrained generation, abstention, medical question answering, inference efficiency.

# Chapter 1 — Introduction

## 1.1 Motivation

Domain-specific medical questions require answers that preserve distinctions between direct causes, risk factors, associations, triggers and uncertain mechanisms. An answer may be plausible but fail to address the requested disease subtype or intermediate causal relationship. A conventional RAG pipeline that retrieves once and generates freely may introduce unsupported material or fail when the initial evidence is insufficient. Multi-round pipelines can broaden retrieval, but repeatedly invoking a local language model introduces additional computational cost. The central objective is therefore not to maximize the number of retrieval operations, but to acquire enough relevant candidate evidence to support a useful response while limiting expensive inference.

## 1.2 Problem statement

Design and evaluate a bounded adaptive retrieval and evidence-constrained generation architecture that improves the measured reliability and computational efficiency of medical question answering relative to selected historical RAG baselines, while explicitly separating answer availability, factual support, grounding, completeness and abstention. The experimental scope is the frozen 132-question development cohort on a local CPU-oriented execution environment.

## 1.3 Research questions

- RQ1: How do candidate evidence coverage, AI-reference claim support, strict supported-and-grounded yield and measured latency differ across eight baseline pipelines and the proposed system?
- RQ2: Does the adaptive retrieval policy reduce retrieval work relative to fixed-round policies, and what are its observed effects on answer availability, completeness and latency?
- RQ3: How do typed graph evidence and document fallback contribute to coverage and completeness under the same B9-v2 controller?
- RQ4: How do evidence identifier compression and alternative local generators affect response availability, token use, measured latency and resource usage?

## 1.4 Contributions

The contributions are an integrated query-aware bounded retrieval controller; source-attributed typed graph and document retrieval with qualified path handling; structured evidence selection and deterministic rendering with abstention; a versioned experimental protocol with per-query traces, model digests and integrity checks; and comparative analysis of quality, coverage, latency, retrieval work and resource measurements. These are system and empirical contributions. Independent novelty claims require a verified related-work comparison.

# Chapter 2 — Background and related work (completion required)

Organize the literature review around single-round lexical/dense/hybrid RAG, graph-based and causal retrieval, adaptive or iterative retrieval, constrained generation and evidence attribution, medical QA evaluation, and resource-efficient local inference. Compare the studied MRD-RAG, GARMLE-G and CDF-RAG papers using their actual published methods, data, quality metrics, latency/throughput and limitations. Do not invent authors, publication venues, dates, paper results or missing efficiency measurements. Verify each paper's bibliographic record and distinguish the thesis's local implementations from results reported by the original authors. This chapter intentionally does not supply unverified citations.

# Chapter 3 — Methodology

## 3.1 Dataset and experimental scope

The canonical question-only input contains 132 unique IDs and questions, comprising 100 pilot and 32 extension queries. Types: direct (19), paraphrased (19), indirect (19), multi-hop (19), comparative (16), risk-factor (16), ambiguous (12), unsupported (12). There are 80 semantic groups and five family-consistent folds of 27, 27, 26, 26 and 26. The folds are not independent test partitions because the entire cohort influenced G1 development. Evaluation-only metadata and AI review labels are not provided to the inference controller. The canonical question SHA-256 is recorded in the final architecture lock. The 24 ambiguous/unsupported records represent a development-set negative category, not a gold answerability standard.

The document corpus consists of provisional source-attributed summaries. The reconciled G1 snapshot records 78 nodes, 52 graph CAUSES edges (including 37 inherited unverified G0 edges), and a separate 24-link typed candidate ledger (15 CAUSES, eight RISK_FACTOR_FOR, one ASSOCIATED_WITH). These counts refer to different representations and must not be added together as if they were disjoint edges. The typed B9 retrieval uses the candidate ledger and excludes links marked as requiring narrowing. Source screening and compatibility approvals do not establish independent clinical gold. Earlier B1–B4 use provisional summaries; B5–B8 use historical G0 graph evidence.

## 3.2 Baseline systems

| ID | Local implementation |
|---|---|
| B1 | Naive lexical RAG |
| B2 | BM25 RAG |
| B3 | Dense vector RAG |
| B4 | Hybrid BM25 and dense RAG |
| B5 | Graph-neighborhood RAG |
| B6 | Directed causal KG-RAG |
| B7 | Legacy CDF-RAG without RL |
| B8 | Legacy CDF-RAG with PPO RL |
| B9 | Proposed B9-v2 adaptive retrieval and constrained rendering |

These are the local experimental systems, not exact reproductions of every result in the original papers. B9 differs in evidence snapshot and generation mechanism; same questions do not make the B1–B9 contrast a controlled causal test of the controller.

## 3.3 Proposed architecture

Let q denote a question, R the retrieval-round budget (3), k the evidence-selection budget (5), E_t the accumulated candidate pool after round t, and A_t the offered set of at most k candidates. Candidate items retain source attribution, type and identifiers. The controller first resolves query intent and targets. Individual/underdetermined inference requests and unsupported comparison scopes abstain. The scheduled retrieval routes are:

| Route | Round 1 | Round 2 | Round 3 |
|---|---|---|---|
| Targeted | Typed direct edges | Targeted documents | Additional documents |
| Pathway | Typed direct edges | Typed two-link path | Qualified document evidence |
| Comparison | Balanced target documents | Additional balanced documents | Additional balanced documents |
| Descriptive/multi-concept | Open document retrieval | Additional documents | Additional documents |

At each round, E_t is formed by adding the retrieved items to the prior pool, deduplicated by identifier. A pathway question requires a strict typed two-link CAUSES path for a strict-path response; otherwise the document fallback is explicitly marked as partial evidence. Comparison retrieval reserves candidates for each identified target. When a candidate set satisfies the structural gate, a local language model is prompted to select a JSON list S_t of relevant identifiers, or return an empty list. The gate is heuristic, not proof of semantic sufficiency.

The validator checks syntax, uniqueness, membership S_t ⊆ A_t, budget and comparison-target coverage. A valid nonempty selection with the required target coverage terminates retrieval. A valid empty selection or insufficient target coverage may trigger further retrieval within the budget; the same candidate-ID set is not sent twice. Invalid JSON fails closed. If no acceptable selection is obtained, the pipeline records abstention with an empty final answer. A transport failure is recorded as an error, not abstention. The renderer copies the selected source-attributed candidate text and adds scope qualifications and evidence IDs/URLs. It does not synthesize unrestricted new medical claims. Citation membership is a structural check, not an independent medical hallucination detector.

### Algorithm (pseudocode)

    route, targets = resolve_query(q)
    if guarded(route): return ABSTAIN
    pool = empty
    attempted = empty
    for action in scheduled_actions(route)[:R]:
        pool = deduplicate(pool + retrieve(action, q, targets))
        offered, groups, scope = construct_candidate_set(pool, route, k)
        if not structural_candidate_gate(offered, groups): continue
        if identifier_set(offered) in attempted: continue
        attempted.add(identifier_set(offered))
        raw = local_llm_select(q, offered, groups, scope)
        selected = parse_strict_json(raw, offered, k)
        if selected is invalid: return ABSTAIN
        if selected is empty or missing_required_target_coverage: continue
        return render_source_attributed_text(selected, scope)
    return ABSTAIN

The final configuration uses Llama 3.2 3B (Ollama tag llama3.2:latest, pinned digest in lock), temperature 0, context budget 2048, output budget 256, original identifiers, top-k 5 and at most three rounds. Inference is rule-based, not a newly trained PPO policy. Exact retrieval-query caching is available; no answer cache is used. An optional compact-ID variant maps identifiers to short local aliases and restores the original identifiers before rendering.

# Chapter 4 — Experimental design and metrics

The archived experiments comprise 8 × 132 baseline outputs and 11 × 132 B9 configuration outputs: 2,508 model-query records, not 2,508 independent questions. The eleven configurations include B9-v1, B9-v2, fixed1, fixed3, graph-off, document-off, original-ID profiling, compact-ID profiling, and three compact-ID generator runs. A single AI-reference protocol (AIJ-MED-HALLUC v1) was used for answer-specific annotations: 1,056 baseline records, 132 initial B9-v2 records, and 372 additional unique B9 cases after exact-context deduplication. The 1,560 case annotations are not independent human clinical judgments or independent statistical samples. Exact-match reuse is documented; the 372 cases map to 591 outputs and 861 outputs reuse the initial B9 review. Review sessions and claim granularity may differ.

The AI review classifies medically material claims as supported, partially supported, contradicted, unsupported or unverifiable, excluding non-factual claims from the factual count. An accepted answer label is Yes for a material contradicted/unsupported claim or fabricated citation, Uncertain when unresolved according to protocol, and No otherwise. A No label can include partially supported, non-factual or empty answers; it is not an accuracy label. B9 completeness and abstention categories are distinct AI judgments. Historical B7/B8 detector classification is agreement against AI labels, not human gold. B9 has no corresponding binary hallucination detector; structural selector validation must not be used to fabricate one.

| Metric | Definition |
|---|---|
| Candidate evidence coverage | queries with candidate evidence / 132 |
| Substantive coverage | responses with ≥1 factual claim / 132 |
| AI-reference hallucination rate | accepted Yes / accepted Yes + No, i.e. Yes/(Yes+No) |
| Substantive hallucination rate | Yes among accepted substantive answers / accepted substantive answers |
| Strict supported-and-grounded yield | substantive accepted No with full grounding, ≥1 supported claim and no partial, contradicted, unsupported or unverifiable claims / 132 |
| Strict complete yield | strict supported-and-grounded answers also rated COMPLETE / 132 (B9 variants only) |
| Mean latency | sum of measured per-query end-to-end times / 132 |
| Throughput | 132 / measured run wall time in seconds |
| P95 latency | empirical interpolated 95th percentile |
| Semantic proxies | cosine similarity using the pinned all-MiniLM-L6-v2 encoder; report available observation counts and text bases |

B9 context relevance is query–retrieved-context cosine similarity and groundedness is answer–evidence cosine similarity; they are not entailment or clinical correctness. Semantic refinement uses query–refined-query pairs; B9-v2 has just one recorded observation. Missing metrics remain unavailable. Report latency on all questions, answered questions and paired common-answered subsets, since cheap abstentions reduce all-query means. CPU and RAM profiling uses sampled Linux system statistics and Ollama telemetry; sampled peaks are not exact peak model memory, and energy was not measured.

# Chapter 5 — Experimental results and analysis

## 5.1 Main comparison

| Model | Substantive n | AI hall. % | Strict yield % | Mean s | P95 s | QPS |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 121 | 11.36 | 54.55 | 17.412 | 26.816 | 0.0574 |
| B2 | 120 | 12.12 | 50.00 | 18.027 | 26.934 | 0.0555 |
| B3 | 117 | 10.61 | 48.48 | 19.634 | 28.386 | 0.0509 |
| B4 | 121 | 15.91 | 49.24 | 19.183 | 27.210 | 0.0521 |
| B5 | 35 | 1.52 | 8.33 | 4.178 | 16.244 | 0.2393 |
| B6 | 31 | 9.09 | 5.30 | 3.962 | 16.612 | 0.2524 |
| B7 | 51 | 14.50 | 7.58 | 27.398 | 77.660 | 0.0365 |
| B8 | 68 | 15.91 | 7.58 | 49.344 | 96.170 | 0.0203 |
| B9 | 99 | 0.00 | 71.97 | 5.331 | 10.688 | 0.1876 |

Source: final table_B1_B9_132.csv. B7 uses 131 accepted binary labels; all other models here have 132. B9 shows 89.20% lower measured mean latency than B8, about 9.26× QPS and 64.39 percentage points higher strict yield. B1 has more AI-supported substantive answers (106 vs 99), so B9 is not superior under every quality definition. B5 and B6 have lower mean latency partly because of frequent abstention. B9 has 95 strict supported-and-grounded responses (95/132), but only 36 complete answers and 33 strict complete answers (33/132 = 25%). Among 99 responses, 53 were rated partial and 10 nonresponsive. The remaining 33 queries abstained; 17 were rated appropriate and 16 inappropriate. The zero AI Yes labels do not establish zero real hallucinations.

## 5.2 Development from B9-v1 to B9-v2

B9-v1 answered 54 questions; B9-v2 answered 99. Paired output status transitions were 54 ok→ok, 45 abstained→ok and 33 abstained→abstained. Answer availability increased by 34.09 percentage points, but the change does not show a factual-quality regression test by itself. Mean latency rose from 2.009 to 5.331 seconds as more requests reached generation. B9-v2 answered 19/19 direct, 19/19 paraphrased, 19/19 indirect, 17/19 multi-hop, 9/16 comparative and 16/16 risk-factor questions. All 24 negative-category inputs abstained, but only 17 abstentions overall were judged appropriate. Eleven responses were qualified partial-pathway responses and only one response was explicitly tagged strict typed path. Do not equate 17 multi-hop-category answers with 17 validated complete chains.

## 5.3 Adaptive retrieval policy

| Policy | Answers | Complete | Strict complete | Rounds | LLM calls | Mean s |
|---|---:|---:|---:|---:|---:|---:|
| Fixed1 | 46 | 24 | 21 | 101 | 46 | 2.507 |
| Fixed3 | 97 | 37 | 33 | 303 | 100 | 4.906 |
| Adaptive | 99 | 36 | 33 | 169 | 100 | 5.331 |

Adaptive reduces recorded rounds by 44.22% relative to fixed3 and recovers two additional candidate answers. It does not reduce observed latency: fixed3 was faster in this run and in the 44-query common-answered subset (5.296 vs 6.984 seconds). Both make 100 selector calls. Selector-driven retry was not exercised: no reference query made multiple calls, no valid empty selection occurred. The demonstrated benefit is route-dependent early stopping/retrieval-work reduction, not a proven generation-time or feedback-loop speedup.

## 5.4 Component ablation

| Component | Answers | Complete | Strict supported & grounded | Mean s |
|---|---:|---:|---:|---:|
| Full B9-v2 | 99 | 36 | 95 | 5.331 |
| Graph evidence off | 99 | 43 | 93 | 6.139 |
| Document evidence off | 15 | 3 | 15 | 0.409 |

The document fallback is essential for broad coverage in this design. Graph evidence preserves a strict typed-path route and the full configuration has a higher strict supported-and-grounded yield than graph-off, but graph-off has seven more complete answers. Graph-off retains graph-derived target routing; this is not a graph-free model. Document-off lacks document-based comparisons. Single-run latency and differing answer composition limit causal interpretation.

## 5.5 Inference optimization and generator comparison

The original-ID versus compact-ID profiled runs had mean latency 5.685 vs 4.370 seconds, mean output tokens 28.03 vs 12.36 and 99 vs 100 answers. Compact IDs reduced measured mean latency by approximately 23.14% and output tokens by 55.90%. Complete responses fell from 36 to 30. Fifty-one question–answer outputs changed. Different initial system RAM and swap states prevent attributing the entire measured speedup to identifier length. The exact-context retrieval microbenchmark showed median times 112.47 ms uncached and 37.77 ms cached for 396 repeated retrieval calls (2.98×), not a measured end-to-end gain. The adaptive live reference used only one retrieval cache hit.

| Compact-ID generator | Answers | Complete | Mean s | QPS | Peak sampled system RAM (GiB) |
|---|---:|---:|---:|---:|---:|
| Llama 3.2 3B | 100 | 30 | 4.328 | 0.2310 | 7.32 |
| Llama 3.2 1B | 27 | 7 | 4.753 | 0.2103 | 4.80 |
| Qwen 2.5 1.5B | 7 | 3 | 2.302 | 0.4341 | 4.42 |

All three had candidate evidence on 100 questions. Llama 3B produces the largest candidate-answer yield per second; the smaller models have more abstentions under the fixed prompt/protocol. Only two questions were answered by all three, so common-answer latency comparisons have very limited support. Token counts depend on tokenizers. Resource measurements are system-wide sampled values rather than exclusive model allocations.

## 5.6 Final configuration and figures

Freeze the original-ID adaptive Llama 3B B9-v2 configuration as a quality-oriented development-set choice: 95 strict supported-and-grounded answers, 36 complete responses, and 5.331 seconds mean latency. Retain compact IDs as an optional latency optimization, not an unequivocal replacement. The fixed3 and graph-off results indicate genuine trade-offs; neither dominance nor optimality has been established. Insert the six generated figures from the release's figures/ directory with captions describing the metric denominators and AI-reference status.

# Chapter 6 — Limitations, conclusion and future work

The cohort is development-exposed, small, and restricted to medical QA. Evidence sources are provisional; the AI reviewer had limited source access and some annotations were reused, with no independent clinical reviewers, inter-rater agreement or held-out test. Strict yield does not require completeness, and the constrained rendering deliberately favors supported wording over free-form synthesis. The graph and document snapshots differ between historical baselines and B9, and latency includes abstention. The resource results are observational single runs with changing system load. Semantic metrics are cosine similarities, not gold relevance/entailment. There is no measured energy consumption or validated clinical deployment.

The experiments demonstrate that the proposed integrated pipeline can produce more source-grounded candidate responses than the tested legacy causal RAG systems, with substantially lower observed mean latency, while exposing a completeness–coverage–efficiency trade-off. The adaptive policy reduces retrieval work relative to fixed three rounds, but does not show an end-to-end latency benefit in the recorded comparison. Document fallback supplies most coverage; constrained generation limits additional unsupported wording but also limits synthesis. Future work should prioritize independent clinically reviewed held-out questions, expanded verified typed evidence, completeness-oriented selection and composition, controlled repeated trials, independent claim-level entailment review, and evaluation of actual selector-driven retries. This system is a research prototype, not a clinical decision-support tool.

## Reproducibility appendix

Use the exact final release and its integrity manifest. The archived experimental tables are the reported results, not future reruns. Commands:

    python export_final_thesis_132.py --verify presentation_20261118/thesis132/final_thesis_results_132/release_20261125T212811325446Z
    python -m unittest discover -s tests -p 'test_thesis132_release_local.py' -v
    python export_final_thesis_132.py

For a new run (not necessary to reproduce the archived tables), use the locked evidence, code and Ollama digest and execute:

    python run_b9_v2_132.py --allow-live --policy adaptive --top-k 5

The new run must not overwrite the historical experiment. Use the final_architecture_lock.json and manifest.json for exact provenance. Complete institutional front matter, related work citations, acknowledgements, bibliography, and supervisor review before submission.
