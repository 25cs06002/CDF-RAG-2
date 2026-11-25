# Chapter 2 — Background and Related Work

This is a focused narrative review for the B9-v2 medical QA research prototype, not a systematic literature review. The citation keys in brackets correspond to [references.bib](references.bib). Published results below are reported by the cited authors on their own tasks and are not measurements of the local B1–B9 systems. Version and access distinctions are recorded in [LITERATURE_SOURCE_AUDIT.md](LITERATURE_SOURCE_AUDIT.md).

## 2.1 Retrieval-augmented generation

Retrieval-augmented generation combines a generator's parametric knowledge with external evidence. Lewis et al. [lewis2020rag] formalized this approach for knowledge-intensive NLP and compared it with parametric and retrieve-and-extract methods. In a conventional retrieve-then-generate pipeline, the evidence is selected before response generation. While this gives the generator access to external information, it does not enforce that every generated assertion is entailed by the retrieved material.

Lexical retrieval uses term-level matching. BM25, grounded in probabilistic information retrieval, incorporates term frequency, inverse document frequency and document-length normalization [robertson2009bm25]. Its advantages include simple indexing and transparent scoring. Its weakness is sensitivity to vocabulary mismatch. Sentence embeddings enable semantic matching across differently worded passages; Sentence-BERT [reimers2019sbert] provides a practical embedding approach for cosine similarity. Dense retrieval can retrieve related concepts without exact term overlap, but semantic similarity alone is not evidence of causal direction or factual entailment. Hybrid retrieval combines these signals to mitigate weaknesses of either in isolation. These distinctions motivate the local B1–B4 baselines and the use of BM25 document fallback in B9.

## 2.2 Graph-based and causal retrieval

Graph structure allows information to be connected through entities and relations rather than only through independent passages. The GraphRAG work of Edge et al. [edge2024graphrag] addresses corpus-level summarization using graph-based organization. The local graph-neighborhood baseline B5 is a simpler graph retrieval implementation, not a reproduction of the complete published GraphRAG system.

Causal question answering further requires distinguishing causal claims from risk factors, triggers and associations. A directed edge is useful as an explicit representation of an asserted relationship; it does not prove causality. Combining graph traversal with source-linked text allows intermediate relationships to remain inspectable. In the B9-v2 design, typed candidate links are retrieved with relation and source information, and incomplete chains are explicitly qualified. A directed two-link path is not treated as proof of an inevitable clinical outcome. The candidate graph is not independently clinically adjudicated.

## 2.3 Adaptive and multi-round retrieval

FLARE (Jiang et al.) [jiang2023flare] uses predicted upcoming text and generation uncertainty to trigger additional retrieval. It is useful when information needs emerge during generation. Self-RAG (Asai et al.) [asai2024selfrag] learns retrieval and critique behavior using self-reflection rather than assuming all requests need the same retrieved context. Both demonstrate that retrieval decisions may be conditional on the query or generation state.

Adaptive-RAG (Jeong et al.) [jeong2024adaptive] instead learns a classifier that routes questions among no-retrieval, single-step and iterative strategies. Its timing analysis (Table 3 of the arXiv version) reports 0.35, 3.08 and 27.18 seconds per query in those respective predicted categories, with 8.60%, 53.33% and 38.07% of requests in each category. These figures motivate conditional retrieval but cannot be compared directly with local B9 timings because models, data and execution conditions differ.

The present architecture uses rules and evidence selection rather than training a complexity classifier or a reflection-token generator. It limits retrieval to three scheduled rounds and stops after a valid nonempty selection covering required targets. Its adaptive behavior is internal to a single input question, not a patient-facing multi-turn diagnostic dialogue. The actual frozen run shows route-dependent stopping; selector-triggered retries were available but not exercised.

## 2.4 Principal medical and causal RAG studies

### 2.4.1 MRD-RAG: Multi-round diagnostic retrieval

Chen et al. [chen2025mrd] organize disease information into disease-information trees (DI-Trees) for 746 modern-medicine diseases and 130 traditional Chinese medicine diseases. The retriever identifies potential diseases using dense indexes, including a pseudo-medical-history representation; the analyzer examines candidate diseases and their distinguishing features, and the doctor module decides whether to inquire further or give a diagnosis. The framework can skip the retrieval/analyzer stages if the latest dialogue turn does not warrant retrieval. The experiment uses four diagnostic datasets, a maximum of three dialogue rounds and a top-five disease retrieval setting. Its GPT evaluation and physician pairwise assessment investigate diagnostic-dialogue quality.

In Table 3, MRD-RAG-PMH with Qwen2-7B-Instruct receives an average GPT score of 3.76/5, compared with 3.26/5 for the corresponding single-round RAG baseline. This is a diagnostic score, not a percentage of correct answers. The inspected preprint did not provide a comparable end-to-end latency or throughput measurement. Its dialogue and disease-tree design is relevant to B9, but B9 answers standalone questions and is not a clinical diagnosis system.

### 2.4.2 GARMLE-G: Guideline-grounded retrieval

Li et al. [li2025garmle] propose generation-augmented retrieval that enriches a medical model's diagnostic output with electronic health record context and retrieves clinical practice guideline passages. The June 2025 preprint evaluates a hypertension prototype. Section IV-E reports Precision@1 of 0.940, MRR of 0.964 and Hits@3 of 1.0 under a BERTScore matching threshold of 0.72. The paper describes a small BERT-based diagnosis component, two embedding models with a combined 568 million parameters, and a vector database. This is architectural evidence of compactness, not measured runtime RAM or latency. The indexed journal abstract also mentions coronary heart disease; the numerical results quoted here belong to the inspected hypertension preprint, not the later journal experiment [li2025garmlejournal].

The idea of returning selected source content rather than unrestricted prose motivates B9's constrained renderer. However, correct retrieval and clinically relevant guideline interpretation remain distinct. The claim of hallucination-free output in the source paper should not be generalized to every retrieval-only system. B9 uses provisional source summaries rather than a clinically certified guideline corpus.

### 2.4.3 CDF-RAG: Causal dynamic feedback

Khatibi et al. [khatibi2025cdf] combine reinforcement-learned query refinement, semantic retrieval, directed causal graph traversal, rewriting and response verification. PPO selects query refinement actions using downstream reward signals. The original paper evaluates CosmosQA, AdversarialQA, MedQA and MedMCQA. Table 1 reports MedMCQA accuracy of 0.90 for CDF-RAG with LLaMA 3-8B and 0.94 with GPT-4. These are the authors' benchmark measurements, not local medical QA gold accuracy. The limitations section identifies dependence on structured graph availability, overhead of GPT-based verification and uncertainty about generalization to varied queries.

This motivates the local B7/B8 implementations and the B9 rule-based alternative. In the present study, B9 avoids free-form generation by selecting source identifiers, not by reimplementing the published CDF-RAG verifier. The source paper's inspected version does not establish a directly comparable end-to-end latency or throughput figure. Consequently the local B8 latency of 49.344 seconds is reported only as an independently recorded local baseline, never as the original authors' timing.

## 2.5 Inference efficiency

A central systems issue is that multiple retrieval stages can repeatedly present overlapping context to a generator. Ou et al. [ou2025accelerating] target redundant processing in adaptive RAG by using cached representations and parallel generation. Their experiments report mean acceleration factors of 2.79 for prefilling and 2.33 for decoding while retaining reported generation quality. These are stage-level acceleration measurements in that paper, not a promise of equivalent end-to-end speedup on CPU.

LLMLingua [jiang2023llmlingua] uses token-level prompt compression to reduce inference cost while retaining important information. B9's compact-ID experiment is a much narrower alternative: the evidence text is preserved and only identifier strings are replaced with short aliases, then restored before rendering. The local experiment shows reduced output token counts and measured latency but fewer complete responses; it is not a replication of LLMLingua. Likewise the exact-query retrieval cache is not KV caching or a reuse of model hidden states.

For the frozen B9 reference, mean observed retrieval time is 0.473 ms per query, while mean observed generation time is 7035.221 ms across 100 generation observations. These are differently conditioned averages. Reducing retrieval rounds from 303 to 169 did not reduce the number of LLM calls (100 for both fixed3 and adaptive), and fixed3 had lower measured mean latency. This illustrates why component-level timing, wall-clock throughput and quality-adjusted yield must be considered separately.

## 2.6 Comparative summary and research gap

| Study | Main strategy | Reported result or evaluation focus | Efficiency evidence in inspected source | Relation to B9 |
|---|---|---|---|---|
| MRD-RAG (2025 preprint) | Disease trees and diagnostic dialogue | GPT score 3.76 vs 3.26, Qwen2-7B | Conditional retrieval bypass; comparable wall time/QPS not located | Bounded rounds and medical evidence, but not dialogue diagnosis |
| GARMLE-G (2025 preprint) | EHR-enriched guideline retrieval | P@1 0.940; MRR 0.964; semantic matching | Lightweight components; comparable wall time/QPS not located | Constrained evidence output, not EHR diagnosis |
| CDF-RAG (2025 preprint) | PPO, causal graph, rewriting and verification | MedMCQA 0.90 with LLaMA 3-8B | Verification overhead identified; comparable wall time/QPS not located | Historical local B7/B8; B9 changes policy and generation |
| Adaptive-RAG (NAACL 2024) | Learned complexity routing | Open-domain QA quality and efficiency | 0.35 / 3.08 / 27.18 s per predicted route | Rule-based medical routes and candidate selection |
| FLARE (EMNLP 2023) | Generation-driven retrieval | Uncertainty-triggered retrieval | No numerical efficiency claim extracted here | Different stopping signal |
| Self-RAG (ICLR 2024) | Learned retrieval and self-critique | QA and factuality | No numerical efficiency claim extracted here | B9 uses structural selection checks instead |
| Retrieval-overlap acceleration (ACL Findings 2025) | Representation reuse, parallel generation | 2.79x prefill; 2.33x decode | Stage-level reported acceleration | Local cache and identifier compression differ |
| LLMLingua (EMNLP 2023) | Prompt token compression | Inference compression and quality | No numerical claim extracted here | B9 shortens IDs, not evidence text |

These works already establish adaptive retrieval, graph-based reasoning, evidence selection and inference acceleration as active research directions. The gap targeted by this thesis is narrower: the integration and measurement of a small local, rule-based bounded controller that combines typed candidate evidence, document fallback, constrained selection, explicit partial-response qualification and abstention, with per-query quality and computational telemetry. The thesis does not claim that no prior system combines these ideas. The empirical research questions concern how the particular implementation behaves and which components account for observed trade-offs. An exhaustive novelty claim would require a broader systematic review.

## 2.7 Evaluation and reporting

Medical QA reliability requires independent evaluation of factual support, answer completeness, uncertainty and relevance. TRIPOD-LLM [gallifant2025tripod] emphasizes transparent reporting and human oversight in biomedical LLM studies. Accordingly the thesis reports dataset exposure, evidence provenance, model configuration and evaluation limitations. AI-reference claim support is not an independently established clinical reference standard. A valid citation ID or high semantic cosine score does not prove entailment. The existing 132-question cohort is development-exposed; held-out clinical adjudication remains future work.

The following methodology chapter describes the local implementations and archived experiments rather than interpreting published benchmark scores as directly comparable measurements.
