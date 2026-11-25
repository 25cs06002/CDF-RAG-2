# Research defense preparation (working notes)

## 90-second explanation

My M.Tech project addresses reliability and computational efficiency in domain-specific RAG, evaluated in medical question answering. Instead of running a fixed expensive retrieval-generation loop, I built a rule-based adaptive controller that routes a query to typed causal graph retrieval or BM25 document retrieval and executes at most three rounds. The local Llama model selects evidence identifiers rather than freely generating medical claims. A structural validator checks the selection and the answer renderer preserves source-attributed text, with qualified partial responses or abstention where necessary. I evaluated eight historical baselines and eleven versions or ablations of the proposed system on the same frozen 132-question development set. B9-v2 produced 99 substantive answers, 95 strict supported-and-grounded answers under the AI reference, and 36 answers rated complete. Compared with the historical PPO-based baseline, its mean latency was 5.33 versus 49.34 seconds and throughput was about 9.26 times higher. The controlled ablations also reveal that adaptive stopping saves retrieval rounds, not necessarily end-to-end time; document fallback is critical to coverage, and compact IDs reduce token and latency costs with a completeness trade-off. These are development-set results from a single AI judge, not independently verified clinical accuracy. Future work would address independent validation and completeness.

## Likely questions and defensible responses

1. **What is novel?** The integration of query-dependent bounded retrieval, typed candidate evidence, structured selection, deterministic rendering and explicit abstention, together with a reproducible local efficiency evaluation. The individual techniques are not claimed to be inventions.
2. **Why not PPO in your final model?** Historical PPO results motivated an inexpensive rule-based policy. B9 is not newly RL-trained. The main comparison also changes generation and evidence, so it cannot isolate the effect of removing PPO.
3. **What does the hallucination result mean?** Zero Yes labels under the AI protocol for B9; not zero actual hallucinations. Empty and partial answers can receive No. Use 95/132 strict yield and 33/132 strict complete yield as complementary metrics.
4. **Why do you claim reliability when only 36 are complete?** Factual support and completeness are separate. 53 partial and 10 nonresponsive responses show the cost of constrained rendering. The complete-answer count is an explicit limitation.
5. **Does adaptive retrieval improve latency?** It reduced total rounds from 303 to 169 against fixed3 but both made 100 LLM calls; fixed3 was faster. The large B8-to-B9 latency gap is a descriptive system-level comparison.
6. **What is the graph contribution?** Full and graph-off both answered 99; full retains typed path capability and has 95 vs 93 strict answers, but graph-off has 43 vs 36 complete answers. This is a trade-off, not dominance. Graph-off still uses graph-derived routing.
7. **Why choose Llama 3B?** With compact IDs it answered 100 vs 27 (Llama 1B) and 7 (Qwen 1.5B). Qwen's high QPS mostly reflects failure to produce valid answers under this selection protocol.
8. **Is the dataset independent?** No. All 132 questions influenced development. Family folds were defined but cannot retroactively create an independent test set. Gold references are unavailable.
9. **How is evidence validation performed?** Strict JSON, identifier membership/uniqueness, selection budget, comparison group coverage; render selected qualified source text. Structural validation is not independent entailment or a medical hallucination detector.
10. **Why are many multi-hop responses partial?** The typed ledger is small and incomplete. Eleven B9-v2 responses explicitly qualify incomplete chains; only one is tagged strict typed path. Never imply a complete chain without evidence.
11. **Why not compact IDs as the primary architecture?** They improve measured latency but reduce complete answers from 36 to 30. Original IDs were selected as a quality-oriented development-set choice, not an optimal solution.
12. **How can results be reproduced?** Use the archived release, its input/output hashes, code and Ollama digest, and the release verification command. Fresh timing can vary. All original runs are preserved.
13. **Is energy efficiency demonstrated?** No. We measured latency, throughput, tokens, sampled CPU/RAM and swap; energy consumption was not measured.
14. **What is your next research objective?** Independent clinical adjudication and a held-out evaluation, then improve answer completeness without losing strict support. Controlled repeated timing and retry exercises are additional priorities.

## Suggested presentation structure (not a finished slide deck)

1. Problem and motivation: reliability plus resource-constrained inference.
2. Research questions and contributions.
3. B9-v2 architecture, route table and selection/abstention mechanism.
4. Dataset and experimental setup: 132 development questions and eight baselines.
5. B1–B9 performance and AI-reference quality; show strict yield and complete count together.
6. Adaptive and component ablation: include negative findings.
7. Compact identifiers and generator trade-off.
8. Conclusions, limitations and future work.

Use the final release's figures and exact metrics. If presentation time is short, combine sections 1–2, 4–5 and 6–7. Adapt timing and formatting to the supervisor's actual instructions.
