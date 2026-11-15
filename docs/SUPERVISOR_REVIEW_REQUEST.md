# Supervisor review request — M.Tech progress presentation

**Subject:** M.Tech progress review — expanded CDF-RAG benchmark, six verified baselines and proposed next experiments

Respected Ma'am,

I am sharing my progress on **Adaptive Multi-Round RAG Architecture for Reliable Domain-Specific Question Answering** ahead of the M.Tech progress presentation on November 18, 2026.

After the previous review, I expanded the experimental design beyond the earlier 22-query RL versus No-RL comparison. The repository now contains a 100-question medical QA pilot across eight query categories, a common experimental interface, six additional retrieval configurations (naive lexical, BM25, dense, hybrid, graph-neighborhood and causal KG), and the existing two CDF-RAG variants (without and with PPO). The code is organized in a common model registry and is covered by automated tests.

**Completed measured work:** six retrieval-only baselines have run on all 100 pilot questions (600 operations), with recorded timing, evidence-return counts, abstentions and errors. This is an operational pilot, not a claim of medically verified accuracy. The 100 questions still require independent adjudication, G0 is small (35 nodes and 37 edges), and the local 100-query end-to-end runs with the legacy RL/No-RL systems are pending. The earlier 22-query experiment showed that PPO raised mean latency from approximately 39.58 s to 50.09 s without improving the recorded retrieval coverage proxy, motivating performance-aware adaptive stopping.

I would appreciate your guidance on the following priorities: whether to first validate a gold benchmark subset and expand the knowledge graph with evidence provenance; whether all baselines should use a common source corpus before ranking model accuracy; and whether to prioritize adaptive evidence sufficiency/early stopping, feedback-loop implementation, or PPO reward redesign as the next novel contribution.

Please let me know any corrections or additional experiments you would like me to incorporate into the thesis work. I will use your feedback to prioritize the remaining implementation and comparative evaluation.

Thank you.

---

**Supporting materials:** [presentation slides source](PRESENTATION_SUPERVISOR_20261118.md); [repository](https://github.com/25cs06002/CDF-RAG-2/tree/research-v2); [measured six-baseline comparison](../research_artifacts/presentation_20261118/retrieval_only_100_v2_multihop.csv).
