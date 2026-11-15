# B1–B8 Model Registry

Use `python3 -m research_v2.pipelines` to inspect all eight models. Use `from research_v2.pipelines import create_pipeline` to instantiate B1–B8. The implementation files remain in their original tested locations to avoid breaking thesis history.

| ID | Method | Implementation |
| --- | --- | --- |
| B1 | Naive lexical overlap RAG | `research_v2/presentation_baselines.py` |
| B2 | BM25 RAG | Same |
| B3 | Dense RAG (local SentenceTransformer or explicit LSA for offline CI) | Same |
| B4 | Hybrid BM25 + Dense via RRF | Same |
| B5 | Undirected neighborhood GraphRAG | Same |
| B6 | Directed causal KG-RAG | Same |
| B7 | Original CDF-RAG No-RL | `research_v2/legacy_no_rl.py` |
| B8 | Original CDF-RAG PPO RL | `research_v2/legacy_rl.py` |

The common interface is `research_v2/base_pipeline.py` (`PipelineRequest`, `BaseRAGPipeline`, `PipelineRecord`), while `research_v2/presentation_matrix.py` handles batch execution and reports.

Source-summary and graph baselines currently use different evidence, so this is exploratory operational comparison, not controlled architecture-only medical accuracy evaluation. B7–B8 require local Neo4j/Ollama, and B8 needs the original PPO checkpoint.
