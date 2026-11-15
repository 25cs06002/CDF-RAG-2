# CDF-RAG-2 — Medical Causal QA Research

**M.Tech project:** Adaptive Multi-Round RAG Architecture for Reliable Domain-Specific Question Answering.

> **Active branch:** `research-v2`. The presentation sprint targets November 18, 2026. This repository extends the upstream [CDF-RAG study](https://arxiv.org/abs/2504.12560) by Khatibi, Wang and Rahmani; their original repository description and credit are preserved in [docs/ORIGINAL_CDF_RAG_README.md](docs/ORIGINAL_CDF_RAG_README.md).

## Start here

The one supported entry point for presentation experiments is:

```bash
cd ~/Folder/CDF-RAG-2
# Use your already-configured virtual environment before running Python commands.
python3 evaluate_all_baselines.py prepare
python3 evaluate_all_baselines.py validate
python3 evaluate_all_baselines.py smoke --encoder lsa
# Full local B1-B8 run: local Ollama, Neo4j and PPO checkpoint are required.
python3 evaluate_all_baselines.py run --models all
python3 evaluate_all_baselines.py report
```

`prepare` freezes a 100-question inference-only set, source-candidate document summaries, and G0 structural reference labels under `presentation_20261118/`. It does not produce medical gold labels. The `run` command executes the six new baseline configurations and the two original legacy adapters, sequentially. It saves **new** run directories (no overwriting historical results). Do not run the local LLM batches concurrently on the same machine.

For a reproducible **retrieval-only** smoke/scale check that does not use local Ollama:

```bash
python3 evaluate_all_baselines.py run --models B1,B2,B3,B4,B5,B6 --retrieval-only --encoder lsa --limit 100
python3 evaluate_all_baselines.py report --allow-partial
```

These runs are **not** equivalent to end-to-end answer generation; the local `sentence` backend uses the existing cached sentence transformer, whereas `lsa` is a separate offline TF-IDF + LSA vector-space baseline.

## Where the code lives

| Need | Location |
| --- | --- |
| **One-command orchestration** | [evaluate_all_baselines.py](evaluate_all_baselines.py) |
| **Model registry** | [research_v2/pipelines/](research_v2/pipelines/) |
| Naive, BM25, Dense, Hybrid, Graph, Causal KG retrievers | [research_v2/presentation_baselines.py](research_v2/presentation_baselines.py) |
| Legacy No-RL and PPO RL adapters | [research_v2/legacy_no_rl.py](research_v2/legacy_no_rl.py), [research_v2/legacy_rl.py](research_v2/legacy_rl.py) |
| **Batch execution and metrics** | [research_v2/presentation_matrix.py](research_v2/presentation_matrix.py) |
| **100-question pilot** | [data/benchmark/medical_causal_pilot100_draft_v1.jsonl](data/benchmark/medical_causal_pilot100_draft_v1.jsonl) |
| Corpus preparation (Phase 2 candidate summaries) | [research_v2/presentation_corpus.py](research_v2/presentation_corpus.py) |
| G0 structural scoring | [research_v2/presentation_g0_reference.py](research_v2/presentation_g0_reference.py) |
| Immutable G0 | [research_artifacts/graph-G0/](research_artifacts/graph-G0/) |
| Phase 2 reviewer handoff and checks | [research_artifacts/experiment-artifacts/phase2/](research_artifacts/experiment-artifacts/phase2/) |
| Legacy 22-query CSVs and earlier scripts | Retained at repository root for traceability |
| CI regression + 600 retrieval-only queries | [GitHub Actions workflow](.github/workflows/presentation-offline-tests.yml) |
| **Supervisor review and presentation** | [docs/PRESENTATION_SUPERVISOR_20261118.md](docs/PRESENTATION_SUPERVISOR_20261118.md) |
| Full repository organization and cleanup policy | [docs/REPOSITORY_MAP.md](docs/REPOSITORY_MAP.md) |

## B1–B8 comparison inventory

| ID | Method | Data/evidence | Status |
| --- | --- | --- | --- |
| B1 | Naive lexical overlap RAG | Phase 2 candidate-source summaries | 100-query retrieval-only run verified |
| B2 | BM25 RAG | Same candidate summaries | 100-query retrieval-only run verified |
| B3 | Dense vector RAG | Candidate summaries; offline LSA encoder | 100-query retrieval-only run verified |
| B4 | Hybrid BM25 + Dense RRF | Same candidate summaries | 100-query retrieval-only run verified |
| B5 | Undirected graph-neighborhood RAG | Archived G0 graph | 100-query retrieval-only run verified |
| B6 | Directed causal KG RAG | Archived G0 graph | 100-query retrieval-only run verified |
| B7 | CDF-RAG No-RL | Existing Neo4j / Ollama | **100-query local run pending** |
| B8 | CDF-RAG PPO RL | Existing Neo4j / Ollama / PPO | **100-query local run pending** |

Six-model results: [research_artifacts/presentation_20261118/retrieval_only_100_v1.csv](research_artifacts/presentation_20261118/retrieval_only_100_v1.csv), from [this GitHub Actions run](https://github.com/25cs06002/CDF-RAG-2/actions/runs/43716390027).

**The preliminary evidence is not clinical ground truth.** In particular, the 100-question set is unreviewed, B1–B4 use AI-assisted source-summary candidates, and B5–B8 use the limited G0 knowledge graph. As a result, comparing evidence-return rates measures baseline behavior and corpus coverage, **not medical answer accuracy**. Scores for the original 22-question RL/No-RL experiments remain a distinct earlier experiment.

## Publication and reproducibility

Before pushing local run outputs, review credentials, private patient data and full third-party copyrighted text. Preserve historical files and signed-off review decisions. Do not label graph-agreement F1 or embedding similarity as externally verified clinical accuracy.

See [PRESENTATION_SPRINT_20261118.md](PRESENTATION_SPRINT_20261118.md) for full deployment and [docs/REPOSITORY_MAP.md](docs/REPOSITORY_MAP.md) for code locations.
