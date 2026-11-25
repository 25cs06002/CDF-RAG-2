# Thesis claim-to-evidence checklist

Use this alongside the manuscript. All paths are relative to the repository root. The pinned release is `presentation_20261118/thesis132/final_thesis_results_132/release_20261125T212811325446Z`.

| Defensible claim | Evidence file / field | Qualification |
|---|---|---|
| 132 questions; 100+32 development cohort | data/benchmark/final_132_v1/README.md | No independent holdout |
| 8 baseline + 11 B9 runs; 2508 records | table_B1_B9_132.csv, table_B9_variants_132.csv | Counts executions, not independent samples |
| B9 99 substantive, 95 strict, 36 complete, 33 strict complete | table_B1_B9_132.csv and table_B9_variants_132.csv | AI reference, not gold accuracy |
| B9 zero AI Yes among 132 | table_B1_B9_132.csv, yes/accepted | Does not establish absence of hallucinations |
| B9 vs B8 5.331 vs 49.344 s; 0.1876 vs 0.0203 QPS | table_B1_B9_132.csv | Different generation/evidence; descriptive |
| Adaptive 169 vs 303 rounds, 100 vs 100 calls | table_round_policy_operational.csv | 44.22% fewer rounds, not faster latency |
| Full vs graph-off 99 vs 99 answers; 36 vs 43 complete | table_component_operational.csv + table_B9_variants_132.csv | Do not claim graph universally improves completeness |
| Document-off 15 vs full 99 answers | table_component_operational.csv | Graph-derived route retained across variants |
| Compact IDs mean 4.370 vs original 5.685 s | table_identifier_resource_profile.csv | Separate single runs; changed answers and initial memory state |
| Compact IDs 30 vs 36 complete | table_B9_variants_132.csv | Quality trade-off |
| Llama 3B compact 100 vs 27 vs 7 answers | table_generator_resource_profile.csv | Model/prompt-specific, no general leaderboard |
| Cache 2.98× retrieval microbenchmark | cache_feedback_diagnostics.json | Not end-to-end speedup |
| SRS 0.9821 from one observation | table_B1_B9_132.csv | Do not compare as equivalent to B7/B8 support |
| 24 negative-category inputs abstain | table_B9_by_query_type.csv | 17 appropriate abstentions by AI reference across cohort; not 100% validated refusal accuracy |
| Primary B9 original IDs | final_architecture_lock.json | Post hoc development-set selection |

## Do not report as established

- 100% medical accuracy, clinical validation, zero true hallucinations or independently human-approved gold.
- Gold retrieval recall, precision, hit@k or diagnostic sensitivity without independently adjudicated gold.
- Energy savings, exact peak model memory, significance or independent generalization without measurements.
- Proven latency gains from adaptive stopping relative to fixed3 or proven retry benefits.
- A new PPO-trained B9 policy; B9 is rule-based.
- All graph-off answers inferior or all multi-hop answers complete.
- A definitive novelty claim without checking published literature.

Before submission check the actual paper citations, institutional thesis template, supervisor's required chapter structure, and source access limitations in both AI review audits. Preserve original files and manifests.
