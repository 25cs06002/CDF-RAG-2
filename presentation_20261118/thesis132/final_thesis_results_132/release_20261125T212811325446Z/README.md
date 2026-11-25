# Final thesis experiment export

Primary: B9-v2 adaptive, original evidence IDs. Optional speed variant: Llama 3B compact IDs.

All tables derive from archived development experiments. No model inference or new labels. Empty metric cells mean unavailable, not zero. Operational ablation tables are historical exports; use table_B9_variants_132.csv for completed variant quality labels. Strict complete yield includes AI completeness and has no B1-B8 equivalent.

## Limitations

- Fixed development-exposed cohort; no independent held-out clinical evaluation or independently human-adjudicated gold.
- AI-reference judgments are single-judge, with limited source access and reused annotations; zero observed Yes labels do not prove zero hallucinations.
- Strict supported-and-grounded yield includes incomplete or irrelevant answers. Completeness is a distinct B9-only AI judgment.
- Graph/corpus and answer generation differ between B1-B8 and B9; descriptive comparisons, not causal attribution to any one component.
- Semantic similarities are cosine proxies, with varying text bases and observation counts; they are not medical entailment or retrieval recall.
- All-query latency includes abstentions; compare answered-only and common-answered subsets too. Single runs have uncontrolled OS/cache/thermal conditions.
- Adaptive reduces retrieval rounds, not necessarily LLM calls or end-to-end latency. Selector-driven retry was not exercised in the reference run.
- Resource samples describe system usage and sampled RSS, not energy or exclusive model memory. Cross-model tokens depend on tokenization.
- Primary configuration selected after observing development results. Compact-ID variant is faster but has fewer complete answers; graph-off has more complete answers.
