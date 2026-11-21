# B1–B8: reproducible 132-query evaluation metrics and comparisons

Root-level scripts created for the user's original 22-query workflow:

- `evaluate_metrics_132.py` — reads verified, existing `B1_queries_132.csv` through `B8_queries_132.csv` **without rerunning models**, calculates per-query operational metrics and per-model summaries in an old-script-compatible `Metric,Value` format.
- `compare_evaluation_132.py` — reviews comparison tables, validates their SHA hashes, prints all eight pipelines and B7/B8 paired differences. Can generate a fresh analysis if passed `--input-dir`.
- `research_v2/thesis132_evaluation.py` — shared validated analysis and CSV writer.
- `tests/test_thesis132_evaluation_local.py` — actual committed CSV integrity tests, output checks, and tamper rejection.

## Run on Ubuntu

```bash
cd ~/Folder/CDF-RAG-2-thesis132
source ../CDF-RAG-2/venv/bin/activate
git fetch origin work/thesis132-g1-target-sufficiency
# after resolving local branch and staged changes:
# git pull --ff-only origin work/thesis132-g1-target-sufficiency

python evaluate_metrics_132.py \
  --input-dir presentation_20261118/thesis132/csv_reports/batch132_csv_20261121T061653054773Z_bfeb15

# Use the evaluation_132_* folder printed above:
python compare_evaluation_132.py --analysis-dir \
  presentation_20261118/thesis132/csv_reports/batch132_csv_20261121T061653054773Z_bfeb15/evaluation_132_YOUR_TIMESTAMP

# Or one command to generate fresh analysis plus print comparison:
python compare_evaluation_132.py --input-dir \
  presentation_20261118/thesis132/csv_reports/batch132_csv_20261121T061653054773Z_bfeb15
```

The scripts reject changed CSVs by checking original export_manifest.json checksums,
verified 132 question order/text, all eight model identities and the original pipeline
summary counts. No live Neo4j, Ollama or PPO dependency.

## Output names

Created in new, non-overwriting `batch132_csv_*/evaluation_132_*/` folder:

```text
evaluation_metrics_B1_132.csv ... evaluation_metrics_B8_132.csv  (8 files)
evaluation_summary_B1_132.csv ... evaluation_summary_B8_132.csv  (8 files)
evaluation_summary_all_models_132.csv
comparison_metrics_B1_B8_132.csv
comparison_by_query_type_132.csv
comparison_by_medical_domain_132.csv
comparison_B7_vs_B8_paired_132.csv
B7_B8_evidence_changed_questions_132.csv
analysis_manifest.json
```

`evaluation_summary_Bx_132.csv` preserves the root-level legacy headers
`Metric,Value` and adds `Unit,Notes`. Missing ground-truth-based metrics (Semantic
Refinement Score, Context Relevance, Groundedness, Hallucination Rate and
Refinement Success Rate) are **blank**, NOT zero or made up. Distinct labeled
measures such as `Detector Flag Rate (Not Gold)` and candidate evidence yield
are available. A candidate returned, a final answer generated or validator
PASS does not imply correctness.

`evaluation_summary_all_models_132.csv` and
`comparison_metrics_B1_B8_132.csv` compare latency, P95, measured throughput,
coverage, answer presence, abstentions, causal edges/paths and detector reporting.
Both all-query and status=OK latency are included: B5/B6 abstentions produce
very low latencies that must not be compared naively to always-answering systems.

`comparison_B7_vs_B8_paired_132.csv` aligns exactly the same query_id and text
for both pipelines and shows B8-only gains, B7-only losses, both/neither,
matched concept, delta per-question latency, direct causes and path counts.
`B7_B8_evidence_changed_questions_132.csv` isolates changed-evidence cases
for manual expert review. The paired summary in `analysis_manifest.json` also
reports the signed net improvement and an *exploratory* exact discordant-pair
test; it is not proof that PPO alone caused the change.

The question set is historical G1-exposed **development data**, not external
medical gold. Source candidate summaries (B1–B4) and G0 (B5–B8) are not the
same evidence base. Automated detector flags are not medical hallucination
labels. The recorded feedback_iterations for B7/B8 are zero by their adapter
implementation and cannot establish the actual number of PPO refinements.

The evaluation scripts **never modify** the original 22-query root CSVs or
the source batch132 export, and produce unique versioned folders for each run.
