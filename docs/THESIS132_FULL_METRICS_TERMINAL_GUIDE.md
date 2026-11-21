# Full 132-question screenshot-style evaluation workflow (B1–B8)

This is a **new** analysis of the successful 1,056 model-query records,
not a rerun of the LLM or Neo4j. It supplements the earlier
\`evaluate_metrics_132.py\` summary and does not modify the 22-question
\`evaluate_metrics_rl.py\` / \`evaluate_metrics_no_rl.py\`.

Files created in repo root:
- \`evaluate_all_metrics_132.py\` — create new, SHA-verified per-pipeline
  \`Metric,Value,Unit,Status,Notes\` summary and detailed metric CSVs.
- \`show_evaluation_132.py\` — screenshot-style quality, retrieval,
  performance, query refinement, and classifier sections.
- \`compare_all_models_132.py\` — eight-model table and B8-minus-B7 table,
  based on the same 132 inputs.
- \`compare_hallucination_132.py\` — hallucination detector report,
  confusion matrix and F1 **only if** reviewed output-specific gold exists.
- \`prepare_hallucination_reviews_132.py\` — create 1,056 **unfilled**
  output-level reviewer slots, not fabricated medical labels.
- \`research_v2/thesis132_full_metrics.py\` — verified calculation engine.
- \`tests/test_thesis132_full_metrics_local.py\` — offline regression tests
  on the real committed 132-query dataset.

## One-by-one terminal commands (Ubuntu)

\`\`\`bash
cd ~/Folder/CDF-RAG-2-thesis132
source ../CDF-RAG-2/venv/bin/activate
export SENTENCE_TRANSFORMERS_HOME="$HOME/.cache/cdf_rag_g0_models"

# 1. Validate calculation code against the existing 1,056 records
python -m unittest discover -s tests -p 'test_thesis132_full_metrics_local.py' -v

# 2. Create all eight full reports (and compute three embedding-similarity PROXIES)
python evaluate_all_metrics_132.py --semantic-proxies

# Alternative quick run (no local embedding model needed; similarity fields empty):
# python evaluate_all_metrics_132.py

# 3. Point to the newest immutable analysis folder. Optionally replace
#    this with the exact path printed by the previous command.
BASE="presentation_20261118/thesis132/csv_reports/batch132_csv_20261121T061653054773Z_bfeb15"
REPORT=$(find "$BASE" -maxdepth 1 -mindepth 1 -type d \
  -name 'full_metrics_132_*' | sort | tail -1)
echo "$REPORT"

# 4. Screenshot-style individual summaries: B1/B2/.../B8
python show_evaluation_132.py --report-dir "$REPORT" --model B1
python show_evaluation_132.py --report-dir "$REPORT" --model B2
python show_evaluation_132.py --report-dir "$REPORT" --model B3
python show_evaluation_132.py --report-dir "$REPORT" --model B4
python show_evaluation_132.py --report-dir "$REPORT" --model B5
python show_evaluation_132.py --report-dir "$REPORT" --model B6
python show_evaluation_132.py --report-dir "$REPORT" --model B7
python show_evaluation_132.py --report-dir "$REPORT" --model B8

# 5. All eight detailed reports in one terminal run
python show_evaluation_132.py --report-dir "$REPORT" --model all

# 6. All-metric B1–B8 comparison, also B7 versus B8
python compare_all_models_132.py --report-dir "$REPORT"

# 7. Hallucination report (predicted detector flags; no invented gold)
python compare_hallucination_132.py --report-dir "$REPORT" --model B7
python compare_hallucination_132.py --report-dir "$REPORT" --model B8
python compare_hallucination_132.py --report-dir "$REPORT" --model all
\`\`\`

No inference (Ollama, Neo4j, PPO) is launched during these commands.
\`--semantic-proxies\` only runs the pinned local sentence encoder.

## Where the CSV files are saved

Every invocation creates a **new** directory:

\`presentation_20261118/thesis132/csv_reports/batch132_csv_20261121T061653054773Z_bfeb15/full_metrics_132_<UTC_TIMESTAMP>_<ID>/\`

Example contents:

\`\`\`text
evaluation_metrics_B1_132_full.csv ... evaluation_metrics_B8_132_full.csv
evaluation_summary_B1_132_full.csv ... evaluation_summary_B8_132_full.csv
hallucination_predictions_B1_132.csv ... hallucination_predictions_B8_132.csv
comparison_all_models_132_full.csv
comparison_B7_vs_B8_paired_132_full.csv
hallucination_classification_summary_132.csv
full_metrics_manifest.json
\`\`\`

If independently approved gold exists for a model, also creates:
\`confusion_matrix_B7_132.csv\` or equivalent.

The source CSVs and all 22-query legacy results are never overwritten.
Outputs carry a SHA-256 manifest; report display scripts reject changes.

## Definitions and *important* limitations

- Causal Retrieval Coverage = fraction with any recorded causal edge/path.
  **Not applicable for B1–B4** because those retrieve documents, not graph
  causal edges. B5/B6 abstentions are counted in the denominator.
- Causal Chain Depth = mean path *node* count for queries with paths, as a
  structural counterpart to the legacy screenshot. Not validated medicine.
- SRS = cosine(query, refined query). A score of 1.0 for unchanged queries
  is *similarity*, not a 100% improvement or success rate.
- Context Relevance cosine = query versus rewritten knowledge if present,
  otherwise versus retrieved evidence text.
- Groundedness cosine = rewritten knowledge if available, otherwise generated
  answer, versus retrieved evidence. This is **not factual correctness,
  claim entailment, or clinical groundedness**. Basis varies by model; the
  per-query CSV records the basis. For B1–B4 retrieved text comes from the
  provisional source-summary candidate corpus, while B5–B8 use **unverified
  G0 graph verbalizations**. Thus do **not** directly rank clinical quality
  using these proxy scores.
- Hallucination Rate = fraction with hallucination among 132 independently
  approved output-level labels; otherwise *N/A*. Automated B7/B8 detector
  flags are separately reported as \`Detector Flag Rate (Not Gold)\`.
  B1–B6 have no detector predictions, so their detector precision/F1 cannot
  be computed from saved records even if you provide gold.
- Precision, Recall, F1, Accuracy and confusion matrices are computed only
  for aligned, approved output-level labels. If only part of 132 is approved,
  the classification explicitly says **partial_review_subset_not_full_132**.
  A partial score must never be reported as full-dataset performance.
- True \`Refinement Success Rate\` requires reviewed refinement-quality labels
  and is N/A. \`Recorded Query Change Rate\` is a different measured field.
- \`Average Throughput\` = 132 queries / recorded model wall seconds.
  \`Minimum/Maximum Throughput (QPS, latency proxy)\` = min/max 1/query latency,
  and must **not** be represented as measured run throughput.
- B5/B6 abstentions make unconditional mean latency misleading; compare
  \`Mean Latency if Answered\` separately.
- The fixed 132 queries were used in earlier G1 development and do not form
  an independent medical test set.

## Optional independently adjudicated hallucination classification

\`\`\`bash
python prepare_hallucination_reviews_132.py

# Complete the reviewer worksheet OUTSIDE code; leave undecided cases pending.
# Never fabricate reviewed labels or use an LLM-only judgment as clinician gold.
# After independently verifying output-specific labels and source evidence:
python evaluate_all_metrics_132.py --semantic-proxies \
  --gold-csv /path/inside/repo/to/independently_approved_gold.csv
\`\`\`

Approved gold rows require: \`model\`, \`query_id\`, \`is_hallucination_gold\`
(Yes/No), \`review_status=approved\`, \`reviewer_id\`,
\`reviewer_qualification\`, \`independence_declaration=Yes\`, and
\`verified_source_evidence\`. Pending, excluded, or uncertain rows are
not counted. No label is filled automatically.

The old 22-query hallucination confusion-matrix result cannot be directly
ported to the new 132-question model outputs without separate assessment.
