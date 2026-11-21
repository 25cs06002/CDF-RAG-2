# Fixed 132-query B1–B8 CSV evaluation

The original `evaluate_batch_queries_no_rl.py` and
`evaluate_batch_queries_rl.py` hard-code 22 queries. **Do not modify those
historical scripts or overwrite their CSVs.**

Use these new entry points in `CDF-RAG-2-thesis132`:

- `evaluate_batch_queries_132.py`: runs existing B1–B8 pipeline adapters
  sequentially on the exact frozen 132 inputs and exports results.
- `export_batch_metrics_132.py`: exports or merges existing completed
  `research_v2.thesis132` sessions without executing models again.
- `research_v2/thesis132_csv_export.py`: audited CSV conversion, per-record
  metrics, pipeline summaries and 132-query comparison matrix.

## Commands

```bash
cd "$HOME/Folder/CDF-RAG-2-thesis132"
source ../CDF-RAG-2/venv/bin/activate
export SENTENCE_TRANSFORMERS_HOME="$HOME/.cache/cdf_rag_g0_models"

python -m research_v2.thesis132_dataset_release
python -m unittest discover -s tests -p 'test_thesis132_csv_export_local.py' -v

# 132 questions for B1-B6, retrieval-only:
python evaluate_batch_queries_132.py --models B1,B2,B3,B4,B5,B6 --mode retrieval_only --encoder sentence --top-k 5

# All eight with generator, Neo4j G0 and PPO (after BOTH G0 preflights pass):
python evaluate_batch_queries_132.py --models all --mode end_to_end --encoder sentence --top-k 5 --allow-live

# Re-export a previous complete 132-session without rerunning:
python export_batch_metrics_132.py --session-dir presentation_20261118/thesis132/runs/batch_YOUR_ID --require-all --fail-on-errors
```

For a single-query runtime smoke, continue to use
`python -m research_v2.thesis132 run --models B7,B8 --mode end_to_end --limit 1 --top-k 5 --allow-live`.
**This CSV exporter rejects smoke/partial-question runs**: every exported model
must have 132 records with the original IDs, wording, order and frozen hashes.

The runner writes the original `records.jsonl`, `manifest.json`,
`summary.json` and the session provenance first. The CSV exporter then
creates a new, unique subdirectory under
`presentation_20261118/thesis132/csv_reports/batch132_csv_*/`:

```text
B1_queries_132.csv
B2_queries_132.csv
B3_queries_132.csv
B4_queries_132.csv
B5_queries_132.csv
B6_queries_132.csv
B7_queries_132.csv
B8_queries_132.csv
pipeline_summary_132.csv
all_models_per_query_132.csv
comparison_by_query_132.csv
export_manifest.json
```

Only models with actual complete 132-query records receive CSVs; the exporter
**does not create fake empty B7/B8 files** if live preflight fails. In that case,
the earlier real CSVs remain and the CLI exits nonzero. To merge separately
completed runs later, repeat `--session-dir` for each session. Each must have
the same query hash, full 132 count, mode, encoder, top-k, code commit, G0 and
document corpus; otherwise the exporter rejects the merge.

Each B-model CSV includes query ID/text/type/domain/fold, original/refined
query, matched concept, retrieved source/document IDs, JSON-serialized
causal edges and paths, answer, rewrites, action history, detector/validator
flags, reported feedback rounds, stage timings, measured per-query latency,
and *reciprocal-latency throughput proxy*. `pipeline_summary_132.csv`
records actual run throughput as 132 / measured run wall seconds, p50 and
p95 latency, evidence yield, answer presence and errors.

**Interpretation safeguards:** `retrieval_success` means a candidate document
or structural evidence was returned, NOT clinically relevant evidence;
`is_hallucination` is a legacy detector's flag or blank if unavailable, not a
measured hallucination ground truth. True medically verified accuracy,
clinical groundedness, semantic refinement score, and context relevance
require an independent reference/evaluator and are **not invented**. B1–B4
use provisional documents while B5–B8 use G0: cross-family evidence yield
comparisons are descriptive, not controlled evidence-accuracy comparisons.
B7/B8 adapter feedback counts can underestimate actual historical PPO steps.
The fixed 132 questions influenced G1 previously and are development data.

Keep each CSV's `export_manifest.json` (SHA256 fingerprints and exact source
session location) for your thesis analysis and supervisor demonstration.
