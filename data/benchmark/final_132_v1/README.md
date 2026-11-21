# THESIS132-FINAL-QUERY-INPUT-V1

**Finalized as a deterministic, 132-row source-screened DEVELOPMENT input for every existing thesis pipeline. Not independently clinically adjudicated gold.**
Prepared 2026-11-21; no questions, IDs, or historical results were changed.

## Canonical files

| File | Intended consumer |
| --- | --- |
| `questions_only.jsonl` | **All B1–B9 inference**: exactly `query_id`, `query`; SHA-256 `a45fcddfc153c0f9c2211c8167e001b03513216bf9191f2f08a4444d52df6c76` |
| `questions_only.csv` | Exact same ID/question records for inspection/import; NEVER use alongside JSONL to double-count |
| `metadata_review_only.csv` | **EVALUATOR ONLY**: original cohort/type/domain, tentative answerability, semantic group, five-fold allocation and screened source references |
| `manifest.json` | Pinned v1/v2 Git blob hashes, counts and output SHA-256 invariants |
| `pipeline_contract.json` | B1–B9 controller maturity and same-input run constraints |
| `source_url_access_audit_20261121.json` | 24 original links checked: 22 opened in web reader, 2 reader HTTP-403/search-indexed with alternative NLM pages. URL access is NOT claim-level verification |

`../v2/medical_causal_132_source_screened_v2.jsonl` is the full, already tracked candidate annotation ledger. `../v2/source_catalog_screened_20261121.json` is its 24-source original catalog. Do **not** leak either into the query inference interface.

## Question cohort and exposure

- 100 original pilot questions + 32 original extension questions, no additions or deletions; 132 unique IDs and unique normalized texts.
- Type counts: direct 19, paraphrased 19, indirect 19, multi-hop 19, comparative 16, risk-factor 16, ambiguous 12, unsupported 12.
- 80 semantic groups, 5 prospective family-consistent folds of 27/27/26/26/26.
- 122 questions have 223 *topical* source references; 10 negative examples deliberately have none. These are sources to investigate, **not** guaranteed supporting passages.
- 24 tasks involve refusing unsupported individual medical inferences. Their **tentative annotation** must not be used to prime model inference or report gold accuracy.
- **All 132 questions influenced earlier G1 development. This remains a development cohort even with family folds.** Folds are useful for prospective tuning/ablation, not a retrospectively independent holdout.
- Expert answers/answerability, causal edge/path gold, exact evidence spans, human clinical quality, and an unseen external test are not available in this release.

## Safe Ubuntu commands

```bash
cd "$HOME/Folder/CDF-RAG-2-thesis132"
git fetch origin work/thesis132-g1-target-sufficiency
git pull --ff-only origin work/thesis132-g1-target-sufficiency
python -m research_v2.benchmark_v2_validation
python -m research_v2.thesis132_dataset_release
python -m unittest discover -s tests -p 'test_thesis132_dataset_release_local.py' -v
python -m research_v2.thesis132 freeze
```

The release validation is now invoked by `thesis132._cohort()`; it rejects a changed, missing or reordered 132-row input. The idempotent freeze writes into the already existing, Git-ignored `presentation_20261118/thesis132/frozen_117175bb60b80e54/`, and its `questions_only.jsonl` must match this tracked release exactly.

## Run all pipelines using identical question ID/text/order

Run separately and save exact parameters:
```bash
# Available B1–B6 retrieval-only protocol, full 132, no --limit:
python -m research_v2.thesis132 run --models B1,B2,B3,B4,B5,B6 --mode retrieval_only --encoder sentence --top-k 5

# B7/B8 ONLY when Neo4j/Ollama and the trained assets are configured:
python -m research_v2.thesis132 run --models B7,B8 --mode end_to_end --encoder sentence --top-k 5 --allow-live

# B9 = bounded candidate controller prototype (no learned PPO, no generator):
export G1_FINAL="$PWD/research_artifacts/graph-G1-reconciled-20261121"
python -m research_v2.thesis132_adaptive --graph-dir "$G1_FINAL" --policy adaptive --top-k 5
```

Check actual CLI `--help` if you changed your local branch. B1–B6 running in retrieval-only mode **cannot** be described as final end-to-end answer accuracy. B7/B8 require live services. B9 currently evaluates retrieval and stopping only; PPO and final grounding remain future implementation.

Each model's `records.jsonl` (from `presentation_20261118/thesis132/runs/` or `.../adaptive/`) must pass:
```bash
python -m research_v2.thesis132_dataset_release --model B9 --check-records /ABSOLUTE/PATH/TO/B9/records.jsonl
```

Any missing, reordered or changed question, or actual `status=error`, fails the checker. Record all pipeline graph/corpus versions, tokenizer, top-k, seed, runtime environment, and exact source hashes; **same questions alone do not make model/corpus comparisons causal or fair**.

## Scientific reference standard

TRIPOD-LLM recommends explicit provenance, data-partitioning, contamination and model-performance reporting (https://www.nature.com/articles/s41591-024-03425-5). For evaluations claiming diagnostic validity, STARD requires a defined, suitable reference standard (https://www.bmj.com/content/351/bmj.h5527). Human independent review of these 132 answers and causal edges remains pending. The adjacent v2 folder can generate 264 empty two-reviewer slots, but external real reviewers, adjudication and supervisor confirmation are required before a separately signed gold release.

## Historical experiment preservation

Do not edit `data/benchmark/medical_causal_pilot100_draft_v1.jsonl`, `data/benchmark/medical_causal_expansion32_draft_v1.jsonl`, `data/benchmark/v2/` or past results. Do not overwrite Git-ignored triage, prior frozen folders or local experiment outputs. Provenance and SHA checks prevent silent mutation.
