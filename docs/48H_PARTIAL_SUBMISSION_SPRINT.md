# CDF-RAG-2 — 48-hour partial-work submission sprint

Goal: defensible partial contribution and controlled results on **existing**
100+32 draft queries only. Do not expand questions, retrain PPO, rewrite
reviewed CSVs, rebuild graph, or run the offline sprint.

## Main constraints and scope

- Historical B1–B8 100-query experiment has **800 recorded model-query
  slots, NOT 800 successful predictions**. Its older B1–B6 executions
  include errors; keep raw archived results and error counts visible.
- Historical B7/B8 G0/Ollama/PPO 100-query runs succeeded; reuse them.
- B1–B4 retrieve provisional AI-assisted source summaries, while B5/B6
  retrieve an **unverified G0** graph. Evidence is not comparable as gold
  relevance and architecture-only medical accuracy cannot be claimed.
- The new B1–B6 comparison is **retrieval only** and preserves the frozen
  candidate corpus, graph, top-k=5, encoder selection and question hashes.
- No B7/B8 execution on the 32-query extension is implied.
- B5/B6 G1-candidate and stabilized shadow experiments already cover
  both query sets (132 total); G1 inherited 37 unverified G0 edges.
- The manual-review analysis covers 31 selected model-query cases and
  132 evidence occurrences representing 58 distinct question-evidence
  assertions. It is NOT a held-out sample or medical gold.

## Sprint execution in repository root

Assume local Python venv active and all original outputs preserved:

```bash
python -m py_compile research_v2/partial_submission_sprint.py
python -m research_v2.partial_submission_sprint verify
python -m research_v2.partial_submission_sprint run100 --encoder sentence
python -m research_v2.partial_submission_sprint run32 --encoder sentence
```

If the sentence-transformer cache is absent, **do not silently fall back**.
Either install the original cached model and rerun a new versioned experiment,
or explicitly use `--encoder lsa` and label LSA as a different method.
Do not compare a sentence-encoder run with an LSA run as identical.

The two run commands print versioned folders under
`presentation_20261118/partial_submission_sprint/`:
`run100_<timestamp>_*/` and `run32_<timestamp>_*/`.
Both produce a `run_index.json` and `100_query_model_summary.csv` or
`32_query_model_summary.csv`, including per-query, per-model records.
**Never edit historical runs.** If an error occurs, read the preserved
per-model `records.jsonl` and `summary.json`; fix its cause before a new run.
Only an error-free six-model cohort is accepted for final consolidation.

After both complete:

```bash
RUN100=$(find presentation_20261118/partial_submission_sprint -maxdepth 1 -type d -name 'run100_*' | sort | tail -1)
RUN32=$(find presentation_20261118/partial_submission_sprint -maxdepth 1 -type d -name 'run32_*' | sort | tail -1)

python -m research_v2.partial_submission_sprint assemble \
  --retrieval100-dir "$RUN100" --run32-dir "$RUN32"
```

This creates one new `submission_<timestamp>_*/` with provenance JSON,
historical, fresh 100 and fresh 32 model CSVs, G0/G1 shadow summary, the
22-query historical RL/No-RL comparison, 20-query G0 structural agreement
(if present), a Markdown results section, report outline and PPT outline.

## Phase-by-phase sprint priorities

### Day 1 — Establish defensible results

- [x] Archive earlier 100 B1–B8 and 22-question RL vs No-RL results.
- [x] G1-candidate graph built and 132-question G0/G1 shadow retrieval done.
- [x] Reviewed changed-case annotations audited (31 cases, 132 rows).
- [x] Completed reviewed-error analysis (17 flagged model-query cases).
- [x] Sprint runner and submission bundle generator committed.
- [ ] Verify exact archived 100-run per-model errors.
- [ ] Run error-free 100-query retrieval-only B1–B6.
- [ ] Run error-free 32-query retrieval-only B1–B6.
- [ ] Assemble and double-check table names, sample sizes, execution modes,
      corpus SHA and presence/absence of independent medical gold.

### Day 2 — Consolidate scientific message and rehearse

- [ ] Inspect the 17 error-priority cases; identify one or two concrete,
      independently supported examples of wrong target and false gain.
- [ ] Limit claims to **covered cases**: graph-expansion target drift,
      stability-first resolver and source-typed evidence retrieval.
- [ ] Report coverage and manual-reviewed relevance as separate axes.
- [ ] State tradeoff: B8 historical G0 structural coverage vs B7 latency,
      without calling either medically more accurate.
- [ ] Prepare 7–9 slides and a partial report, using exact reproduced tables.
- [ ] Prepare supervisor questions and caveats: evidence sufficiency,
      feedback agent, G1/Neo4j migration, independent gold annotations.
- [ ] Version files and preserve reviewed CSVs; do not auto-commit sensitive
      or unlicensed medical passages to public GitHub.

## Thesis claim — strongest current bounded wording

> We identify and quantify a graph-expansion-induced query-target drift
> failure in causal RAG retrieval, construct a review-gated typed medical
> causal graph, and implement a stability-aware query resolver. Across the
> existing draft query sets, this improves retrieval evidence coverage over
> the frozen G0 baseline; human-reviewed error analysis reveals remaining
> target-selection and evidence-validity failures. The results motivate
> an evidence-sufficiency-guided adaptive retrieval system, which remains
> future work.

This is a **partial result and engineering/research-method contribution**.
It does not establish clinical validity, independent held-out generalization
or PPO superiority.

## Preferred 8-slide story

1. Problem and stakes: unreliable domain QA, causal direction, latency.
2. Prior literature and gap: MRD-RAG, guideline-based retrieval and CDF-RAG.
3. Pipeline family B1–B8 and experimental controls.
4. Data/graph: 100+32 draft questions; G0 35/37; G1 78/52 + 24 typed.
5. Historical B7/B8 and 100-query B1–B8: run status and modes visible.
6. Repaired B1–B6 retrieval-only 100/32 (when complete), graph-vs-doc evidence basis.
7. Core result: B5/B6 G0 vs G1 legacy vs stabilized; 31-case review and errors.
8. Claim, limitations, and 3 next-step questions for supervisor.

## Suggested 8–12 page partial report

Abstract, introduction, problem statement and research questions,
literature comparison, existing architecture, datasets and provenance,
baselines and experimental controls, G0/G1 graph design, retrieved results,
human-reviewed error analysis, explicit contributions and limitations,
planned adaptive feedback/early stopping/PPO ablations, references, artifact
provenance appendix.

## Submission guardrails

Do not write “all eight models were successful” for the archived matrix;
run statuses must be explicit. Do not combine 22-, 100- and 132-query
scores in one averaged number. Do not call graph-snapshot micro-F1
`clinical F1`. Do not compare legacy end-to-end latencies with
retrieval-only submillisecond times. Report reviewer source checks
as attestation, not independent medical expert approval.
