# CDF-RAG-2 — Unified M.Tech Research Repository

**Canonical branch: master**. The former `research-v2` and G1 branches are retained as historical development references.
**Research title:** Adaptive Multi-Round Retrieval-Augmented Generation Architecture for Reliable Domain-Specific Question Answering.
**Initial study domain:** medical causal question answering.
**Status:** preliminary retrieval experiments and G1 failure analysis completed; full evidence-sufficiency-driven adaptive multi-round model NOT yet validated.

This repository retains two different research lines without falsely combining
their scores: the original **22-query No-RL/PPO experiment**, and later
**100-question eight-model / 132-question retrieval and G1 experiments**.

The original CDF-RAG paper belongs to **Elahe Khatibi, Ziyu Wang, and Amir M.
Rahmani**. Its README, published ablations and citations are preserved separately
at [docs/ORIGINAL_CDF_RAG_README.md](docs/ORIGINAL_CDF_RAG_README.md).
**Do not present upstream paper's F1 values as results of this M.Tech project.**

## One authoritative source of truth

- **Experimental provenance:** [research_artifacts/unified/experiment_registry_v1.json](research_artifacts/unified/experiment_registry_v1.json)
- **Offline consistency verifier:** [research_v2/unified_integrity.py](research_v2/unified_integrity.py)
- **Research roadmap:** [docs/MTP_UNIFIED_ROADMAP.md](docs/MTP_UNIFIED_ROADMAP.md)
- **Datasets:** [data/benchmark/](data/benchmark/) — frozen pilot100,
  expansion32 and separately labeled [source-screened v2](data/benchmark/v2/README.md).
- **Eight-model evaluation:** [research_v2/pipelines/](research_v2/pipelines/).
- **Original 22-query implementations and PPO training:** [agents/](agents/),
  original `evaluate_*_rl*.py`/`evaluate_*_no_rl*.py` scripts,
  original CSVs and legacy summary under
  [presentation_20261118/legacy_results/](presentation_20261118/legacy_results/).
- **Immutable G0:** [research_artifacts/graph-G0/](research_artifacts/graph-G0/).
- **Existing G1 code and review methods:** `research_v2/g1_*.py`,
  `research_v2/phase2m_*.py`.
- **Selected archived 100-query runs:** `presentation_20261118/runs/`
  and [end-to-end matrix](presentation_20261118/metrics/matrix_20261116T111224165031Z/pipeline_comparison.csv).
- **Newer 132-query aggregate results:** completed and locally audited,
  but **full source run folders not yet committed** to GitHub. Do not
  describe them as already available in the remote repository. See
  [local import instructions](docs/MTP_UNIFIED_ROADMAP.md).

**Older documents** such as `docs/PRESENTATION_SUPERVISOR_20261118.md`
and `PRESENTATION_SPRINT_20261118.md` are historical snapshots. Their
then-current "pending" labels do not supersede the newer verified runs.

## Actual experiment inventory — do NOT pool into one score

| Experiment | Dataset | Systems | Evidence/status | Findings |
|---|---|---|---|---|
| Historical deterministic feasibility | 22 selected queries | No-RL vs PPO-RL | Tracked historical CSV; proxies, not medical gold | PPO 39.58 s -> 50.09 s mean latency, near-identical groundedness proxies |
| Selected end-to-end pilot | 100 draft queries | B1–B8 | Eight tracked error-free runs; end-to-end | B7 49/100 versus B8 65/100 graph evidence, with B8 slower |
| Structural G0 subset | 20 mapped draft queries | B7/B8 | Reference is G0 snapshot, NOT medical gold | Edge micro-F1 0.872 versus 0.971, not clinical F1 |
| New retrieval-only extension | 100+32 draft queries | B1–B6 | 792 executions and zero errors, technically audited **locally** | Document yield B1–B4 132/132; G0 graph B5 49, B6 41 |
| G1 expansion shadow comparison | 132 draft queries | B5/B6 G0, G1 legacy and stable | Audited **local** developmental experiment | B5 49->53->55; B6 41->35->44 evidence-yield |
| Selected changed-case review | 31 model-query cases / 20 questions | B5/B6 | 132 evidence occurrences / 58 assertions; local non-holdout | 17 flagged cases over 12 questions |
| Source-screened benchmark v2 | Same 132 development questions | Annotation/validator | 132 annotated; 24 source pages; CI passed | **0 independent clinician-approved gold questions** |

**Important:** Retrieval coverage is not document relevance, causal truth,
answer accuracy, or hallucination reduction. B1–B4 use candidate source
summaries while B5–B8 use G0, so evidence-count comparisons are not
controlled medical-accuracy comparisons. B5/B6 abstentions should not be
confused with software errors. End-to-end latency is not the same as
retrieval-only latency. B7/B8 retrieval method differences confound attribution
of improved evidence yield exclusively to PPO.

## Verification — fast, safe, no LLM or Neo4j

```bash
python -m research_v2.unified_integrity --json
python -m research_v2.benchmark_v2_validation
python -m unittest discover -s tests -p 'test_mtp_unified_integrity.py'
python -m unittest discover -s tests -p 'test_benchmark_v2*_local.py'
```

The historical run files and source datasets are immutable. The verifier
checks the selected archived 100 run IDs rather than assuming "newest file"
equals "best result".

## Running experiments

**Existing results:** First use the registry; do not repeat expensive B7/B8
LLM/PPO runs just to generate a summary.

**B1–B8 registry and original pilot runner:**
```bash
python evaluate_all_baselines.py validate
python evaluate_all_baselines.py smoke --encoder sentence
```

Full end-to-end B7/B8 runs require the legacy Neo4j, Ollama and PPO
checkpoint environment. Historical 100-query results are already archived.

**Reproduce B1–B6 retrieval-only 100 + 32 if intentionally needed:**
```bash
python -m research_v2.partial_submission_sprint verify
python -m research_v2.partial_submission_sprint doctor --encoder sentence
python -m research_v2.partial_submission_sprint run100 --encoder sentence
python -m research_v2.partial_submission_sprint run32 --encoder sentence
```

Run configurations and data hashes must be preserved. Switching to the
`lsa` encoder is a distinct experiment, not a silent substitute.

**Development benchmark annotations and safe input export:**
```bash
python -m research_v2.benchmark_v2_validation
python -m research_v2.benchmark_v2_export
```

V2's AI-generated candidate response rubrics must **not** be used as gold
answers. External, qualified reviewers must adjudicate question-specific
answers, directed edges, evidence spans and review disagreements.

## Local-only evidence migration

The successful 132-query submission bundle, reviewed G1 annotations and
case-study artifacts created on the author's Ubuntu machine are not all
committed remotely.
To import without affecting the original local repository, create a separate
Git worktree from `origin/master` and use `--source-repo` to read the audited
outputs from the original Ubuntu checkout. Only the allowlisted results
are copied; no input experiment or review file is overwritten. See
[the complete commands](docs/MTP_UNIFIED_ROADMAP.md#import-commands-for-a-separate-safe-git-worktree).
 They cannot be reconstructed losslessly from aggregate
metrics. Use the opt-in, hash-checked importer
`research_v2/unified_import_local.py` to stage nonsensitive aggregate
files. Verify source rights, private content and hashes before pushing.
Never force-push, reset, clean or delete the original experimental history.

## Final-thesis work remaining

1. Repair concept/subtype ambiguity and unsupported evidence gains;
   evaluate on an independently reviewed question set.
2. Build an evidence-sufficiency and source-typed verification gate,
   with calibrated abstention and provenance.
3. Implement bounded adaptive multi-round feedback and early stopping.
4. Retrain/compare PPO with actual retrieval feedback and latency-aware
   rewards; isolate PPO from other pipeline differences.
5. Run matched corpus/generator/hardware experiments and document
   correctness, latency, throughput, memory, ablations and uncertainty.
6. Obtain external medical adjudication and an independent held-out split.
7. Write and defend the complete thesis after validated final results.

See [docs/MTP_UNIFIED_ROADMAP.md](docs/MTP_UNIFIED_ROADMAP.md)
for detailed phases and how each current artifact maps to the MTP plan.

## Branch provenance and safety

The unified merge commit has both `master` (original legacy history) and
`g1-query-evidence-foundation` as parents. `research-v2` is an ancestor
of the latter. Old branches are retained as read-only historical references.
Legacy scratch code in the old master is preserved by Git history; its
annotated replacement lives in
`research_artifacts/legacy22/scripts/temp_hallucination_report.py`.
`NEW_README.md` was superseded, not silently used as active instructions.
