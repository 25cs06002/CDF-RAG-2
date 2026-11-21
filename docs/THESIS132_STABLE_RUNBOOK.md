# Fixed 132-query MTP: verified local execution sequence

Use exactly the original 100 pilot + 32 extension questions. Preserve legacy
22/100 artifacts. No additional question can enter the main 132 cohort.
The 132 texts influenced G1 development: separately adjudicated labels may be
independent, but no subset is a historically unseen external question test.
Prospective semantic-family-group folds must retain the exposure disclosure.

This runbook NEVER reruns the Thesis132 Experimental Foundation GitHub Actions
workflow. It uses individual local commands only. The development branch is
work/thesis132-g1-target-sufficiency; changes have not been merged into master.
Do not reset/clean original local research files.

## 0. FIRST GitHub fetch, safe isolated checkout, GitHub pull

    cd "$HOME/Folder/CDF-RAG-2"
    git status --short
    git worktree list
    git fetch origin work/thesis132-g1-target-sufficiency
    # Only when the target path and local branch do not exist:
    git worktree add -b thesis132-exec-20261121 ../CDF-RAG-2-thesis132 origin/work/thesis132-g1-target-sufficiency
    cd ../CDF-RAG-2-thesis132
    git pull --ff-only origin work/thesis132-g1-target-sufficiency
    git rev-parse --short HEAD

If a worktree already exists, inspect it, switch into that actual worktree
and pull only when its branch and status are correct. Never force-reset.
Work in the new worktree; do not modify the original.

    if test -f ../CDF-RAG-2/venv/bin/activate; then
      source ../CDF-RAG-2/venv/bin/activate
    elif test -f ../CDF-RAG-2/.venv/bin/activate; then
      source ../CDF-RAG-2/.venv/bin/activate
    else
      python3 -m venv .venv
      source .venv/bin/activate
    fi
    python -m pip install 'pydantic>=2,<3' requests python-dotenv

This minimal install is for offline integrity tests. Full dense/SentenceTransformer,
Ollama, PPO and Neo4j runs need the complete, compatible original environment,
model assets, service configuration and dependencies.

## 1. Targeted offline validation, fixed dataset freeze

    python -m compileall -q research_v2 tests
    python -m research_v2.unified_integrity --json
    python -m research_v2.benchmark_v2_validation
    python -m unittest discover -s tests -p 'test_g1_target_resolution_local.py' -v
    python -m unittest discover -s tests -p 'test_g1_typed_local.py' -v
    python -m unittest discover -s tests -p 'test_thesis132_evidence_local.py' -v
    python -m unittest discover -s tests -p 'test_thesis132_gold_scoring_local.py' -v
    python -m unittest discover -s tests -p 'test_thesis132_gold_template_local.py' -v
    python -m unittest discover -s tests -p 'test_thesis132_legacy_metrics_local.py' -v
    python -m research_v2.thesis132 freeze
    python -m research_v2.benchmark_v2_export

STOP at any failure. Freeze is idempotent and checks exactly 132 IDs/texts,
manifest hashes and prospective semantic-family-group folds. It does not
produce clinical gold or a historically unseen holdout.

## 2. Required local-only G1 snapshot and 17-case triage

Find actual source folders in the original checkout:

    find ../CDF-RAG-2/research_artifacts -type f -path '*/G1_CANDIDATE_*/manifest.json' -print
    find ../CDF-RAG-2/presentation_20261118/g1_error_triage -name priority_cases.csv -print

Set the two variables to EXISTING, matched folders from your Nov 17 G1 run,
rather than copying placeholders literally. Verify the manifest hash against
its corresponding saved paired run. Keep local reviewer material private.

    export G1_SOURCE="/ABSOLUTE/PATH/TO/REAL/G1_CANDIDATE_DIR"
    export TRIAGE_SOURCE="/ABSOLUTE/PATH/TO/REAL/triage_DIR"
    test -f "$G1_SOURCE/typed_relationships.csv"
    test -f "$TRIAGE_SOURCE/priority_cases.csv"
    sha256sum "$G1_SOURCE/manifest.json"
    mkdir -p research_artifacts/local_g1_snapshots presentation_20261118/g1_error_triage
    cp -a "$G1_SOURCE" research_artifacts/local_g1_snapshots/
    cp -a "$TRIAGE_SOURCE" presentation_20261118/g1_error_triage/
    export G1="$PWD/research_artifacts/local_g1_snapshots/$(basename "$G1_SOURCE")"
    export TRIAGE="$PWD/presentation_20261118/g1_error_triage/$(basename "$TRIAGE_SOURCE")"

If files are absent, STOP here. They are not in GitHub. No fabricated graph
or triage file may be substituted for the archived study.

## 3. Stage 1: 8 wrong targets + 6 unsupported gains + 3 other flags

    python -m research_v2.thesis132_g1 --graph-dir "$G1" --triage-dir "$TRIAGE" --top-k 5

Inspect the printed diagnostic directory: per_query.csv must contain
264 B5/B6 x 132 rows; reviewed_17_cases.csv must contain all 17 matched
cases; summary.json must include the frozen G1 and query hashes. Review
newly retained evidence, new withholds, lost previous candidates and
previously correct cases. These are candidate-retrieval changes, not
clinically certified improvements.

## 4. Stage 2: passage-level graph verification and typed status

    python -m research_v2.thesis132_evidence prepare --graph-dir "$G1"

The printed new workspace contains edge_adjudication.csv, ALL candidates
marked pending. Two real independent reviewers must verify source URL,
exact passage in a locally stored licensed source snapshot, relation
direction, distinction between causal/risk/association, and evidence scope.
An independent adjudicator marks EACH relation verified, uncertain or
excluded, with actual reviewer IDs, date and rationale.

    export EVIDENCE_WORKSPACE="/ABSOLUTE/PATH/TO/PRINTED/G1_EVIDENCE_REVIEW_DIR"
    python -m research_v2.thesis132_evidence audit --graph-dir "$G1" --workspace "$EVIDENCE_WORKSPACE"

An incomplete worksheet exits nonzero, as intended. Freeze only after actual
independent review AND externally verified supervisor approval:

    python -m research_v2.thesis132_evidence freeze \
      --graph-dir "$G1" --workspace "$EVIDENCE_WORKSPACE" \
      --signoff "$EVIDENCE_WORKSPACE/supervisor_signoff.json"

Required supervisor signoff JSON fields are supervisor_name, approval_date,
reviewer_credentials_checked, source_authenticity_checked,
causal_directions_checked and independent_external_adjudication_completed.
Booleans must be true only after real external validation. No software
can independently authenticate a clinician, a full source page or licensing.

## 5. Stage 3: 132 independent reference answers, not new questions

    python -m research_v2.benchmark_v2_validation --prepare-review
    python -m research_v2.thesis132_gold_template

The first creates 264 unfilled independent reviewer slots; the second
creates 132 deliberately EMPTY reference placeholders. This is NOT gold.
After genuine, separately conducted double-review and adjudication:

    python -m research_v2.benchmark_v2_review_gate \
      --review-file "/ACTUAL/COMPLETED/independent_adjudication_worksheet.csv"

An independent supervisor must authenticate reviewer qualifications,
claim-level source passages, relations, all gold paths, verified answers
and individual answerability. Only then create a separate, NEW versioned
gold_132_EXTERNALLY_ADJUDICATED.jsonl with the exact schema defined by
research_v2/thesis132_gold_scoring.py, never editing the 132 original texts.
Keep clinical workspaces gitignored. The scorer rejects pending/partial gold.

## 6. Stages 4 and 5: evidence-aware B9 candidate ablation

B9 is currently an evidence-typed retrieval prototype, not a fully
source-verified medical generator. Use matched graph, candidate document
corpus, top-k and frozen query inputs for all fixed/adaptive policies:

    python -m research_v2.thesis132_adaptive --graph-dir "$G1" --policy fixed1 --top-k 5
    python -m research_v2.thesis132_adaptive --graph-dir "$G1" --policy fixed2 --top-k 5
    python -m research_v2.thesis132_adaptive --graph-dir "$G1" --policy fixed3 --top-k 5
    python -m research_v2.thesis132_adaptive --graph-dir "$G1" --policy adaptive --top-k 5

Use the FOUR ACTUAL printed output paths:

    python -m research_v2.thesis132_adaptive_compare \
      --run-dirs "/ACTUAL/b9_fixed1" "/ACTUAL/b9_fixed2" \
        "/ACTUAL/b9_fixed3" "/ACTUAL/b9_adaptive"

Compare candidate evidence retention, abstentions, rounds, early stops and
retrieval latency. No gold relevance/groundedness can be claimed until
the independent review is finished. B9 is not a final full hybrid RAG yet.

## 7. Stage 6: new cost-aware PPO is blocked pending real feedback

Historical B8 uses the earlier PPO checkpoint. A NEW PPO requires a real
retrieval-feedback state/action/reward environment, latency and compute
penalties, externally verified evidence quality, and prospective family-
grouped policy training. Do not claim a new PPO is trained or execute an
invented new PPO command. Compare against no refinement, fixed rounds and
rule-based adaptive under the SAME generator, corpus, graph and hardware.

## 8. Stage 7: all B1-B8 on frozen 132 (Ubuntu only)

Retrieval smoke then FULL B1-B6, using installed SentenceTransformer assets:

    python -m research_v2.thesis132 run --models B1,B2,B3,B4,B5,B6 \
      --mode retrieval_only --encoder sentence --top-k 5 --limit 3
    python -m research_v2.thesis132 run --models B1,B2,B3,B4,B5,B6 \
      --mode retrieval_only --encoder sentence --top-k 5

Do not misreport the three-query smoke as a 132 result. B1-B4 candidate
summary documents and B5-B6 G0 graph edges have DIFFERENT evidence bases.
The current full runner explicitly preserves that confound.

For end-to-end B1-B8, first confirm local Ollama and Neo4j services, G0
runtime lock, original .env (never commit) and PPO checkpoint. Only if needed:

    if test -f ../CDF-RAG-2/.env && ! test -f .env; then
      cp ../CDF-RAG-2/.env .env
    fi
    python -m research_v2.ollama_generation check

Then separate smoke and FULL end-to-end sessions:

    python -m research_v2.thesis132 run \
      --models B1,B2,B3,B4,B5,B6,B7,B8 --mode end_to_end \
      --encoder sentence --top-k 5 --allow-live --limit 2
    python -m research_v2.thesis132 run \
      --models B1,B2,B3,B4,B5,B6,B7,B8 --mode end_to_end \
      --encoder sentence --top-k 5 --allow-live

Review every model's summary and 132 records. Any error or aborted graph
drift is an actual failure. Resume ONLY between completed models using
the existing --session-dir option with unchanged run hashes. Do not
overwrite outputs or treat errors as abstentions.

## 9. 22-era metrics extended descriptively to all 132

Identify a full, NOT smoke, 132 session folder printed by the runner:

    export SESSION="/ABSOLUTE/PATH/TO/FULL_BATCH_SESSION"
    python -m research_v2.thesis132 summarize --session-dir "$SESSION"
    python -m research_v2.thesis132_analysis --session-dir "$SESSION"
    python -m research_v2.thesis132_legacy_metrics --session-dir "$SESSION"

The bridge emits the original 22-era metric names and directly measurable
132-query latency, throughput, answer length and graph/evidence counts.
Context relevance, medical groundedness, clinical hallucination rate,
semantic refinement and refinement success are NULL until a valid separate
evaluator or human reference exists; they are not falsely set to zero
or copied from retrieval yield. Old 22 values cannot be causally compared
with new metrics when the methods/evidence change.

ONLY after a REAL reviewed gold release and complete independent human
answer judgments CSV:

    python -m research_v2.thesis132_gold_scoring \
      --gold "/ACTUAL/gold_132_EXTERNALLY_ADJUDICATED.jsonl" \
      --session-dir "$SESSION" --top-k 5 \
      --answer-judgments "/ACTUAL/answer_judgments.csv"

Judgment CSV columns: model,query_id,answer_correct,grounded,clinically_safe,
reviewer_id, with complete yes/no human decisions for every evaluated
model-query pair. The scorer verifies query IDs/texts, file SHA, external
review declarations, source-span metadata, directed relations, and
semantic-family grouped confidence intervals. It CANNOT authenticate
reviewer identity or source truth by itself. The 132 remains retrospective.

## 10. Stage 8 thesis reproducibility and signoff

Every improvement claim must point to an actual code commit, source
corpus/graph SHA, the exact 132 input SHA, real run ID, hardware/model
config, measured latency/throughput, output SHA, answer/evidence scoring
definition, uncertainty interval and adjudication status.

Before an architecture-only comparison, use a genuinely matched evidence
corpus, retrieval budgets, generator and hardware. Do not merge old
22-query, 100-query and 132-query scores. Do not publish private records
or medical/source texts without permission. A clinical gold-standard
release, verified path expansion, new PPO training and final supervisor
approval remain dependent on actual external human work and new local
computational runs.
