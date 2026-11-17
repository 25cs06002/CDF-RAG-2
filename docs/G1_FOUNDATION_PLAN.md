# G1 foundation — preservation, review and fast presentation experiments

**Development branch:** `g1-query-evidence-foundation`. The workflow is
manually executable and does not change the `research-v2` offline sprint CI.

## What already exists (do not redo)
- 100 unreviewed pilot questions in `data/benchmark/medical_causal_pilot100_draft_v1.jsonl`.
- A separate original 22-query-era experimental history (original CSVs);
  do not treat historical queries as a reviewed test split.
- G0, 35 nodes / 37 unreviewed CAUSES edges; B1–B8 archived runs.
- Original 15-source + eight-source registries.
- Phase2K/2M multi-hop review and 52 source tasks, zero independently
  adjudicated links according to `phase2m_full_adjudication_handoff_v1/handoff_manifest.json`.
- Existing `graph_gap_triage.py` experimental shadow target resolver.

## New outputs
- `data/benchmark/medical_causal_expansion32_draft_v1.jsonl`
  — **separate** 32-question pilot expansion (four per query type), NOT
  appended to frozen 100, NOT gold or split/test. The eventual 650 target
  remains a plan, not completed data.
- `python -m research_v2.g1_foundation prepare` creates a new
  `research_artifacts/experiment-artifacts/g1_review_workspaces/G1_REVIEW_*/`
  directory containing:
  - `concept_inventory.csv`: every distinct draft target from 132 questions,
    flags exact G0 presence and candidate source IDs.
  - `source_review.csv`: the pre-existing 15+8 source candidates plus two
    newly checked official pages, all **unreviewed**; passage locators required.
  - `typed_relationship_review.csv`: intentionally EMPTY for reviewer-approved
    source-supported relationships.
  - `alias_review.csv`: intentionally EMPTY; no unsafe automatic equivalences.
  - `manifest.json`: frozen SHA256 inputs and approval warnings.

## Human review tasks (REQUIRED before G1 build)
1. **Concept inventory**: Review absent concepts, de-duplicate true synonyms
   only with a cited source. Never map "type 2 diabetes" to G0's broader
   "Diabetes", or CAD to broad "heart disease", as a synonym.
2. **Sources**: Open the actual source pages (check section, exact locator,
   date and scientific scope). Fill `source_review.csv`; approve only real
   checks with name/date. Medical expert validation is separate.
3. **Relationships**: Fill `typed_relationship_review.csv` with independently
   supported `subject, relation, object, source_id, passage_locator,
   support_summary, reviewer, review_date`. Use one of `CAUSES`,
   `RISK_FACTOR_FOR`, `ASSOCIATED_WITH`, `TRIGGERS_EXACERBATION`,
   `PROTECTS_AGAINST`, `IS_A`, `UNCERTAIN`. Never promote risk factors,
   correlations, or symptom triggers to CAUSES.
4. **Aliases**: Fill `alias_review.csv` only for reviewed synonym equivalence,
   with source/reviewer. Aliases are used by a **separate experimental
   query-only retriever**, never by historical B1–B8.
5. **Question review**: Independently review new draft questions for ambiguity,
   evidence and family leakage; keep their output and gold labels separate.

Run `python -m research_v2.g1_foundation audit --folder '<actual directory>'`
to check typed source/review provenance structure. It does **not** verify
medical correctness or visit URLs.

Only after approvals:
`python -m research_v2.g1_foundation build --folder '<actual directory>'`

That produces `graphs/G1_CANDIDATE_*/nodes.csv`, `edges.csv` (only
approved *new* CAUSES edges, plus inherited G0), `typed_relationships.csv`,
`approved_aliases.csv`, and `manifest.json`. Inherited 37 G0 edges remain
unverified. No direct Neo4j mutation.

## Controlled retrieval-only shadow experiment
Run `python -m research_v2.g1_query_interpreter --top-k 5`
against frozen G0, and optionally:
```bash
python -m research_v2.g1_query_interpreter --queries data/benchmark/medical_causal_expansion32_draft_v1.jsonl
python -m research_v2.g1_query_interpreter --graph-dir research_artifacts/experiment-artifacts/g1_review_workspaces/G1_REVIEW_.../graphs/G1_CANDIDATE_... --alias-file .../approved_aliases.csv
```
These are **new shadow conditions**, NOT reruns or replacements of B5/B6.
They never see draft target labels or gold answers at inference. They can
abstain on missing/ambiguous targets; report both recovered and lost yield.

## Expensive reruns: NOT YET
- Do NOT rerun archived 100-query B7/B8, or B1–B6 full generation.
- After source review and G1 snapshot lock, run B5/B6 retrieval-only on
  new G1; then optional small end-to-end smoke with matching Ollama settings.
- B7/B8 cannot use new G1 simply by changing a CSV; their Neo4j graph,
  runtime lock and PPO-related preflight must be migrated/versioned first.
- Later: independently reviewed quality labels, family-safe splits,
  matched-evidence-budget comparisons and cache/early-stop ablation.

## Authoritative source access verification (optional, local only)
The generated `source_review.csv` includes 25 *candidate* sources
(15 primary, 8 additional, 2 separately verified official landing pages).
The existing Phase 2M handoff includes 52 source review tasks, **zero
independently adjudicated source records**; use those worksheets instead of
replacing past work with a generic source list.

To check URL availability without downloading or publishing complete pages:

```bash
python -m research_v2.g1_source_access --folder "$REVIEW" --limit 25 --timeout 12
```

The `source_access_*/source_access.csv` records HTTP status, response
location and a fetched-byte checksum, with `source_claims_verified=false`.
Some sites block scripted access. Failed requests are not proof that a source
is invalid. Open the links manually and record the precise passage/scope.

## Focused local validation (does not run the offline sprint)
```bash
python -m unittest tests.test_g1_foundation_local -v
```
This checks dataset separation, inventory creation, approval gating,
unsafe synonym handling and a few query-intent cases. It does not
execute Ollama, Neo4j, PPO, or the large CI suite.

## Typed G1 retrieval experiment (after at least one approved relationship)
Run only after building a candidate G1 snapshot:
```bash
python -m research_v2.g1_typed_retrieval --graph-dir "$GRAPH" --top-k 5
python -m research_v2.g1_typed_retrieval --graph-dir "$GRAPH" \
  --queries data/benchmark/medical_causal_expansion32_draft_v1.jsonl --top-k 5
```
This adds **experimental** `B5_G1_TYPED` (one-hop typed evidence) and
`B6_G1_TYPED` (typed one-hop evidence and fully causal two-hop paths).
Only a reviewed `CAUSES` relation appears in `causal_edges`; risk factors,
associations and triggers stay labeled in `retrieved_documents`. The
37 inherited unreviewed G0 edges do not count as independently verified
typed evidence in this run. Do not replace archived B5/B6 scores with
these new, different-evidence evaluations.

A query needing a risk factor or association may legitimately produce an
evidence snippet instead of a causal path. A query needing a causal chain
may legitimately abstain until each causal arrow is supported. This is a
deliberate quality guard, not an execution error.

## Focused typed-retrieval test (not offline sprint)
```bash
python -m unittest tests.test_g1_foundation_local tests.test_g1_typed_local -v
```
The synthetic tests confirm that risk-factor edges are returned as typed
source evidence rather than silently relabeled as causal graph edges.

## Completed Phase 2M independent handoff: validate first

Do not overwrite the original handoff or treat a manually edited CSV as
independently certified medical gold. For your completed 52 source and 30
link task files, run:

```bash
python -m research_v2.phase2m_independent_audit \
 --folder research_artifacts/experiment-artifacts/phase2/phase2m_full_adjudication_handoff_v1 \
 --require-complete
```

Creates `PHASE2M_AUDIT_*/review_audit.json` and
`g1_link_import_candidates.csv`. If there are errors, correct the
reviewed CSVs and rerun the auditor; source IDs and relationship
directions must agree. Proposals are **not** automatically imported
into G1. The new G1 inventory still needs approved source records
and reviewed concept nodes; unknown source IDs must be crosswalked
into that ledger without losing provenance.

Use a versioned checksum snapshot before pushing reviewed worksheets.
Review all Git-staged files for private information, personal IDs,
unlicensed full-text excerpts or secrets because the GitHub repository
may be public.

## After the 52/30 Phase2M audit passes

Use the **current local audit directory** (not a stale GitHub copy). The
converter verifies that the audit SHA256 matches the two actually reviewed
CSV inputs before creating a new versioned G1 review workspace.

```bash
AUDIT=$(find research_artifacts/experiment-artifacts/g1_review_workspaces \
  -maxdepth 1 -type d -name 'PHASE2M_AUDIT_*' | sort | tail -1)
python -m research_v2.phase2m_g1_handoff --audit-dir "$AUDIT"
```

Output: a new `G1_REVIEW_*/` directory with existing 25 candidate sources,
**all 52 source task review events** (even partial/unsupported), 24
typed relationship **proposals** (not automatically approved), newly
encountered intermediate concepts, a source-ID crosswalk and a link
provenance file retaining **all** original supporting source IDs. The
G1 CSV currently uses one primary source ID for each typed relationship;
`phase2m_edge_provenance.csv` retains every additional source ID for
subsequent evidence attribution or a later multi-source schema.

Manual review required:
- `source_review.csv`: check original URL, actual source title, scope,
  cited passage, review notes. Only approve genuinely supporting sources.
- `concept_inventory.csv`: for newly needed disease and intermediate nodes,
  use approved sources to document existence/identity, then approve.
- `typed_relationship_review.csv`: review each typed arrow, confirm the
  supporting source(s), and approve where justified.
- `phase2m_edge_provenance.csv`: cross-reference every extra supporting
  source to ensure no evidence provenance is omitted.
- `alias_review.csv`: remains empty until terminology equivalence is reviewed.

Run `python -m research_v2.g1_foundation audit --folder "$REVIEW"`
after manual approvals; build G1 only after at least one approved edge
and its source/concepts pass the review gate.

A structural pass is not clinical correctness, and a candidate G1
is not medical gold. Frozen G0 and the historical 100-query experiment
remain unchanged.

## After stable-G1 shadow: hand off changed retrieval cases for relevance review

The first 100+32 query shadow found that target locking restores original
G0 evidence on all originally lost B5/B6 cases, but the stable shadow
retains some evidence-yield losses because it intentionally abstains for
ambiguous, underdetermined or unsafe broad-to-specific interpretations.

A stronger research comparison needs **manual relevance and direction
labels**, rather than treating a retrieved graph edge as medically correct.
Do not build gold accuracy from an automatically passing structural audit.
For the existing G1, paired runs, and stable run, execute:

```bash
python -m research_v2.g1_relevance_handoff prepare \
  --graph-dir "$GRAPH" \
  --stable-dir "$STABLE" \
  --paired-dirs "$P100" "$P32" \
  --scope stable_changes
```

This produces `review_cases.csv` (one row per changed model/query),
`review_evidence.csv` (retrieved edges for each of frozen G0,
G1 legacy, G1 stable and G1 typed), and `manifest.json`.
Original `G0_EDGE_####` IDs are interpreted by graph-specific order and
mapped to the underlying cause/effect, so an inherited G0 statement is
not silently labeled verified. New G1 CAUSES edges are joined to typed
edge provenance and the Phase2M source crosswalk.

For a broader queue of changed *legacy or stabilized* results, use
`--scope any_changes`; for all 132×2 model-query cases use `--scope all`.
No graph/model execution is required to generate this handoff.

Human tasks: First label query intent and target selection; then examine
each *returned* edge and any linked authoritative source passage.
`review_evidence_relevance` is one of `relevant`, `irrelevant`,
`uncertain`. `review_relation_direction` is `correct`, `incorrect`,
`uncertain`; `review_source_checked` is `checked`,
`not_checked`, `not_applicable`. A graph edge inherited from G0
must **not** be called verified just because it is present in G1.
Record real reviewer identifiers, actual dates and explicit notes.
Use `python -m research_v2.g1_relevance_handoff audit --folder "$REVIEW_QUEUE"`
to assess annotation progress. This audit never labels medical gold and
never modifies historical runs.

The changed-case queue is selected *because* retrieval behavior differs;
its metrics are useful for error analysis but cannot estimate unseen-query
clinical accuracy. Before claiming retrieval accuracy, construct
independently adjudicated question/evidence relevance labels and genuinely
held-out query families. The published 32 extension questions are draft,
not independently held out from the G1 concept inventory.
