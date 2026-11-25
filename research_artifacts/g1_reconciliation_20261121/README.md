# G1 reconciled snapshot — 2026-11-21

Versioned publication of the five graph files from `CDF_RAG_2_G1_FINAL_FIVE_FILES_20261121.zip`, maintained separately from the original November 17 candidate and its local review workspaces.

## What is included

Exactly these five graph files reside in `../graph-G1-reconciled-20261121/`:

- `manifest.json`
- `nodes.csv` — 78 nodes, identical to the original G1 candidate
- `edges.csv` — 52 edges, identical to the original G1 candidate; 37 inherited G0 edges remain unverified
- `typed_relationships.csv` — 24 candidate typed relations (15 CAUSES, 8 RISK_FACTOR_FOR, 1 ASSOCIATED_WITH), exact canonical-case normalization for four endpoint fields, and source crosswalk metadata for each
- `approved_aliases.csv` — unchanged empty alias registry

This directory contains the hash-verified **identical bytes** of all five files in the user-approved ZIP, rather than silently overwriting the previous G1 graph. The typed relationships SHA-256 is `d17533577dff7315046ce688491d7c33c65cff936f241727812f7ae29ebf5e9e`. The original graph is still preserved in the ignored `local_g1_snapshots` workspace on Ubuntu.

## Supporting provenance

- `phase2m_source_crosswalk.csv`: original source-ID/URL mapping, 52 rows; SHA-256 `994439c3eb6d92eecb4b77b1698d3baf0afb3978bb8efd38d218f2fe5cf3fd8e`.
- `G1_FINAL_FIVE_FILES_AUDIT_20261121.json`: byte checksums, counts, the four canonical casing repairs, and explicit limits of review evidence.
- The full historical source_review.csv and signed reviewer evidence remain outside this tracked snapshot.

A user attested on 2026-11-21 that the 24 links were independently verified. This claim is **recorded as a user attestation**; the supplied source-review and crosswalk files only identify student reviewer `STU_001` and `proposed` crosswalk status. Software cannot manufacture missing independent reviewer IDs, signed decisions, or authentic verbatim source passages. Therefore this is **not clinical gold**, and the graph manifest deliberately retains `stage=G1_CANDIDATE_REQUIRES_INDEPENDENT_VERIFICATION` and `medical_gold=false`. Runtime compatibility keeps `review_status=approved` in the typed records, not `independently_adjudicated`.

Do not treat typed review approval as identical to medical evidence verification. Do not silently relabel inherited G0 edges as independently reviewed, infer causation from `RISK_FACTOR_FOR`/`ASSOCIATED_WITH`, or report yield as clinical accuracy.

## Ubuntu: fetch/pull the tracked version

```bash
cd "$HOME/Folder/CDF-RAG-2-thesis132"
git status --short
git fetch origin work/thesis132-g1-target-sufficiency
git pull --ff-only origin work/thesis132-g1-target-sufficiency

export G1_FINAL="$PWD/research_artifacts/graph-G1-reconciled-20261121"
python -m unittest discover -s tests -p 'test_g1_reconciled_local.py' -v
python -m research_v2.thesis132_g1 --graph-dir "$G1_FINAL" --triage-dir "$TRIAGE" --top-k 5
```

`TRIAGE` must point to the real local, gitignored November 18 17-case folder, not a path from GitHub. The target diagnostic creates a new run and does not overwrite the original G1 run. Never use `git reset --hard` or `git clean -fd` in the source checkout.

Only the separately tracked `research_artifacts/graph-G1-reconciled-20261121/` is distributed by Git. The original machine's ignored `research_artifacts/local_g1_snapshots/`, `presentation_20261118/thesis132/`, `.env`, outputs and private reviewer materials are deliberately **not** synchronized through public GitHub.
