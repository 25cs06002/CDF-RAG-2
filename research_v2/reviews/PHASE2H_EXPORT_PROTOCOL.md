# Phase 2H — Strict reviewed-candidate export

The Phase 2G first 30-question human-decision worksheet now has a separate,
**non-gold** structured export path. Reviewed source worksheets remain outside
this repository.

## Input contract

A review folder contains claim_review.csv, question_review.csv and
review_manifest.json. Use a frozen copy. SHA256SUMS.txt supplies optional
file-integrity verification.

The exporter invokes audit_review_workspace(strict=True) **before** any files
are written. This checks immutable field preservation, reviewer workflow
completeness and recorded source IDs, not clinical correctness or reviewer identity.

## Commands

    cd ~/Folder/CDF-RAG-2
    source venv/bin/activate
    FREEZE_DIR="$HOME/Folder/CDF-RAG-2-experiment-artifacts/phase2/reviewed30_frozen_v1"
    (cd "$FREEZE_DIR" && sha256sum -c SHA256SUMS.txt)
    OUT="$HOME/Folder/CDF-RAG-2-experiment-artifacts/phase2/reviewed30_export_$(date -u +%Y%m%dT%H%M%SZ)"
    python3 -m research_v2.reviewed_export --review-dir "$FREEZE_DIR" --checksums-file "$FREEZE_DIR/SHA256SUMS.txt" --output-dir "$OUT"
    (cd "$OUT" && sha256sum -c SHA256SUMS.txt)
    python3 -m unittest discover -s tests -p 'test_reviewed_export.py' -v

The output directory must not exist; output is never directly imported into G1.

## Export files

- reviewed_relationships_v1.jsonl — accept/revise-reviewed relationship
  **candidates**, preserving original and corrected triples
- unresolved_relationships_v1.jsonl — rejected / needs-more-evidence claims
- reviewed_questions_v1.jsonl — answerable question **drafts**, not gold answers
- unresolved_questions_v1.jsonl — questions with no sufficiently established
  answer in the registered source
- export_profile.json — counts, input hashes, strict audit and warnings
- SHA256SUMS.txt — independent checksums of outputs

All question records carry question/concept family fields and unset split.
No train/validation/test assignment, medical gold or graph edge import occurs.

## Adjudication warnings

PILOT_005 and PILOT_020 are marked answerable, but corresponding proposed
relationships EVP006:R01 and EVP006:R02 need more evidence. The exporter
preserves the answer drafts but flags them for separate causal-edge/path
adjudication.

The asthma onset questions PILOT_006 and PILOT_021 remain unresolved because
an asthma-trigger source does not establish initial disease causes.

Revisions EVP003:R02 and EVP015:R02 retain original and effective candidates.
A compound environmental exposure cannot be silently split into multiple
causal graph edges without separately evidenced review records.

The reviewer identity and medical accuracy of these candidates are not
independently verified by code. Resolve source gaps, adjudicate with domain
expertise as appropriate and protect held-out labels from PPO and RAG inputs.
