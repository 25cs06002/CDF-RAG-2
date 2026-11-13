# Phase 2I-C3 — Editable 94-source review workspace

This creates an independent editable copy of the verified Phase 2I-C2
handoff, **not** a new medical gold benchmark. Existing 70-question
decisions, 94 source review tasks, and 30 link placeholders are copied
without changing any recorded review. The C2 input is checksum-bound and
must remain untouched.

## Ubuntu

```bash
cd ~/Folder/CDF-RAG-2
source venv/bin/activate
git switch research-v2
git pull --ff-only origin research-v2
python3 -m unittest discover -s tests -p 'test_editable_combined_review.py' -v
python3 -m unittest discover -s tests -p 'test_*.py' -q

BASE="$HOME/Folder/CDF-RAG-2-experiment-artifacts/phase2/review70_audit_passed_v1"
SUPP="$HOME/Folder/CDF-RAG-2-experiment-artifacts/phase2/indirect5_supplement_reviewed_v1"
COMBINED="$HOME/Folder/CDF-RAG-2-experiment-artifacts/phase2/review70_combined94_v1"
EDITABLE="$HOME/Folder/CDF-RAG-2-experiment-artifacts/phase2/review70_editable94_v1"

python3 -m research_v2.editable_combined_review \
  --combined-dir "$COMBINED" --baseline-dir "$BASE" \
  --supplement-dir "$SUPP" --output-dir "$EDITABLE"
python3 -m research_v2.editable_combined_review \
  --combined-dir "$COMBINED" --baseline-dir "$BASE" \
  --supplement-dir "$SUPP" --audit-dir "$EDITABLE"
```

Initial counts match C2; original decisions are NOT automatically
promoted. On `$EDITABLE/source_review.csv`, verify what each new ADD
source actually supports. On `$EDITABLE/question_review.csv`, reconsider
only PILOT_034, PILOT_035, PILOT_036, PILOT_038, PILOT_044 if their
source support is sufficient. A source marked `partial` cannot be cited
as sole support for an `answerable` question. Record real reviewer
identity, actual date, exact evidence passages, and rationale. Retain
`needs_more_evidence` when the evidence is incomplete.

Do not change question/source candidate fields (IDs, URLs, original
needs_additional_sources flag). The auditor enforces the original
source-discovery flag even when new evidence is attached. The validator
checks only structure and claimed support relationships; no medical expert
adjudication, clinical accuracy, causal gold or train/test split.

Use non-strict audit for incomplete worksheets. Strict audit requires
all question, source, and link decisions; do not force these to
`supports` merely to pass strict mode. Preserve the C2 snapshot.
