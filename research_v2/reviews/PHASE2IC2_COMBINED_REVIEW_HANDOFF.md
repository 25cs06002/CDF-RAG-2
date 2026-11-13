# Phase 2I-C2 — Combined 70-question / 94-source review handoff

**Important:** This is a **read-only migrated snapshot**, not a medically
adjudicated benchmark, not an editable 94-source review validator, and
not a new set of approved answers.

It verifies the existing 70-question audit and the strict audit for five
supplemental source reviews, then copies the exact existing review decisions
into a **new directory**. Five extra sources are appended to the corresponding
questions and to the source worksheet. The original source and question
decisions are neither upgraded nor discarded. Review counts: 70 questions,
94 source-review tasks (89 + 5), 30 unverified link slots.

It does not modify `remaining_review_workspace.py` (which users might have
locally modified for diagnostics), the untracked reconciliation script, or
the original 30-question frozen export.

## Ubuntu commands

```bash
cd ~/Folder/CDF-RAG-2
source venv/bin/activate
git switch research-v2
git pull --ff-only origin research-v2
python3 -m unittest discover -s tests -p 'test_combined_review_handoff.py' -v
python3 -m unittest discover -s tests -p 'test_*.py' -q

BASE="$HOME/Folder/CDF-RAG-2-experiment-artifacts/phase2/review70_audit_passed_v1"
SUPP="$HOME/Folder/CDF-RAG-2-experiment-artifacts/phase2/indirect5_supplement_reviewed_v1"
COMBINED="$HOME/Folder/CDF-RAG-2-experiment-artifacts/phase2/review70_combined94_v1"

python3 -m research_v2.combined_review_handoff \
  --baseline-dir "$BASE" --supplement-dir "$SUPP" --output-dir "$COMBINED"
python3 -m research_v2.combined_review_handoff \
  --baseline-dir "$BASE" --supplement-dir "$SUPP" --audit-dir "$COMBINED"
```

The output contains `question_review.csv`, `source_review.csv`,
`link_review.csv`, `migration_manifest.json`. Its audit rejects edits
to combined review fields because this is a checksum-traceable handoff.
**Do not edit** combined CSVs, or promote any `partial` source to
`supports` without medical/source re-review.

## Following milestone

Implement a separate validated **editable** 94-source workspace that can
record new review outcomes and re-adjudicate the five indirect questions.
The previous 70-question snapshot remains a historical audit state. Any
source originally marked partial is only partial after migration. Neither
migration nor its validation proves clinical correctness or establishes gold.
