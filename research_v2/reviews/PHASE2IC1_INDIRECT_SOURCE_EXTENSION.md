# Phase 2I-C1 — Supplemental sources for five unresolved indirect questions

This is a *supplemental review workspace*, **not** an automatic migration of
the old 70-question review, a new answerability judgement, or medical gold.
Keep the original `review70_audit_passed_v1` snapshot immutable.

New versioned candidate registry:
`data/benchmark/indirect5_source_extension_v1.jsonl`.
Each additional URL is a source **candidate**, not automatically validated
evidence. The five tasks cover PILOT_034, PILOT_035, PILOT_036, PILOT_038,
and PILOT_044. The original source remains relevant, especially SRC014's
lung cancer risk-factor exposures. For asthma, differentiate airway
narrowing from causes of onset or attack triggers.

Run from repository root:

```bash
cd ~/Folder/CDF-RAG-2
source venv/bin/activate
git switch research-v2
git pull --ff-only origin research-v2
python3 -m unittest discover -s tests -p 'test_source_extension.py' -v
BASE="$HOME/Folder/CDF-RAG-2-experiment-artifacts/phase2/review70_audit_passed_v1"
OUT="$HOME/Folder/CDF-RAG-2-experiment-artifacts/phase2/indirect5_supplement_v1"
python3 -m research_v2.indirect_source_extension --baseline-dir "$BASE" --output-dir "$OUT"
python3 -m research_v2.indirect_source_extension --baseline-dir "$BASE" --audit-dir "$OUT"
```

Expected five pending supplemental-source review tasks. Open
`$OUT/supplemental_source_review.csv` and inspect the actual source
passages. Fill only `review_decision`, `source_passage_locator`,
`supported_claim_summary`, `limitation_note`, `reviewer_id`,
`reviewed_date`, `review_note`. Use actual review identity/date; do
not automatically approve candidates. Then rerun audit; strict mode
requires all five decisions.

Do not edit `$BASE` or copy new source IDs into the old CSV: the old
workspace intentionally rejects unregistered IDs. **Next stage (2I-C2)**
is an explicit, validated migration into a new versioned combined
workspace after supplemental-source review. That stage is not implemented
here. No G1 import or benchmark gold has been created.
