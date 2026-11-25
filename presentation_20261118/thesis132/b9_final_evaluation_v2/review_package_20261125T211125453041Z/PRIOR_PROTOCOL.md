# External AI reviewer: B9-v2, 132 medical RAG answers

Evaluate every record in B9_review_inputs_132.jsonl. The other files in this
package are supporting context. No access to the user's machine or GitHub is
needed. The thesis is Adaptive Multi-Round RAG Architecture for Reliable
Domain-Specific Question Answering. B9-v2 uses bounded graph/document retrieval,
Ollama evidence-ID selection and deterministic rendering of candidate summaries.
The fixed 132 questions are development-exposed. G1 and document summaries are
provisional, not independently clinically verified. Do not infer medical
correctness from pipeline status, citations or the fact that B9 is proposed.

## Protocol: AIJ-MED-HALLUC v1, matching the B1-B8 AI reference

For ALL 132 records, extract atomic medically meaningful claims. Verify each
against actually inspected authoritative medical references, with source URL,
title, passage or finding, access mode and reasoning. Keep qualifiers and
distinguish causes, risk factors, association, triggers, subtype and direction.
Classify each claim SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED, UNSUPPORTED,
UNVERIFIABLE or NON_FACTUAL. UNSUPPORTED requires adequate search, not merely
absence of retrieved support. If unable to verify, use UNVERIFIABLE. Do not
invent quotations or source access; distinguish paraphrases and exact quotes.
Verify citations against the actual retrieved evidence separately.

Answer-level label Yes if any CONTRADICTED or UNSUPPORTED material claim or
fabricated citation; otherwise Uncertain if any UNVERIFIABLE claim; otherwise
No. Partial support alone is not automatically Yes. Empty answer has no
claims, label No for hallucination only, but may be inappropriate abstention.
Use AI_AUTO_ACCEPTED for Yes/No with medium/high confidence; AI_UNRESOLVED for
Uncertain with low confidence. Human status always NOT_REVIEWED.

Grounding: FULLY_GROUNDED, PARTIALLY_GROUNDED, NOT_GROUNDED, NO_FACTUAL_CLAIMS,
NOT_APPLICABLE_EMPTY_ANSWER. Check evidence entailment and citation scope, not
mere ID membership. Completeness: COMPLETE, PARTIAL, NONRESPONSIVE,
APPROPRIATE_ABSTENTION, INAPPROPRIATE_ABSTENTION, UNCERTAIN. An externally
correct but irrelevant answer is not necessarily hallucinated. A partial
pathway is not automatically complete. Review all empty answers separately.
Question category is only a development descriptor, not clinical ground truth.

## Deliver four UTF-8 files

1. ai_verified_hallucination_labels_132.csv: use the exact header and IDs
from B9_ai_label_template_132.csv; fill all labels and counts. Preserve exact
final_response_sha256. Count claims by verdict. Set fabricated_citation true
or false. Document rationale and source evidence; keep uncertainty. Preserve
protocol_version and label_provenance_statement as prefilled.

2. medical_claim_verification_132.jsonl: 132 objects with model, query_id,
query, final_response, final_response_sha256, ai_final_label, ai_review_status,
human_review_status, confidence, evidence_grounding_status, answer_completeness,
fabricated_citation (boolean), claims (array). Every claim object has claim
(exact assertion), verdict, reasoning, evidence (array of objects with
source_url, source_title, access_mode, passage_or_finding, evidence_kind
[exact_quote or paraphrase], supports_or_contradicts). Supported, partial
and contradicted claims require traceable evidence. Unsupported/unverifiable
can have empty evidence only with a reason. No claims for empty answers.

3. ai_hallucination_evaluation_summary.csv: one B9 row with model,
total_responses, auto_accepted_total, auto_accepted_yes, auto_accepted_no,
uncertain_unresolved, substantive_answers, substantive_answers_accepted,
externally_supported_answers, fully_grounded_answers. Substantive means at
least one factual claim (anything except NON_FACTUAL); externally_supported
means substantive accepted No, per historical metric. Compute, do not guess.

4. ai_evaluation_audit.json: protocol_version (AIJ-MED-HALLUC v1), evaluator
(model/version), annotation_date, human_clinical_review ("none"),
scope ({"total_responses":132,"records_processed":132}), review_input_sha256
(SHA256 of B9_review_inputs_132.jsonl bytes), run_manifest_sha256 (from
review_package_manifest.json), output_sha256 (mapping of exact filenames of
the other three outputs to actual SHA256 digests), access_limitations (array),
label_provenance_statement (exact prefilled statement in label template).

No human approval. No detector predictions for B9. Check every exact answer,
ID, hash, claim count and aggregate. If you cannot finish or compute hashes,
report partial results and limitations; do not fabricate. Return the four
files, not merely a text summary. Process in batches if necessary.
