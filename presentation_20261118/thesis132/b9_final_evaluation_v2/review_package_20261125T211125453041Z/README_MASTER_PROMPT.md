# B9 experiment variants: deduplicated external AI review
Review every unique record in review_cases.jsonl using the same AIJ-MED-HALLUC v1
protocol as the previously reviewed B9-v2. This is a development-exposed medical
RAG experiment, not independent clinical validation. Variant names and timing
are withheld; all cases use model=B9. review_id uniquely identifies a case;
query_id can repeat. Do not skip any case. Do not infer correctness from citations.
The original protocol is attached as PRIOR_PROTOCOL.md; follow its claim,
grounding, completeness and uncertainty rules, with the output schema below
instead of its original fixed-132 instructions. Preserve exact query, answer,
IDs and answer SHA256. Prior annotations were single-AI judgments and source
access was limited; use the same sentence-level claim granularity (split
independent assertions when needed) and report actual source access.
Check each medically material claim against authoritative sources actually
inspected; distinguish causes, association, risk, triggers and subtypes.
Do not fabricate evidence, quotations, hashes or human approvals. Empty
answers have no factual claims and hallucination No; separately assess
abstention appropriateness. Partial pathways are not complete by default.
A true but irrelevant answer is not necessarily a hallucination.

Return THREE UTF-8 files:
1. ai_variant_labels.csv, exact header in review_label_template.csv. Fill
every row, preserve review_id, model, query_id, final_response_sha256,
protocol_version and provenance. Use AI_AUTO_ACCEPTED or AI_UNRESOLVED and
NOT_REVIEWED, with Yes/No/Uncertain according to PRIOR_PROTOCOL.md.
2. ai_variant_claims.jsonl, one object per review_id. Fields: review_id,
model, query_id, query, final_response, final_response_sha256, ai_final_label,
ai_review_status, human_review_status, confidence, evidence_grounding_status,
answer_completeness, fabricated_citation (boolean), claims (array). Each
claim: claim, verdict, reasoning, evidence (array of objects: source_url,
source_title, access_mode, passage_or_finding, evidence_kind [exact_quote
or paraphrase], supports_or_contradicts). Follow the previous protocol's
claim counts and evidence requirements.
3. ai_variant_audit.json: protocol_version, evaluator, annotation_date,
human_clinical_review="none", scope={"total_responses":N,"records_processed":N}
where N is the actual number of cases; review_input_sha256 (SHA256 of exact
review_cases.jsonl bytes), output_sha256 (mapping of the two actual output
filenames to SHA256), access_limitations (array), label_provenance_statement
(exact template string). Do not invent hashes; compute with a file tool.
Return partial work honestly if incomplete. Use a batch workflow if needed.
No summary CSV required: the local validator calculates all aggregates.
