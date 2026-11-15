"""Build a provisional, source-attributed text corpus for B1-B4.

Consumes existing Phase 2 AI-assisted evidence *summaries*, not original
copyrighted page text. These summaries have NOT been independently medically
adjudicated and MUST NOT be used as medical gold. Do not use question text,
gold answers, target question IDs or reviewer metadata as retrieved content.

Run:
  python3 -m research_v2.presentation_corpus prepare
"""
import argparse
import csv
import json
from pathlib import Path

from .experiment_runner import PROJECT_ROOT, file_sha256

WORKSPACE = PROJECT_ROOT / "presentation_20261118"
OUTPUT = WORKSPACE / "corpus/provisional_source_summaries_v1"
PROPOSALS = PROJECT_ROOT / "data/benchmark/evidence_proposals15_v2.jsonl"
SOURCE_REVIEWS = (
    PROJECT_ROOT
    / "research_artifacts/experiment-artifacts/phase2"
    / "phase2m_full_adjudication_handoff_v1"
    / "all52_source_review_tasks.csv"
)
SOURCE_REGISTRIES = (
    PROJECT_ROOT / "data/benchmark/source_candidates15_v2.jsonl",
    PROJECT_ROOT / "data/benchmark/additional_source_candidates8_v1.jsonl",
)


def records_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_candidate_documents():
    source_urls = {}
    for source_path in SOURCE_REGISTRIES:
        for row in records_jsonl(source_path):
            source_urls[row["source_id"]] = row["url"]

    documents, provenance, seen = [], [], set()

    def add(text, url, source_id, origin, note):
        text = " ".join(str(text).split())
        url = str(url or "").strip()
        if len(text) < 40 or not url.startswith("https://"):
            return
        if "?" in url and "token=" in url.lower():
            raise ValueError("Do not record access tokens in source URLs")
        key = (url.casefold(), text.casefold())
        if key in seen:
            return
        seen.add(key)
        doc_id = f"PHASE2_CAND_{len(documents) + 1:04d}"
        documents.append({
            "document_id": doc_id,
            "text": text,
            "source_id": url,
        })
        provenance.append({
            "document_id": doc_id,
            "candidate_source_id": source_id,
            "origin": origin,
            "scope_note": note,
            "independent_human_review": False,
            "medical_gold": False,
            "retrieved_text_role": "paraphrased_source_candidate_summary",
        })

    for item in records_jsonl(PROPOSALS):
        source_id = item.get("candidate_source_id")
        url = source_urls.get(source_id)
        for relation in item.get("proposed_relationships", []):
            add(
                relation.get("source_support_summary", ""),
                url, source_id,
                "phase2_evidence_proposal",
                item.get("unresolved_scope_issue"),
            )

    with SOURCE_REVIEWS.open(newline="", encoding="utf-8") as handle:
        for task in csv.DictReader(handle):
            # The original_claim is a paraphrased AI-assisted candidate summary.
            # Do NOT include original_passage or independent reviewer fields.
            add(
                task.get("original_claim", ""),
                task.get("source_url", ""),
                task.get("source_id", ""),
                "phase2_source_review_worklist",
                task.get("original_limitation", ""),
            )
    if not documents:
        raise ValueError("No candidate source summaries available")
    return documents, provenance


def prepare(output=OUTPUT):
    output = Path(output).expanduser().resolve()
    if not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("Corpus must stay inside the repository")
    if output.exists():
        raise FileExistsError(f"Preserve existing provisional corpus: {output}")

    documents, provenance = build_candidate_documents()
    output.mkdir(parents=True, exist_ok=False)
    docfile = output / "documents.jsonl"
    docfile.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in documents),
        encoding="utf-8",
    )
    (output / "provenance.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in provenance),
        encoding="utf-8",
    )
    manifest = {
        "corpus_role": "provisional_phase2_source_candidate_summaries",
        "document_count": len(documents),
        "unique_source_urls": len({d["source_id"] for d in documents}),
        "documents_sha256": file_sha256(docfile),
        "proposal_source_sha256": file_sha256(PROPOSALS),
        "review_task_source_sha256": file_sha256(SOURCE_REVIEWS),
        "human_verified": False,
        "medical_gold_created": False,
        "direct_source_page_text_ingested": False,
        "warning": (
            "AI-assisted paraphrases are candidate evidence, not independently "
            "verified medical claims. Check source scope, medical correctness, "
            "licensing and provenance before a clinical accuracy study."
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **manifest}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare"])
    args = parser.parse_args()
    if Path.cwd().resolve() != PROJECT_ROOT:
        parser.error("Run from CDF-RAG-2 project root")
    prepare()


if __name__ == "__main__":
    main()
