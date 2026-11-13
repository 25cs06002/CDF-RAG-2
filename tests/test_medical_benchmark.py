"""Synthetic fixtures only; none are real clinical gold annotations."""

import json
import tempfile
import unittest
from pathlib import Path
from pydantic import ValidationError

from research_v2.medical_benchmark import (
    MedicalBenchmarkRecord, audit_benchmark, load_benchmark,
)


def draft(**overrides):
    data = {
        "query_id": "D001",
        "query": "Example medical question?",
        "query_type": "direct",
        "difficulty": "easy",
        "review_status": "draft",
    }
    data.update(overrides)
    return data


def reviewed(**overrides):
    data = {
        **draft(),
        "review_status": "source_reviewed",
        "reviewer_id": "synthetic_reviewer",
        "answerable": True,
        "gold_answer": "Synthetic answer, not a clinical claim.",
        "gold_entities": ["Concept A"],
        "evidence_sources": [{
            "source_id": "S1",
            "title": "Synthetic fixture",
            "url": "https://example.org/synthetic",
            "source_type": "other",
        }],
        "question_family": "qfam_a",
        "concept_family": "cfam_a",
        "source_family": "sfam_a",
        "split": "train",
    }
    data.update(overrides)
    return data


class TestMedicalBenchmark(unittest.TestCase):
    def test_draft_is_accepted_not_verified(self):
        record = MedicalBenchmarkRecord.model_validate(draft())
        result = audit_benchmark([record])
        self.assertEqual(result["review_status_counts"], {"draft": 1})
        self.assertFalse(result["clinical_correctness_independently_verified_by_code"])

    def test_source_reviewed_annotation(self):
        record = MedicalBenchmarkRecord.model_validate(reviewed())
        self.assertEqual(audit_benchmark([record], strict=True)["split_counts"], {"train": 1})

    def test_reviewed_relationship_requires_citation(self):
        item = reviewed(gold_causal_edges=[{
            "cause": "A", "effect": "B", "relation": "CAUSES"
        }])
        with self.assertRaisesRegex(ValidationError, "needs evidence IDs"):
            MedicalBenchmarkRecord.model_validate(item)

    def test_unknown_citation_rejected(self):
        item = draft(gold_causal_edges=[{
            "cause": "A", "effect": "B", "relation": "RISK_FACTOR_FOR",
            "evidence_source_ids": ["missing"]
        }])
        with self.assertRaisesRegex(ValidationError, "Unknown evidence source ID"):
            MedicalBenchmarkRecord.model_validate(item)

    def test_unsupported_requires_reason(self):
        with self.assertRaisesRegex(ValidationError, "abstention_reason"):
            MedicalBenchmarkRecord.model_validate(reviewed(answerable=False))

    def test_unsupported_can_have_no_evidence(self):
        record = MedicalBenchmarkRecord.model_validate(reviewed(
            answerable=False, abstention_reason="Not answerable within corpus.",
            gold_entities=[], gold_answer=None, evidence_sources=[],
        ))
        self.assertFalse(record.answerable)

    def test_reviewed_requires_reviewer(self):
        with self.assertRaisesRegex(ValidationError, "reviewer_id"):
            MedicalBenchmarkRecord.model_validate(reviewed(reviewer_id=None))

    def test_draft_cannot_have_final_split(self):
        with self.assertRaisesRegex(ValidationError, "Draft cannot"):
            MedicalBenchmarkRecord.model_validate(draft(split="train"))

    def test_duplicate_query_ids_detected(self):
        items = [MedicalBenchmarkRecord.model_validate(draft()),
                 MedicalBenchmarkRecord.model_validate(draft(query="Other?"))]
        with self.assertRaisesRegex(ValueError, "Duplicate query_id"):
            audit_benchmark(items)

    def test_duplicate_normalized_question_detected(self):
        items = [MedicalBenchmarkRecord.model_validate(draft()),
                 MedicalBenchmarkRecord.model_validate(draft(
                     query_id="D002", query="  EXAMPLE   medical  QUESTION? "))]
        with self.assertRaisesRegex(ValueError, "Duplicate normalized question"):
            audit_benchmark(items)

    def test_cross_split_concept_family_rejected(self):
        a = MedicalBenchmarkRecord.model_validate(reviewed())
        b = MedicalBenchmarkRecord.model_validate(reviewed(
            query_id="D002", query="Different query?", split="test",
            question_family="qfam_b", source_family="sfam_b"))
        with self.assertRaisesRegex(ValueError, "concept_family leakage"):
            audit_benchmark([a, b], strict=True)

    def test_cross_split_source_family_rejected(self):
        a = MedicalBenchmarkRecord.model_validate(reviewed())
        b = MedicalBenchmarkRecord.model_validate(reviewed(
            query_id="D002", query="Different query?", split="validation",
            question_family="qfam_b", concept_family="cfam_b"))
        with self.assertRaisesRegex(ValueError, "source_family leakage"):
            audit_benchmark([a, b], strict=True)

    def test_cross_split_paraphrase_family_rejected(self):
        a = MedicalBenchmarkRecord.model_validate(reviewed())
        b = MedicalBenchmarkRecord.model_validate(reviewed(
            query_id="D002", query="Different query?", split="test",
            concept_family="cfam_b", source_family="sfam_b"))
        with self.assertRaisesRegex(ValueError, "question_family leakage"):
            audit_benchmark([a, b], strict=True)

    def test_strict_rejects_drafts(self):
        item = MedicalBenchmarkRecord.model_validate(draft())
        with self.assertRaisesRegex(ValueError, "reviewed and split"):
            audit_benchmark([item], strict=True)

    def test_reviewed_same_group_same_split_allowed(self):
        a = MedicalBenchmarkRecord.model_validate(reviewed())
        b = MedicalBenchmarkRecord.model_validate(reviewed(
            query_id="D002", query="A paraphrase?"))
        result = audit_benchmark([a, b], strict=True)
        self.assertEqual(result["total_questions"], 2)

    def test_jsonl_reports_line_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bench.jsonl"
            path.write_text(json.dumps(draft()) + "\n" + '{"bad":true}\n')
            with self.assertRaisesRegex(ValueError, "line 2"):
                load_benchmark(path)

    def test_unknown_relation_type_rejected(self):
        with self.assertRaises(ValidationError):
            MedicalBenchmarkRecord.model_validate(draft(
                gold_causal_edges=[{"cause":"A", "effect":"B", "relation":"PROVEN"}]
            ))


if __name__ == "__main__":
    unittest.main()
