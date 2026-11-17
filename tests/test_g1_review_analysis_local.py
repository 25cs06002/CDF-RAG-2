"""Focused synthetic review-analysis checks; no offline sprint, LLM or Neo4j."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from research_v2.experiment_runner import PROJECT_ROOT
from research_v2.g1_relevance_handoff import CASE_FIELDS, EVIDENCE_FIELDS
from research_v2.g1_review_analysis import analyze


def write(path,columns,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def case(case_id,transition,g0,legacy,stable,typed):
    return {
        "case_id":case_id,"question_set":"synthetic/questions.jsonl",
        "query_id":case_id.split("_",1)[-1],"question":"What is the cause?",
        "phase2m_graph_construction_seed":"False","family":"B5",
        "g0_target":"","g1_legacy_target":"","g1_stable_target":"Mock condition",
        "stable_selection_reason":"test","g0_evidence":str(g0),
        "g1_legacy_evidence":str(legacy),"g1_stable_evidence":str(stable),
        "g1_typed_evidence":str(typed),"original_G1_change":transition,
        "g0_to_stable_change":transition,
        "review_query_intent":"answerable","review_stable_target":"appropriate",
        "reviewer":"TEST_REVIEWER","review_date":"2026-11-17",
        "review_note":"Synthetic evidence review, not a medical statement.",
    }


def evidence(c,stage,doc,provenance,checked):
    return {
        "case_id":c["case_id"],"query_id":c["query_id"],"family":"B5",
        "stage":stage,"document_id":doc,"subject":"Mock exposure",
        "relation":("CAUSES" if provenance!="inherited_G0_unverified"
                    else "G0_UNVERIFIED_CAUSAL_EDGE"),
        "object":"Mock condition","evidence_provenance":provenance,
        "original_supporting_source_ids":"",
        "g1_supporting_source_ids":("TEST_SOURCE" if checked else ""),
        "source_urls":("https://example.org/placeholder" if checked else ""),
        "reviewer_paraphrase":("Synthetic source claim" if checked else ""),
        "review_evidence_relevance":"relevant",
        "review_relation_direction":"correct",
        "review_source_checked":"checked" if checked else "not_checked",
        "reviewer":"TEST_REVIEWER","review_date":"2026-11-17",
        "review_note":"Synthetic review, no actual source was checked.",
    }


class AnalysisTests(unittest.TestCase):
    def test_collapses_identical_evidence_across_stages(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as root:
            root=Path(root)
            c=case("B5_P001","gained",False,True,True,False)
            e=[
                evidence(c,"G1_LEGACY","G0_EDGE_0038","G1_new_candidate_CAUSES",True),
                evidence(c,"G1_STABLE","G0_EDGE_0038","G1_new_candidate_CAUSES",True),
            ]
            write(root/"review_cases.csv",CASE_FIELDS,[c])
            write(root/"review_evidence.csv",EVIDENCE_FIELDS,e)
            (root/"manifest.json").write_text(json.dumps({
                "mode":"manual_relevance_and_target_review_not_medical_gold",
                "review_case_count":1,"evidence_review_rows":2,
            }))
            _,report=analyze(root,root)
            self.assertEqual(report["reviewed_evidence_occurrences"],2)
            self.assertEqual(report["distinct_question_evidence_assertions"],1)
            self.assertEqual(report["by_family"]["B5"]["G1_STABLE"]
                             ["cases_with_relevant_direction_correct_checked_G1_candidate"],1)

    def test_rejects_incomplete_relevance_decisions(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as root:
            root=Path(root)
            c=case("B5_P001","gained",False,True,True,False)
            e=[
                evidence(c,"G1_LEGACY","G0_EDGE_0038","G1_new_candidate_CAUSES",True),
                evidence(c,"G1_STABLE","G0_EDGE_0038","G1_new_candidate_CAUSES",True),
            ]
            e[0]["review_evidence_relevance"]=""
            write(root/"review_cases.csv",CASE_FIELDS,[c])
            write(root/"review_evidence.csv",EVIDENCE_FIELDS,e)
            (root/"manifest.json").write_text(json.dumps({
                "mode":"manual_relevance_and_target_review_not_medical_gold",
                "review_case_count":1,"evidence_review_rows":2,
            }))
            with self.assertRaisesRegex(ValueError,"Missing/invalid evidence review"):
                analyze(root,root)


if __name__=="__main__":
    unittest.main()
