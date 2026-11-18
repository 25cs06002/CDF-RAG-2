"""Synthetic, lightweight acceptance tests for the MTP submission audit."""
import unittest
from collections import Counter, defaultdict

from research_v2.presentation_matrix import validate_inputs
from research_v2.partial_submission_sprint import Q100, _expansion, MODEL32, MODEL100
from research_v2.submission_readiness import (
    _validate_exported_rows, _slide_notes, _model_map
)


def sample():
    q100,meta=validate_inputs(Q100)
    q32,ext=_expansion()
    types={**{q.query_id:meta[q.query_id]["query_type"] for q in q100},**ext}
    records=[]
    for model in MODEL32:
        for query in [*q100,*q32]:
            records.append({
                "model_id":model,"query_id":query.query_id,
                "query_type":types[query.query_id],
                "status":"ok","returned_evidence":"True",
                "retrieval_ms":"0.5",
            })
    summary=[{
        "model":model,"questions":"132","evidence_returned":"132",
        "abstained":"0","errors":"0","mean_retrieval_ms":"0.5",
        "evidence_basis":(
            "provisional_AI_assisted_source_summaries" if model in MODEL32[:4]
            else "G0_unverified_graph"),
        "mode":"retrieval_only",
    } for model in MODEL32]
    categorized=defaultdict(list)
    for record in records:
        categorized[(record["model_id"],record["query_type"])].append(record)
    cat=[{
        "model":model,"query_type":typ,"questions":str(len(rows)),
        "evidence_returned":str(len(rows)),"abstained":"0",
    } for (model,typ),rows in categorized.items()]
    return summary,records,cat


class ReadinessTests(unittest.TestCase):
    def test_combined_792_records_pass_strict_consistency(self):
        summary,detail,cat=sample()
        _validate_exported_rows(None,summary,detail,cat)

    def test_rejects_duplicate_question_model(self):
        summary,detail,cat=sample()
        detail[1]=dict(detail[0])
        with self.assertRaisesRegex(ValueError,"Duplicate model/query"):
            _validate_exported_rows(None,summary,detail,cat)

    def test_rejects_misreported_evidence(self):
        summary,detail,cat=sample()
        summary[0]["evidence_returned"]="131"
        with self.assertRaisesRegex(ValueError,"Evidence yield"):
            _validate_exported_rows(None,summary,detail,cat)

    def test_slide_factsheet_generates_from_verified_counts(self):
        historical={
            model:{
                "status_ok":"100","status_abstained":"0","status_error":"0",
                "evidence_returned":"100","mean_total_ms":"2000.0",
            } for model in MODEL100
        }
        clean={
            model:{"evidence_returned":"132","abstained":"0",
                   "mean_retrieval_ms":"1.5"} for model in MODEL32
        }
        graph={
            "B5":{"G0_yield_132":"49","G1_legacy_yield_132":"53",
                  "G1_stable_yield_132":"55"},
            "B6":{"G0_yield_132":"41","G1_legacy_yield_132":"35",
                  "G1_stable_yield_132":"44"},
        }
        review={
            model:{"selected_changed_cases":"16",
                   "G1_stable_cases_with_evidence":"11",
                   "cases_with_relevant_direction_correct_evidence":"6"}
            for model in ("B5","B6")
        }
        manifest={
            "reviewed_evidence_occurrences":132,"reviewed_distinct_assertions":58,
            "attention_cases":17,
            "review_flags":{"wrong_stable_target":8,
                            "stable_target_uncertain":5,
                            "gain_without_relevant_direction_correct_evidence":6},
        }
        text=_slide_notes(None,manifest,historical,clean,graph,review)
        self.assertIn("| B5 | 49 | 53 | 55 |",text)
        self.assertIn("independently",text)
        self.assertIn("132",text)


if __name__=="__main__":
    unittest.main()
