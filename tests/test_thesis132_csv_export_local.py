"""Offline integration tests for fixed-132 per-model CSV export.

Test records are SYNTHETIC. They do not demonstrate actual medical QA quality
or results for B1-B8; all live pipelines still require local execution.
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from research_v2.experiment_runner import PROJECT_ROOT as ROOT, file_sha256
from research_v2.schema import PipelineRecord, StageTimings, EvidenceDocument
from research_v2.thesis132_csv_export import export, result_to_row
from research_v2.thesis132_dataset_release import RELEASE, verify_release


def _jsonl(path, rows):
    path.write_text(
        "".join(row.model_dump_json() + "\n" for row in rows),
        encoding="utf-8",
    )


def create_synthetic_session(base: Path, model: str = "B1", faulty=False):
    audit = verify_release()
    records = [
        json.loads(line)
        for line in (RELEASE/"questions_only.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()
    ]
    session = base/"session"
    folder = session/"records"/"synthetic"
    folder.mkdir(parents=True)
    run_id = "SYNTHETIC_TEST_ONLY"
    simulated = [
        PipelineRecord(
            run_id=run_id, query_id=q["query_id"], pipeline_id="naive_rag",
            original_query=q["query"], graph_version="G0",
            retrieved_documents=[EvidenceDocument(document_id="TEST_DOC",
                                                  text="synthetic candidate")],
            final_answer=None, hallucination_detected=None,
            status="error" if faulty and index == 0 else "ok",
            error_message="SyntheticError" if faulty and index == 0 else None,
            timings=StageTimings(total_ms=10.0, retrieval_ms=6.0),
        ) for index, q in enumerate(records)
    ]
    _jsonl(folder/"records.jsonl", simulated)
    commit = "synthetic_test_commit"
    (folder/"manifest.json").write_text(json.dumps({
        "input_sha256":audit["query_sha256"],
        "git_commit":commit,"run_id":run_id
    }),encoding="utf-8")
    (folder/"summary.json").write_text(json.dumps({
        "total_queries":132,"successful":131 if faulty else 132,
        "abstained":0,"errors":1 if faulty else 0,
        "aborted":False,"wall_time_seconds":13.2,
    }),encoding="utf-8")
    (session/"session.json").write_text(json.dumps({
        "schema":"THESIS132_SESSION_V1",
        "dataset_sha256":audit["query_sha256"],
        "query_count":132,"mode":"end_to_end","encoder":"sentence",
        "top_k":5,"code_commit":commit,"candidate_corpus_sha256":"synthetic",
        "G0_nodes_sha256":"synthetic","G0_edges_sha256":"synthetic",
        "runs":{model:{"folder":"records/synthetic",
                       "run_id":run_id,
                       "record_sha256":file_sha256(folder/"records.jsonl")}},
    }),encoding="utf-8")
    return session, folder, simulated


class Thesis132CSVExportTests(unittest.TestCase):
    def test_no_detector_flag_or_quality_score_is_fabricated(self):
        rec = PipelineRecord(
            run_id="test", query_id="Q1", pipeline_id="bm25_rag",
            original_query="Sample question", status="ok",
            timings=StageTimings(total_ms=10, retrieval_ms=5)
        )
        row = result_to_row("B2",rec,
                            {"query_type":"direct","medical_domain":"general_medicine",
                             "semantic_group":"test_group","fold":"0"},
                            1,"end_to_end")
        self.assertIsNone(row["is_hallucination"])
        self.assertIsNone(row["validator_passed"])
        self.assertEqual(row["query_throughput_qps_proxy"],100)
        self.assertFalse(row["retrieval_success"])
        self.assertIsNone(row["generation_ms"])

    def test_132_rows_separate_csv_and_summary(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            base=Path(temp)
            session, folder, records = create_synthetic_session(base)
            out, report=export([session],output_root=base/"csv")
            self.assertEqual(report["models_present"],["B1"])
            self.assertEqual(report["models_missing"],[
                "B2","B3","B4","B5","B6","B7","B8"])
            with (out/"B1_queries_132.csv").open(newline="",encoding="utf-8") as f:
                per=list(csv.DictReader(f))
            self.assertEqual(len(per),132)
            self.assertEqual(per[0]["query_id"],records[0].query_id)
            self.assertEqual(per[-1]["query_id"],records[-1].query_id)
            self.assertEqual(per[0]["is_hallucination"],"")
            self.assertFalse((out/"B2_queries_132.csv").exists())
            with (out/"pipeline_summary_132.csv").open(newline="",encoding="utf-8") as f:
                summary=list(csv.DictReader(f))
            self.assertEqual(len(summary),1)
            self.assertEqual(summary[0]["status_ok"],"132")
            self.assertEqual(summary[0]["independent_medical_accuracy"],"")
            self.assertEqual(len((out/"comparison_by_query_132.csv").read_text(
                encoding="utf-8").splitlines()),133)
            self.assertEqual(report["generated_csv_sha256"]["B1_queries_132.csv"],
                             file_sha256(out/"B1_queries_132.csv"))

    def test_reject_missing_models_when_require_all(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            base=Path(temp)
            session, _, _ = create_synthetic_session(base)
            with self.assertRaisesRegex(ValueError,"missing model"):
                export([session],base/"csv",require_all=True)

    def test_reject_inference_errors_when_requested(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            base=Path(temp)
            session, _, _ = create_synthetic_session(base,faulty=True)
            out, report=export([session],base/"csv")
            self.assertEqual(report["models_with_errors"],["B1"])
            with self.assertRaisesRegex(ValueError,"execution errors"):
                export([session],base/"csv2",fail_on_errors=True)

    def test_reject_changed_question_or_tampered_hash(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            base=Path(temp)
            session, folder, recs = create_synthetic_session(base)
            recs[2] = recs[2].model_copy(update={"original_query":"Altered question"})
            _jsonl(folder/"records.jsonl",recs)
            with self.assertRaisesRegex(ValueError,"Modified experimental records"):
                export([session],base/"csv")
            session_metadata=json.loads((session/"session.json").read_text(encoding="utf-8"))
            session_metadata["runs"]["B1"]["record_sha256"]=file_sha256(
                folder/"records.jsonl")
            (session/"session.json").write_text(json.dumps(session_metadata),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"wording changed"):
                export([session],base/"csv")


if __name__ == "__main__":
    unittest.main()
