import json,tempfile,unittest,zipfile
from pathlib import Path
from research_v2.thesis132_b9_final import ROOT,review_plan,prepare,signature,semantic_pairs,RUNS
class FinalTests(unittest.TestCase):
    def test_real_archives_and_plan(self):
        rows,hashes,prior,cases,links=review_plan()
        self.assertEqual(len(rows),1452)
        self.assertEqual(len(hashes),11)
        self.assertEqual(len(prior),132)
        self.assertTrue(all(r["review_source"]=="EXACT_PRIOR_AI_REVIEW" for r in links if r["run"]=="B9_v2"))
        self.assertLess(len(cases),1452-132)
        self.assertEqual(len(links),len(rows))
        print("Unique review cases:",len(cases),"reused prior rows:",sum(r["review_source"]=="EXACT_PRIOR_AI_REVIEW" for r in links))
        self.assertNotEqual(signature(rows[0]),signature(dict(rows[0],final_response="changed")))
        self.assertNotEqual(signature(rows[0]),signature(dict(rows[0],retrieved_documents=[{"text":"changed"}])))
    def test_package(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            d=prepare(Path(tmp)/"package")
            with zipfile.ZipFile(d.with_suffix(".zip")) as z:
                self.assertNotIn("local_review_mapping.csv",z.namelist())
                self.assertIn("review_cases.jsonl",z.namelist())
                self.assertIsNone(z.testzip())
    def test_semantic_bases(self):
        r=dict(query="q",final_response="a",refined_query=None,rewritten_knowledge=None,retrieved_documents=[{"text":"d"}])
        self.assertEqual(semantic_pairs(r)["grounding_cosine_proxy"],("a","d"))
        self.assertEqual(semantic_pairs(r)["srs_cosine_proxy"],("q",""))
