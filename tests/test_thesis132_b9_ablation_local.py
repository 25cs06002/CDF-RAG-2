import unittest
from research_v2.thesis132_b9_ablation import read_run,matching,metrics,paired
from research_v2.thesis132_b9_review import RUN,META,readcsv

class AblationTests(unittest.TestCase):
    def test_archived_adaptive(self):
        m,s,rows=read_run(RUN,"adaptive")
        meta={r["query_id"]:r for r in readcsv(META)}
        rows=[dict(r,query_type=meta[r["query_id"]]["query_type"]) for r in rows]
        q=metrics(rows,s["wall_time_seconds"])
        self.assertEqual(q["answered"],99)
        self.assertEqual(q["negative_answered"],0)
        self.assertEqual(q["partial_path_answers"],11)
        self.assertEqual(q["total_selector_calls"],100)
        self.assertEqual(paired(rows,rows)["mean_common_answered_latency_delta_ms"],0)
        matching(dict(adaptive=m,fixed1=m,fixed3=m))
        with self.assertRaises(ValueError):matching(dict(adaptive=m,fixed1=dict(m,top_k=4)))
    def test_pair_uses_common_answered(self):
        a=[dict(query_id="q1",status="ok",answer="a",latency_ms=10,rounds=1,selector_calls=1),
           dict(query_id="q2",status="abstained",answer="",latency_ms=1,rounds=0,selector_calls=0)]
        b=[dict(a[0],latency_ms=8),dict(a[1],status="ok",answer="b",latency_ms=20)]
        r=paired(a,b)
        self.assertEqual(r["both_answered"],1)
        self.assertEqual(r["mean_common_answered_latency_delta_ms"],-2)
        self.assertEqual(r["mean_all_query_latency_delta_ms"],8.5)
