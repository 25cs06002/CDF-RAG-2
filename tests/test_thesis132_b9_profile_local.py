import json,unittest
from research_v2.thesis132_b9_profile import compact_prompt,restore_selection,Selector,Monitor,profile_summary
class ProfileTests(unittest.TestCase):
    def test_roundtrip_and_group_mapping(self):
        p="Select\n"+json.dumps({"items":[{"id":"LONG_A","text":"a"},{"id":"LONG_B","text":"b"}],"groups":{"target":["LONG_B"]}})
        s,m=compact_prompt(p);o=json.loads(s.split("\n",1)[1])
        self.assertEqual(o["groups"],{"target":["E2"]})
        self.assertEqual(json.loads(restore_selection('{"selected_ids":["E2"]}',m)),{"selected_ids":["LONG_B"]})
        self.assertEqual(json.loads(restore_selection('{"selected_ids":[]}',m)),{"selected_ids":[]})
        self.assertEqual(restore_selection('{"selected_ids":["LONG_B"]}',m),"INVALID_COMPACT_SELECTION")
    def test_telemetry(self):
        sel=Selector("compact_ids",{},lambda p:('{"selected_ids":["E1"]}',dict(eval_count=4,eval_duration=200000000,http_attempts=1)))
        sel.query_id="Q"
        answer=sel("Select\n"+json.dumps({"items":[{"id":"LONG_A"}],"groups":{}}))
        self.assertEqual(json.loads(answer)["selected_ids"],["LONG_A"])
        self.assertEqual(sel.events[0]["query_id"],"Q")
        with Monitor(.01) as m:pass
        s=profile_summary(sel.events,m.samples)
        self.assertEqual(s["decode_tokens_per_second"],20)
        self.assertGreater(s["peak_sampled_system_ram_used_kib"],0)
    def test_unknown_or_duplicate_never_valid(self):
        from research_v2.thesis132_b9_v2 import parse_selection
        m={"E1":"LONG_A"}
        for raw in ('{"selected_ids":["E2"]}','{"selected_ids":["E1","E1"]}','{"selected_ids":["LONG_A"]}','junk'):
            self.assertIsNone(parse_selection(restore_selection(raw,m),["LONG_A"],5))
