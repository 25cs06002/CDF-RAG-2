import unittest,json
from unittest.mock import patch
from research_v2.thesis132_b9_generators import check_reference,settings,comparable,MODELS
from research_v2.thesis132_b9_profile import load_profile,OUT

class GeneratorTests(unittest.TestCase):
    def test_frozen_config(self):
        ref=check_reference()
        with patch("research_v2.thesis132_b9_generators.generation_settings",return_value=ref["generation_settings"]):
            for k,v in MODELS.items():
                cfg=settings(k,ref)
                self.assertEqual(cfg["model"],v)
                self.assertEqual(cfg["num_ctx"],2048)
            self.assertEqual(ref["generation_settings"]["model"],"llama3.2:latest")
        with patch("research_v2.thesis132_b9_generators.generation_settings",return_value=dict(ref["generation_settings"],num_ctx=1024)):
            with self.assertRaises(ValueError):settings("llama1b",ref)
    def test_archived_compact_control(self):
        m,s,r=load_profile(OUT/"compact_ids_20261125T201909732485Z_07af85","compact_ids")
        self.assertEqual(len(r),132)
        self.assertEqual(s["answered"],100)
        self.assertEqual(s["total_selector_calls"],100)
    def test_matching_configuration(self):
        ref=check_reference()
        keys=("reference_manifest_sha256","inputs","dataset_sha256","selector_variant","policy","top_k","runner_sha256","profiler_sha256","host","resource_interval_seconds","initial_model_state","ollama_version","reference_labels_used_for_inference")
        a={k:None for k in keys};a["reference_labels_used_for_inference"]=False;a["generation_settings"]=ref["generation_settings"]
        b=dict(a,generation_settings=dict(a["generation_settings"],model="qwen2.5:1.5b"))
        comparable(a,b)
        with self.assertRaises(ValueError):comparable(a,dict(b,top_k=3))
