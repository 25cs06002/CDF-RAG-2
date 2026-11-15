"""Lightweight validation of central B1-B8 pipeline registry."""
import unittest

from research_v2.pipelines import MODEL_REGISTRY, create_pipeline
from research_v2.base_pipeline import BaseRAGPipeline


class ModelRegistryTests(unittest.TestCase):
    def test_registry_lists_exactly_eight_pipelines(self):
        self.assertEqual(set(MODEL_REGISTRY), {f"B{i}" for i in range(1, 9)})
        self.assertEqual(len({x.implementation for x in MODEL_REGISTRY.values()}), 8)

    def test_graph_models_are_real_retrievers(self):
        for model_id in ("B5", "B6"):
            with self.subTest(model_id=model_id):
                model = create_pipeline(model_id)
                self.assertIsInstance(model, BaseRAGPipeline)
                self.assertEqual(model.pipeline_id, MODEL_REGISTRY[model_id].implementation)

    def test_document_models_require_source_data(self):
        for model_id in ("B1", "B2", "B3", "B4"):
            with self.subTest(model_id=model_id):
                with self.assertRaisesRegex(ValueError, "frozen source-attributed"):
                    create_pipeline(model_id)

    def test_legacy_models_can_be_instantiated_without_running_ollama(self):
        for model_id in ("B7", "B8"):
            with self.subTest(model_id=model_id):
                model = create_pipeline(model_id)
                self.assertIsInstance(model, BaseRAGPipeline)

    def test_canonical_modules_and_legacy_imports_share_class_identity(self):
        from research_v2.pipelines.baselines import BaselinePipeline as CanonicalBaseline
        from research_v2.presentation_baselines import BaselinePipeline as OldBaseline
        from research_v2.pipelines.legacy_no_rl import LegacyNoRLPipeline as CanonicalNoRL
        from research_v2.legacy_no_rl import LegacyNoRLPipeline as OldNoRL
        from research_v2.pipelines.legacy_rl import LegacyRLPipeline as CanonicalRL
        from research_v2.legacy_rl import LegacyRLPipeline as OldRL
        self.assertIs(CanonicalBaseline, OldBaseline)
        self.assertIs(CanonicalNoRL, OldNoRL)
        self.assertIs(CanonicalRL, OldRL)
        self.assertTrue(issubclass(CanonicalRL, CanonicalNoRL))

    def test_unknown_model_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown pipeline ID"):
            create_pipeline("B9")


if __name__ == "__main__":
    unittest.main()
