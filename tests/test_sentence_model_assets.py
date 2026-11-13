import tempfile
import unittest

from pathlib import Path

from research_v2.sentence_model_assets import (
    fingerprint_snapshot,
)


class TestSentenceModelAssets(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        self.root = Path(self.tmp.name)

        (self.root / "1_Pooling").mkdir()

        (self.root / "modules.json").write_text(
            '{"modules":[]}',
            encoding="utf-8",
        )

        (
            self.root / "model.safetensors"
        ).write_bytes(
            b"synthetic-model-weights"
        )

        (
            self.root / "1_Pooling/config.json"
        ).write_text(
            '{"pooling":true}',
            encoding="utf-8",
        )

    def test_fingerprint_is_stable(self):
        first = fingerprint_snapshot(
            self.root
        )
        second = fingerprint_snapshot(
            self.root
        )

        self.assertEqual(
            first["snapshot_sha256"],
            second["snapshot_sha256"],
        )

        self.assertTrue(
            first["modules_json_present"]
        )

        self.assertIn(
            "model.safetensors",
            first["weight_files"],
        )

    def test_modified_weights_change_fingerprint(self):
        before = fingerprint_snapshot(
            self.root
        )

        (
            self.root / "model.safetensors"
        ).write_bytes(
            b"different-model-weights"
        )

        after = fingerprint_snapshot(
            self.root
        )

        self.assertNotEqual(
            before["snapshot_sha256"],
            after["snapshot_sha256"],
        )

    def test_added_file_changes_fingerprint(self):
        before = fingerprint_snapshot(
            self.root
        )

        (
            self.root / "tokenizer.json"
        ).write_text(
            '{"tokens":[]}',
            encoding="utf-8",
        )

        after = fingerprint_snapshot(
            self.root
        )

        self.assertNotEqual(
            before["snapshot_sha256"],
            after["snapshot_sha256"],
        )

    def test_nested_asset_is_included(self):
        result = fingerprint_snapshot(
            self.root
        )

        paths = [
            item["path"]
            for item in result["files"]
        ]

        self.assertIn(
            "1_Pooling/config.json",
            paths,
        )

    def test_empty_snapshot_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                ValueError,
                "No model files",
            ):
                fingerprint_snapshot(
                    Path(tmp)
                )


if __name__ == "__main__":
    unittest.main()
