"""Non-destructive checks for local-result import path safety."""
import tempfile
import unittest
from pathlib import Path

from research_v2.unified_import_local import ROOT, _inside, REG


class LocalImportSafetyTests(unittest.TestCase):
    def test_registry_exists_and_is_tracked(self):
        self.assertTrue(REG.is_file())

    def test_source_paths_must_remain_inside_explicit_checkout(self):
        self.assertEqual(_inside("data/benchmark",ROOT),
                         (ROOT/"data/benchmark").resolve())
        with self.assertRaisesRegex(ValueError,"escapes"):
            _inside("../unrelated-folder",ROOT)

    def test_output_root_cannot_escape_checkout(self):
        with self.assertRaisesRegex(ValueError,"escapes"):
            _inside("/tmp/foreign-output-dir",ROOT)

    def test_optional_external_input_is_read_only_and_explicit(self):
        with tempfile.TemporaryDirectory() as scratch:
            external=Path(scratch)
            path=external/"presentation_20261118"/"dummy"
            path.parent.mkdir(parents=True)
            path.write_text("test")
            self.assertEqual(_inside("presentation_20261118/dummy",external),
                             path.resolve())
            with self.assertRaisesRegex(ValueError,"escapes"):
                _inside(ROOT/"readme.md",external)


if __name__=="__main__":
    unittest.main()
