import json,tempfile,unittest
from pathlib import Path
from research_v2.thesis132_release import export,verify,ROOT,readcsv

class ReleaseTests(unittest.TestCase):
    def test_real_export(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            d=export(Path(temp)/"release",plots=False)
            verify(d)
            models=readcsv(d/"table_B1_B9_132.csv")
            variants={r["run"]:r for r in readcsv(d/"table_B9_variants_132.csv")}
            self.assertEqual(len(models),9)
            self.assertEqual(len(variants),11)
            self.assertEqual(models[-1]["substantive_answers"],"99")
            self.assertEqual(variants["B9_v2"]["complete_answers"],"36")
            self.assertEqual(variants["llama3b"]["complete_answers"],"30")
            self.assertEqual(json.loads((d/"final_architecture_lock.json").read_text())["primary"],"B9_v2")
            (d/"README.md").write_text("tampered")
            with self.assertRaises(ValueError):verify(d)
