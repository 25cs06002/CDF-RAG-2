"""Synthetic-only tests for Phase 2I-C1 supplemental evidence review."""
import csv
import tempfile
import unittest
from pathlib import Path
from research_v2.indirect_source_extension import create, audit

class TestSourceExtension(unittest.TestCase):
    def baseline(self, root):
        base = root / "baseline"
        base.mkdir()
        for name in ("question_review.csv","source_review.csv","link_review.csv","review_manifest.json"):
            (base / name).write_text("SYNTHETIC BASELINE FILE\n")
        return base

    def test_create_and_audit_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = self.baseline(root)
            work = root / "supplement"
            manifest = create(work, base)
            self.assertFalse(manifest["existing_reviews_migrated"])
            result = audit(work, base)
            self.assertEqual(result["source_tasks"], 5)
            self.assertEqual(result["decisions"]["pending"], 5)
            self.assertFalse(result["gold_labels_created"])

    def test_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = self.baseline(root)
            work = root / "supplement"
            create(work, base)
            with self.assertRaises(FileExistsError):
                create(work, base)

    def test_source_editable_but_immutable_locked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = self.baseline(root)
            work = root / "supplement"
            create(work, base)
            csvpath = work / "supplemental_source_review.csv"
            with csvpath.open(newline="") as f:
                reader = csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
            rows[0]["candidate_source_url"] = "https://example.org/tampered"
            with csvpath.open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "Original supplemental candidate modified"):
                audit(work,base)

    def test_baseline_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            base=self.baseline(root); work=root/"supplement"
            create(work,base)
            (base/"question_review.csv").write_text("ALTERED")
            with self.assertRaisesRegex(ValueError,"Baseline review snapshot modified"):
                audit(work,base)

if __name__ == "__main__":
    unittest.main()
