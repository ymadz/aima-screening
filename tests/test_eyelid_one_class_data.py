"""Public normal-group eligibility and conservative split checks."""
import unittest
import csv
from pathlib import Path
import tempfile
import numpy as np
from src.eyelid_one_class_data import normal_groups, archived_workflow_source
from src.local_eyelid_validation import sha256


class PublicScopeTests(unittest.TestCase):
    def test_nail_reset_bridge_requires_exact_snapshot_and_every_eyelid_field(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); parent = root/"data/raw/local/annotations"
            current = parent/"manifests/annotation_manifest.csv"
            archived = parent/"provenance/nail_annotation_reset_2026-10-13/manifests_before_reset/annotation_manifest.csv"
            current.parent.mkdir(parents=True); archived.parent.mkdir(parents=True)
            rows = [{"modality": "eyelid", "subject_id": str(i), "status": "approved"} for i in range(140)]
            def write(path, items):
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(items)
            write(archived, rows+[{"modality": "nail", "subject_id": "n", "status": "approved"}])
            digest = sha256(archived)
            write(current, rows+[{"modality": "nail", "subject_id": "n", "status": "pending"}])
            self.assertEqual(archived_workflow_source(root, current, digest), archived)
            changed = [dict(r) for r in rows]; changed[0]["status"] = "pending"; write(current, changed)
            with self.assertRaisesRegex(ValueError, "Eyelid rows changed"):
                archived_workflow_source(root, current, digest)
            with self.assertRaisesRegex(ValueError, "missing or damaged"):
                archived_workflow_source(root, current, "0"*64)

    def test_group_mean_and_mixed_labels_excluded(self):
        x = np.arange(12*18, dtype=float).reshape(12, 18)
        ids = [str(i) for i in range(12)]
        labels = np.r_[np.zeros(10), np.ones(2)]
        groups = ["same", "same", *[f"g{i}" for i in range(2, 9)], "mixed", "mixed", "positive"]
        values, names, audit = normal_groups(x, ids, labels, groups, set(ids), "eyes")
        self.assertEqual(len(names), 8)
        self.assertEqual(audit["mixed_label_groups_excluded"], 1)
        np.testing.assert_array_equal(values[names.index("eyes/same")], x[:2].mean(0))
        self.assertNotIn("eyes/mixed", names)
        self.assertNotIn("eyes/positive", names)

    def test_conservative_group_cannot_cross_public_split(self):
        x = np.ones((12, 18)); ids = [str(i) for i in range(12)]
        with self.assertRaisesRegex(ValueError, "crosses"):
            normal_groups(x, ids, np.zeros(12), ["same", "same", *ids[2:]], set(ids[1:]), "cp")
