"""Public normal-group eligibility and conservative split checks."""
import unittest
import numpy as np
from src.eyelid_one_class_data import normal_groups


class PublicScopeTests(unittest.TestCase):
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
