"""Deterministic full-fold smoke and fitting/threshold leakage regressions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.eyelid_one_class import CANDIDATES, TRACKS, normal_threshold
from src.eyelid_one_class_run import crossfit_outer, verify, check_model, check_expected_scope
from src.local_eyelid_validation import sha256


class NestedScopeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.x = rng.normal(size=(45, 18)); self.y = np.r_[np.zeros(36), np.ones(9)].astype(int)
        self.x[36:] += 4
        self.ids = [f"local/{i}" for i in range(45)]
        self.test_ids = [f"local/test{i}" for i in range(6)]
        self.test_x = rng.normal(size=(6, 18))
        self.public = {"eyes": rng.normal(size=(12, 18))+1, "cp": rng.normal(size=(15, 18))+2,
            "eyes_ids": np.array([f"eyes/{i}" for i in range(12)]),
            "cp_ids": np.array([f"cp/{i}" for i in range(15)])}

    def fold(self, name, x=None, heldout=None):
        return crossfit_outer(name, self.x if x is None else x, self.y, self.ids,
            self.test_ids, self.test_x if heldout is None else heldout, self.public)

    def test_complete_deterministic_smoke_every_track(self):
        positive_ids = set(self.ids[36:])
        for name in [c["id"] for c in CANDIDATES]+list(TRACKS):
            with self.subTest(name=name):
                a = self.fold(name); b = self.fold(name)
                np.testing.assert_array_equal(a["inner_scores"], b["inner_scores"])
                np.testing.assert_array_equal(a["outer_scores"], b["outer_scores"])
                self.assertTrue(np.isfinite(a["outer_scores"]).all())
                for scope in a["inner_scopes"]:
                    fitted = set(scope["training_ids"]) | set(scope["adaptation_ids"] or [])
                    self.assertFalse(fitted & positive_ids)
                    self.assertFalse(fitted & set(scope["forbidden_ids"]))

    def test_heldout_values_and_training_positive_features_cannot_affect_policy_A(self):
        original = self.fold("mahalanobis")
        heldout = self.fold("mahalanobis", heldout=self.test_x+1000)
        changed = self.x.copy(); changed[self.y == 1] -= 1000
        positive = self.fold("mahalanobis", x=changed)
        for result in [heldout, positive]:
            np.testing.assert_array_equal(result["inner_scores"][self.y == 0], original["inner_scores"][self.y == 0])
            self.assertEqual(normal_threshold(result["inner_scores"][self.y == 0], .85),
                             normal_threshold(original["inner_scores"][self.y == 0], .85))

    def test_public_heldout_group_collision_fails(self):
        self.public["cp_ids"] = self.public["cp_ids"].astype(object)
        self.public["cp_ids"][0] = "local/test0"
        with self.assertRaisesRegex(ValueError, "Held-out"):
            self.fold("C1")

    def test_checksum_failure_precedes_model_deserialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary); (output/"item.txt").write_text("before")
            (output/"checksum_manifest.json").write_text(json.dumps({"item.txt": sha256(output/"item.txt")}))
            (output/"item.txt").write_text("after")
            with patch("src.eyelid_one_class_run.joblib.load") as load:
                with self.assertRaisesRegex(ValueError, "Checksum"):
                    verify(output)
                load.assert_not_called()

    def test_saved_statistics_and_complete_source_scope_every_track(self):
        lookup = dict(zip(self.ids, self.x)); labels = dict(zip(self.ids, self.y))
        for source in ("eyes", "cp"):
            lookup.update(dict(zip(self.public[source+"_ids"], self.public[source])))
            labels.update({s: 0 for s in self.public[source+"_ids"]})
        for name in [c["id"] for c in CANDIDATES]+list(TRACKS):
            saved = {}
            def save(stage, model):
                saved[stage] = model
                return stage
            result = crossfit_outer(name, self.x, self.y, self.ids, self.test_ids, self.test_x, self.public, save)
            for scope in result["inner_scopes"]:
                model = saved[scope["model_path"]]
                check_expected_scope(name, model, [self.ids[i] for i in scope["local_normal_indices"]], self.public)
                check_model(model, lookup, labels, scope["forbidden_ids"])
