"""Synthetic scope, orientation, pooling and threshold regression checks."""
import numpy as np
import unittest
from sklearn.metrics import confusion_matrix

from src.eyelid_color_data import ROBUST_NAMES
from src.eyelid_one_class import (CANDIDATES, adapt_normal, assisted_threshold,
    balanced_sources, calibration, fit_normal, inner_assignments, matrix, normal_threshold)


def sample(n=36):
    return np.random.default_rng(13).normal(size=(n, 18)), [f"local/{i}" for i in range(n)]


class OneClassTests(unittest.TestCase):
    def check_candidate(self, candidate):
        x, ids = sample()
        a = fit_normal(candidate, x, np.zeros(len(x)), ids, ["local/test"])
        b = fit_normal(candidate, x, np.zeros(len(x)), ids, ["local/test"])
        np.testing.assert_array_equal(a.score(x), b.score(x))
        assert a.score(np.full((1, 18), 12))[0] > np.median(a.score(x))
        np.testing.assert_allclose(a.scaler.mean_, x.mean(axis=0), atol=1e-14)
        with self.assertRaisesRegex(ValueError, "non-anemic"):
            fit_normal(candidate, x, np.ones(len(x)), ids, [])
        with self.assertRaisesRegex(ValueError, "Held-out"):
            fit_normal(candidate, x, np.zeros(len(x)), ids, [ids[0]])


    def test_schema_and_no_imputation(self):
        x, _ = sample()
        with self.assertRaises(ValueError): matrix(x, tuple(reversed(ROBUST_NAMES)))
        x[0, 0] = np.nan
        with self.assertRaises(ValueError): matrix(x)


    def test_equal_source_mass_and_adaptation_scope(self):
        x, ids = sample()
        public = x[:12]+5
        public_ids = [f"cp/{i}" for i in range(12)]
        a = balanced_sources([(x, ids), (public, public_ids)], ["local/test"])
        np.testing.assert_allclose(a.scaler.mean_, .5*x.mean(0)+.5*public.mean(0))
        assert np.isclose(a.weights[:36].sum(), .5)
        b = adapt_normal(balanced_sources([(public, public_ids)], []), x, np.zeros(36), ids, ["local/test"])
        np.testing.assert_allclose(b.scaler.mean_, x.mean(0))
        with self.assertRaises(ValueError): balanced_sources([(x, ids), (x, ids)], [])
        with self.assertRaises(ValueError): adapt_normal(a, x, np.zeros(36), ids, [])


    def test_cross_fit_assignments_ignore_positive_feature_values_and_count(self):
        a = inner_assignments(np.r_[np.zeros(36), np.ones(9)])
        b = inner_assignments(np.r_[np.zeros(36), np.ones(15)])
        np.testing.assert_array_equal(a[:36], b[:36])
        assert set(a[:36]) == {0, 1, 2}


    def test_normal_and_assisted_thresholds_with_ties(self):
        normals = np.array([1., 1., 2., 2., 3., 3., 4., 4., 5., 5.])
        positives = np.array([3., 6., 7.])
        a = normal_threshold(normals, .8)
        assert a["threshold"] == 5 and a["inner_normal_specificity"] == .8
        y = np.r_[np.zeros(10), np.ones(3)]
        b = assisted_threshold(y, np.r_[normals, positives], .8)
        assert b["inner_specificity"] == 1 and b["inner_sensitivity"] == 2/3
        assert confusion_matrix(y, np.r_[normals, positives] >= b["threshold"]).tolist() == [[10, 0], [1, 2]]
        assert normal_threshold(normals, .8) == a
        assert calibration(normals)["scale"] > 0

    def test_candidates(self):
        for candidate in CANDIDATES:
            with self.subTest(candidate=candidate["id"]):
                self.check_candidate(candidate["id"])
