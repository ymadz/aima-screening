"""Independent sklearn verification of tied-score bootstrap and metrics."""
import unittest
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from src.eyelid_one_class_evaluation import evaluate, draws, ranking_samples


class ScoreMetricTests(unittest.TestCase):
    def test_unbounded_scores_and_confusion(self):
        result = evaluate(np.array([0, 0, 1, 1]), np.array([-3., 1., .5, 7.]), 1.)
        self.assertEqual(result["confusion_matrix_labels_0_1"], [[1, 1], [1, 1]])
        self.assertEqual(result["roc_auc"], .75)

    def test_bootstrap_matches_sklearn_with_ties_and_duplicates(self):
        y = np.array([0, 1, 0, 1, 0, 1])
        score = np.array([.2, .2, -3, 4, 3, 4.])
        counts = draws(y, 80)
        auc, ap = ranking_samples(y, score, counts)
        for i in range(80):
            idx = np.r_[np.repeat(np.flatnonzero(y == 0), counts[0][i].astype(int)),
                        np.repeat(np.flatnonzero(y == 1), counts[1][i].astype(int))]
            self.assertAlmostEqual(auc[i], roc_auc_score(y[idx], score[idx]), places=14)
            self.assertAlmostEqual(ap[i], average_precision_score(y[idx], score[idx]), places=14)
        a = ranking_samples(y, score, draws(y, 80))
        np.testing.assert_array_equal(a[0], auc)
        np.testing.assert_array_equal(a[1], ap)
