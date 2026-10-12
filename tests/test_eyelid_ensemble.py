"""Synthetic alignment, selection, threshold and leakage checks."""
import unittest
import numpy as np
import pandas as pd
from src.eyelid_ensemble import align_predictions, average_scores, training_policy, overlap


class EnsembleTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'subject_id': [f'S{i}' for i in range(10)],
                             'label': [0, 1] * 5, 'fold': np.repeat(np.arange(1, 6), 2)})

    def test_alignment_reorders_by_identity(self):
        a, b = align_predictions(self.frame(), self.frame().iloc[::-1])
        pd.testing.assert_frame_equal(a, b)

    def test_alignment_fails_closed(self):
        for field, value in [('subject_id', 'S1'), ('label', 1), ('fold', 2)]:
            b = self.frame(); b.loc[0, field] = value
            with self.assertRaises(ValueError): align_predictions(self.frame(), b)
        with self.assertRaises(ValueError): align_predictions(self.frame(), self.frame().iloc[:-1])

    def test_average_is_label_free(self):
        np.testing.assert_allclose(average_scores(np.array([.2, .8]), np.array([.6, .4]), .75), [.3, .7], rtol=0, atol=2e-16)

    def test_invalid_probability_inputs(self):
        for a, b, w in [([np.nan], [.1], .5), ([1.1], [.1], .5), ([.1], [.2, .3], .5), ([.1], [.1], 2)]:
            with self.assertRaises(ValueError): average_scores(np.array(a), np.array(b), w)

    def test_weight_ties_prefer_equal_and_training_only_thresholds(self):
        y = np.array([0] * 10 + [1] * 5); p = np.linspace(.01, .99, 15)
        policy = training_policy(y, p, p, [str(i) for i in range(15)], ['heldout'])
        self.assertEqual(policy['selected_lab_weight'], .5)
        for candidate in policy['candidates'].values():
            for target, rule in candidate['rules'].items():
                self.assertGreaterEqual(rule['inner_specificity'], float(target))
                self.assertEqual(rule['inner_sensitivity'], 1)

    def test_subject_leakage_is_rejected(self):
        with self.assertRaises(ValueError):
            training_policy(np.array([0, 0, 0, 1, 1, 1]), np.ones(6)*.2, np.ones(6)*.3,
                            list('abcdef'), ['a'])

    def test_known_error_overlap(self):
        y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        a = np.array([.8, .8, .2, .2, .8, .8, .2, .2])
        b = np.array([.8, .2, .8, .2, .8, .2, .8, .2])
        result = overlap(y, a, b, np.full(8, .5), np.full(8, .5))
        self.assertEqual(list(result['anemia_overlap'].values()), [1, 1, 1, 1])
        self.assertEqual(list(result['non_anemia_overlap'].values()), [1, 1, 1, 1])
        self.assertEqual(result['binary_agreement'], .5)


if __name__ == '__main__': unittest.main()
