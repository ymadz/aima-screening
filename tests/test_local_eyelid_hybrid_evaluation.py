"""Independent hand-calculated threshold and paired-resampling checks."""
import unittest
import numpy as np
from src.local_eyelid_hybrid_evaluation import evaluate, conditional_intervals


class HybridEvaluationTests(unittest.TestCase):
    def test_fold_thresholds_are_applied_per_subject(self):
        y=np.array([0,0,1,1]);p=np.array([.4,.7,.3,.8]);t=np.array([.5,.8,.2,.9])
        actual=evaluate(y,p,t)
        self.assertEqual(actual['confusion_matrix_labels_0_1'],[[2,0],[1,1]])
        self.assertEqual(actual['sensitivity'],.5)
        self.assertEqual(actual['balanced_accuracy'],.75)
        self.assertEqual(actual['precision'],1)
        with self.assertRaises(ValueError):evaluate(y,p,np.array([.5,.5,np.nan,.5]))

    def test_identical_models_have_zero_paired_differences(self):
        y=np.array([0,0,0,0,1,1]);p=np.array([.1,.8,.2,.4,.7,.9])
        actual=conditional_intervals(y,p,.5,p,replicates=20)
        for name,row in actual['paired_delta_exp02_minus_fixed_exp01'].items():
            self.assertEqual(row['estimate'],0)
            self.assertEqual(row['interval'],[0,0])
        self.assertEqual(actual['paired_accuracy_discordance']['mcnemar_exact_two_sided_p'],1)
