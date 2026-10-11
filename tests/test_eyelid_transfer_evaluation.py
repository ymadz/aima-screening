import unittest
import numpy as np
from src.eyelid_transfer_evaluation import fast_metrics, compare_scores
from src.local_eyelid_hybrid_evaluation import evaluate, METRICS


class TransferEvaluationTests(unittest.TestCase):
    def test_bootstrap_metric_engine_matches_reference(self):
        y = np.array([0, 0, 1, 1, 0, 1]); p = np.array([.1, .6, .7, .4, .9, .6]); t = np.array([.5] * 6)
        one, other = fast_metrics(y, p, t), evaluate(y, p, t)
        for k in METRICS: self.assertEqual(one[k], other[k])

    def test_identical_scores_have_zero_paired_deltas(self):
        y = np.tile([0, 1], 5); p = np.linspace(.1, .9, 10); t = np.full(10, .5)
        registry = {n: (p, t) for n in ['exp01_fixed', 'exp02_fixed', 'exp02_selected', 'lab_nested_selected', 'A_fixed']}
        r = compare_scores(y, np.repeat(np.arange(1, 6), 2), registry, replicates=20)
        for k in METRICS:
            self.assertEqual(r['paired_deltas']['A_fixed']['exp01_fixed'][k]['estimate'], 0)
            self.assertEqual(r['paired_deltas']['A_fixed']['exp01_fixed'][k]['interval'], [0, 0])

if __name__ == '__main__': unittest.main()
