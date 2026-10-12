"""Nested isolation, guarded resume and metric/checksum smoke tests."""
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.cp_transfer_models import CANDIDATES, OBJECTIVES
from src.cp_transfer_run import fit_outer, classical_fit
from src.cp_transfer_reproduce import checksum_guard, check_fusion
from src.cp_transfer_evaluation import regression_metrics
from src.preserve_local_experiment import tree_checksums


class CPRunTests(unittest.TestCase):
    def test_outer_api_and_heldout_inputs_cannot_change_inner_decisions(self):
        self.assertNotIn('validation_labels', inspect.signature(fit_outer).parameters)
        rng = np.random.default_rng(10)
        x = rng.normal(size=(40, 594))
        y = np.array([0] * 28 + [1] * 12)
        ids = [f's{i}' for i in range(40)]
        maps = torch.tensor(rng.normal(size=(40, 2, 1, 1, 1)), dtype=torch.float32)
        heldout = torch.ones((8, 2, 1, 1, 1))
        features = {o: x.copy() for o in OBJECTIVES}
        outside = {o: rng.normal(size=(8, 594)) for o in OBJECTIVES}
        initial = {o: {} for o in OBJECTIVES}
        calls = []

        def fit(state, cache, labels, tr, va, subjects, strategy, directory, seed, cache_hash, fixed_epochs=None):
            self.assertEqual(subjects, ids)
            self.assertEqual(len(cache), 40)
            self.assertTrue(np.array_equal(labels, y))
            self.assertFalse(set(tr) & set(va if va is not None else []))
            directory.mkdir(parents=True, exist_ok=True)
            calls.append((strategy, seed, subjects))
            return strategy, {'best_epoch': fixed_epochs or 2}

        def predict(model, cache):
            return (.3 + .4 * torch.sigmoid(cache[:, 0, 0, 0, 0])).numpy().astype(float)

        with tempfile.TemporaryDirectory() as temporary, patch('src.cp_transfer_run.fit_model', side_effect=fit), patch('src.cp_transfer_run.predict', side_effect=predict):
            root = Path(temporary)
            a = fit_outer(maps, y, ids, features, heldout, outside, initial, root / 'first', 'hash', 1)
            b = fit_outer(maps, y, ids, features, heldout * 99, {o: z * 99 for o, z in outside.items()}, initial, root / 'second', 'hash', 1)
            self.assertEqual(a, b)
            self.assertEqual(set(a['candidates']), set(CANDIDATES))
            self.assertEqual(len(calls), 48)
            self.assertFalse(np.array_equal(np.load(root / 'first/cp_classification_C/outer_probabilities.npy'),
                                           np.load(root / 'second/cp_classification_C/outer_probabilities.npy')))

    def test_classical_resume_training_scalers_and_tamper_rejection(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(40, 594))
        y = np.tile([0, 1], 20)
        tr, va = np.arange(30), np.arange(30, 40)
        ids = [str(i) for i in range(40)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.joblib'
            first, _ = classical_fit(x, y, tr, va, ids, path)
            check_fusion(first, x[tr])
            second, _ = classical_fit(x, y, tr, va, ids, path)
            np.testing.assert_array_equal(first.predict_proba(x[va]), second.predict_proba(x[va]))
            changed = x.copy()
            changed[tr] += 1
            with self.assertRaises(ValueError):
                classical_fit(changed, y, tr, va, ids, path)
            hashes = tree_checksums(Path(directory))
            (Path(directory) / 'checksum_manifest.json').write_text(json.dumps(hashes))
            checksum_guard(Path(directory))
            path.write_bytes(path.read_bytes() + b'changed')
            with self.assertRaises(ValueError):
                checksum_guard(Path(directory))

    def test_regression_metrics_units_bias_and_repeatability(self):
        hb = np.array([7., 9., 11., 13., 15.])
        result = regression_metrics(hb, hb + 2, replicates=50)
        self.assertEqual(result['mae_g_dl'], 2.)
        self.assertEqual(result['bias_prediction_minus_lab_g_dl'], 2.)
        self.assertEqual(result, regression_metrics(hb, hb + 2, replicates=50))

    def test_paired_evaluation_reports_matches_and_uncertainty(self):
        from src.cp_transfer_evaluation import comparison
        from src.eyelid_color_verify import check_metrics
        y = np.tile([0, 0, 0, 1], 10)
        folds = np.tile(np.arange(1, 6), 8)
        rng = np.random.default_rng(9)
        scores = {'exp04_robust_selected': (rng.random(40), np.full(40, .5))}
        for objective in OBJECTIVES:
            for strategy in ('A', 'B', 'C', 'selected'):
                scores[f'{objective}_{strategy}_threshold'] = (rng.random(40), np.full(40, .5))
        result = comparison(y, folds, scores, replicates=30)
        self.assertEqual(len(result['paired_matched_cp_minus_imagenet']), 8)
        self.assertNotIn('robust_none_selected', result['models'])
        for name, (p, t) in scores.items():
            check_metrics(y, p, t, result['models'][name])
            self.assertEqual(len(result['models'][name]['per_fold']), 5)
