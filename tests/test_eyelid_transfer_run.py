import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from src.eyelid_transfer_run import train_local, private_destination


class TransferRunTests(unittest.TestCase):
    def test_output_guard_rejects_prior_experiments_and_other_paths(self):
        root = Path('/private/tmp/example')
        for dest in [root / 'data/processed/local_eyelid_classification_2026-10-10_v1', root / 'raw/exp03']:
            with self.assertRaises(ValueError): private_destination(root, dest)

    def test_nested_trainer_passes_only_outer_training_labels(self):
        y = np.tile([0, 1], 10); folds = np.repeat(np.arange(1, 6), 4)
        ids = [f's{i}' for i in range(20)]; rng = np.random.default_rng(22)
        data = {'local_y': torch.tensor(y), 'local_folds': folds, 'local_subjects': ids, 'local_lab': rng.normal(size=(20, 6))}
        maps = torch.rand(20, 2, 1, 1, 1)
        calls = []
        def fit(initial, cache, labels, train, validation, subjects, strategy, directory, seed, cache_hash, fixed_epochs=None):
            self.assertEqual(len(labels), 16); self.assertEqual(len(cache), 16)
            self.assertEqual(len(set(subjects)), 16)
            fold = int(directory.parts[-3].split('_')[-1])
            self.assertFalse(set(subjects) & {ids[i] for i in np.flatnonzero(folds == fold)})
            if validation is not None: self.assertFalse(set(train) & set(validation))
            directory.mkdir(parents=True, exist_ok=True)
            calls.append(strategy)
            return object(), {'best_epoch': 2}
        with tempfile.TemporaryDirectory() as folder:
            with patch('src.eyelid_transfer_run.fit_model', side_effect=fit), patch('src.eyelid_transfer_run.predict', side_effect=lambda m, c: c[:, 0, 0, 0, 0].numpy().astype(float)):
                train_local(data, maps, {}, Path(folder), 'test')
            self.assertEqual(len(calls), 40)
            for fold in range(1, 6):
                r = json.loads((Path(folder) / f'fold_{fold}/selection.json').read_text())
                self.assertEqual(r['selected_strategy'], 'A')
                self.assertEqual(r['strategies']['A']['outer_refit_epochs'], 2)

if __name__ == '__main__': unittest.main()
