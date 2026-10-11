import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from src.eyelid_transfer_neural import AnemiaMobileNet, fit_model, prefix_cache, predict, runtime, state_hash


class TransferNeuralTests(unittest.TestCase):
    def setUp(self):
        runtime(77)
        self.model = AnemiaMobileNet().eval()

    def test_cached_prefix_preserves_full_network_predictions(self):
        x = torch.rand(2, 3, 224, 224)
        cache = prefix_cache(self.model, x)
        with torch.no_grad():
            np.testing.assert_allclose(self.model(x).sigmoid().numpy(), predict(self.model, cache), rtol=0, atol=1e-7)
        self.assertEqual(cache.shape[:2], (2, 2))

    def test_frozen_head_strategy_and_late_layer_strategy(self):
        self.model.configure('A')
        self.assertTrue(all(not p.requires_grad for p in self.model.features.parameters()))
        self.assertTrue(all(p.requires_grad for p in self.model.head.parameters()))
        self.model.configure('B'); self.model.training_mode()
        self.assertTrue(all(not p.requires_grad for p in self.model.features[:-2].parameters()))
        self.assertTrue(all(p.requires_grad for p in self.model.features[-2:].parameters()))
        self.assertTrue(all(not m.training for m in self.model.modules() if isinstance(m, torch.nn.BatchNorm2d)))

    def test_actual_training_checkpoint_reproduction_and_resume_guards(self):
        initial = {k: v.clone() for k, v in self.model.state_dict().items()}
        x = torch.rand(6, 3, 224, 224); cache = prefix_cache(self.model, x)
        y = np.array([0, 0, 0, 1, 1, 1]); train = np.arange(6); ids = [str(i) for i in train]
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            fitted, result = fit_model(initial, cache, y, train, None, ids, 'B', directory, 77, 'smoke', fixed_epochs=2)
            self.assertEqual(result['training_status'], 'complete')
            self.assertEqual(result['best_epoch'], 2)
            self.assertNotEqual(state_hash(fitted.state_dict()), state_hash(initial))
            for k, value in initial.items():
                if k.startswith('features.') and int(k.split('.')[1]) < len(fitted.features) - 2:
                    self.assertTrue(torch.equal(value, fitted.state_dict()[k]))
                if 'running_' in k or 'num_batches_tracked' in k:
                    self.assertTrue(torch.equal(value, fitted.state_dict()[k]))
            loaded, _ = fit_model(initial, cache, y, train, None, ids, 'B', directory, 77, 'smoke', fixed_epochs=2)
            np.testing.assert_array_equal(predict(fitted, cache), predict(loaded, cache))
            with self.assertRaises(ValueError):
                fit_model(initial, cache, y, train, None, ids, 'B', directory, 78, 'smoke', fixed_epochs=2)
            with self.assertRaises(ValueError):
                fit_model(initial, cache, y, train, train, ids, 'B', directory, 77, 'smoke')

if __name__ == '__main__': unittest.main()
