"""Synthetic tests of manual tissue isolation and immutable feature definitions."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from src.local_eyelid_hybrid_features import manual_roi_tensor, FrozenMobileNet, MEAN
from src.local_eyelid_classifier import extract_masked_lab


class HybridFeatureTests(unittest.TestCase):
    def pair(self):
        image = np.full((30, 80, 3), 100, dtype=np.uint8)
        mask = np.zeros((30, 80), dtype=np.uint8)
        mask[8:22, 5:75] = 255
        mask[8:11, 5:8] = 0
        return image, mask

    def test_outside_mask_never_influences_either_feature_path(self):
        image, mask = self.pair()
        other = image.copy(); other[mask == 0] = [255, 0, 0]
        torch.testing.assert_close(manual_roi_tensor(image, mask), manual_roi_tensor(other, mask), rtol=0, atol=0)
        self.assertEqual(extract_masked_lab(image, mask), extract_masked_lab(other, mask))
        tensor = manual_roi_tensor(image, mask)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertTrue(torch.isfinite(tensor).all())
        self.assertTrue(torch.equal(tensor[:, 0, :], torch.zeros(3, 224)))

    def test_exact_normalization_and_mask_validation(self):
        image, mask = self.pair()
        tensor = manual_roi_tensor(image, mask)
        expected = (torch.tensor([100/255]*3)-torch.tensor(MEAN))/torch.tensor([.229,.224,.225])
        torch.testing.assert_close(tensor[:,112,112], expected)
        with self.assertRaises(ValueError):
            manual_roi_tensor(image, np.zeros_like(mask))
        with self.assertRaises(ValueError):
            manual_roi_tensor(image, np.ones_like(mask))

    def test_weights_fail_closed_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'weights.pth'; path.write_bytes(b'wrong weights')
            with self.assertRaises(ValueError):
                FrozenMobileNet(path)
