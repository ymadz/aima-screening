"""Synthetic-only checks for segmentation model and positive-class metrics."""

import unittest

import numpy as np
import torch

from src.segmentation_model import MobileNetSmallUNet, dice_bce_loss
from src.train_public_segmentation import binary_metrics


class SegmentationModelTests(unittest.TestCase):
    def test_forward_backward_and_binary_geometry(self) -> None:
        model = MobileNetSmallUNet(pretrained_encoder=False)
        image = torch.randn(1, 3, 64, 64)
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, 20:35, 15:45] = 1
        logits = model(image)
        self.assertEqual(tuple(logits.shape), tuple(mask.shape))
        loss = dice_bce_loss(logits, mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.head.weight.grad)

    def test_shape_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            dice_bce_loss(torch.zeros(1, 1, 8, 8), torch.zeros(1, 8, 8))

    def test_positive_class_metrics(self) -> None:
        predicted = np.array([[1, 0], [1, 0]])
        truth = np.array([[1, 1], [0, 0]])
        metrics = binary_metrics(predicted, truth)
        self.assertAlmostEqual(metrics["dice"], 0.5)
        self.assertAlmostEqual(metrics["iou"], 1 / 3)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
