"""Synthetic checks for prediction-only local segmentation boundaries."""

import unittest

import numpy as np
import torch

from src.local_segmentation_pilot import (
    _letterbox_image,
    local_subject_split,
    predict_local_mask,
)
from src.segmentation_model import MobileNetSmallUNet


class LocalPilotTests(unittest.TestCase):
    def test_subject_assignment_is_stable(self) -> None:
        self.assertEqual(local_subject_split("synthetic-001"),
                         local_subject_split("synthetic-001"))
        self.assertIn(local_subject_split("synthetic-002"),
                      {"train", "validation", "test"})

    def test_letterbox_and_prediction_use_image_only(self) -> None:
        image = np.full((80, 60, 3), 100, dtype=np.uint8)
        square, geometry = _letterbox_image(image, 64)
        self.assertEqual(square.shape, (64, 64, 3))
        self.assertEqual(geometry, (8, 0, 48, 64))
        model = MobileNetSmallUNet(pretrained_encoder=False)
        mask, seconds = predict_local_mask(model, image, 64, torch.device("cpu"))
        self.assertEqual(mask.shape, image.shape[:2])
        self.assertEqual(mask.dtype, np.dtype(bool))
        self.assertGreaterEqual(seconds, 0)
        cropped, _ = predict_local_mask(
            model, image, 64, torch.device("cpu"), (0.2, 0.3, 0.8, 0.7)
        )
        self.assertFalse(cropped[:24].any())
        self.assertFalse(cropped[56:].any())


if __name__ == "__main__":
    unittest.main()
