"""Unit checks for geometry and metrics used by the SAM 2.1 nail workflow."""

import unittest

import numpy as np

from src.sam2_nail_segmentation import (
    binary_scores,
    bbox_from_mask,
    largest_component,
    points_from_mask,
)


class SAM2NailWorkflowTests(unittest.TestCase):
    def test_box_is_padded_and_clipped_to_image(self) -> None:
        mask = np.zeros((100, 200), dtype=bool)
        mask[5:25, 180:200] = True
        box = bbox_from_mask(mask)
        np.testing.assert_array_equal(box, np.array([177, 2, 200, 28], dtype=np.float32))

    def test_empty_mask_cannot_create_a_prompt(self) -> None:
        with self.assertRaises(ValueError):
            bbox_from_mask(np.zeros((20, 30), dtype=bool))

    def test_point_prompts_have_one_positive_and_four_background_points(self) -> None:
        mask = np.zeros((100, 120), dtype=bool)
        mask[30:70, 40:80] = True
        box = bbox_from_mask(mask)
        points, labels = points_from_mask(mask, box)
        self.assertEqual(points.shape, (5, 2))
        self.assertEqual(labels.tolist(), [1, 0, 0, 0, 0])
        self.assertTrue(mask[round(points[0, 1]), round(points[0, 0])])
        for x, y in points[1:]:
            self.assertFalse(mask[round(y), round(x)])

    def test_binary_metrics(self) -> None:
        target = np.array([[True, True], [False, False]])
        prediction = np.array([[True, False], [True, False]])
        self.assertEqual(binary_scores(prediction, target), {
            "dice": 0.5, "iou": 1 / 3, "precision": 0.5, "recall": 0.5,
        })

    def test_largest_component_removes_speckles_and_reports_count(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[3:10, 4:11] = True
        mask[15, 15] = True
        cleaned, count = largest_component(mask)
        self.assertEqual(count, 2)
        self.assertEqual(int(cleaned.sum()), 49)
        self.assertFalse(cleaned[15, 15])


if __name__ == "__main__":
    unittest.main()
