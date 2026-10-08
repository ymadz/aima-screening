"""Synthetic checks and optional public-data regression checks."""

from pathlib import Path
import unittest

import numpy as np

from src.public_segmentation_data import (
    PublicPair,
    assign_subject_splits,
    discover_public_pairs,
    letterbox_pair,
    load_public_pair,
    reconstruct_mask,
)


PUBLIC_ROOT = Path(__file__).resolve().parents[1] / "data/raw/dataset anemia"


class MaskReconstructionTests(unittest.TestCase):
    def test_zero_alpha_is_foreground_evidence(self) -> None:
        image = np.zeros((20, 20, 4), dtype=np.uint8)
        image[7:12, 5:15, 3] = 255
        mask, encoding = reconstruct_mask(image)
        self.assertEqual(encoding, "alpha")
        self.assertEqual(int(mask.sum()), 50)

    def test_nonzero_alpha_must_not_label_whole_frame(self) -> None:
        image = np.full((20, 20, 4), 255, dtype=np.uint8)
        image[7:12, 5:15, :3] = (125, 150, 200)
        image[7:12, 5:15, 3] = 191
        mask, encoding = reconstruct_mask(image)
        self.assertEqual(encoding, "exact_white_background")
        self.assertEqual(int(mask.sum()), 50)

    def test_black_is_not_assumed_background(self) -> None:
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        image[7:12, 5:15, :] = 150
        with self.assertRaisesRegex(ValueError, "exact-white"):
            reconstruct_mask(image)

    def test_letterbox_keeps_mask_binary_and_aligned(self) -> None:
        image = np.full((40, 30, 3), 128, dtype=np.uint8)
        mask = np.zeros((40, 30), dtype=bool)
        mask[20:30, 10:20] = True
        output, binary, box = letterbox_pair(image, mask, 64)
        self.assertEqual(output.shape, (64, 64, 3))
        self.assertEqual(binary.shape, (64, 64))
        self.assertEqual(set(np.unique(binary)), {0, 1})
        self.assertEqual(box, (8, 0, 48, 64))

    def test_similarity_group_cannot_cross_splits(self) -> None:
        pairs = [
            PublicPair("C", str(index), Path(f"{index}.jpg"), Path(f"{index}.png"))
            for index in range(30)
        ]
        split = assign_subject_splits(pairs, [("C/1", "C/2", 5)], seed=123)
        self.assertEqual(split["C/1"], split["C/2"])
        self.assertEqual(len(split), 30)
        self.assertEqual(set(split.values()), {"train", "validation", "test"})
        self.assertEqual(split, assign_subject_splits(pairs, [("C/1", "C/2", 5)], 123))


@unittest.skipUnless(PUBLIC_ROOT.is_dir(), "Eyes Defy Anemia source not present")
class AvailablePublicDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pairs = discover_public_pairs(PUBLIC_ROOT)
        cls.by_key = {pair.key: pair for pair in cls.pairs}

    def test_one_palpebral_pair_per_public_subject(self) -> None:
        self.assertEqual(len(self.pairs), 218)
        self.assertEqual(len(self.by_key), 218)
        self.assertTrue(self.by_key["India/7"].segmentation_path.name.endswith("_papebral.png"))
        self.assertTrue(self.by_key["Italy/95"].segmentation_path.name.endswith("_palplebral.png"))
        self.assertEqual(self.by_key["India/8"].segmentation_path.name.count("(1)"), 0)

    def test_opaque_alpha_exception_recovers_small_mask(self) -> None:
        _, mask, evidence = load_public_pair(self.by_key["Italy/15"])
        self.assertEqual(evidence["encoding"], "exact_white_background")
        self.assertAlmostEqual(float(mask.mean()), 0.0643, places=3)

    def test_16_bit_alpha_pair_and_typo_pairs_align(self) -> None:
        for key in ("India/2", "India/7", "Italy/95"):
            image, mask, evidence = load_public_pair(self.by_key[key])
            self.assertEqual(image.shape[:2], mask.shape)
            self.assertTrue(0 < mask.mean() < 0.30)
            self.assertEqual(evidence["encoding"], "alpha")


if __name__ == "__main__":
    unittest.main()
