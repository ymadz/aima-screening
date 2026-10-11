import unittest
import numpy as np
from PIL import Image, ImageCms
from src.eyelid_transfer_data import harmonized_label, convert_srgb, public_split
from src.local_eyelid_hybrid_features import manual_roi_tensor


class TransferDataTests(unittest.TestCase):
    def test_strict_cutoffs_and_pregnancy_ambiguity(self):
        self.assertEqual(harmonized_label(30, 'M', 12.9), (1, 'eligible'))
        self.assertEqual(harmonized_label(30, 'M', 13), (0, 'eligible'))
        self.assertEqual(harmonized_label(30, 'F', 10.49), (1, 'eligible'))
        self.assertIsNone(harmonized_label(30, 'F', 10.5)[0])
        self.assertIsNone(harmonized_label(30, 'F', 11.99)[0])
        self.assertEqual(harmonized_label(30, 'F', 12), (0, 'eligible'))
        self.assertIsNone(harmonized_label(66, 'M', 10)[0])
        self.assertIsNone(harmonized_label(30, 'M', np.nan)[0])
        with self.assertRaises(ValueError): harmonized_label(30, 'X', 12)
        with self.assertRaises(ValueError): harmonized_label(30, 'M', 120)

    def test_icc_conversion_and_background_isolation(self):
        x = np.full((20, 30, 3), 130, np.uint8)
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
        np.testing.assert_array_equal(convert_srgb(Image.fromarray(x), profile), x)
        mask = np.zeros((20, 30), np.uint8); mask[3:15, 6:23] = 255
        other = x.copy(); other[mask == 0] = 255
        np.testing.assert_array_equal(manual_roi_tensor(x, mask).numpy(), manual_roi_tensor(other, mask).numpy())

    def test_duplicate_candidates_remain_in_one_partition(self):
        ids = [f'{c}/{i}' for c in ['India', 'Italy'] for i in range(20)]
        y = np.tile([0, 1], 20)
        countries = ['India'] * 20 + ['Italy'] * 20
        tr, va, groups = public_split(ids, y, countries, [(ids[0], ids[1], 4)])
        self.assertFalse(set(tr) & set(va))
        self.assertEqual(set(tr) | set(va), set(range(40)))
        self.assertFalse(set(np.array(groups)[tr]) & set(np.array(groups)[va]))

if __name__ == '__main__': unittest.main()
