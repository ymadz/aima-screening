"""Scientific conversion/geometry checks on synthetic pixels only."""
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image,ImageCms
from src.eyelid_color_data import color_features,FEATURE_SETS,render_pair
from src.eyelid_transfer_data import convert_srgb
from src.local_eyelid_classifier import extract_masked_lab

class ColorDataTests(unittest.TestCase):
    def test_primary_colors_lab_units_and_rgb_order(self):
        for rgb,expected in [([255,0,0],[53,80,67]),([0,255,0],[88,-86,83]),([255,255,255],[100,0,0]),([0,0,0],[0,0,0])]:
            image=np.tile(np.uint8(rgb),(4,4,1));mask=np.full((4,4),255,np.uint8);mask[0]=0
            v=color_features(image,mask)['original']
            np.testing.assert_allclose(v[[0,2,4]],expected,atol=1)
            np.testing.assert_allclose(v[[1,3,5]],0,rtol=0,atol=2e-14)
    def test_original_exact_background_exclusion_and_reproducibility(self):
        rng=np.random.default_rng(42);image=rng.integers(0,256,(10,12,3),dtype=np.uint8);mask=np.zeros((10,12),np.uint8);mask[2:8,3:10]=255
        one=color_features(image,mask);image[mask==0]=255;two=color_features(image,mask)
        np.testing.assert_array_equal(one['original'],list(extract_masked_lab(image,mask).values()))
        for k in FEATURE_SETS:
            self.assertEqual(one[k].shape,(len(FEATURE_SETS[k]),));np.testing.assert_array_equal(one[k],two[k]);self.assertTrue(np.isfinite(one[k]).all())
    def test_redness_percentiles_and_regional_orientation(self):
        image=np.zeros((4,6,3),np.uint8);image[:,:3]=[200,50,50];image[:,3:]=[50,100,100];mask=np.full((4,6),255,np.uint8);mask[0]=0
        a=color_features(image,mask);b=color_features(image[:,::-1].copy(),mask)
        expected=np.log((image[mask>0,0].astype(float)+1)/(image[mask>0,1].astype(float)+1))
        self.assertAlmostEqual(a['robust'][-2],np.median(expected))
        np.testing.assert_allclose(a['regional'][-3:],-b['regional'][-3:])
        thin=color_features(image[:,:1],mask[:,:1]);np.testing.assert_array_equal(thin['regional'][-3:],0)
    def test_invalid_pixels_and_masks_fail_closed(self):
        good=np.zeros((4,4,3),np.uint8);mask=np.full((4,4),255,np.uint8)
        for image,m in [(good.astype(float),mask),(good,mask[:2]),(good,np.zeros((4,4),np.uint8)),(good,np.ones((4,4),np.uint8))]:
            with self.assertRaises(ValueError): color_features(image,m)
    def test_icc_transform_is_applied_and_invalid_profile_fails(self):
        rgb=np.full((5,5,3),[240,50,90],np.uint8);profile=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
        np.testing.assert_array_equal(convert_srgb(Image.fromarray(rgb),profile),rgb)
        with self.assertRaises(Exception):convert_srgb(Image.fromarray(rgb),b'invalid ICC bytes')

    def test_non_srgb_profile_changes_pixels(self):
        p=Path('/System/Library/ColorSync/Profiles/Display P3.icc')
        if not p.exists():self.skipTest('System Display P3 profile absent')
        image=np.full((3,3,3),[180,80,60],np.uint8)
        managed=convert_srgb(Image.fromarray(image),p.read_bytes())
        self.assertEqual(managed.dtype,np.uint8);self.assertFalse(np.array_equal(image,managed))
    def test_color_transform_preserves_mask_geometry(self):
        rgb=np.zeros((4,4,3),np.uint8);mask=np.zeros((4,4),np.uint8);mask[1:3,1:3]=255
        row={'annotation_image_path':'test.tif','mask_path':'mask.png'}
        with patch('src.eyelid_color_data.load_manual_pair',return_value=(rgb,mask)),patch('src.eyelid_color_data.local_pixels',return_value=(rgb.copy(),mask.copy(),{})):
            a,b,m,info=render_pair(Path('.'),row);np.testing.assert_array_equal(m,mask)
        with patch('src.eyelid_color_data.load_manual_pair',return_value=(rgb,mask)),patch('src.eyelid_color_data.local_pixels',return_value=(rgb.copy(),np.zeros_like(mask),{})):
            with self.assertRaises(AssertionError):render_pair(Path('.'),row)
