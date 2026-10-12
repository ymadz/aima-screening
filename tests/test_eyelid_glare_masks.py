"""Synthetic scientific mask contracts; no patient fixtures."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from src.eyelid_glare_masks import propose_glare, validate_glare, panel

class GlareMaskTests(unittest.TestCase):
    def setUp(self):
        self.rgb=np.full((100,160,3),[175,70,85],dtype=np.uint8)
        self.anatomy=np.zeros((100,160),np.uint8);self.anatomy[15:85,20:140]=255
    def test_empty_is_pending_and_not_human_review(self):
        mask,q=propose_glare(self.rgb,self.anatomy)
        self.assertEqual(int(mask.sum()),0)
        self.assertEqual(q['status'],'AUTO_GENERATED_PENDING_REVIEW')
        self.assertFalse(q['verified_no_glare'])
    def test_bright_red_clipping_alone_is_not_glare(self):
        self.rgb[:]=[255,80,90]
        mask,q=propose_glare(self.rgb,self.anatomy)
        self.assertEqual(int(mask.sum()),0)
    def test_uniform_pale_tissue_not_removed(self):
        self.rgb[:]=[245,235,230]
        mask,q=propose_glare(self.rgb,self.anatomy)
        self.assertEqual(int(mask.sum()),0);self.assertEqual(q['status'],'UNCERTAIN')
    def test_white_local_reflection_inside_only(self):
        self.rgb[45:50,75:80]=255;self.rgb[2:8,2:8]=255
        mask,q=propose_glare(self.rgb,self.anatomy)
        self.assertTrue(mask[46,76]);self.assertFalse(mask[4,4]);self.assertEqual(q['outside_roi_pixels'],0)
        self.assertGreater(q['removed_fraction'],0)
    def test_alignment_binary_overlap_and_total_removal(self):
        zero=np.zeros_like(self.anatomy)
        self.assertTrue(validate_glare(self.rgb,self.anatomy,zero)['empty_proposal'])
        for wrong in [zero[:10],np.ones_like(zero),self.anatomy.copy()]:
            with self.assertRaises(ValueError):validate_glare(self.rgb,self.anatomy,wrong)
        zero[0,0]=255
        with self.assertRaises(ValueError):validate_glare(self.rgb,self.anatomy,zero)
    def test_private_panel(self):
        mask,q=propose_glare(self.rgb,self.anatomy)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'panel.png';panel(self.rgb,self.anatomy,mask,{'review_id':'synthetic-001',**q},p)
            with Image.open(p) as im:self.assertEqual(im.size,(1440,390))
