"""Regression checks for the observed trusted-cache string loading failure."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from src.eyelid_color_verify import load_verified_sex
from src.local_eyelid_validation import sha256

class ColorVerifyTests(unittest.TestCase):
    def test_legacy_object_strings_and_future_unicode_match(self):
        with tempfile.TemporaryDirectory() as temp:
            for dtype in (object,str):
                p=Path(temp)/f'{dtype.__name__}.npz';np.savez(p,sex=np.array(['M','F','F'],dtype=dtype))
                actual=load_verified_sex(p,sha256(p));np.testing.assert_array_equal(actual,['M','F','F']);self.assertEqual(actual.dtype.kind,'U')
    def test_hash_failure_prevents_pickle_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'input.npz';np.savez(p,sex=np.array(['M'],dtype=object))
            with patch('src.eyelid_color_verify.np.load') as loader:
                with self.assertRaises(ValueError):load_verified_sex(p,'incorrect')
                loader.assert_not_called()
    def test_unknown_metadata_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'input.npz';np.savez(p,sex=np.array(['unknown'],dtype=object))
            with self.assertRaises(ValueError):load_verified_sex(p,sha256(p))
