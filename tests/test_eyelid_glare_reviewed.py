import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from PIL import Image
from src.eyelid_glare_reviewed import selected,ORDER,inputs,reproduce
from src.local_eyelid_validation import sha256
from src.preserve_local_experiment import tree_checksums

class ReviewedGlareTests(unittest.TestCase):
    def test_selection_prespecified_groups_and_ties(self):
        scores=dict.fromkeys(ORDER,.5)
        self.assertEqual(selected(scores),'original_robust_none')
        self.assertEqual(selected(scores,'glare_excluded'),'glare_excluded_robust_none')
        with self.assertRaises(ValueError):selected({})
    def test_complete_explicit_owner_review_and_mask_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';review=root/'review';source.mkdir();review.mkdir()
            subjects=np.array([f'synthetic{i}' for i in range(119)])
            cache={'subjects':subjects,'labels':np.array([1]*19+[0]*100),'folds':np.arange(119)%5+1}
            for p in ['original','proposed_excluded']:
                for f,n in [('robust',18),('regional',22)]:cache[p+'_'+f]=np.zeros((119,n))
            np.savez(source/'features.npz',**cache)
            package=source/'synthetic_review.txt';package.write_text('No patient fixtures')
            records=[]
            for i,sid in enumerate(subjects):
                mask=source/f'{i}.png';values=np.zeros((3,3),np.uint8)
                if i<96:values[1,1]=255
                Image.fromarray(values).save(mask)
                records.append({'subject_id':sid,'reviewer':'synthetic_owner','status':'APPROVED_GLARE' if i<96 else 'REVIEWED_NO_GLARE','reviewed_mask_path':str(mask),'reviewed_mask_sha256':sha256(mask),'review_package_path':str(package),'review_package_sha256':sha256(package)})
            pd.DataFrame(records).to_csv(review/'human_glare_review_register_private.csv',index=False)
            for path in [source,review]:(path/'checksum_manifest.json').write_text(json.dumps(tree_checksums(path)))
            _,_,rows=inputs(source,review);self.assertEqual(len(rows),119)
            (source/'0.png').write_bytes(b'changed')
            with self.assertRaises(ValueError):inputs(source,review)
    def test_tampering_prevents_model_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'changed.txt').write_text('tampered')
            (root/'checksum_manifest.json').write_text(json.dumps({'changed.txt':'wrong'}))
            with patch('src.eyelid_glare_reviewed.joblib.load') as loader:
                with self.assertRaises(ValueError):reproduce(root,root)
                loader.assert_not_called()
