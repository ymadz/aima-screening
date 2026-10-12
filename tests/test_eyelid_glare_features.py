import tempfile
import unittest
import subprocess
from pathlib import Path
import numpy as np
from src.eyelid_color_data import color_features
from src.eyelid_glare_features import extract,require_verified_review,REGIONAL_NAMES,region_masks
from src.eyelid_glare_prepare import destination,validate_fresh_inputs

class GlareFeatureTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(17)
        self.rgb=rng.integers(30,230,(50,80,3),dtype=np.uint8)
        self.anatomy=np.zeros((50,80),np.uint8);self.anatomy[10:40,10:70]=255
    def test_original_features_exact_and_empty_exclusion_identity(self):
        x=extract(self.rgb,self.anatomy);z=extract(self.rgb,self.anatomy,np.zeros_like(self.anatomy))
        np.testing.assert_array_equal(x['robust'],color_features(self.rgb,self.anatomy)['robust'])
        np.testing.assert_array_equal(x['regional'],z['regional']);self.assertEqual(len(x['regional']),len(REGIONAL_NAMES))
    def test_geometry_rotation_and_flip_invariance(self):
        x=extract(self.rgb,self.anatomy)
        for rgb,mask in [(self.rgb[:,::-1],self.anatomy[:,::-1]),(np.rot90(self.rgb),np.rot90(self.anatomy))]:
            np.testing.assert_allclose(x['regional'],extract(rgb.copy(),mask.copy())['regional'],rtol=0,atol=1e-12)
    def test_exclusion_ignores_only_selected_pixels(self):
        glare=np.zeros_like(self.anatomy);glare[20:23,30:33]=255
        x=extract(self.rgb,self.anatomy,glare)
        retained=self.anatomy.copy();retained[glare>0]=0
        np.testing.assert_array_equal(x['robust'],color_features(self.rgb,retained)['robust'])
    def test_missing_review_blocks_all_masks_including_empty(self):
        for rows in [[],[{'status':'AUTO_GENERATED_PENDING_REVIEW'}],[{'status':'REVIEWED_NO_GLARE'}]]:
            with self.assertRaises(ValueError):require_verified_review(rows)
        require_verified_review([{'status':'REVIEWED_NO_GLARE','reviewer':'owner','reviewed_mask_sha256':'a'*64}])
    def test_degenerate_region_fails(self):
        a=np.zeros((3,3),np.uint8);a[1,1]=255
        with self.assertRaises(ValueError):region_masks(a)
    def test_private_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();subprocess.run(['git','init','-q',str(root)],check=True)
            (root/'.gitignore').write_text('data/processed/*\n')
            good=root/'data/processed/local_eyelid_exp04b_glare_regional_test_v1'
            self.assertEqual(destination(root,good),good)
            with self.assertRaises(ValueError):destination(root,root/'results/test')
            good.mkdir(parents=True)
            with self.assertRaises(FileExistsError):destination(root,good)

    def test_fresh_audit_typed_labels_without_weakening_provenance(self):
        old={k:'a'*64 for k in ['original_source_sha256','annotation_image_sha256','roi_sha256','mask_sha256']}
        old['derived_anemia_label']='1'
        data={'subjects':['synthetic'],'rows':{'synthetic':old}}
        fresh={'synthetic':dict(old,derived_anemia_label=1)}
        validate_fresh_inputs(fresh,data)
        fresh['synthetic']['derived_anemia_label']=0
        with self.assertRaises(ValueError):validate_fresh_inputs(fresh,data)
        fresh['synthetic']=dict(old,mask_sha256='changed')
        with self.assertRaises(ValueError):validate_fresh_inputs(fresh,data)
