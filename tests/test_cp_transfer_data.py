import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from src.cp_transfer_data import tissue,audit_public,verify_official_v2
from src.cp_transfer_prepare import destination
from src.local_eyelid_hybrid_features import manual_roi_tensor

class CPDataTests(unittest.TestCase):
    def test_alpha_policy_binary_equivalence_and_outside_invariance(self):
        rgb=np.full((30,50,3),[180,80,90],np.uint8);alpha=np.zeros((30,50),np.uint8);alpha[5:25,8:42]=255;alpha[4,8:42]=100
        x,mask=tissue(rgb,alpha);self.assertFalse(mask[4].any())
        torch.testing.assert_close(x,manual_roi_tensor(rgb,mask),rtol=0,atol=0)
        changed=rgb.copy();changed[alpha<255]=255
        torch.testing.assert_close(x,tissue(changed,alpha)[0],rtol=0,atol=0)
        with self.assertRaises(ValueError):tissue(rgb,np.zeros_like(alpha))
        with self.assertRaises(ValueError):tissue(rgb,alpha[:3])
    def test_public_schema_duplicate_alias_hb_and_subject_limits(self):
        rng=np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for folder in ['Anaemic','Non-Anaemic','All Categories']:(root/folder).mkdir()
            records=[];parts=[]
            for i in range(20):
                sid=f'Image_{i+1:03d}';label=i<10;hb=8.5 if label else 12.5;folder='Anaemic' if label else 'Non-Anaemic';text='Anemic' if label else 'Non-anemic'
                values=np.zeros((30,50,4),np.uint8);values[:,:,:3]=rng.integers(0,255,(30,50,3),dtype=np.uint8);values[5:25,8:42,3]=255;values[4,8:42,3]=100
                path=root/folder/(sid+'.png');Image.fromarray(values).save(path);(root/'All Categories'/path.name).write_bytes(path.read_bytes())
                records.append({'Image_ID':sid,'File_Name':path.name,'Folder_Label':text,'Hb_g_dl':hb});parts.append({'image_id':sid,'filename':path.name,'folder_label':text,'hb_g_dl':hb,'group_id':i+1})
            with pd.ExcelWriter(root/'Cp-Anaemic Dataset.xlsx') as writer:
                pd.DataFrame(records).to_excel(writer,sheet_name='Cp-Anaemic Dataset',index=False)
                pd.DataFrame(parts[:10]).to_excel(writer,sheet_name='Anaemic',index=False)
                pd.DataFrame(parts[10:]).to_excel(writer,sheet_name='Non-Anaemic',index=False)
            rows,audit=audit_public(root);self.assertEqual(len(rows),20);self.assertEqual(audit['duplicate_alias_files_not_loaded'],20)
            self.assertIn('not_verified',audit['unique_subjects'])
            tr,va=audit['training_indices'],audit['validation_indices'];self.assertFalse(set(tr)&set(va))
            (root/'All Categories/Image_001.png').write_bytes(b'changed')
            with self.assertRaises(ValueError):audit_public(root)
    def test_official_archive_hash_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'bad.zip';path.write_bytes(b'bad')
            with self.assertRaises(ValueError):verify_official_v2(Path(temp),path,'wrong')
    def test_new_private_destination(self):
        import subprocess
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();subprocess.run(['git','init','-q',str(root)],check=True);(root/'.gitignore').write_text('data/processed/*\n')
            path=root/'data/processed/local_eyelid_classification_exp05_cp_transfer_test_v1'
            self.assertEqual(destination(root,path),path)
            path.mkdir(parents=True)
            with self.assertRaises(FileExistsError):destination(root,path)
            with self.assertRaises(ValueError):destination(root,root/'results/test')
