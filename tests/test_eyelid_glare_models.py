import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
from src.eyelid_glare_models import CANDIDATES,fit_outer,select
from src.eyelid_glare_features import FEATURES
from src.eyelid_glare_evaluation import compare
from src.eyelid_glare_reproduce import reproduce
from src.eyelid_color_verify import check_metrics
from src.eyelid_glare_run import check_exp4_comparator

class GlareModelTests(unittest.TestCase):
    def test_frozen_grid_and_ties(self):
        self.assertEqual(len(CANDIDATES),4)
        self.assertEqual(select({c['id']:.5 for c in CANDIDATES}),'robust_none')
        with self.assertRaises(ValueError):select({})
    def test_nested_partitions_scaling_saved_prediction_and_decisions(self):
        rng=np.random.default_rng(14);y=np.array([0]*48+[1]*12)
        x={k:rng.normal(size=(60,len(names))) for k,names in FEATURES.items()}
        valid={k:np.full((3,len(names)),100.) for k,names in FEATURES.items()}
        subjects=[f'synthetic{i}' for i in range(60)]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);first=root/'first';first.mkdir()
            p,rec=fit_outer(x,y,subjects,valid,first)
            for c in CANDIDATES:
                model=joblib.load(first/c['id']/'outer.joblib')
                np.testing.assert_array_equal(model.predict_proba(valid[c['feature']])[:,1],p[c['id']])
                np.testing.assert_allclose(model['scaler'].mean_,x[c['feature']].mean(0),rtol=0,atol=1e-12)
                for part in rec['inner_partitions']:
                    tr=part['train_indices'];va=part['validation_indices']
                    self.assertFalse(set(tr)&set(va));self.assertEqual(set(tr)|set(va),set(range(60)))
                    self.assertFalse(set(part['training_subjects'])&set(part['validation_subjects']))
                    model=joblib.load(first/c['id']/f"inner_{part['inner_fold']}.joblib")
                    np.testing.assert_allclose(model['scaler'].mean_,x[c['feature']][tr].mean(0),rtol=0,atol=1e-12)
            second=root/'second';second.mkdir()
            _,rec2=fit_outer(x,y,subjects,{k:v+100 for k,v in valid.items()},second)
            self.assertEqual(rec,rec2)
            with self.assertRaises(ValueError):fit_outer(x,y,['same']*60,valid,second)
            with self.assertRaises(ValueError):fit_outer(dict(x,proposed_excluded_robust=x['robust']),y,subjects,valid,second)
    def test_metrics_and_conditional_comparison(self):
        y=np.array([0]*10+[1]*4);folds=np.array([1,2]*7)
        p=np.linspace(.1,.9,14);t=np.full(14,.5)
        result=compare(y,folds,{'robust_none_selected':(p,t),'copy':(p,t)},replicates=20)
        check_metrics(y,p,t,result['models']['copy'])
        self.assertEqual(result['paired_vs_unchanged_exp4_robust']['copy']['sensitivity']['interval'],[0.,0.])
    def test_hash_tampering_blocks_deserialization(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'example.txt').write_text('changed')
            (root/'checksum_manifest.json').write_text(json.dumps({'example.txt':'wrong'}))
            with patch('src.eyelid_glare_reproduce.joblib.load') as loader:
                with self.assertRaises(ValueError):reproduce(root,root)
                loader.assert_not_called()

    def test_comparator_uses_authoritative_binary_and_json_thresholds(self):
        folds=np.arange(1,6);p=np.arange(5)/10.;t=np.full(5,.42647694300000005)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);np.savez(root/'candidate_predictions.npz',A_robust_lr_c1_none=p)
            for fold in range(1,6):
                d=root/f'fold_{fold}';d.mkdir()
                (d/'selection.json').write_text(json.dumps({'candidates':{'A_robust_lr_c1_none':{'rules':{'0.8':{'threshold':float(t[0])}}}}}))
            check_exp4_comparator(root,folds,(p,t))
            with self.assertRaises(AssertionError):check_exp4_comparator(root,folds,(p,np.nextafter(t,0)))
