"""Synthetic nested selection and training-only transformation checks."""
import tempfile,unittest
from pathlib import Path
import joblib,numpy as np
from src.eyelid_color_data import FEATURE_SETS
from src.eyelid_color_models import CANDIDATES,BY_ID,select,threshold_rule,fit_outer

class ColorModelTests(unittest.TestCase):
    def test_threshold_tradeoffs_and_fallback(self):
        y=np.array([0]*5+[1]*2);p=np.array([.1,.2,.3,.4,.8,.7,.9])
        self.assertEqual(threshold_rule(y,p,.8)['threshold'],.7)
        self.assertEqual(threshold_rule(y,p,.9)['inner_specificity'],1)
        self.assertGreater(threshold_rule(y,np.ones(7))['threshold'],1)
        with self.assertRaises(ValueError):threshold_rule(y,p*np.nan)
    def test_candidate_count_selection_and_ties(self):
        self.assertEqual(len(CANDIDATES),36)
        scores=dict.fromkeys(BY_ID,.5);self.assertEqual(select(scores),'A_original_lr_c1_none')
        self.assertEqual(select(scores,'B'),'B_original_lr_c1_none')
        with self.assertRaises(ValueError):select({})
    def test_nested_predictions_saved_and_scalers_train_only(self):
        rng=np.random.default_rng(8);n=60;y=np.array([0]*48+[1]*12);subjects=[f'synth{i}' for i in range(n)]
        train={f'{p}_{f}':rng.normal(size=(n,len(names))) for p in ('A','B') for f,names in FEATURE_SETS.items()}
        valid={k:np.full((3,v.shape[1]),100.) for k,v in train.items()}
        with tempfile.TemporaryDirectory() as temp:
            prob,rec=fit_outer(train,y,subjects,valid,Path(temp))
            for c in CANDIDATES:
                outer=joblib.load(Path(temp)/c['id']/'outer.joblib')
                np.testing.assert_array_equal(outer.predict_proba(valid[c['key']])[:,1],prob[c['id']])
                if c['family']=='logistic':np.testing.assert_allclose(outer['scaler'].mean_,train[c['key']].mean(0),atol=1e-12)
                for part in rec['inner_partitions']:
                    tr=part['train_indices'];va=part['validation_indices'];self.assertFalse(set(tr)&set(va));self.assertEqual(set(tr)|set(va),set(range(n)))
                    if c['family']=='logistic':
                        inner=joblib.load(Path(temp)/c['id']/f"inner_{part['inner_fold']}.joblib")
                        np.testing.assert_allclose(inner['scaler'].mean_,train[c['key']][tr].mean(0),atol=1e-12)
            second=Path(temp)/'second';second.mkdir()
            _,rec2=fit_outer(train,y,subjects,{k:v+100 for k,v in valid.items()},second)
            self.assertEqual(rec,rec2)
            with self.assertRaises(ValueError):fit_outer(train,y,['same']*n,valid,second)
