"""Synthetic fold/selection tests; no private fixtures."""
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
from src.local_eyelid_hybrid_model import HybridProjection, fit_outer_training, select_candidate, select_threshold


class HybridModelTests(unittest.TestCase):
    def test_projection_keeps_lab_and_fits_training_only(self):
        rng=np.random.default_rng(4);train=rng.normal(size=(40,582));valid=rng.normal(100,1,size=(10,582))
        projection=HybridProjection().fit(train)
        np.testing.assert_allclose(projection.scaler_.mean_,train[:,6:].mean(axis=0))
        result=projection.transform(valid).to_numpy()
        np.testing.assert_array_equal(result[:,:6],valid[:,:6])
        self.assertEqual(result.shape,(10,14))
        np.testing.assert_allclose(projection.scaler_.mean_,train[:,6:].mean(axis=0))

    def test_threshold_objective_and_deterministic_ties(self):
        y=np.array([0,0,0,0,0,1,1]);p=np.array([.1,.2,.3,.4,.8,.7,.9])
        selected=select_threshold(y,p)
        self.assertEqual(selected['threshold'],.7)
        self.assertEqual(selected['inner_sensitivity'],1)
        self.assertEqual(selected['inner_specificity'],.8)
        self.assertEqual(select_candidate({'unweighted':.5,'balanced':.5}),'unweighted')
        all_tied=select_threshold(y,np.ones(7))
        self.assertGreater(all_tied['threshold'],1)
        self.assertEqual(all_tied['inner_specificity'],1)
        with self.assertRaises(ValueError):select_threshold(np.zeros(7),p)

    def test_nested_saved_models_exclude_outer_and_reproduce(self):
        rng=np.random.default_rng(44);x=rng.normal(size=(60,582));y=np.array([0]*48+[1]*12)
        valid=rng.normal(size=(8,582));subjects=[f'synthetic_{i}' for i in range(60)]
        with tempfile.TemporaryDirectory() as temporary:
            model,probability,record=fit_outer_training(x,y,subjects,valid,Path(temporary))
            np.testing.assert_array_equal(joblib.load(Path(temporary)/'outer_model.joblib').predict_proba(valid)[:,1],probability)
            self.assertEqual(model.named_steps['projection'].training_samples_,60)
            for partition in record['inner_partitions']:
                training=partition['training_subjects'];validation=partition['validation_subjects']
                self.assertFalse(set(training)&set(validation))
                self.assertEqual(set(training)|set(validation),set(subjects))
                indices=[subjects.index(s) for s in training]
                fitted=joblib.load(Path(temporary)/f"inner_unweighted_{partition['inner_fold']}.joblib")
                np.testing.assert_allclose(fitted.named_steps['projection'].scaler_.mean_,x[indices,6:].mean(axis=0))
            # Changing outer feature values cannot change training model selection or thresholds.
            _,_,second=fit_outer_training(x,y,subjects,valid+100)
            self.assertEqual(record,second)
            with self.assertRaises(ValueError):fit_outer_training(x,y,['same']*60,valid)
