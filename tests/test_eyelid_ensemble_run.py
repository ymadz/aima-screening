"""Synthetic source-integrity, inner-fold scope and metric checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.eyelid_ensemble_run import checksum_guard, matched_scope, metrics_with_uncertainty, calculate, positive_scores
from src.preserve_local_experiment import tree_checksums


class RunTests(unittest.TestCase):
    def test_positive_class_consistency(self):
        class Model:
            classes_=np.array([0,1])
            def predict_proba(self,x):return np.tile([.2,.8],(len(x),1))
        model=Model();np.testing.assert_array_equal(positive_scores(model,np.zeros((3,1))),[.8,.8,.8])
        model.classes_=np.array([1,0])
        with self.assertRaises(ValueError):positive_scores(model,np.zeros((3,1)))
    def scope(self):
        y=np.array([0]*12+[1]*6);ids=[str(i) for i in range(18)]
        parts=[]
        for i,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261011).split(ids,y),1):
            parts.append({'inner_fold':i,'train_indices':tr.tolist(),'validation_indices':va.tolist(),
                          'training_subjects':[ids[j] for j in tr],'validation_subjects':[ids[j] for j in va]})
        return {'training_subjects':ids,'training_labels':y.tolist(),'validation_subjects':['heldout'],'inner_partitions':parts},ids,y

    def test_matched_scope_and_inner_leakage(self):
        r,ids,y=self.scope();matched_scope(r,r,ids,y,['heldout'])
        b=copy.deepcopy(r);b['inner_partitions'][0]['validation_subjects'][0]='other'
        with self.assertRaises(ValueError):matched_scope(r,b,ids,y,['heldout'])
        b=copy.deepcopy(r);b['training_labels'][0]=1
        with self.assertRaises(ValueError):matched_scope(r,b,ids,y,['heldout'])

    def test_checksum_coverage_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);(d/'a.txt').write_text('source')
            (d/'checksum_manifest.json').write_text(json.dumps(tree_checksums(d)))
            checksum_guard(d)
            (d/'a.txt').write_text('changed')
            with self.assertRaises(ValueError):checksum_guard(d)

    def test_metric_confusion_and_paired_intervals(self):
        y=np.array([0,0,1,1]*5);folds=np.repeat(np.arange(1,6),4)
        p=np.tile([.1,.8,.9,.2],5);t=np.full(20,.5)
        scores={'exp04_robust_historical':(p,t),'exp05_cp_C_historical':(p,t)}
        m=metrics_with_uncertainty(y,folds,scores,replicates=20)
        self.assertEqual(m['models']['exp04_robust_historical']['confusion_matrix_labels_0_1'],[[5,5],[5,5]])
        for result in m['paired_differences']['exp04_robust_historical']['exp05_cp_C_historical'].values():
            self.assertEqual(result['estimate'],0);self.assertEqual(result['interval_95'],[0,0])

    def test_outer_labels_do_not_choose_its_weight_or_cutoff(self):
        n=50;ids=[str(i) for i in range(n)];y=np.tile([0,0,0,1,1],10);folds=np.repeat(np.arange(1,6),10)
        p=np.linspace(.05,.95,n);q=p[::-1].copy();records={}
        for f in range(1,6):
            tr,va=np.flatnonzero(folds!=f),np.flatnonzero(folds==f)
            records[str(f)]={'train_indices':tr,'validation_indices':va,'training_subjects':[ids[i] for i in tr],
                             'heldout_subjects':[ids[i] for i in va],'inner':{'lab':p[tr],'cp':q[tr]},'partitions':[],'models':[]}
        data={'subjects':ids,'labels':y,'lab':(p,np.full(n,.5)),'cp':(q,np.full(n,.5)),
              'historical_primary':{},'records':records}
        _,before=calculate(data);data['labels']=y.copy();data['labels'][folds==1]=1-y[folds==1]
        _,after=calculate(data)
        self.assertEqual(before['1'],after['1'])


if __name__=='__main__':unittest.main()
