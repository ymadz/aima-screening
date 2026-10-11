"""Complete synthetic artifact reproduction, tamper checks and output isolation."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.local_eyelid_classifier import FEATURES, make_subject_folds
from src.local_eyelid_hybrid_features import PREPROCESSING, private_output_path
from src.local_eyelid_hybrid_model import DESIGN, fit_outer_training
from src.local_eyelid_hybrid_evaluation import evaluate
from src.local_eyelid_hybrid_run import write_csv, write_json, verify_run
from src.preserve_local_experiment import tree_checksums


class HybridRunTests(unittest.TestCase):
    def test_output_guard_requires_new_ignored_exp02_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);subprocess.run(['git','init','-q',str(root)],check=True)
            (root/'.gitignore').write_text('data/processed/*\n')
            good=root/'data/processed/local_eyelid_classification_exp02_hybrid_synthetic_v1'
            self.assertEqual(private_output_path(root,good),good.resolve())
            for bad in [root/'data/processed/local_eyelid_classification_2026-10-10_v1',root/'raw/output',root/'results/local_eyelid_classification_exp02_hybrid_bad']:
                with self.assertRaises(ValueError):private_output_path(root,bad)
            good.mkdir(parents=True)
            with self.assertRaises(FileExistsError):private_output_path(root,good)

    def test_all_saved_pipelines_and_decisions_reproduce_and_tampering_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary);rng=np.random.default_rng(99)
            subjects=[f'synthetic_{i:03d}' for i in range(119)]
            y=np.array([1]*19+[0]*100);x=rng.normal(size=(119,582))
            fold_rows=make_subject_folds(dict(zip(subjects,y)))
            assignments={r['subject_id']:int(r['fold']) for r in fold_rows}
            folds=np.array([assignments[s] for s in subjects]);p=np.zeros(119);t=np.zeros(119);baseline=np.full(119,.2)
            write_csv(output/'subject_folds.csv',fold_rows)
            write_csv(output/'baseline_features.csv',[{'subject_id':s,**dict(zip(FEATURES,row[:6]))} for s,row in zip(subjects,x)])
            np.savez_compressed(output/'features.npz',features=x,subjects=np.array(subjects))
            write_json(output/'experiment_config.json',{'design':DESIGN,'preprocessing':PREPROCESSING})
            for number in range(1,6):
                folder=output/f'fold_{number}';folder.mkdir();train=folds!=number;valid=folds==number
                _,p[valid],record=fit_outer_training(x[train],y[train],[s for s,use in zip(subjects,train) if use],x[valid],folder)
                t[valid]=record['threshold'];write_json(folder/'selection.json',record)
            rows=[{'subject_id':s,'label':int(label),'fold':int(fold),'hybrid_probability':float(score),
                   'selected_threshold':float(threshold),'fixed_prediction':int(score>=.5),'selected_prediction':int(score>=threshold),'baseline_probability':float(base)}
                  for s,label,fold,score,threshold,base in zip(subjects,y,folds,p,t,baseline)]
            write_csv(output/'oof_predictions.csv',rows)
            write_json(output/'metrics.json',{'pooled_oof':{'fixed_0_5':evaluate(y,p,.5),'inner_selected':evaluate(y,p,t)},
                       'baseline_fixed_0_5':evaluate(y,baseline,.5),
                       'per_fold':{str(n):{'fixed_0_5':evaluate(y[folds==n],p[folds==n],.5),'inner_selected':evaluate(y[folds==n],p[folds==n],t[folds==n])} for n in range(1,6)}})
            write_json(output/'checksum_manifest.json',tree_checksums(output))
            self.assertEqual(verify_run(output)['fold_pipelines'],35)
            prediction_path=output/'oof_predictions.csv'
            prediction_path.write_text(prediction_path.read_text().replace(subjects[0],'tampered_subject',1))
            with self.assertRaises(ValueError):verify_run(output)
            # Even an updated checksum cannot hide altered subject coverage.
            hashes=tree_checksums(output);hashes.pop('checksum_manifest.json')
            (output/'checksum_manifest.json').write_text(json.dumps(hashes))
            with self.assertRaises(ValueError):verify_run(output)
