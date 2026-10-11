"""Boundary and metric checks for private experiment orchestration."""
import tempfile,unittest,subprocess,json
from unittest.mock import patch
from pathlib import Path
import numpy as np
from src.eyelid_color_run import destination
from src.eyelid_color_reproduce import check_metrics,reproduce
from src.local_eyelid_hybrid_evaluation import evaluate

class ColorRunTests(unittest.TestCase):
    def test_private_destination_excludes_prior_and_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();subprocess.run(['git','init','-q',str(root)],check=True)
            (root/'.gitignore').write_text('data/processed/*\n')
            good=root/'data/processed/local_eyelid_classification_exp04_color_optimization_test_v1'
            self.assertEqual(destination(root,good),good)
            with self.assertRaises(ValueError):destination(root,root/'data/processed/local_eyelid_classification_2026-10-10_v1')
            with self.assertRaises(ValueError):destination(root,root/'results/local_eyelid_classification_exp04_color_optimization_test_v1')
            good.mkdir(parents=True)
            with self.assertRaises(FileExistsError):destination(root,good)
    def test_independent_metrics_check_counts_ranking_and_thresholds(self):
        y=np.array([0,0,0,1,1]);p=np.array([.1,.7,.2,.8,.4]);t=np.array([.5]*5)
        m=evaluate(y,p,t);check_metrics(y,p,t,m)
        changed=dict(m);changed['roc_auc']=0
        with self.assertRaises(ValueError):check_metrics(y,p,t,changed)

    def test_checksum_failure_prevents_deserialization(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'artifact.txt').write_text('changed')
            (root/'checksum_manifest.json').write_text(json.dumps({'artifact.txt':'incorrect'}))
            with patch('src.eyelid_color_reproduce.joblib.load') as loader:
                with self.assertRaises(ValueError):reproduce(root,root)
                loader.assert_not_called()
