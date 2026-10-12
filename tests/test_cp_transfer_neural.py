import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from src.eyelid_transfer_neural import AnemiaMobileNet,prefix_cache,state_hash,load_checkpoint,runtime
from src.cp_transfer_neural import adult_initializer,fit_regression,regression_predict,REGRESSION_SETTINGS,embeddings
from src.cp_transfer_models import fusion_pipeline,select,CANDIDATES

class CPNeuralTests(unittest.TestCase):
    def test_new_adult_head_never_inherits_public_output(self):
        a={'features.example':torch.tensor([1.]),'head.weight':torch.ones(1,576),'head.bias':torch.ones(1)}
        b={k:v.clone() for k,v in a.items()};b['head.weight']*=999;b['head.bias']*=999
        aa,bb=adult_initializer(a,17),adult_initializer(b,17)
        torch.testing.assert_close(aa['head.weight'],bb['head.weight'],rtol=0,atol=0)
        torch.testing.assert_close(aa['features.example'],a['features.example'],rtol=0,atol=0)
        self.assertFalse(torch.equal(aa['head.weight'],a['head.weight']))
    def test_regression_smoke_checkpoint_reproduction_and_train_target_scaling(self):
        runtime(12);model=AnemiaMobileNet().eval();x=torch.rand(20,3,224,224);cache=prefix_cache(model,x);hb=np.linspace(7.,15.,20);tr=np.arange(16);va=np.arange(16,20)
        settings=dict(REGRESSION_SETTINGS,max_epochs=2,patience=2)
        with tempfile.TemporaryDirectory() as temp:
            fitted,info=fit_regression(model.state_dict(),cache,hb,tr,va,[f'synthetic{i}' for i in range(20)],Path(temp),12,'synthetic_cache',settings)
            self.assertEqual(info['target_mean_training_only'],float(hb[tr].mean()));self.assertEqual(info['target_scale_training_only'],float(hb[tr].std()))
            np.testing.assert_array_equal(regression_predict(fitted,cache[va],info['target_mean_training_only'],info['target_scale_training_only']),regression_predict(load_checkpoint(Path(temp)/'best.pt'),cache[va],info['target_mean_training_only'],info['target_scale_training_only']))
            self.assertEqual(embeddings(fitted,cache).shape,(20,576))
            again,second=fit_regression(model.state_dict(),cache,hb,tr,va,[f'synthetic{i}' for i in range(20)],Path(temp),12,'synthetic_cache',settings);self.assertEqual(info,second)
            for name,value in model.features[:-2].state_dict().items():torch.testing.assert_close(value,fitted.features[:-2].state_dict()[name],rtol=0,atol=0)
            changed=hb.copy();changed[va]+=1
            with self.assertRaises(ValueError):fit_regression(model.state_dict(),cache,changed,tr,va,[f'synthetic{i}' for i in range(20)],Path(temp),12,'synthetic_cache',settings)
    def test_fusion_training_only_projection_and_grid_ties(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(40,594));y=np.array([0]*32+[1]*8);model=fusion_pipeline().fit(x,y)
        self.assertEqual(model['projection'].training_samples_,40)
        np.testing.assert_allclose(model['projection'].embedding_scaler_.mean_,x[:,18:].mean(0),rtol=0,atol=1e-12)
        self.assertEqual(model['projection'].transform(np.full((3,594),100.)).shape,(3,26))
        self.assertEqual(len(CANDIDATES),9);self.assertEqual(select(dict.fromkeys(CANDIDATES,.5)),'cp_classification_A');self.assertEqual(select(dict.fromkeys(CANDIDATES,.5),'imagenet_control'),'imagenet_control_A')
