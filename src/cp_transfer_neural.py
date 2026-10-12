"""CP regression pretraining and explicitly replaced adult heads."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from .eyelid_transfer_neural import AnemiaMobileNet,runtime,state_hash,atomic_save,load_checkpoint
from .local_eyelid_validation import sha256

REGRESSION_SETTINGS={'max_epochs':24,'patience':4,'head_lr':.001,'tail_lr':.0001,'weight_decay':.001,'batch_size':32,'loss':'SmoothL1 beta1 on training-standardized Hb; checkpoint minimizes public validation MAE'}


def adult_initializer(public_state: dict[str,torch.Tensor],seed: int) -> dict[str,torch.Tensor]:
    """Discard pediatric classification/regression head; retain backbone only."""
    runtime(seed);state=copy.deepcopy(public_state);head=nn.Linear(576,1)
    state['head.weight']=head.weight.detach().clone();state['head.bias']=head.bias.detach().clone()
    return state


def regression_predict(model: AnemiaMobileNet,cache: torch.Tensor,mean: float,scale: float) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        values=[model(cache[i:i+32,0],cached=True).numpy() for i in range(0,len(cache),32)]
    return np.concatenate(values).astype(float)*scale+mean


def embeddings(model: AnemiaMobileNet,cache: torch.Tensor) -> np.ndarray:
    """CP/ImageNet backbone only, frozen eval with no local-label fitting."""
    model.eval()
    with torch.no_grad():
        pieces=[model.pool(model.features[-2:](cache[i:i+32,0])).flatten(1).numpy() for i in range(0,len(cache),32)]
    values=np.concatenate(pieces).astype(float)
    if values.shape!=(len(cache),576) or not np.isfinite(values).all():raise ValueError('Invalid frozen embeddings')
    return values


def fit_regression(initial_state: dict[str,torch.Tensor],cache: torch.Tensor,hb: np.ndarray,train: np.ndarray,validation: np.ndarray,ids: list[str],directory: Path,seed: int,cache_hash: str,
                   settings: dict[str,Any] | None=None) -> tuple[AnemiaMobileNet,dict]:
    """Public train/validation only; resumable guarded optimization in g/dL."""
    runtime(seed);settings=dict(REGRESSION_SETTINGS if settings is None else settings);train=np.asarray(train,dtype=int);validation=np.asarray(validation,dtype=int);hb=np.asarray(hb,dtype=float)
    if len(ids)!=len(cache) or len(set(ids))!=len(ids) or not len(train) or not len(validation) or set(train)&set(validation) or not np.isfinite(hb).all() or np.any((hb<=0)|(hb>=25)):
        raise ValueError('Invalid public regression support or leakage')
    mean,scale=float(hb[train].mean()),float(hb[train].std())
    if scale<=0:raise ValueError('No Hb variation in public training')
    signature={'train_ids':[ids[i] for i in train],'validation_ids':[ids[i] for i in validation],'training_hb_g_dl':hb[train].tolist(),'validation_hb_g_dl':hb[validation].tolist(),'settings':settings,
               'seed':seed,'cache_sha256':cache_hash,'initializer_state_sha256':state_hash(initial_state),'target_mean_training_only':mean,'target_scale_training_only':scale}
    digest=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest();directory.mkdir(parents=True,exist_ok=True)
    info_path=directory/'training.json';best=directory/'best.pt';last=directory/'last.pt'
    if info_path.exists():
        info=json.loads(info_path.read_text())
        if info['signature_sha256']!=digest or sha256(best)!=info['checkpoint_sha256']:raise ValueError('Regression completed-fit signature differs')
        return load_checkpoint(best),info
    model=AnemiaMobileNet();model.load_state_dict(initial_state);model.configure('public')
    optimizer=torch.optim.AdamW([{'params':model.head.parameters(),'lr':settings['head_lr']},{'params':model.features[-2:].parameters(),'lr':settings['tail_lr']}],weight_decay=settings['weight_decay'])
    loss_function=nn.SmoothL1Loss(beta=1.);targets=torch.tensor((hb-mean)/scale,dtype=torch.float32)
    history=[];start=1;best_key=(float('inf'),float('inf'));best_epoch=0;stale=0
    if last.exists():
        saved=torch.load(last,weights_only=True,map_location='cpu')
        if saved['signature_sha256']!=digest:raise ValueError('Regression resume provenance mismatch')
        model.load_state_dict(saved['state']);optimizer.load_state_dict(saved['optimizer']);history=saved['history'];start=saved['epoch']+1;best_key=tuple(saved['best_key']);best_epoch=saved['best_epoch'];stale=saved['stale']
    for epoch in range(start,settings['max_epochs']+1):
        if stale>=settings['patience']:break
        torch.manual_seed(seed+epoch);generator=torch.Generator().manual_seed(seed+epoch);indices=train[torch.randperm(len(train),generator=generator).numpy()];model.training_mode();total=0.
        for offset in range(0,len(indices),settings['batch_size']):
            batch=indices[offset:offset+settings['batch_size']];flip=torch.randint(2,(len(batch),),generator=generator);optimizer.zero_grad(set_to_none=True)
            loss=loss_function(model(cache[torch.tensor(batch),flip],cached=True),targets[batch]);loss.backward();optimizer.step();total+=float(loss.detach())*len(batch)
        prediction=regression_predict(model,cache[validation],mean,scale);mae=float(np.abs(prediction-hb[validation]).mean());rmse=float(np.sqrt(((prediction-hb[validation])**2).mean()));key=(mae,rmse)
        if key<best_key:
            best_key=key;best_epoch=epoch;stale=0;atomic_save({'state':model.state_dict(),'state_sha256':state_hash(model.state_dict()),'epoch':epoch,'signature_sha256':digest},best)
        else:stale+=1
        history.append({'epoch':epoch,'training_standardized_smooth_l1':total/len(train),'validation_mae_g_dl':mae,'validation_rmse_g_dl':rmse})
        atomic_save({'state':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'signature_sha256':digest,'history':history,'best_key':best_key,'best_epoch':best_epoch,'stale':stale},last)
    info={**signature,'signature_sha256':digest,'best_epoch':best_epoch,'history':history,'checkpoint_sha256':sha256(best),'status':'complete'}
    info_path.write_text(json.dumps(info,indent=2)+'\n');return load_checkpoint(best),info
