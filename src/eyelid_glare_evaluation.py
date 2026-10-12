"""Shared conditional bootstrap and descriptive coefficient stability."""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
from .eyelid_glare_models import CANDIDATES
from .eyelid_glare_features import FEATURES
from .eyelid_transfer_evaluation import fast_metrics
from .local_eyelid_hybrid_evaluation import evaluate,wilson,METRICS,LIMITATION


def compare(y: np.ndarray, folds: np.ndarray, registry: dict, replicates: int=2000) -> dict:
    strata=[np.flatnonzero(y==c) for c in (0,1)]
    if not all(len(s) for s in strata): raise ValueError('Both classes required')
    rng=np.random.default_rng(20261011)
    draws=[np.concatenate([rng.choice(s,len(s),replace=True) for s in strata]) for _ in range(replicates)]
    values={};models={}
    for name,(p,t) in registry.items():
        t=np.broadcast_to(t,y.shape);m=evaluate(y,p,t)
        (tn,fp),(fn,tp)=m['confusion_matrix_labels_0_1']
        m.update(true_positives=tp,false_negatives=fn,true_negatives=tn,false_positives=fp)
        sample={k:[] for k in METRICS}
        for idx in draws:
            stats=fast_metrics(y[idx],p[idx],t[idx])
            for k in METRICS:sample[k].append(stats[k])
        intervals={k:np.quantile([v for v in sample[k] if v is not None],[.025,.975]).tolist() if any(v is not None for v in sample[k]) else None for k in METRICS}
        intervals.update(sensitivity=wilson(tp,int((y==1).sum())),specificity=wilson(tn,int((y==0).sum())),accuracy=wilson(tp+tn,len(y)),precision=wilson(tp,tp+fp))
        m['intervals_95']=intervals
        m['per_fold']={str(f):evaluate(y[folds==f],p[folds==f],t[folds==f]) for f in sorted(set(folds))}
        models[name]=m;values[name]=sample
        print(f'Calculated uncertainty: {name}',flush=True)
    baseline='robust_none_selected';paired={}
    for name in registry:
        if name==baseline:continue
        paired[name]={}
        for k in METRICS:
            deltas=[a-b for a,b in zip(values[name][k],values[baseline][k]) if a is not None and b is not None]
            paired[name][k]={'estimate':models[name][k]-models[baseline][k] if models[name][k] is not None and models[baseline][k] is not None else None,
                             'interval':np.quantile(deltas,[.025,.975]).tolist() if deltas else None}
    return {'models':models,'paired_vs_unchanged_exp4_robust':paired,'replicates':replicates,'seed':20261011,'limitations':LIMITATION}


def stability(output: Path) -> dict:
    """Describe fixed-feature coefficients in training-standardized units only."""
    result={}
    for c in CANDIDATES:
        coefficients=[];means=[];scales=[]
        for fold in range(1,6):
            model=joblib.load(output/f'fold_{fold}'/c['id']/'outer.joblib')
            coefficients.append(model['classifier'].coef_[0]);means.append(model['scaler'].mean_);scales.append(model['scaler'].scale_)
        co=np.asarray(coefficients);mu=np.asarray(means);sd=np.asarray(scales)
        rows=[]
        for j,name in enumerate(FEATURES[c['feature']]):
            rows.append({'feature':name,'coefficient_by_fold':co[:,j].tolist(),
                         'dominant_sign_fraction':float(max((co[:,j]>0).sum(),(co[:,j]<0).sum())/5),
                         'mean_standardized_coefficient':float(co[:,j].mean()),
                         'top5_abs_coefficient_folds':int(sum(j in np.argsort(np.abs(v))[-5:] for v in co)),
                         'training_scaler_mean_range':[float(mu[:,j].min()),float(mu[:,j].max())],
                         'training_scaler_scale_range':[float(sd[:,j].min()),float(sd[:,j].max())]})
        result[c['id']]={'feature_rows':rows,'same_sign_all_five':sum(r['dominant_sign_fraction']==1 for r in rows),
                         'same_sign_at_least_four':sum(r['dominant_sign_fraction']>=.8 for r in rows),
                         'limitations':'Correlated predictors and overlapping training folds; coefficient/rank stability does not establish causality or external robustness. No features chosen from these summaries.'}
    return result
