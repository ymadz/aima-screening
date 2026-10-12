"""Small fixed regional comparison using only outer-training decisions."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import warnings
import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from .eyelid_color_models import pipeline as original_pipeline, threshold_rule
from .eyelid_glare_features import FEATURES

CANDIDATES = [{'id': f'{f}_{weight}', 'feature': f,
               'original_pipeline': f'A_robust_lr_c1_{weight}'}
              for f in ('robust','regional') for weight in ('none','balanced')]
BY_ID = {c['id']:c for c in CANDIDATES}
DESIGN = {'candidates':CANDIDATES,'outer_folds':'unchanged119-subject fivefold assignments seed20261010',
          'inner_folds':3,'inner_seed':20261011,'model':'Exact Experiment4 C1 L2 logistic none/balanced with training-only StandardScaler',
          'selection':'highest pooled inner OOF AP, ties frozen candidate order',
          'threshold':'maximize pooled inner sensitivity subject to specificity>=0.8, ties specificity then higher threshold',
          'specificity_constraint':.8,'fixed_threshold':.5,
          'glare_exclusion_models':'BLOCKED_PENDING_HUMAN_GLARE_REVIEW; never use diagnostic proposal features',
          'feature_selection':'none; fixed18/22 features; stability is descriptive only',
          'scope':'Exploratory repeated development subjects, no external test'}


def select(scores: dict[str,float]) -> str:
    if set(scores)!=set(BY_ID) or not np.isfinite(list(scores.values())).all():
        raise ValueError('Incomplete candidate scores')
    return max(BY_ID,key=lambda name:(scores[name],-list(BY_ID).index(name)))


def check_features(features: dict[str,np.ndarray], n: int) -> None:
    if set(features)!=set(FEATURES): raise ValueError('Original anatomy feature schema differs')
    for name,x in features.items():
        if x.shape!=(n,len(FEATURES[name])) or not np.isfinite(x).all():
            raise ValueError('Invalid/nonfinite feature matrix')


def fit_outer(features: dict[str,np.ndarray], labels: np.ndarray, subjects: list[str],
              validation: dict[str,np.ndarray], output: Path) -> tuple[dict[str,np.ndarray],dict[str,Any]]:
    """Outer validation labels are deliberately absent from this fitting API."""
    labels=np.asarray(labels)
    if len(subjects)!=len(labels) or len(set(subjects))!=len(subjects) or set(labels)!={0,1} or min(np.bincount(labels))<3:
        raise ValueError('Invalid subject/class support')
    check_features(features,len(labels));check_features(validation,len(next(iter(validation.values()))))
    splits=list(StratifiedKFold(3,shuffle=True,random_state=20261011).split(subjects,labels))
    partitions=[{'inner_fold':i,'train_indices':tr.tolist(),'validation_indices':va.tolist(),
                 'training_subjects':[subjects[j] for j in tr],'validation_subjects':[subjects[j] for j in va]}
                for i,(tr,va) in enumerate(splits,1)]
    details={};probabilities={};scores={}
    for c in CANDIDATES:
        directory=output/c['id'];directory.mkdir(mode=0o700)
        x=features[c['feature']];inner=np.full(len(labels),np.nan)
        for i,(tr,va) in enumerate(splits,1):
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                model=original_pipeline(c['original_pipeline']).fit(x[tr],labels[tr])
            inner[va]=model.predict_proba(x[va])[:,1]
            joblib.dump(model,directory/f'inner_{i}.joblib',compress=3)
        scores[c['id']]=float(average_precision_score(labels,inner))
        details[c['id']]={'inner_probabilities':inner.tolist(),'inner_ap':scores[c['id']],
                          'rule':threshold_rule(labels,inner,.8)}
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            model=original_pipeline(c['original_pipeline']).fit(x,labels)
        joblib.dump(model,directory/'outer.joblib',compress=3)
        probabilities[c['id']]=model.predict_proba(validation[c['feature']])[:,1]
    return probabilities,{'training_subjects':subjects,'training_labels':labels.tolist(),
                          'inner_partitions':partitions,'candidates':details,'selected':select(scores)}
