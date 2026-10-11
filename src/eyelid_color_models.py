"""Bounded nested classical selection; no outer labels enter the fitting API."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .eyelid_color_data import FEATURE_SETS
from .local_eyelid_hybrid_model import PARAMETERS, SEED

MODEL_SETTINGS = [
    {'name': 'lr_c1_none', 'family': 'logistic', 'C': 1.0, 'class_weight': None},
    {'name': 'lr_c1_balanced', 'family': 'logistic', 'C': 1.0, 'class_weight': 'balanced'},
    {'name': 'lr_c01_none', 'family': 'logistic', 'C': 0.1, 'class_weight': None},
    {'name': 'lr_c01_balanced', 'family': 'logistic', 'C': 0.1, 'class_weight': 'balanced'},
    {'name': 'lgb_none', 'family': 'lightgbm', 'class_weight': None},
    {'name': 'lgb_balanced', 'family': 'lightgbm', 'class_weight': 'balanced'},
]
CANDIDATES = [{**m, 'rendering': p, 'features': f, 'key': f'{p}_{f}', 'id': f"{p}_{f}_{m['name']}"}
              for p in ('A', 'B') for f in FEATURE_SETS for m in MODEL_SETTINGS]
BY_ID = {c['id']: c for c in CANDIDATES}
SPECIFICITIES = (.7, .8, .9)
DESIGN = {'candidate_count': len(CANDIDATES), 'candidates': CANDIDATES,
          'outer_folds': 'unchanged original five subject folds, seed20261010',
          'inner_folds': 3, 'inner_seed': SEED,
          'configuration_selection': 'highest pooled inner OOF AP; exact ties prefer earliest frozen candidate',
          'threshold_selection': 'maximum inner OOF sensitivity at specificity>=constraint; ties specificity then higher cutoff',
          'primary_specificity_constraint': .8, 'secondary_specificity_constraints': [.7, .9],
          'fixed_threshold': .5, 'lightgbm_parameters': PARAMETERS,
          'lr_parameters': {'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 2000, 'tol': .0001},
          'scaling': 'training-only StandardScaler for logistic; no learned transform for trees',
          'oversampling': 'none', 'calibration': 'none', 'augmentation': 'none',
          'policy_outputs': ['A_selected', 'B_selected', 'joint_selected'],
          'scope': 'Exploratory internal development; prior experiments informed design on these subjects'}


class NamedColumns(TransformerMixin, BaseEstimator):
    """Keep fixed feature names for LightGBM without fitting a transformation."""
    def __init__(self, names: tuple[str, ...]):
        self.names = names

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> NamedColumns:
        self.n_features_in_ = len(self.names)
        if x.shape[1] != self.n_features_in_: raise ValueError('Feature schema differs')
        return self

    def transform(self, x: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(x, columns=self.names)


def pipeline(candidate_id: str) -> Pipeline:
    c = BY_ID[candidate_id]
    if c['family'] == 'logistic':
        return Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(
            penalty='l2', solver='lbfgs', C=c['C'], max_iter=2000, tol=.0001,
            class_weight=c['class_weight'], random_state=20261010))])
    return Pipeline([('columns', NamedColumns(FEATURE_SETS[c['features']])),
                     ('classifier', LGBMClassifier(**PARAMETERS, class_weight=c['class_weight']))])


def threshold_rule(y: np.ndarray, p: np.ndarray, constraint: float = .8) -> dict[str, float]:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or set(y) != {0, 1} or not np.isfinite(p).all() or np.any((p<0)|(p>1)) or not 0 <= constraint <= 1:
        raise ValueError('Invalid threshold inputs')
    choices = []
    for t in np.unique(np.r_[0., p, np.nextafter(p.max(), np.inf)]):
        z = p >= t; sens, spec = float(z[y==1].mean()), float((~z[y==0]).mean())
        if spec >= constraint: choices.append((sens, spec, float(t)))
    sensitivity, specificity, threshold = max(choices)
    return {'threshold': threshold, 'inner_sensitivity': sensitivity, 'inner_specificity': specificity,
            'specificity_constraint': constraint}


def select(scores: dict[str, float], rendering: str | None = None) -> str:
    eligible = [c['id'] for c in CANDIDATES if rendering is None or c['rendering'] == rendering]
    if set(scores) != set(BY_ID) or not all(np.isfinite(s) for s in scores.values()):
        raise ValueError('Missing/nonfinite prespecified candidate scores')
    return max(eligible, key=lambda name: (scores[name], -list(BY_ID).index(name)))


def checked_features(features: dict[str, np.ndarray], n: int) -> None:
    if set(features) != {f'{p}_{f}' for p in ('A','B') for f in FEATURE_SETS}:
        raise ValueError('Feature cache schema differs')
    for key, x in features.items():
        if x.shape != (n, len(FEATURE_SETS[key[2:]])) or not np.isfinite(x).all():
            raise ValueError('Nonfinite/invalid feature matrix')


def fit_outer(features: dict[str, np.ndarray], labels: np.ndarray, subjects: list[str],
              validation_features: dict[str, np.ndarray], output: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit all frozen candidates, choose within outer training, save all models."""
    labels = np.asarray(labels, dtype=int)
    if len(subjects) != len(labels) or len(set(subjects)) != len(subjects) or set(labels) != {0,1} or min(np.bincount(labels)) < 3:
        raise ValueError('Invalid subject or class support')
    checked_features(features, len(labels))
    checked_features(validation_features, len(next(iter(validation_features.values()))))
    splits = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(subjects, labels))
    records = [{'inner_fold': i+1, 'train_indices': tr.tolist(), 'validation_indices': va.tolist(),
                'training_subjects': [subjects[j] for j in tr], 'validation_subjects': [subjects[j] for j in va],
                'training_class_counts': np.bincount(labels[tr], minlength=2).tolist(),
                'validation_class_counts': np.bincount(labels[va], minlength=2).tolist()}
               for i,(tr,va) in enumerate(splits)]
    scores, details, outer_probabilities = {}, {}, {}
    for c in CANDIDATES:
        directory = output/c['id'];directory.mkdir(mode=0o700)
        x = features[c['key']];inner = np.full(len(labels), np.nan)
        for i, (tr,va) in enumerate(splits,1):
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                fitted = pipeline(c['id']).fit(x[tr],labels[tr])
            inner[va] = fitted.predict_proba(x[va])[:,1]
            joblib.dump(fitted,directory/f'inner_{i}.joblib',compress=3)
        if not np.isfinite(inner).all(): raise ValueError('Incomplete inner OOF coverage')
        scores[c['id']] = float(average_precision_score(labels,inner))
        details[c['id']] = {'inner_probabilities': inner.tolist(), 'inner_ap': scores[c['id']],
                            'rules': {str(s):threshold_rule(labels,inner,s) for s in SPECIFICITIES}}
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            fitted = pipeline(c['id']).fit(x,labels)
        joblib.dump(fitted,directory/'outer.joblib',compress=3)
        outer_probabilities[c['id']] = fitted.predict_proba(validation_features[c['key']])[:,1]
    policies = {f'{p}_selected':select(scores,p) for p in ('A','B')}
    policies['joint_selected'] = select(scores)
    return outer_probabilities, {'training_subjects':subjects,'training_labels':labels.tolist(),
                                'inner_partitions':records,'candidates':details,'selected_policies':policies}
