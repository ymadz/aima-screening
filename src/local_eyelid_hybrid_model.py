"""Prespecified nested training; outer labels never enter model/threshold selection."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .local_eyelid_classifier import FEATURES

SEED = 20261011
CANDIDATES = ('unweighted', 'balanced')
PARAMETERS = {'objective': 'binary', 'n_estimators': 100, 'learning_rate': 0.03,
              'max_depth': 2, 'num_leaves': 3, 'min_child_samples': 10,
              'reg_lambda': 5.0, 'min_split_gain': 0.05, 'max_bin': 31,
              'subsample': 1.0, 'colsample_bytree': 1.0, 'n_jobs': 1,
              'deterministic': True, 'force_col_wise': True,
              'random_state': SEED, 'verbosity': -1}
DESIGN = {'outer_folds': 'exact Experiment 1 five subject assignments, seed 20261010',
          'inner_folds': 3, 'inner_seed': SEED, 'embedding_scaling': 'training-fit StandardScaler',
          'pca_components': 8, 'pca_solver': 'full', 'lab_features': list(FEATURES),
          'candidates': list(CANDIDATES), 'parameters': PARAMETERS,
          'selection': 'highest pooled inner OOF average precision; ties prefer unweighted',
          'threshold_objective': 'maximum inner OOF sensitivity with specificity >= 0.80; ties greater specificity then higher threshold',
          'fixed_threshold': 0.5, 'augmentation': 'none', 'oversampling': 'none',
          'early_stopping': 'none', 'calibration': 'none; scores are research probabilities',
          'outer_results_tuning': 'none; one prespecified run'}


def valid_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 582 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected finite six LAB + 576 embedding features')
    return x


class HybridProjection(TransformerMixin, BaseEstimator):
    """Keep LAB untouched and learn embedding scaling/PCA only during fit."""
    def __init__(self, n_components: int = 8):
        self.n_components = n_components

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> HybridProjection:
        x = valid_features(x)
        if len(x) <= self.n_components:
            raise ValueError('Insufficient training support for PCA')
        self.scaler_ = StandardScaler().fit(x[:, 6:])
        self.pca_ = PCA(n_components=self.n_components, svd_solver='full').fit(self.scaler_.transform(x[:, 6:]))
        self.training_samples_ = len(x)
        return self

    def transform(self, x: np.ndarray) -> pd.DataFrame:
        x = valid_features(x)
        values = np.concatenate([x[:, :6], self.pca_.transform(self.scaler_.transform(x[:, 6:]))], axis=1)
        return pd.DataFrame(values, columns=[*FEATURES, *[f'embedding_pc_{i+1}' for i in range(self.n_components)]])


def make_pipeline(candidate: str) -> Pipeline:
    if candidate not in CANDIDATES:
        raise ValueError('Unknown prespecified candidate')
    return Pipeline([('projection', HybridProjection()),
                     ('classifier', LGBMClassifier(**PARAMETERS, class_weight='balanced' if candidate == 'balanced' else None))])


def select_candidate(scores: dict[str, float]) -> str:
    if set(scores) != set(CANDIDATES) or not all(np.isfinite(v) for v in scores.values()):
        raise ValueError('Missing/nonfinite candidate scores')
    return max(CANDIDATES, key=lambda name: (scores[name], -CANDIDATES.index(name)))


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if labels.shape != probabilities.shape or labels.ndim != 1 or set(labels) != {0, 1} or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Invalid inner threshold inputs')
    # >maximum is an explicit all-negative fallback even if a score equals 1.
    candidates = np.unique(np.r_[0.0, probabilities, np.nextafter(probabilities.max(), np.inf)])
    choices = []
    for threshold in candidates:
        predicted = probabilities >= threshold
        sensitivity = float(predicted[labels == 1].mean())
        specificity = float((~predicted[labels == 0]).mean())
        if specificity >= 0.80:
            choices.append((sensitivity, specificity, float(threshold)))
    sensitivity, specificity, threshold = max(choices)
    return {'threshold': threshold, 'inner_sensitivity': sensitivity, 'inner_specificity': specificity}


def fit_outer_training(x: np.ndarray, labels: np.ndarray, subjects: list[str],
                       validation_x: np.ndarray, models_dir: Path | None = None) -> tuple[Pipeline, np.ndarray, dict[str, Any]]:
    """Choose settings entirely inside outer training; API takes no outer labels."""
    x, validation_x = valid_features(x), valid_features(validation_x)
    labels = np.asarray(labels)
    if len(labels) != len(x) or len(subjects) != len(x) or len(set(subjects)) != len(subjects) or set(labels) != {0, 1} or min(np.bincount(labels)) < 3:
        raise ValueError('Invalid subject/class support for inner CV')
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    splits = list(splitter.split(x, labels))
    assignment = np.zeros(len(labels), dtype=int)
    records = []
    for number, (training, validation) in enumerate(splits, 1):
        if set(training) & set(validation) or set(labels[training]) != {0, 1} or set(labels[validation]) != {0, 1}:
            raise ValueError('Inner subject leakage or missing class')
        assignment[validation] = number
        records.append({'inner_fold': number, 'training_subjects': [subjects[i] for i in training],
                        'validation_subjects': [subjects[i] for i in validation],
                        'training_class_counts': np.bincount(labels[training], minlength=2).tolist(),
                        'validation_class_counts': np.bincount(labels[validation], minlength=2).tolist()})
    inner_probabilities, scores = {}, {}
    for candidate in CANDIDATES:
        probabilities = np.full(len(labels), np.nan)
        for number, (training, validation) in enumerate(splits, 1):
            pipeline = make_pipeline(candidate).fit(x[training], labels[training])
            probabilities[validation] = pipeline.predict_proba(x[validation])[:, 1]
            if models_dir is not None:
                joblib.dump(pipeline, models_dir / f'inner_{candidate}_{number}.joblib')
        if not np.isfinite(probabilities).all():
            raise ValueError('Incomplete inner OOF prediction coverage')
        inner_probabilities[candidate] = probabilities.tolist()
        scores[candidate] = float(average_precision_score(labels, probabilities))
    selected = select_candidate(scores)
    threshold = select_threshold(labels, np.array(inner_probabilities[selected]))
    model = make_pipeline(selected).fit(x, labels)
    if models_dir is not None:
        joblib.dump(model, models_dir / 'outer_model.joblib')
    return model, model.predict_proba(validation_x)[:, 1], {
        'training_subjects': subjects, 'training_labels': labels.tolist(),
        'inner_assignment': assignment.tolist(), 'inner_partitions': records,
        'inner_oof_probabilities': inner_probabilities, 'inner_average_precision': scores,
        'selected_candidate': selected, **threshold,
        'projection_dimensions': 14, 'training_samples': len(x),
        'pca_explained_variance_ratio': model.named_steps['projection'].pca_.explained_variance_ratio_.tolist()}
