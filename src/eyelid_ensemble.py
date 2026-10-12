"""Small, training-only ensembles and strict paired prediction alignment."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.metrics import average_precision_score

from .eyelid_color_models import threshold_rule
from .local_eyelid_hybrid_evaluation import wilson

WEIGHTS = (0.5, 0.75, 0.25)  # LAB weight; equal average wins exact AP ties.
TARGETS = (0.8, 0.85, 0.9)
DESIGN = {
    'id': 'AIMA-EXP06',
    'lab': 'Exp4 A_robust_lr_c1_none; predefined comparison, not E4 primary',
    'cp': 'Exp5 cp_classification_C; individual candidate, not joint primary',
    'weights_lab': list(WEIGHTS),
    'weight_selection': 'maximum pooled inner OOF average precision; frozen order ties',
    'threshold_selection': 'maximum inner sensitivity at specificity >= target; ties specificity then higher cutoff',
    'specificity_targets': list(TARGETS), 'primary_target': 0.8,
    'fixed_rule': 'equal raw probability average; fixed research cutoff 0.5 also reported',
    'inner': 'reuse verified matched three-fold inner models/predictions within each unchanged outer training fold',
    'outer': 'unchanged five subject folds, seed 20261010',
    'calibration': 'none; score scales differ (LAB unweighted, CP balanced); probabilities are uncalibrated',
    'stacking': 'not implemented; 19 positives and high correlation support a smaller investigation first',
    'scope': 'exploratory repeated-cohort analysis; gate and base-model choice informed by historical outcomes',
}


def align_predictions(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require complete identical subject sets and labels/folds; never inner join."""
    for frame in (left, right):
        if not {'subject_id', 'label', 'fold'} <= set(frame):
            raise ValueError('Missing identity/label/fold columns')
        if frame.subject_id.isna().any() or (frame.subject_id.astype(str).str.strip() == '').any():
            raise ValueError('Blank subject IDs')
        if frame.subject_id.duplicated().any():
            raise ValueError('Duplicate subject IDs')
        if set(frame.label) != {0, 1} or set(frame.fold) != set(range(1, 6)):
            raise ValueError('Invalid binary labels or outer folds')
    if set(left.subject_id) != set(right.subject_id):
        raise ValueError('Subject sets differ')
    a = left.sort_values('subject_id').reset_index(drop=True)
    b = right.set_index('subject_id').loc[a.subject_id].reset_index()
    if not np.array_equal(a[['label', 'fold']], b[['label', 'fold']]):
        raise ValueError('Labels or fold assignments differ')
    return a, b


def average_scores(lab: np.ndarray, cp: np.ndarray, lab_weight: float) -> np.ndarray:
    """Combine aligned finite positive-class scores without using any labels."""
    a, b = np.asarray(lab, dtype=float), np.asarray(cp, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not len(a) or not 0 <= lab_weight <= 1:
        raise ValueError('Invalid score alignment or weight')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any((a < 0) | (a > 1) | (b < 0) | (b > 1)):
        raise ValueError('Scores must be positive-class probabilities in [0,1]')
    return lab_weight * a + (1 - lab_weight) * b


def training_policy(labels: np.ndarray, lab: np.ndarray, cp: np.ndarray,
                    training_ids: list[str], heldout_ids: list[str]) -> dict[str, Any]:
    """No held-out labels accepted. Inputs must be matched inner OOF scores."""
    y = np.asarray(labels)
    if len(training_ids) != len(y) or len(set(training_ids)) != len(y):
        raise ValueError('Invalid training identity scope')
    if len(set(heldout_ids)) != len(heldout_ids) or set(training_ids) & set(heldout_ids):
        raise ValueError('Training/heldout subject leakage')
    if set(y) != {0, 1} or min(np.bincount(y.astype(int))) < 3:
        raise ValueError('Insufficient class support')
    candidates = {}
    for w in WEIGHTS:
        p = average_scores(lab, cp, w)
        if p.shape != y.shape:
            raise ValueError('Training scores and labels differ')
        candidates[str(w)] = {'inner_ap': float(average_precision_score(y, p)),
                              'rules': {str(t): threshold_rule(y, p, t) for t in TARGETS}}
    chosen = max(WEIGHTS, key=lambda w: (candidates[str(w)]['inner_ap'], -WEIGHTS.index(w)))
    return {'selected_lab_weight': chosen, 'candidates': candidates,
            'endpoints': {'lab': {str(t): threshold_rule(y, lab, t) for t in TARGETS},
                          'cp': {str(t): threshold_rule(y, cp, t) for t in TARGETS}}}


def overlap(labels: np.ndarray, lab: np.ndarray, cp: np.ndarray,
            lab_thresholds: np.ndarray, cp_thresholds: np.ndarray) -> dict[str, Any]:
    """Describe historical error overlap, not a fitted union policy."""
    y = np.asarray(labels)
    average_scores(lab, cp, 0.5)
    if y.shape != lab.shape or set(y) != {0, 1}:
        raise ValueError('Invalid paired labels')
    for thresholds in (lab_thresholds, cp_thresholds):
        if thresholds.shape != y.shape or not np.isfinite(thresholds).all() or np.any((thresholds < 0) | (thresholds > np.nextafter(1., np.inf))):
            raise ValueError('Invalid paired thresholds')
    a, b = lab >= lab_thresholds, cp >= cp_thresholds
    positive, negative = {}, {}
    keys = ('both_positive', 'lab_only_positive', 'cp_only_positive', 'both_negative')
    conditions = (a & b, a & ~b, ~a & b, ~a & ~b)
    for value, dest in ((1, positive), (0, negative)):
        dest.update({k: int(np.sum((y == value) & c)) for k, c in zip(keys, conditions)})
    n = len(y)
    discordant = int(np.sum(a != b))
    gained = int(np.sum((b == y) & (a != y)))
    lost = int(np.sum((b != y) & (a == y)))
    return {'n': n, 'anemia': int(y.sum()), 'anemia_overlap': positive, 'non_anemia_overlap': negative,
            'pearson': float(np.corrcoef(lab, cp)[0, 1]),
            'spearman': float(spearmanr(lab, cp).statistic),
            'binary_agreement': float(np.mean(a == b)),
            'agreement_wilson_95': wilson(n - discordant, n),
            'disagreement_rate': discordant / n, 'disagreement_wilson_95': wilson(discordant, n),
            'paired_correctness': {'cp_correct_lab_wrong': gained, 'lab_correct_cp_wrong': lost,
                                   'mcnemar_exact_p': float(binomtest(gained, gained + lost).pvalue) if gained + lost else 1.0},
            'anemia_disagreement_wilson_95': wilson(positive[keys[1]] + positive[keys[2]], int(y.sum())),
            'non_anemia_disagreement_wilson_95': wilson(negative[keys[1]] + negative[keys[2]], int((y == 0).sum())),
            'limitations': 'Conditional descriptive overlap on reused development subjects; union counts are not ensemble performance or independent model selection.'}
