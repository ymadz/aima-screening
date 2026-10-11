"""Conditional OOF uncertainty and paired comparisons on fixed subject scores."""
from __future__ import annotations

from typing import Any
import numpy as np
from scipy.stats import norm, binomtest
from .local_eyelid_classifier import binary_metrics

METRICS = ('accuracy', 'sensitivity', 'specificity', 'balanced_accuracy', 'precision', 'f1', 'roc_auc', 'average_precision')
LIMITATION = 'Conditional on observed OOF predictions; stratification fixes prevalence. Model-refit uncertainty and dependence from overlapping CV training partitions are not captured. No external validation.'


def evaluate(labels: np.ndarray, probabilities: np.ndarray, thresholds: float | np.ndarray) -> dict[str, Any]:
    result = binary_metrics(labels, probabilities, threshold=0.5)
    thresholds = np.broadcast_to(np.asarray(thresholds, dtype=float), labels.shape)
    if not np.isfinite(thresholds).all() or np.any(thresholds < 0) or np.any(thresholds > np.nextafter(1.0, np.inf)):
        raise ValueError('Invalid fixed or per-subject training-selected threshold')
    predicted = probabilities >= thresholds
    tn=int(np.sum((labels==0)&~predicted));fp=int(np.sum((labels==0)&predicted))
    fn=int(np.sum((labels==1)&~predicted));tp=int(np.sum((labels==1)&predicted))
    ratio=lambda a,b: a/b if b else None
    result.update(confusion_matrix_labels_0_1=[[tn,fp],[fn,tp]],accuracy=ratio(tn+tp,len(labels)),
                  sensitivity=ratio(tp,tp+fn),specificity=ratio(tn,tn+fp),precision=ratio(tp,tp+fp),
                  f1=ratio(2*tp,2*tp+fp+fn),research_threshold=float(thresholds[0]) if np.all(thresholds==thresholds[0]) else 'fold-specific inner-selected')
    result['balanced_accuracy']=(result['sensitivity']+result['specificity'])/2
    return result


def wilson(successes: int, total: int) -> list[float] | None:
    if not total:return None
    z=float(norm.ppf(.975));p=successes/total;denominator=1+z*z/total
    center=(p+z*z/(2*total))/denominator
    radius=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [max(0.0,float(center-radius)),min(1.0,float(center+radius))]


def conditional_intervals(labels: np.ndarray, probabilities: np.ndarray, thresholds: float | np.ndarray,
                          baseline_probabilities: np.ndarray, replicates: int = 2000, seed: int = 20261011) -> dict[str, Any]:
    """Use identical resampled subjects for paired Exp2 minus fixed Exp1 deltas."""
    point=evaluate(labels,probabilities,thresholds);baseline=evaluate(labels,baseline_probabilities,.5)
    thresholds=np.broadcast_to(thresholds,labels.shape)
    strata=[np.flatnonzero(labels==v) for v in (0,1)]
    if not all(len(s) for s in strata):raise ValueError('Both classes needed')
    rng=np.random.default_rng(seed);values={k:[] for k in METRICS};differences={k:[] for k in METRICS}
    for _ in range(replicates):
        indices=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata])
        one=evaluate(labels[indices],probabilities[indices],thresholds[indices])
        other=evaluate(labels[indices],baseline_probabilities[indices],.5)
        for name in METRICS:
            if one[name] is not None:values[name].append(one[name])
            if one[name] is not None and other[name] is not None:differences[name].append(one[name]-other[name])
    interval=lambda rows: np.quantile(rows,[.025,.975]).tolist() if rows else None
    intervals={name:interval(rows) for name,rows in values.items()}
    (tn,fp),(fn,tp)=point['confusion_matrix_labels_0_1']
    intervals.update(accuracy=wilson(tn+tp,len(labels)),sensitivity=wilson(tp,tp+fn),specificity=wilson(tn,tn+fp),precision=wilson(tp,tp+fp))
    correct2=(probabilities>=thresholds)==labels;correct1=(baseline_probabilities>=.5)==labels
    gained=int(np.sum(correct2&~correct1));lost=int(np.sum(~correct2&correct1))
    return {'confidence':.95,'intervals':intervals,'bootstrap_replicates':replicates,'seed':seed,
            'bootstrap_defined_counts':{name:len(rows) for name,rows in values.items()},
            'paired_delta_exp02_minus_fixed_exp01':{name:{'estimate':point[name]-baseline[name] if point[name] is not None and baseline[name] is not None else None,'interval':interval(differences[name]),'defined_replicates':len(differences[name])} for name in METRICS},
            'paired_accuracy_discordance':{'exp02_correct_exp01_wrong':gained,'exp02_wrong_exp01_correct':lost,'mcnemar_exact_two_sided_p':float(binomtest(gained,gained+lost,.5).pvalue) if gained+lost else 1.0},
            'limitations':LIMITATION}
