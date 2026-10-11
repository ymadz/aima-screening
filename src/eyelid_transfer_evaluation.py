"""Paired, conditional subject-score uncertainty for versioned comparisons."""
from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from .local_eyelid_hybrid_evaluation import evaluate, wilson, LIMITATION, METRICS


def fast_metrics(y: np.ndarray, p: np.ndarray, t: np.ndarray) -> dict[str, float | None]:
    z = p >= t
    tn = int(np.sum((y == 0) & ~z)); fp = int(np.sum((y == 0) & z))
    fn = int(np.sum((y == 1) & ~z)); tp = int(np.sum((y == 1) & z))
    sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
    return {"accuracy": (tn + tp) / len(y), "sensitivity": sensitivity, "specificity": specificity,
            "balanced_accuracy": (sensitivity + specificity) / 2,
            "precision": tp / (tp + fp) if tp + fp else None,
            "f1": 2 * tp / (2 * tp + fp + fn), "roc_auc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p))}


def compare_scores(y: np.ndarray, folds: np.ndarray, registry: dict[str, tuple[np.ndarray, np.ndarray]],
                   replicates: int = 2000, seed: int = 20261011) -> dict[str, Any]:
    """Reuse identical subject bootstrap draws for every model/operating rule."""
    models = {}
    values = {name: {k: [] for k in METRICS} for name in registry}
    draws = []
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(y == v) for v in (0, 1)]
    for _ in range(replicates):
        draws.append(np.concatenate([rng.choice(s, len(s), replace=True) for s in strata]))
    for name, (p, thresholds) in registry.items():
        p, thresholds = np.asarray(p), np.broadcast_to(thresholds, y.shape)
        point = evaluate(y, p, thresholds)
        (tn, fp), (fn, tp) = point['confusion_matrix_labels_0_1']
        point.update(recall=point['sensitivity'], true_negatives=tn, false_positives=fp, false_negatives=fn, true_positives=tp)
        for idx in draws:
            m = fast_metrics(y[idx], p[idx], thresholds[idx])
            for k in METRICS:
                values[name][k].append(m[k])
        intervals = {}
        for k in METRICS:
            defined = [v for v in values[name][k] if v is not None]
            intervals[k] = np.quantile(defined, [.025, .975]).tolist() if defined else None
        intervals.update(accuracy=wilson(tn + tp, len(y)), sensitivity=wilson(tp, tp + fn),
                         specificity=wilson(tn, tn + fp), precision=wilson(tp, tp + fp))
        point['uncertainty'] = {'confidence': .95, 'intervals': intervals, 'bootstrap_replicates': replicates,
                                'defined_replicates': {k: sum(v is not None for v in values[name][k]) for k in METRICS},
                                'seed': seed, 'limitations': LIMITATION}
        point['per_fold'] = {str(f): evaluate(y[folds == f], p[folds == f], thresholds[folds == f]) for f in sorted(set(folds))}
        models[name] = point
        print(f'Completed uncertainty for {name}', flush=True)
    paired = {}
    comparators = ['exp01_fixed', 'exp02_fixed', 'exp02_selected', 'lab_nested_selected']
    for name in registry:
        if name in comparators:
            continue
        paired[name] = {}
        for baseline in comparators:
            paired[name][baseline] = {}
            for k in METRICS:
                delta = [a - b for a, b in zip(values[name][k], values[baseline][k]) if a is not None and b is not None]
                one, other = models[name][k], models[baseline][k]
                paired[name][baseline][k] = {'estimate': one - other if one is not None and other is not None else None,
                                              'interval': np.quantile(delta, [.025, .975]).tolist() if delta else None,
                                              'defined_replicates': len(delta)}
    return {'models': models, 'paired_deltas': paired, 'limitations': LIMITATION,
            'bootstrap_seed': seed, 'bootstrap_replicates': replicates}
