"""CP transfer comparisons with frozen subject-score uncertainty."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .eyelid_color_run import score_registry as color_registry
from .eyelid_glare_reviewed import score_registry as glare_registry
from .eyelid_glare_evaluation import compare
from .local_eyelid_hybrid_evaluation import evaluate

EXP4 = 'local_eyelid_classification_exp04_color_optimization_2026-10-12_v1'
EXP4B = 'local_eyelid_exp04b_reviewed_comparison_2026-10-12_v1'


def historical(root: Path, subjects: list[str], y: np.ndarray, folds: np.ndarray) -> dict:
    """Reuse exact binary/JSON evidence where available, never fit a comparator."""
    for name, filename in ((EXP4, 'features.npz'), (EXP4B, 'features_used.npz')):
        with np.load(root / 'data/processed' / name / filename, allow_pickle=False) as cache:
            np.testing.assert_array_equal(cache['subjects'], subjects)
            np.testing.assert_array_equal(cache['labels'], y)
            np.testing.assert_array_equal(cache['folds'], folds)
    old = color_registry(root / 'data/processed' / EXP4, root, subjects, y, folds)
    glare = glare_registry(root / 'data/processed' / EXP4B)
    result = {name: old[name] for name in (
        'exp01_fixed', 'lab_nested_selected', 'exp02_fixed', 'exp02_selected',
        'exp03_selected', 'exp04_joint_selected_fixed', 'exp04_joint_selected')}
    result['exp04_robust_fixed'] = (old['ablation_A_robust_lr_c1_none'][0], np.full(len(y), .5))
    result['exp04_robust_selected'] = old['ablation_A_robust_lr_c1_none']
    for rule in ('fixed', 'selected'):
        result[f'exp04b_glare_weighted_{rule}'] = glare[f'glare_excluded_robust_balanced_{rule}']
    result['exp04b_nested_selected'] = glare['full_selected']
    return result


def registry(output: Path, data: dict, previous: dict) -> dict:
    """Assemble all predefined candidates, objective policies, and joint policy."""
    from .cp_transfer_models import CANDIDATES, OBJECTIVES
    folds = np.asarray(data['local_folds'])
    n = len(folds)
    result = dict(previous)
    result['cp_public_only_fixed'] = (np.load(output / 'public_only/classification.npy'), np.full(n, .5))
    names = list(CANDIDATES) + [f'{o}_selected' for o in OBJECTIVES] + ['joint_selected']
    scores = {name: np.full(n, np.nan) for name in names}
    thresholds = {name: np.full(n, np.nan) for name in names}
    for fold in range(1, 6):
        record = json.loads((output / f'fold_{fold}/selection.json').read_text())
        idx = folds == fold
        for name in CANDIDATES:
            scores[name][idx] = np.load(output / f'fold_{fold}' / name / 'outer_probabilities.npy')
            thresholds[name][idx] = record['candidates'][name]['threshold']
        for name, chosen in record['selected_policies'].items():
            scores[name][idx] = scores[chosen][idx]
            thresholds[name][idx] = thresholds[chosen][idx]
    for name in names:
        if not np.isfinite(scores[name]).all() or not np.isfinite(thresholds[name]).all():
            raise ValueError('Incomplete outer predictions/thresholds')
        result[name + '_fixed'] = (scores[name], np.full(n, .5))
        result[name + '_threshold'] = (scores[name], thresholds[name])
    return result


def comparison(y: np.ndarray, folds: np.ndarray, scores: dict, replicates: int = 2000) -> dict:
    # The existing validated comparator uses this name for the unchanged robust
    # benchmark. Alias only inside its API; public names retain their meaning.
    temporary = dict(scores)
    temporary['robust_none_selected'] = scores['exp04_robust_selected']
    measured = compare(y, folds, temporary, replicates)
    measured['models'].pop('robust_none_selected')
    measured['paired_vs_unchanged_exp4_robust'].pop('exp04_robust_selected')
    measured['paired_matched_cp_minus_imagenet'] = matched_pretraining_deltas(y, scores, replicates)
    measured['benchmark'] = 'exp04_robust_selected; predefined comparison, not E4 primary nested selection'
    return measured


def regression_metrics(hb: np.ndarray, prediction: np.ndarray, replicates: int = 2000) -> dict:
    hb, prediction = np.asarray(hb, dtype=float), np.asarray(prediction, dtype=float)
    if hb.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError('Invalid Hb predictions')
    residual = prediction - hb
    rng = np.random.default_rng(20261012)
    estimates = []
    for _ in range(replicates):
        sample = residual[rng.integers(0, len(hb), len(hb))]
        estimates.append([np.abs(sample).mean(), np.sqrt((sample ** 2).mean()), sample.mean()])
    intervals = np.quantile(estimates, [.025, .975], axis=0).T.tolist()
    return {'n': len(hb), 'mae_g_dl': float(mean_absolute_error(hb, prediction)),
            'rmse_g_dl': float(np.sqrt(mean_squared_error(hb, prediction))),
            'bias_prediction_minus_lab_g_dl': float(residual.mean()), 'r2': float(r2_score(hb, prediction)),
            'intervals_95': dict(zip(('mae_g_dl', 'rmse_g_dl', 'bias_g_dl'), intervals)),
            'bootstrap_replicates': replicates, 'seed': 20261012,
            'limitation': 'Conditional on saved predictions; no retraining, refit, or source measurement uncertainty.'}


def score_distribution(y: np.ndarray, p: np.ndarray) -> dict:
    return {str(c): {'n': int((y == c).sum()),
                    'quantiles_0_25_50_75_100': np.quantile(p[y == c], [0, .25, .5, .75, 1]).tolist()}
            for c in (0, 1)}


def figures(output: Path, y: np.ndarray, scores: dict, metrics: dict) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve
    names = ['exp04_robust_selected', 'exp04b_glare_weighted_selected', 'cp_public_only_fixed',
             'cp_classification_selected_threshold', 'cp_regression_selected_threshold',
             'imagenet_control_selected_threshold', 'joint_selected_threshold']
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for name in names:
        p, t = scores[name]
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        label = name.replace('_threshold', '').replace('_selected', '')
        axes[0, 0].plot(fpr, tpr, label=label)
        axes[0, 1].plot(recall, precision, label=label)
        m = metrics['models'][name]
        axes[0, 2].scatter(m['specificity'], m['sensitivity'], label=label)
    axes[0, 0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0, 0].set(xlabel='False-positive rate', ylabel='Sensitivity', title='OOF ROC')
    axes[0, 1].axhline(float(y.mean()), linestyle='--', color='gray')
    axes[0, 1].set(xlabel='Recall', ylabel='Precision', title='OOF precision–recall')
    axes[0, 2].set(xlabel='Specificity', ylabel='Sensitivity', title='Training-selected rules', xlim=(0, 1), ylim=(0, 1))
    axes[0, 0].legend(fontsize=6)
    axes[0, 2].legend(fontsize=6)
    for ax, name in zip(axes[1], names[3:5] + ['joint_selected_threshold']):
        matrix = np.asarray(metrics['models'][name]['confusion_matrix_labels_0_1'])
        ax.imshow(matrix, cmap='Blues', vmin=0, vmax=100)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha='center', va='center')
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Non-anemia', 'Anemia'],
               yticklabels=['Non-anemia', 'Anemia'], xlabel='Predicted', ylabel='Lab reference', title=name)
    fig.suptitle('Experiment 5: repeated 119-subject development cohort; no external test')
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(output / f'evaluation.{ext}', dpi=160)
    plt.close(fig)


def report(output: Path, metrics: dict, public: dict, regression: dict) -> None:
    lines = ['# Experiment 5 — CP transfer exploratory internal evaluation', '',
             'Same 119 AIMA subjects (19 anemia/100 non-anemia), unchanged manual anatomy and five outer folds. '
             'Nine predefined candidates, separate pediatric/adult heads, training-only inner selection. '
             'Earlier experiments informed this design; this is repeated-cohort development.', '',
             '| Model/rule | Accuracy | Sensitivity | Specificity | Balanced accuracy | Precision | F1 | ROC-AUC | AP | TP | FN | TN | FP |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, m in metrics['models'].items():
        cells = [f'{m[k]:.6f}' if m[k] is not None else 'undefined' for k in
                 ('accuracy', 'sensitivity', 'specificity', 'balanced_accuracy', 'precision', 'f1', 'roc_auc', 'average_precision')]
        cells += [str(m[k]) for k in ('true_positives', 'false_negatives', 'true_negatives', 'false_positives')]
        lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'metrics.json contains per-fold metrics, 95% Wilson/conditional bootstrap intervals and paired '
              'differences against the predefined E4 robust comparison. E4 primary nested selection remains separate. '
              'No outer candidate result determines the primary policy.', '',
              'Public validation is image-group-held development selected for checkpoint stopping, with unknown true '
              'patient linkage. public_validation/metrics.json records classification at fixed .5 and regression in g/dL. '
              'public_only/metrics.json records unadapted pediatric scores and unadapted Hb on AIMA; no local calibration. '
              'Pediatric scores are not calibrated adult anemia probabilities.', '',
              'The common opaque tissue crop/mean background does not remove the observed pediatric/adult, acquisition, '
              'ROI geometry, color-profile or resolution differences. V1 was rejected for conflicting Hb among exact '
              'repeated images; V2 preserves its original pediatric target. See audit/ and domain_comparison/.', '',
              'Only 19 anemia cases: one detection changes sensitivity by 5.3 percentage points. Conditional intervals '
              'omit training/model-selection uncertainty and overlapping folds. An inner specificity constraint of .80 '
              'does not guarantee .80 specificity on the outer fold. No clinical validation or deployment claim follows.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')


def matched_pretraining_deltas(y: np.ndarray, scores: dict, replicates: int) -> dict:
    """Paired CP-minus-ImageNet comparisons under the same adaptation design."""
    from .eyelid_transfer_evaluation import fast_metrics
    from .local_eyelid_hybrid_evaluation import METRICS
    strata = [np.flatnonzero(y == c) for c in (0, 1)]
    rng = np.random.default_rng(20261011)
    draws = [np.concatenate([rng.choice(s, len(s), replace=True) for s in strata]) for _ in range(replicates)]
    pairs = [(f'{objective}_{strategy}_threshold', f'imagenet_control_{strategy}_threshold')
             for objective in ('cp_classification', 'cp_regression') for strategy in ('A', 'B', 'C', 'selected')]
    names = set(name for pair in pairs for name in pair)
    values = {}
    for name in sorted(names):
        p, t = scores[name]
        values[name] = [fast_metrics(y[idx], p[idx], t[idx]) for idx in draws]
    result = {}
    for name, baseline in pairs:
        point, other = evaluate(y, *scores[name]), evaluate(y, *scores[baseline])
        result[name + '_minus_' + baseline] = {}
        for k in METRICS:
            differences = [a[k] - b[k] for a, b in zip(values[name], values[baseline]) if a[k] is not None and b[k] is not None]
            result[name + '_minus_' + baseline][k] = {
                'estimate': point[k] - other[k] if point[k] is not None and other[k] is not None else None,
                'interval': np.quantile(differences, [.025, .975]).tolist() if differences else None}
    return result
