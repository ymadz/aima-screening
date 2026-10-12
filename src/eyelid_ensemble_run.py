"""Reuse sealed fold models for exploratory ensembles; never fit a model.

Run and verification use explicit read-only sources and exclusive private outputs.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from typing import Any

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from .eyelid_ensemble import DESIGN, WEIGHTS, TARGETS, align_predictions, average_scores, overlap, training_policy
from .eyelid_color_verify import check_model
from .eyelid_color_models import BY_ID
from .cp_transfer_reproduce import check_fusion, source_guard
from .local_eyelid_hybrid_evaluation import evaluate, wilson, METRICS, LIMITATION
from .local_eyelid_validation import sha256
from .preserve_local_experiment import tree_checksums

EXP4 = 'local_eyelid_classification_exp04_color_optimization_2026-10-12_v1'
EXP5 = 'local_eyelid_classification_exp05_cp_transfer_2026-10-12_v1'
LAB = 'A_robust_lr_c1_none'
CP = 'cp_classification_C'


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def checksum_guard(directory: Path) -> dict[str, str]:
    expected = json.loads((directory / 'checksum_manifest.json').read_text())
    actual = tree_checksums(directory)
    actual.pop('checksum_manifest.json', None)
    if expected != actual:
        raise ValueError('Checksum coverage or content changed; no deserialization')
    return expected


def positive_scores(model: Any, features: np.ndarray) -> np.ndarray:
    """Fail closed when a saved classifier uses a different class order."""
    if list(model.classes_) != [0, 1]:
        raise ValueError('Positive class differs')
    return model.predict_proba(features)[:, 1]


def matched_scope(a: dict, b: dict, ids: list[str], y: np.ndarray, heldout: list[str]) -> None:
    """Enforce equal inner partitions, train-only labels and heldout exclusion."""
    for record in (a, b):
        if record['training_subjects'] != ids or record['training_labels'] != y.tolist():
            raise ValueError('Outer training scope differs')
        if record['validation_subjects'] != heldout or set(ids) & set(heldout):
            raise ValueError('Outer validation scope/leakage differs')
    if a['inner_partitions'] != b['inner_partitions']:
        raise ValueError('Inner folds incompatible')
    splits = StratifiedKFold(3, shuffle=True, random_state=20261011).split(ids, y)
    for record, (tr, va) in zip(a['inner_partitions'], splits):
        if (record['training_subjects'] != [ids[i] for i in tr]
                or record['validation_subjects'] != [ids[i] for i in va]
                or record['train_indices'] != tr.tolist() or record['validation_indices'] != va.tolist()
                or set(record['training_subjects']) & set(record['validation_subjects'])):
            raise ValueError('Inner subject membership/leakage differs')


def load_bases(root: Path) -> dict:
    """Hash-first, reproduce both matched candidates' 40 pipelines, no fitting."""
    e4, e5 = (root / 'data/processed' / name for name in (EXP4, EXP5))
    checksum_guard(e4); checksum_guard(e5); source_guard(e5, root)
    a, b = align_predictions(pd.read_csv(e4/'oof_predictions.csv', dtype={'subject_id': str}, float_precision='round_trip'),
                             pd.read_csv(e5/'oof_predictions.csv', dtype={'subject_id': str}, float_precision='round_trip'))
    subjects, y, folds = a.subject_id.tolist(), a.label.to_numpy(), a.fold.to_numpy()
    if len(y) != 119 or np.bincount(y).tolist() != [100, 19]:
        raise ValueError('Approved cohort differs')
    with np.load(e4/'features.npz', allow_pickle=False) as cache:
        if subjects != cache['subjects'].tolist(): raise ValueError('LAB feature subject order differs')
        np.testing.assert_array_equal(y, cache['labels']); np.testing.assert_array_equal(folds, cache['folds'])
        lab_x = cache['A_robust']
    with np.load(e5/'frozen_features.npz', allow_pickle=False) as cache:
        cp_x = cache['cp_classification']
    np.testing.assert_array_equal(cp_x[:, :18], lab_x)
    lab_p = a.ablation_A_robust_lr_c1_none_probability.to_numpy()
    lab_t = a.ablation_A_robust_lr_c1_none_threshold.to_numpy()
    cp_p = b.cp_classification_C_threshold_probability.to_numpy()
    cp_t = b.cp_classification_C_threshold_threshold.to_numpy()
    np.testing.assert_array_equal(lab_p, b.exp04_robust_selected_probability)
    np.testing.assert_array_equal(lab_t, b.exp04_robust_selected_threshold)
    records = {}
    for f in range(1, 6):
        tr, va = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        ids, heldout = [subjects[i] for i in tr], [subjects[i] for i in va]
        ra = json.loads((e4/f'fold_{f}/selection.json').read_text())
        rb = json.loads((e5/f'fold_{f}/selection.json').read_text())
        matched_scope(ra, rb, ids, y[tr], heldout)
        inner = {}
        model_paths = []
        for name, matrix, directory, record in [('lab', lab_x, e4/f'fold_{f}'/LAB, ra),
                                                  ('cp', cp_x, e5/f'fold_{f}'/CP, rb)]:
            p = np.full(len(tr), np.nan)
            for i, part in enumerate(record['inner_partitions'], 1):
                tt, vv = np.asarray(part['train_indices']), np.asarray(part['validation_indices'])
                path = directory / f'inner_{i}.joblib'
                model = joblib.load(path)
                if name == 'lab': check_model(model, BY_ID[LAB], matrix[tr[tt]])
                else: check_fusion(model, matrix[tr[tt]])
                p[vv] = positive_scores(model, matrix[tr[vv]])
                model_paths.append({'path': str(path), 'sha256': sha256(path)})
            candidate = LAB if name == 'lab' else CP
            np.testing.assert_allclose(p, record['candidates'][candidate]['inner_probabilities'], rtol=0, atol=1e-14)
            inner[name] = p
            path = directory / 'outer.joblib'; model = joblib.load(path)
            if name == 'lab': check_model(model, BY_ID[LAB], matrix[tr])
            else: check_fusion(model, matrix[tr])
            np.testing.assert_allclose(positive_scores(model, matrix[va]), (lab_p if name == 'lab' else cp_p)[va], rtol=0, atol=1e-14)
            model_paths.append({'path': str(path), 'sha256': sha256(path)})
        records[str(f)] = {'train_indices': tr, 'validation_indices': va, 'training_subjects': ids,
                           'heldout_subjects': heldout, 'inner': inner, 'partitions': ra['inner_partitions'],
                           'models': model_paths}
    return {'subjects': subjects, 'labels': y, 'folds': folds, 'lab': (lab_p, lab_t), 'cp': (cp_p, cp_t),
            'historical_primary': {'exp04_primary_historical': (a.exp04_joint_selected_probability.to_numpy(), a.exp04_joint_selected_threshold.to_numpy()),
                                   'exp05_primary_historical': (b.joint_selected_threshold_probability.to_numpy(), b.joint_selected_threshold_threshold.to_numpy())},
            'records': records, 'features': {'lab': lab_x, 'cp': cp_x}}


def calculate(data: dict) -> tuple[dict, dict]:
    """All learned decisions are completed before outer evaluation is invoked."""
    n = len(data['subjects']); y = data['labels']
    scores = dict(data['historical_primary'])
    scores['exp04_robust_historical'] = data['lab']
    scores['exp05_cp_C_historical'] = data['cp']
    names = ('lab', 'cp', 'equal', 'lab75', 'lab25', 'weighted')
    for name in names:
        for rule in ('fixed', *[f'spec{t}' for t in TARGETS]):
            scores[name+'_'+rule] = (np.full(n, np.nan), np.full(n, np.nan))
    decisions = {}
    for f, record in data['records'].items():
        tr, va = record['train_indices'], record['validation_indices']
        policy = training_policy(y[tr], record['inner']['lab'], record['inner']['cp'],
                                 record['training_subjects'], record['heldout_subjects'])
        decisions[f] = {'policy': policy, 'outer_fold': int(f), 'training_subjects': record['training_subjects'],
                        'heldout_subjects': record['heldout_subjects'], 'inner_partitions': record['partitions'],
                        'read_only_model_references': record['models']}
        weights = {'equal': .5, 'lab75': .75, 'lab25': .25, 'weighted': policy['selected_lab_weight']}
        for name in names:
            p = data[name][0][va] if name in ('lab', 'cp') else average_scores(data['lab'][0][va], data['cp'][0][va], weights[name])
            rules = policy['endpoints'][name] if name in ('lab', 'cp') else policy['candidates'][str(weights[name])]['rules']
            for rule in ('fixed', *[f'spec{t}' for t in TARGETS]):
                threshold = .5 if rule == 'fixed' else rules[rule[4:]]['threshold']
                scores[name+'_'+rule][0][va] = p; scores[name+'_'+rule][1][va] = threshold
    if any(not np.isfinite(v).all() for pair in scores.values() for v in pair):
        raise ValueError('Incomplete heldout scores or thresholds')
    return scores, decisions


def metrics_with_uncertainty(y: np.ndarray, folds: np.ndarray, scores: dict, replicates: int = 2000) -> dict:
    """Shared stratified draws; reuse ranking draws across threshold policies."""
    rng = np.random.default_rng(20261011); strata = [np.flatnonzero(y == c) for c in (0, 1)]
    draws = np.array([np.concatenate([rng.choice(s, len(s), replace=True) for s in strata]) for _ in range(replicates)])
    yy = y[draws]; models, values, rankings = {}, {}, {}
    for name, (p, t) in scores.items():
        m = evaluate(y, p, t); (tn, fp), (fn, tp) = m['confusion_matrix_labels_0_1']
        m.update(true_positives=tp, false_negatives=fn, true_negatives=tn, false_positives=fp)
        key = p.tobytes()
        if key not in rankings:
            rankings[key] = {'roc_auc': np.array([roc_auc_score(y[ix], p[ix]) for ix in draws]),
                             'average_precision': np.array([average_precision_score(y[ix], p[ix]) for ix in draws])}
        zz = p[draws] >= t[draws]
        tpv, tnv = ((yy == 1) & zz).sum(1), ((yy == 0) & ~zz).sum(1)
        fpv, fnv = len(strata[0])-tnv, len(strata[1])-tpv
        sensitivity, specificity = tpv/len(strata[1]), tnv/len(strata[0])
        vals = {'accuracy': (tpv+tnv)/len(y), 'sensitivity': sensitivity, 'specificity': specificity,
                'balanced_accuracy': (sensitivity+specificity)/2,
                'precision': np.divide(tpv, tpv+fpv, out=np.full(replicates, np.nan), where=tpv+fpv > 0),
                'f1': 2*tpv/(2*tpv+fpv+fnv), **rankings[key]}
        values[name] = vals
        ci = {k: np.quantile(v[np.isfinite(v)], [.025, .975]).tolist() if np.isfinite(v).any() else None for k,v in vals.items()}
        ci.update(sensitivity=wilson(tp, tp+fn), specificity=wilson(tn, tn+fp), accuracy=wilson(tp+tn, len(y)), precision=wilson(tp, tp+fp))
        m['intervals_95'] = ci
        m['per_fold'] = {str(f): evaluate(y[folds == f], p[folds == f], t[folds == f]) for f in range(1, 6)}
        models[name] = m
        print('Evaluated '+name, flush=True)
    paired = {}
    for name in scores:
        paired[name] = {}
        for base in ('exp04_robust_historical', 'exp05_cp_C_historical'):
            paired[name][base] = {}
            for k in METRICS:
                delta = values[name][k]-values[base][k]; delta = delta[np.isfinite(delta)]
                paired[name][base][k] = {'estimate': models[name][k]-models[base][k] if models[name][k] is not None and models[base][k] is not None else None,
                                        'interval_95': np.quantile(delta, [.025, .975]).tolist() if len(delta) else None}
    return {'models': models, 'paired_differences': paired, 'bootstrap_replicates': replicates,
            'seed': 20261011, 'limitations': LIMITATION}


def figures(output: Path, data: dict, scores: dict, agreement: dict) -> None:
    y = data['labels']; p, t = data['lab']; q, u = data['cp']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    categories = (p >= t).astype(int)*2+(q >= u).astype(int)
    for c, label in enumerate(('Both negative', 'CP only', 'LAB only', 'Both positive')):
        idx = categories == c
        axes[0].scatter(p[idx], q[idx], label=label, s=18, alpha=.7)
    axes[0].set(xlabel='LAB anemia score', ylabel='CP hybrid anemia score', title=f'Pearson r={agreement["pearson"]:.3f}')
    axes[0].legend(fontsize=7)
    for ax, values, title in [(axes[1], agreement['anemia_overlap'], '19 anemia subjects'),
                              (axes[2], agreement['non_anemia_overlap'], '100 non-anemia subjects')]:
        ax.bar(['Both +', 'LAB only +', 'CP only +', 'Both −'], list(values.values()))
        ax.set_title(title); ax.set_ylabel('Subjects'); ax.tick_params(axis='x', rotation=25)
    fig.tight_layout(); fig.savefig(output/'overlap.png', dpi=180); fig.savefig(output/'overlap.svg'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in ('lab_spec0.8', 'cp_spec0.8', 'equal_spec0.8', 'weighted_spec0.8'):
        p, _ = scores[name]; fpr, tpr, _ = roc_curve(y, p); precision, recall, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, label=name); axes[1].step(recall, precision, where='post', label=name)
    axes[0].plot([0,1], [0,1], 'k:', alpha=.4)
    axes[0].set(xlabel='False-positive rate', ylabel='Sensitivity', title='Exploratory outer OOF ROC')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Exploratory outer OOF precision–recall')
    for ax in axes: ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output/'roc_pr.png', dpi=180); fig.savefig(output/'roc_pr.svg'); plt.close(fig)


def run(root: Path, output: Path, preflight: Path) -> dict:
    os.umask(0o077); root, output = root.resolve(), output.resolve()
    if not output.is_relative_to(root/'data/processed') or not output.name.startswith('local_eyelid_classification_exp06_ensemble_v'):
        raise ValueError('Require private versioned Experiment 6 output')
    if output.exists(): raise FileExistsError('Never overwrite a prior experiment')
    if subprocess.run(['git','check-ignore','-q',str(output/'private.csv')], cwd=root).returncode:
        raise ValueError('Output is not ignored')
    for name in ('exp04', 'exp05'):
        log = (preflight/(name+'_verification.log')).read_text()
        if '"status": "reproduced_without_refitting"' not in log:
            raise ValueError('Complete fresh baseline reproduction must precede ensemble run')
    start = time.perf_counter(); data = load_bases(root); output.mkdir(mode=0o700)
    code_root = Path(__file__).resolve().parents[1]
    environment = {'python': platform.python_version(), 'packages': {n: importlib.metadata.version(n) for n in ('numpy','scipy','pandas','scikit-learn','torch','torchvision','joblib','matplotlib')}, 'threads': 1}
    provenance = {'design': DESIGN, 'base_commit': '0943363b874c612be6ac241de8bd3819c55ccc50',
                  'code_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=code_root, text=True).strip(),
                  'source_root': str(root), 'preflight': str(preflight), 'seed': 20261011,
                  'baseline_checksum_manifests': {name: sha256(root/'data/processed'/name/'checksum_manifest.json') for name in (EXP4, EXP5)},
                  'sources_before_read': json.loads((preflight/'preservation_before.json').read_text())['sources'],
                  'positive_class': 'classes_[1] == 1 == laboratory-derived anemia', 'refitted_models': 0,
                  'evaluation_code_sha256': {str(p.relative_to(code_root)):sha256(p) for p in (code_root/'src/eyelid_ensemble.py',code_root/'src/eyelid_ensemble_run.py')}}
    write_json(output/'experiment_manifest.json', provenance); write_json(output/'environment.json', environment)
    for p in preflight.iterdir():
        if p.is_file(): shutil.copy2(p, output/p.name)
    pseudonyms = {s: f'E06-{i+1:03d}' for i,s in enumerate(data['subjects'])}
    pd.DataFrame({'deidentified_id': list(pseudonyms.values()), 'source_subject_id': data['subjects']}).to_csv(output/'identity_key_private.csv', index=False)
    agreement = overlap(data['labels'], *[data[k][0] for k in ('lab','cp')], *[data[k][1] for k in ('lab','cp')])
    rng = np.random.default_rng(20261011); correlations = []
    for _ in range(2000):
        ix = np.concatenate([rng.choice(np.flatnonzero(data['labels']==c), int(np.sum(data['labels']==c)), replace=True) for c in (0,1)])
        correlations.append([np.corrcoef(data['lab'][0][ix],data['cp'][0][ix])[0,1],spearmanr(data['lab'][0][ix],data['cp'][0][ix]).statistic])
    agreement['conditional_correlation_intervals_95'] = dict(zip(('pearson','spearman'),np.quantile(correlations,[.025,.975],axis=0).T.tolist()))
    write_json(output/'prediction_alignment.json', {'subjects':119,'labels_folds_positive_class_matched':True,'matched_inner_partitions':True,'pipelines_reproduced_without_refitting':40,'model_input_features_same_robust18':True})
    write_json(output/'error_overlap.json', agreement)
    write_json(output/'ensemble_feasibility.json', {'decision':'small_simple_investigation_justified','evidence':agreement,'gate_scope':'exploratory; informed by historical outer outcomes','stacking':'omitted; small positive support, high correlation, added complexity','new_neural_training':False})
    scores, decisions = calculate(data)
    output_frame = pd.DataFrame({'deidentified_id':[pseudonyms[s] for s in data['subjects']], 'label':data['labels'], 'fold':data['folds']})
    for name,(p,t) in scores.items():
        output_frame[name+'_probability'] = p; output_frame[name+'_threshold'] = t
    output_frame.to_csv(output/'oof_predictions_private.csv', index=False)
    for f, record in data['records'].items():
        directory = output/f'fold_{f}'; directory.mkdir()
        write_json(directory/'decision_private.json', decisions[f])
        pd.DataFrame({'deidentified_id':[pseudonyms[s] for s in record['training_subjects']], 'label':data['labels'][record['train_indices']],
                      'lab_inner_oof':record['inner']['lab'], 'cp_inner_oof':record['inner']['cp']}).to_csv(directory/'inner_predictions_private.csv', index=False)
    result = metrics_with_uncertainty(data['labels'], data['folds'], scores)
    write_json(output/'metrics.json', result); figures(output,data,scores,agreement)
    write_json(output/'run_manifest.json', {'status':'ensemble_evaluation_completed','refitted_models':0,'elapsed_seconds':time.perf_counter()-start,'all_results_exploratory':True})
    checksum_guard(root/'data/processed'/EXP4); checksum_guard(root/'data/processed'/EXP5)
    return {'subjects':119,'strategies':list(scores),'output':str(output),'elapsed_seconds':time.perf_counter()-start}


def verify(root: Path, output: Path) -> dict:
    """Reproduce every decision, score and metric without changing the bundle."""
    checksum_guard(output); data = load_bases(root); scores, decisions = calculate(data)
    frame = pd.read_csv(output/'oof_predictions_private.csv', float_precision='round_trip')
    keys = pd.read_csv(output/'identity_key_private.csv', dtype=str)
    if keys.source_subject_id.tolist() != data['subjects'] or keys.deidentified_id.tolist() != frame.deidentified_id.tolist():
        raise ValueError('Private prediction/subject identity mapping differs')
    np.testing.assert_array_equal(frame.label, data['labels']); np.testing.assert_array_equal(frame.fold,data['folds'])
    for f, record in decisions.items():
        if json.loads((output/f'fold_{f}/decision_private.json').read_text()) != record:
            raise ValueError('Training-selected weight or threshold changed')
        inner=pd.read_csv(output/f'fold_{f}/inner_predictions_private.csv',float_precision='round_trip')
        scope=data['records'][f]
        pseudonyms=dict(zip(keys.source_subject_id,keys.deidentified_id))
        if inner.deidentified_id.tolist()!=[pseudonyms[s] for s in scope['training_subjects']]:
            raise ValueError('Inner prediction subject linkage differs')
        np.testing.assert_array_equal(inner.label,data['labels'][scope['train_indices']])
        for name in ('lab','cp'):np.testing.assert_array_equal(inner[name+'_inner_oof'],scope['inner'][name])
    for name,(p,t) in scores.items():
        np.testing.assert_array_equal(frame[name+'_probability'],p);np.testing.assert_array_equal(frame[name+'_threshold'],t)
    saved=json.loads((output/'metrics.json').read_text())
    if metrics_with_uncertainty(data['labels'],data['folds'],scores) != saved:
        raise ValueError('Metric/interval reproduction differs')
    checksum_guard(output)
    return {'status':'reproduced_without_refitting','models_reproduced':40,'all_weights_thresholds_predictions_metrics_intervals_verified':True,'subjects':119}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--verify', action='store_true')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        result=verify(args.source_root,args.output) if args.verify else run(args.source_root,args.output,args.preflight)
    print(json.dumps(result,indent=2))
