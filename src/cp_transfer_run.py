"""Isolated, resumable CP pretraining and adult nested adaptation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from .cp_transfer_data import PREPROCESSING, SEED
from .cp_transfer_models import CANDIDATES, DESIGN, INNER_SEED, OBJECTIVES, fusion_pipeline, select
from .cp_transfer_neural import adult_initializer, embeddings, fit_regression, regression_predict
from .cp_transfer_evaluation import (historical, registry, comparison, regression_metrics,
                                     score_distribution, figures, report)
from .eyelid_color_models import threshold_rule
from .eyelid_color_run import prior_unchanged
from .eyelid_transfer_neural import (initial_model, prefix_cache, fit_model, predict,
                                     runtime, state_hash)
from .local_eyelid_hybrid_evaluation import evaluate
from .local_eyelid_validation import sha256
from .preserve_local_experiment import tree_checksums


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def array_hash(x: np.ndarray) -> str:
    values = np.ascontiguousarray(x)
    return hashlib.sha256(str(values.shape).encode() + str(values.dtype).encode() + values.tobytes()).hexdigest()


def source_guard(output: Path, root: Path) -> dict:
    config = json.loads((output / 'preparation_config.json').read_text())
    if sha256(output / 'inputs.pt') != config['inputs_sha256']:
        raise ValueError('Prepared tensor cache changed')
    for filename, digest in config['source_hashes'].items():
        if sha256(Path(filename)) != digest:
            raise ValueError('Authoritative source changed since preparation')
    prior_unchanged(root, config['previous_checksums'])
    return config


def data_guard(data: dict, audit: dict) -> None:
    y, folds = data['local_y'].numpy(), np.asarray(data['local_folds'])
    if len(y) != 119 or np.bincount(y, minlength=2).tolist() != [100, 19]:
        raise ValueError('Local cohort differs')
    if len(set(data['local_subjects'])) != 119 or set(folds) != set(range(1, 6)):
        raise ValueError('Invalid local subject folds')
    public_y = data['public_y'].numpy()
    np.testing.assert_array_equal(public_y, data['public_hb'].numpy() < 11)
    tr, va = np.asarray(data['public_train']), np.asarray(data['public_validation'])
    if set(tr) & set(va) or set(tr) | set(va) != set(range(len(public_y))):
        raise ValueError('Public partition overlap/missing records')
    groups = np.asarray(data['public_groups'])
    if set(groups[tr]) & set(groups[va]):
        raise ValueError('Public duplicate/curation group leakage')
    np.testing.assert_array_equal(tr, audit['training_indices'])
    np.testing.assert_array_equal(va, audit['validation_indices'])
    for idx in (tr, va):
        if set(public_y[idx]) != {0, 1}:
            raise ValueError('Public class support absent')


def classical_fit(x: np.ndarray, y: np.ndarray, train: np.ndarray, validation: np.ndarray,
                  ids: list[str], path: Path) -> tuple:
    """Persist each train-only fitted projection with guarded fold provenance."""
    signature = {'train_subjects': [ids[i] for i in train],
                 'validation_subjects': [ids[i] for i in validation],
                 'training_labels': y[train].tolist(), 'training_features_sha256': array_hash(x[train]),
                 'design': DESIGN['local_C']}
    path.parent.mkdir(parents=True, exist_ok=True)
    info_path = path.with_suffix('.json')
    if path.exists() or info_path.exists():
        info = json.loads(info_path.read_text())
        if any(info[k] != v for k, v in signature.items()) or sha256(path) != info['model_sha256']:
            raise ValueError('Completed classical fold provenance changed')
        return joblib.load(path), info
    model = fusion_pipeline().fit(x[train], y[train])
    temporary = path.with_suffix('.partial')
    joblib.dump(model, temporary, compress=3)
    os.replace(temporary, path)
    info = {**signature, 'model_sha256': sha256(path), 'training_samples': len(train)}
    write_json(info_path, info)
    return model, info


def fit_outer(train_maps: torch.Tensor, y: np.ndarray, ids: list[str],
              train_features: dict[str, np.ndarray], validation_maps: torch.Tensor,
              validation_features: dict[str, np.ndarray], initial_states: dict,
              directory: Path, cache_hash: str, outer_fold: int) -> dict:
    """The API has no outer labels; decisions use outer-training inner OOF only."""
    y = np.asarray(y, dtype=int)
    if len(ids) != len(y) or len(set(ids)) != len(ids) or set(y) != {0, 1}:
        raise ValueError('Invalid outer-training subjects')
    if len(train_maps) != len(y) or any(len(x) != len(y) for x in train_features.values()):
        raise ValueError('Outer training tensor lengths differ')
    inner = list(StratifiedKFold(3, shuffle=True, random_state=INNER_SEED).split(ids, y))
    partitions = [{'inner_fold': i, 'train_indices': tr.tolist(), 'validation_indices': va.tolist(),
                   'training_subjects': [ids[j] for j in tr], 'validation_subjects': [ids[j] for j in va],
                   'training_class_counts': np.bincount(y[tr], minlength=2).tolist(),
                   'validation_class_counts': np.bincount(y[va], minlength=2).tolist()}
                  for i, (tr, va) in enumerate(inner, 1)]
    records, candidate_scores = {}, {}
    for candidate in CANDIDATES:
        objective, strategy = candidate.rsplit('_', 1)
        location = directory / candidate
        probabilities = np.full(len(y), np.nan)
        epochs = []
        for number, (tr, va) in enumerate(inner, 1):
            if strategy == 'C':
                model, _ = classical_fit(train_features[objective], y, tr, va, ids,
                                         location / f'inner_{number}.joblib')
                probabilities[va] = model.predict_proba(train_features[objective][va])[:, 1]
            else:
                model, info = fit_model(initial_states[objective], train_maps, y, tr, va, ids,
                                        strategy, location / f'inner_{number}',
                                        SEED + outer_fold * 100 + number, cache_hash)
                probabilities[va] = predict(model, train_maps[va])
                epochs.append(info['best_epoch'])
        if not np.isfinite(probabilities).all():
            raise ValueError('Missing inner OOF scores')
        candidate_scores[candidate] = float(average_precision_score(y, probabilities))
        detail = {'inner_probabilities': probabilities.tolist(), 'inner_ap': candidate_scores[candidate],
                  **threshold_rule(y, probabilities)}
        if strategy == 'C':
            model, _ = classical_fit(train_features[objective], y, np.arange(len(y)),
                                     np.array([], dtype=int), ids, location / 'outer.joblib')
            outer_p = model.predict_proba(validation_features[objective])[:, 1]
        else:
            refit = max(1, int(np.floor(np.median(epochs) + .5)))
            model, _ = fit_model(initial_states[objective], train_maps, y, np.arange(len(y)), None,
                                 ids, strategy, location / 'outer', SEED + outer_fold * 100 + 99,
                                 cache_hash, fixed_epochs=refit)
            outer_p = predict(model, validation_maps)
            detail.update(inner_best_epochs=epochs, outer_refit_epochs=refit,
                          adult_initializer_state_sha256=state_hash(initial_states[objective]))
        np.save(location / 'outer_probabilities.npy', outer_p)
        records[candidate] = detail
        print(json.dumps({'fold': outer_fold, 'candidate': candidate, 'completed': True,
                          'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576}), flush=True)
    selected = {f'{o}_selected': select(candidate_scores, o) for o in OBJECTIVES}
    selected['joint_selected'] = select(candidate_scores)
    return {'outer_fold': outer_fold, 'training_subjects': ids, 'training_labels': y.tolist(),
            'inner_partitions': partitions, 'candidates': records, 'selected_policies': selected}


def run(root: Path, output: Path, weights: Path) -> dict:
    os.umask(0o077)
    runtime(SEED)
    root, output = root.resolve(), output.resolve()
    if not output.is_relative_to(root / 'data/processed') or not output.name.startswith('local_eyelid_classification_exp05_cp_transfer_'):
        raise ValueError('Require the prepared private versioned E5 output')
    if subprocess.run(['git', 'check-ignore', '-q', str(output / 'private.pt')], cwd=root).returncode:
        raise ValueError('Output must be ignored')
    if (output / 'checksum_manifest.json').exists():
        from .cp_transfer_reproduce import reproduce
        return reproduce(root, output)
    prepared = source_guard(output, root)
    code_root = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=code_root, text=True):
        raise ValueError('Require clean tested implementation commit before fitting')
    files = sorted((code_root / 'src').glob('*.py'))
    config = {'experiment': 'CP-AnemiC V2 public-to-local transfer, E5', 'design': DESIGN,
              'preprocessing': PREPROCESSING, 'inputs_sha256': prepared['inputs_sha256'],
              'preparation_evidence_sha256': {str(p.relative_to(output)): sha256(p)
                  for folder in ('audit', 'domain_comparison') for p in sorted((output / folder).rglob('*')) if p.is_file()},
              'preparation_config_sha256': sha256(output / 'preparation_config.json'),
              'weights_path': str(weights.resolve()), 'weights_sha256': sha256(weights),
              'code_sha256': {str(p.relative_to(code_root)): sha256(p) for p in files},
              'methods_sha256': sha256(code_root / 'docs/experiments/eyelid-exp05-cp-transfer.md'),
              'training_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code_root, text=True).strip(),
              'source_root': str(root), 'worktree': str(code_root), 'seed': SEED,
              'cohort': {'subjects': 119, 'anemia': 19, 'non_anemia': 100, 'classifier_holdout': 'none; all119 owner-approved development'},
              'resource_policy': 'Single CPU/native thread, label-independent cached early maps, per-epoch atomic best/last checkpoints; disk preflight >=2.5 GiB'}
    config_path = output / 'experiment_config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError('Frozen experiment code/configuration changed')
    else:
        if shutil.disk_usage(output).free < 2.5 * 1024 ** 3:
            raise ValueError('Insufficient safe disk margin for all checkpoints')
        write_json(config_path, config)
        for path in files:
            destination = output / 'implementation' / path.relative_to(code_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        shutil.copy2(code_root / 'requirements.txt', output / 'implementation/requirements.txt')
        shutil.copy2(code_root / 'docs/experiments/eyelid-exp05-cp-transfer.md', output / 'implementation/frozen_methods.md')
        write_json(output / 'environment.json', {'python': platform.python_version(), 'platform': platform.platform(),
                   'native_threads': 1, 'versions': {n: importlib.metadata.version(n) for n in
                   ('numpy', 'Pillow', 'opencv-python', 'scikit-learn', 'scipy', 'joblib', 'torch', 'torchvision', 'matplotlib')}})
    started = time.monotonic()
    data = torch.load(output / 'inputs.pt', map_location='cpu', weights_only=True)
    data_guard(data, json.loads((output / 'audit/public_audit.json').read_text()))
    imagenet = initial_model(weights, SEED)
    path = output / 'prefix_maps.pt'
    if path.exists():
        metadata = json.loads((output / 'prefix_cache.json').read_text())
        if metadata['sha256'] != sha256(path) or metadata['inputs_sha256'] != prepared['inputs_sha256']:
            raise ValueError('Prefix cache changed')
        maps = torch.load(path, map_location='cpu', weights_only=True)
    else:
        print('Computing frozen ImageNet prefix maps, original and flip views', flush=True)
        maps = {name: prefix_cache(imagenet, data[f'{name}_x']) for name in ('public', 'local')}
        torch.save(maps, path)
        write_json(output / 'prefix_cache.json', {'sha256': sha256(path), 'inputs_sha256': prepared['inputs_sha256'],
                   'batch_size': 16, 'label_independent': True})
    cache_hash = sha256(path)
    public_y, public_hb = data['public_y'].numpy(), data['public_hb'].numpy()
    tr, va = np.asarray(data['public_train']), np.asarray(data['public_validation'])
    public_class, class_info = fit_model(copy.deepcopy(imagenet.state_dict()), maps['public'], public_y,
                                        tr, va, data['public_ids'], 'public', output / 'public_classification', SEED, cache_hash)
    print(f"Public classification complete, best epoch {class_info['best_epoch']}", flush=True)
    public_reg, reg_info = fit_regression(copy.deepcopy(imagenet.state_dict()), maps['public'], public_hb,
                                         tr, va, data['public_ids'], output / 'public_regression', SEED, cache_hash)
    print(f"Public Hb regression complete, best epoch {reg_info['best_epoch']}", flush=True)
    hb_mean, hb_scale = reg_info['target_mean_training_only'], reg_info['target_scale_training_only']
    for directory in ('public_validation', 'public_only'):
        (output / directory).mkdir(exist_ok=True)
    public_p = predict(public_class, maps['public'][va])
    public_h = regression_predict(public_reg, maps['public'][va], hb_mean, hb_scale)
    np.save(output / 'public_validation/classification.npy', public_p)
    np.save(output / 'public_validation/regression.npy', public_h)
    public_metrics = {'classification_fixed_0.5': evaluate(public_y[va], public_p, .5),
                      'score_distribution': score_distribution(public_y[va], public_p),
                      'regression': regression_metrics(public_hb[va], public_h),
                      'training_mean_baseline': regression_metrics(public_hb[va], np.full(len(va), hb_mean)),
                      'class_counts_0_1': np.bincount(public_y[va], minlength=2).tolist(),
                      'scope': 'Public group-held-image development; checkpoint selected here; true patient linkage unavailable'}
    write_json(output / 'public_validation/metrics.json', public_metrics)
    local_y, local_hb = data['local_y'].numpy(), data['local_hb'].numpy()
    local_p = predict(public_class, maps['local'])
    local_h = regression_predict(public_reg, maps['local'], hb_mean, hb_scale)
    np.save(output / 'public_only/classification.npy', local_p)
    np.save(output / 'public_only/regression.npy', local_h)
    unadapted = {'classification_fixed_0.5': evaluate(local_y, local_p, .5),
                 'score_distribution': score_distribution(local_y, local_p),
                 'regression': regression_metrics(local_hb, local_h),
                 'public_training_mean_baseline': regression_metrics(local_hb, np.full(len(local_y), hb_mean)),
                 'scope': 'Unadapted CP pediatric model on AIMA adult labels/Hb, no local calibration or selection'}
    write_json(output / 'public_only/metrics.json', unadapted)
    models = dict(zip(OBJECTIVES, (public_class, public_reg, imagenet)))
    features = {name: np.c_[data['local_lab'].numpy(), embeddings(model, maps['local'])]
                for name, model in models.items()}
    np.savez_compressed(output / 'frozen_features.npz', **features)
    folds, subjects = np.asarray(data['local_folds']), data['local_subjects']
    pd.DataFrame({'subject_id': subjects, 'label': local_y, 'fold': folds}).to_csv(output / 'subject_folds.csv', index=False)
    for fold in range(1, 6):
        directory = output / f'fold_{fold}'
        directory.mkdir(exist_ok=True)
        if (directory / 'selection.json').exists():
            print(f'Fold {fold} complete, guarded model records retained', flush=True)
            continue
        tr, va = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        initial = {name: adult_initializer(model.state_dict(), SEED + fold * 100) for name, model in models.items()}
        record = fit_outer(maps['local'][tr], local_y[tr], [subjects[i] for i in tr],
                           {k: x[tr] for k, x in features.items()}, maps['local'][va],
                           {k: x[va] for k, x in features.items()}, initial, directory, cache_hash, fold)
        record['validation_subjects'] = [subjects[i] for i in va]
        write_json(directory / 'selection.json', record)
    previous = historical(root, subjects, local_y, folds)
    scores = registry(output, data, previous)
    metrics = comparison(local_y, folds, scores)
    write_json(output / 'metrics.json', metrics)
    frame = pd.DataFrame({'subject_id': subjects, 'label': local_y, 'fold': folds})
    for name, (p, t) in scores.items():
        frame[name + '_probability'] = p
        frame[name + '_threshold'] = t
    frame.to_csv(output / 'oof_predictions.csv', index=False)
    figures(output, local_y, scores, metrics)
    report(output, metrics, public_metrics, unadapted)
    source_guard(output, root)
    instructions = ('# Checksum-first saved-model reproduction\n\n'
                    'Use the recorded clean implementation commit/environment. No fitting or new threshold selection.\n\n'
                    '```sh\nOMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 '
                    '/Users/madz/Documents/aima-screening/aima_env/bin/python -m src.cp_transfer_reproduce '
                    f'--source-root {root} --output {output} --fresh-inputs\n```\n\n'
                    'Verifies every file before deserialization, all182 fitted models, all saved inner/outer/public '
                    'scores, train-only transformations/heads/stopping/thresholds, metrics/intervals and prior checksums. '
                    'Fresh mode rerenders all public/local inputs and reproduces both-view frozen prefix maps. '
                    'Private verification reports belong outside this sealed directory.\n')
    (output / 'REPRODUCTION.md').write_text(instructions)
    manifest = {'status': 'training_and_evaluation_complete', 'training_commit': config['training_commit'],
                'public_models': 2, 'local_neural_models': 120, 'local_classical_models': 60,
                'subjects': 119, 'anemia': 19, 'elapsed_seconds_this_invocation': time.monotonic() - started,
                'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576,
                'disk_bytes_before_seal': sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
                'previous_experiments_unchanged': True, 'output': str(output)}
    write_json(output / 'run_manifest.json', manifest)
    for p in output.rglob('*'):
        p.chmod(0o700 if p.is_dir() else 0o600)
    write_json(output / 'checksum_manifest.json', tree_checksums(output))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source_root, args.output, args.weights), indent=2))
