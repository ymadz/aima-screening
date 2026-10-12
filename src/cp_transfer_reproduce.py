"""Checksum-first independent CP saved-model verification, never retraining."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from .cp_transfer_data import SEED, audit_public, tissue
from .cp_transfer_models import CANDIDATES, INNER_SEED, OBJECTIVES, select
from .cp_transfer_neural import adult_initializer, embeddings, regression_predict, REGRESSION_SETTINGS
from .cp_transfer_prepare import sha256_tensor
from .cp_transfer_run import source_guard, data_guard, array_hash
from .cp_transfer_evaluation import historical, registry, comparison, regression_metrics, score_distribution
from .eyelid_color_models import threshold_rule
from .eyelid_color_verify import check_metrics
from .eyelid_color_data import color_features
from .eyelid_transfer_data import local_pixels
from .eyelid_transfer_neural import (SETTINGS, initial_model, prefix_cache, load_checkpoint,
                                     predict, runtime, state_hash)
from .local_eyelid_validation import sha256, load_manual_pair
from .preserve_local_experiment import tree_checksums


def checksum_guard(output: Path) -> dict:
    expected = json.loads((output / 'checksum_manifest.json').read_text())
    actual = tree_checksums(output)
    actual.pop('checksum_manifest.json', None)
    if actual != expected:
        raise ValueError('Artifact checksum coverage mismatch; refuse deserialization')
    return expected


def verified_checkpoint(directory: Path, info: dict):
    path = directory / 'best.pt'
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if sha256(path) != info['checkpoint_sha256'] or payload['signature_sha256'] != info['signature_sha256'] or payload['epoch'] != info['best_epoch']:
        raise ValueError('Checkpoint epoch, signature or file digest differs from training record')
    return load_checkpoint(path)


def check_frozen(model, initial: dict, strategy: str) -> None:
    for name, value in initial.items():
        fixed = (name.startswith('features.') and
                 (strategy == 'A' or int(name.split('.')[1]) < len(model.features) - 2))
        fixed = fixed or 'running_' in name or 'num_batches_tracked' in name
        if fixed and not torch.equal(value, model.state_dict()[name]):
            raise ValueError('Frozen backbone/BatchNorm changed')


def check_class_info(info: dict, y: np.ndarray, ids: list[str], tr: np.ndarray,
                     va: np.ndarray, initial: dict, strategy: str, cache_hash: str,
                     epochs: int | None = None) -> None:
    expected = {'train_subjects': [ids[i] for i in tr], 'validation_subjects': [ids[i] for i in va],
                'training_labels': y[tr].tolist(), 'validation_labels': y[va].tolist(),
                'strategy': strategy, 'settings': SETTINGS[strategy],
                'epochs': epochs or SETTINGS[strategy]['max_epochs'], 'seed': info['seed'],
                'cache_sha256': cache_hash, 'initializer_state_sha256': state_hash(initial)}
    if any(info[k] != value for k, value in expected.items()):
        raise ValueError('Class fold provenance/settings mismatch')
    digest = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    if digest != info['signature_sha256'] or info['pos_weight_training_only'] != float((y[tr] == 0).sum() / (y[tr] == 1).sum()):
        raise ValueError('Class weighting/signature differs')
    if len(va):
        best = max(info['history'], key=lambda row: (row['validation_ap'], -row['validation_weighted_bce']))['epoch']
        if best != info['best_epoch']:
            raise ValueError('Checkpoint not selected from inner/public validation')
    elif info['best_epoch'] != epochs or len(info['history']) != epochs:
        raise ValueError('Outer refit differs from inner epoch decision')


def check_fusion(model, x: np.ndarray) -> None:
    projection = model['projection']
    np.testing.assert_allclose(projection.embedding_scaler_.mean_, x[:, 18:].mean(0), rtol=0, atol=1e-13)
    np.testing.assert_allclose(projection.embedding_scaler_.var_, x[:, 18:].var(0), rtol=1e-12, atol=1e-13)
    transformed = projection.embedding_scaler_.transform(x[:, 18:])
    np.testing.assert_allclose(projection.pca_.mean_, transformed.mean(0), rtol=0, atol=1e-13)
    np.testing.assert_allclose(projection.pca_.components_ @ projection.pca_.components_.T, np.eye(8), rtol=0, atol=1e-13)
    np.testing.assert_allclose(model['scaler'].mean_, projection.transform(x).mean(0), rtol=0, atol=1e-13)
    np.testing.assert_allclose(model['scaler'].var_, projection.transform(x).var(0), rtol=1e-12, atol=1e-13)
    c = model['classifier']
    if projection.training_samples_ != len(x) or projection.pca_.n_components != 8 or c.C != .1 or c.class_weight != 'balanced' or c.solver != 'lbfgs':
        raise ValueError('Fusion model training support/config differs')


def fresh_inputs(root: Path, output: Path, data: dict, config: dict) -> None:
    records, audit = audit_public(Path(config['public_root']))
    saved = json.loads((output / 'audit/public_audit.json').read_text())
    for key, value in audit.items():
        if saved[key] != value:
            raise ValueError('Fresh public audit differs')
    for i, row in enumerate(records):
        with Image.open(row['path']) as image:
            pixels = np.asarray(image).copy()
        x, _ = tissue(pixels[:, :, :3], pixels[:, :, 3])
        if sha256_tensor(x) != config['tensor_sha256']['cp/' + row['image_id']] or not torch.equal(x, data['public_x'][i]):
            raise ValueError('Public fresh tissue tensor differs')
    rows = {r['subject_id']: r for r in json.loads((output / 'audit/fresh_inputs.json').read_text())}
    for i, sid in enumerate(data['local_subjects']):
        row = rows[sid]
        rgb, mask, _ = local_pixels(root, row)
        x, _ = tissue(rgb, mask)
        if sha256_tensor(x) != config['tensor_sha256']['local/' + sid] or not torch.equal(x, data['local_x'][i]):
            raise ValueError('Local fresh tissue tensor differs')
        raw, raw_mask = load_manual_pair(root / row['annotation_image_path'], root / row['mask_path'])
        np.testing.assert_array_equal(color_features(raw, raw_mask)['robust'], data['local_lab'][i].numpy())
        if (i + 1) % 20 == 0:
            print(f'Fresh anatomical input verified {i+1}/119', flush=True)


def reproduce(root: Path, output: Path, fresh: bool = False) -> dict:
    os.umask(0o077)
    runtime(SEED)
    cv2.setNumThreads(1)
    expected = checksum_guard(output)
    prepared = source_guard(output, root)
    config = json.loads((output / 'experiment_config.json').read_text())
    code_root = Path(__file__).resolve().parents[1]
    for path, digest in config['code_sha256'].items():
        if sha256(code_root / path) != digest or sha256(output / 'implementation' / path) != digest:
            raise ValueError('Use recorded implementation source for reproduction')
    data = torch.load(output / 'inputs.pt', map_location='cpu', weights_only=True)
    data_guard(data, json.loads((output / 'audit/public_audit.json').read_text()))
    imagenet = initial_model(Path(config['weights_path']), SEED)
    maps = torch.load(output / 'prefix_maps.pt', map_location='cpu', weights_only=True)
    cache_hash = sha256(output / 'prefix_maps.pt')
    if fresh:
        fresh_inputs(root, output, data, prepared)
        for name in ('public', 'local'):
            if not torch.equal(prefix_cache(imagenet, data[name + '_x']), maps[name]):
                raise ValueError('Fresh full-network prefix maps differ')
    tr, va = np.asarray(data['public_train']), np.asarray(data['public_validation'])
    public_y, hb = data['public_y'].numpy(), data['public_hb'].numpy()
    class_info = json.loads((output / 'public_classification/training.json').read_text())
    check_class_info(class_info, public_y, data['public_ids'], tr, va, imagenet.state_dict(), 'public', cache_hash)
    if class_info['seed'] != SEED:
        raise ValueError('Public classification seed differs')
    public_class = verified_checkpoint(output / 'public_classification', class_info)
    check_frozen(public_class, imagenet.state_dict(), 'public')
    reg_info = json.loads((output / 'public_regression/training.json').read_text())
    public_reg = verified_checkpoint(output / 'public_regression', reg_info)
    check_frozen(public_reg, imagenet.state_dict(), 'public')
    info = json.loads((output / 'public_regression/training.json').read_text())
    mean, scale = float(hb[tr].mean()), float(hb[tr].std())
    reg_signature = {'train_ids': [data['public_ids'][i] for i in tr], 'validation_ids': [data['public_ids'][i] for i in va],
                     'training_hb_g_dl': hb[tr].tolist(), 'validation_hb_g_dl': hb[va].tolist(),
                     'settings': REGRESSION_SETTINGS, 'seed': SEED, 'cache_sha256': cache_hash,
                     'initializer_state_sha256': state_hash(imagenet.state_dict()),
                     'target_mean_training_only': mean, 'target_scale_training_only': scale}
    if any(info[k] != v for k, v in reg_signature.items()) or hashlib.sha256(json.dumps(reg_signature, sort_keys=True).encode()).hexdigest() != info['signature_sha256']:
        raise ValueError('Regression target standardization or provenance differs')
    if info['best_epoch'] != min(info['history'], key=lambda r: (r['validation_mae_g_dl'], r['validation_rmse_g_dl']))['epoch']:
        raise ValueError('Regression checkpoint selection differs')
    models = dict(zip(OBJECTIVES, (public_class, public_reg, imagenet)))
    features = {}
    with np.load(output / 'frozen_features.npz', allow_pickle=False) as saved:
        for objective, model in models.items():
            features[objective] = np.c_[data['local_lab'].numpy(), embeddings(model, maps['local'])]
            np.testing.assert_array_equal(features[objective], saved[objective])
    public_metrics = json.loads((output / 'public_validation/metrics.json').read_text())
    local_metrics = json.loads((output / 'public_only/metrics.json').read_text())
    for domain, idx, target_y, target_hb, measured in (
        ('public_validation', va, public_y[va], hb[va], public_metrics),
        ('public_only', np.arange(119), data['local_y'].numpy(), data['local_hb'].numpy(), local_metrics)):
        domain_maps = maps['public'][idx] if domain == 'public_validation' else maps['local']
        p = predict(public_class, domain_maps)
        h = regression_predict(public_reg, domain_maps, mean, scale)
        np.testing.assert_array_equal(p, np.load(output / domain / 'classification.npy'))
        np.testing.assert_array_equal(h, np.load(output / domain / 'regression.npy'))
        check_metrics(target_y, p, .5, measured['classification_fixed_0.5'])
        if regression_metrics(target_hb, h) != measured['regression'] or score_distribution(target_y, p) != measured['score_distribution']:
            raise ValueError('Public-only/validation metrics differ')
    verified_neural, verified_classical = 2, 0
    folds, y, subjects = np.asarray(data['local_folds']), data['local_y'].numpy(), data['local_subjects']
    for fold in range(1, 6):
        tr, va = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        ids, train_y = [subjects[i] for i in tr], y[tr]
        directory = output / f'fold_{fold}'
        record = json.loads((directory / 'selection.json').read_text())
        if record['training_subjects'] != ids or record['training_labels'] != train_y.tolist() or record['validation_subjects'] != [subjects[i] for i in va] or set(ids) & set(record['validation_subjects']):
            raise ValueError('Outer subjects/leakage mismatch')
        partitions = list(StratifiedKFold(3, shuffle=True, random_state=INNER_SEED).split(ids, train_y))
        scores = {}
        for candidate in CANDIDATES:
            objective, strategy = candidate.rsplit('_', 1)
            initial = adult_initializer(models[objective].state_dict(), SEED + fold * 100)
            location = directory / candidate
            inner_p = np.full(len(tr), np.nan)
            epochs = []
            for number, (ii, jj) in enumerate(partitions, 1):
                saved_partition = record['inner_partitions'][number - 1]
                if saved_partition['train_indices'] != ii.tolist() or saved_partition['validation_indices'] != jj.tolist() or saved_partition['training_subjects'] != [ids[i] for i in ii] or saved_partition['validation_subjects'] != [ids[j] for j in jj]:
                    raise ValueError('Inner folds differ')
                if strategy == 'C':
                    model_path = location / f'inner_{number}.joblib'
                    provenance = json.loads(model_path.with_suffix('.json').read_text())
                    if provenance['train_subjects'] != [ids[i] for i in ii] or provenance['validation_subjects'] != [ids[j] for j in jj] or provenance['training_features_sha256'] != array_hash(features[objective][tr[ii]]) or provenance['training_labels'] != train_y[ii].tolist():
                        raise ValueError('Classical inner provenance differs')
                    model = joblib.load(model_path)
                    check_fusion(model, features[objective][tr[ii]])
                    inner_p[jj] = model.predict_proba(features[objective][tr[jj]])[:, 1]
                    verified_classical += 1
                else:
                    info = json.loads((location / f'inner_{number}/training.json').read_text())
                    check_class_info(info, train_y, ids, ii, jj, initial, strategy, cache_hash)
                    if info['seed'] != SEED + fold * 100 + number:
                        raise ValueError('Inner training seed differs')
                    model = verified_checkpoint(location / f'inner_{number}', info)
                    check_frozen(model, initial, strategy)
                    inner_p[jj] = predict(model, maps['local'][tr[jj]])
                    epochs.append(info['best_epoch'])
                    verified_neural += 1
            detail = record['candidates'][candidate]
            np.testing.assert_array_equal(inner_p, detail['inner_probabilities'])
            rule = threshold_rule(train_y, inner_p)
            scores[candidate] = float(average_precision_score(train_y, inner_p))
            if detail['inner_ap'] != scores[candidate] or any(detail[k] != v for k, v in rule.items()):
                raise ValueError('Inner threshold/selection differs')
            if strategy == 'C':
                model = joblib.load(location / 'outer.joblib')
                provenance = json.loads((location / 'outer.json').read_text())
                if provenance['train_subjects'] != ids or provenance['validation_subjects'] or provenance['training_features_sha256'] != array_hash(features[objective][tr]):
                    raise ValueError('Classical outer provenance differs')
                check_fusion(model, features[objective][tr])
                outer_p = model.predict_proba(features[objective][va])[:, 1]
                verified_classical += 1
            else:
                epoch = max(1, int(np.floor(np.median(epochs) + .5)))
                if detail['inner_best_epochs'] != epochs or detail['outer_refit_epochs'] != epoch or detail['adult_initializer_state_sha256'] != state_hash(initial):
                    raise ValueError('Adult head initialization/refit decision differs')
                info = json.loads((location / 'outer/training.json').read_text())
                check_class_info(info, train_y, ids, np.arange(len(tr)), np.array([], dtype=int), initial, strategy, cache_hash, epoch)
                if info['seed'] != SEED + fold * 100 + 99:
                    raise ValueError('Outer seed differs')
                model = verified_checkpoint(location / 'outer', info)
                check_frozen(model, initial, strategy)
                outer_p = predict(model, maps['local'][va])
                verified_neural += 1
            np.testing.assert_array_equal(outer_p, np.load(location / 'outer_probabilities.npy'))
        selected = {f'{o}_selected': select(scores, o) for o in OBJECTIVES}
        selected['joint_selected'] = select(scores)
        if selected != record['selected_policies']:
            raise ValueError('Primary policies differ')
        print(f'Reproduced all 36 models, decisions and predictions for fold {fold}', flush=True)
    scores = registry(output, data, historical(root, subjects, y, folds))
    saved = json.loads((output / 'metrics.json').read_text())
    frame = __import__('pandas').read_csv(output / 'oof_predictions.csv', dtype={'subject_id': str}, float_precision='round_trip')
    if frame.subject_id.tolist() != subjects:
        raise ValueError('Saved OOF subject order differs')
    for name, (p, t) in scores.items():
        np.testing.assert_array_equal(frame[name + '_probability'], p)
        np.testing.assert_array_equal(frame[name + '_threshold'], t)
        check_metrics(y, p, t, saved['models'][name])
        for fold in range(1, 6):
            idx = folds == fold
            check_metrics(y[idx], p[idx], t[idx], saved['models'][name]['per_fold'][str(fold)])
    if comparison(y, folds, scores) != saved:
        raise ValueError('Conditional uncertainty or paired metrics differ')
    if verified_neural != 122 or verified_classical != 60:
        raise ValueError('Incomplete saved-model coverage')
    checksum_guard(output)
    source_guard(output, root)
    return {'status': 'reproduced_without_refitting', 'neural_models': verified_neural, 'classical_models': verified_classical,
            'checksum_files': len(expected), 'fresh_source_inputs_and_both_view_prefix_maps': fresh,
            'inner_outer_predictions_selections_thresholds_and_all_metrics_intervals_verified': True,
            'previous_experiments_and_original_sources_unchanged': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fresh-inputs', action='store_true')
    args = parser.parse_args()
    print(json.dumps(reproduce(args.source_root, args.output, args.fresh_inputs), indent=2))
