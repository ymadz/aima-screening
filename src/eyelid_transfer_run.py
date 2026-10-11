"""Independent Experiment 3 execution, nested selection and reproduction CLI."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from .eyelid_transfer_data import PREPROCESSING, SEED, prepare_inputs
from .local_eyelid_hybrid_features import validated_inputs
from .eyelid_transfer_evaluation import compare_scores
from .eyelid_transfer_neural import (DESIGN, initial_model, prefix_cache, fit_model,
                                     predict, load_checkpoint, runtime)
from .local_eyelid_classifier import fit_training_fold
from .local_eyelid_hybrid_evaluation import evaluate
from .local_eyelid_hybrid_model import select_threshold
from .local_eyelid_validation import sha256
from .preserve_local_experiment import tree_checksums


def private_destination(source_root: Path, output: Path) -> Path:
    root, output = source_root.resolve(), output.resolve()
    if not output.is_relative_to(root / 'data/processed') or not output.name.startswith('local_eyelid_classification_exp03_transfer_'):
        raise ValueError('Require a distinct private Experiment 3 output')
    if subprocess.run(['git', 'check-ignore', '-q', str(output / 'private.pt')], cwd=root).returncode:
        raise ValueError('Output must be Git-ignored')
    return output


def sources_unchanged(source_root: Path, output: Path) -> None:
    audit = json.loads((output / "public_audit.json").read_text())
    snapshots = audit["sources"] + [r[k] for r in audit["records_private"] for k in ("image", "mask")]
    if any(sha256(Path(item["path"])) != item["sha256"] for item in snapshots):
        raise ValueError("Public source changed since preparation")
    validated_inputs(source_root, source_root / "data/processed/local_eyelid_classification_2026-10-10_v1",
                     source_root / "data/processed/local_eyelid_readiness_2026-10-10_v2")


def previous_unchanged(source_root: Path, output: Path) -> None:
    prior = json.loads((output / 'previous_experiment_checksums.json').read_text())
    for name, hashes in prior.items():
        if tree_checksums(source_root / 'data/processed' / name) != hashes:
            raise ValueError('Historical experiment changed')


def historical_predictions(source_root: Path, subjects: list[str], labels: np.ndarray, folds: np.ndarray) -> dict[str, np.ndarray]:
    paths = {'exp01': 'local_eyelid_classification_2026-10-10_v1',
             'exp02': 'local_eyelid_classification_exp02_hybrid_2026-10-11_v1'}
    result = {}
    for name, folder in paths.items():
        frame = pd.read_csv(source_root / 'data/processed' / folder / 'oof_predictions.csv').set_index('subject_id')
        if len(frame) != len(subjects) or set(frame.index) != set(subjects):
            raise ValueError('Historical OOF subject mismatch')
        frame = frame.loc[subjects]
        np.testing.assert_array_equal(frame.label.to_numpy(), labels)
        np.testing.assert_array_equal(frame.fold.to_numpy(), folds)
        result[name] = frame['lab_logistic_probability' if name == 'exp01' else 'hybrid_probability'].to_numpy()
        if name == 'exp02': result['exp02_thresholds'] = frame.selected_threshold.to_numpy()
    return result


def train_local(data: dict[str, Any], maps: torch.Tensor, initial: dict[str, torch.Tensor],
                output: Path, cache_hash: str) -> None:
    """Finish folds independently; training functions receive outer-training data only."""
    y, folds, subjects = data['local_y'].numpy(), data['local_folds'], data['local_subjects']
    for fold in range(1, 6):
        directory = output / f'fold_{fold}'; directory.mkdir(exist_ok=True)
        if (directory / 'selection.json').exists():
            print(json.dumps({'fold': fold, 'status': 'already_complete'}), flush=True)
            continue
        training, validation = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        ids = [subjects[i] for i in training]; train_y = y[training]; train_maps = maps[training]
        inner = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(ids, train_y))
        partitions = [{'inner_fold': i + 1, 'training_subjects': [ids[j] for j in tr],
                       'validation_subjects': [ids[j] for j in va], 'train_indices': tr.tolist(), 'validation_indices': va.tolist(),
                       'training_class_counts': np.bincount(train_y[tr], minlength=2).tolist(),
                       'validation_class_counts': np.bincount(train_y[va], minlength=2).tolist()} for i, (tr, va) in enumerate(inner)]
        records = {'outer_fold': fold, 'training_subjects': ids, 'training_labels': train_y.tolist(),
                   'validation_subjects': [subjects[i] for i in validation],
                   'inner_partitions': partitions, 'strategies': {}}
        for strategy in ('A', 'B'):
            probabilities = np.full(len(training), np.nan); epochs = []
            for number, (tr, va) in enumerate(inner, 1):
                model, info = fit_model(initial, train_maps, train_y, tr, va, ids, strategy,
                                        directory / strategy / f'inner_{number}', SEED + fold * 100 + number,
                                        cache_hash)
                probabilities[va] = predict(model, train_maps[va]); epochs.append(info['best_epoch'])
            if not np.isfinite(probabilities).all(): raise ValueError('Incomplete inner OOF')
            refit_epochs = max(1, int(np.floor(np.median(epochs) + .5)))
            model, info = fit_model(initial, train_maps, train_y, np.arange(len(training)), None,
                                    ids, strategy, directory / strategy / 'outer', SEED + fold * 100 + 99,
                                    cache_hash, fixed_epochs=refit_epochs)
            scores = predict(model, maps[validation])
            np.save(directory / strategy / 'outer_probabilities.npy', scores)
            records['strategies'][strategy] = {'inner_probabilities': probabilities.tolist(),
                'inner_ap': float(average_precision_score(train_y, probabilities)),
                'inner_best_epochs': epochs, 'outer_refit_epochs': refit_epochs,
                **select_threshold(train_y, probabilities)}
            print(json.dumps({'outer_fold': fold, 'strategy': strategy, 'training_status': 'complete',
                              'refit_epochs': refit_epochs}), flush=True)
        # The simpler comparison uses unchanged LAB features/configuration and new inner fits.
        lab = data['local_lab']; lab_probabilities = np.full(len(training), np.nan)
        lab_dir = directory / 'lab_nested'; lab_dir.mkdir(exist_ok=True)
        for number, (tr, va) in enumerate(inner, 1):
            model_path = lab_dir / f'inner_{number}.joblib'
            model = joblib.load(model_path) if model_path.exists() else fit_training_fold(lab[training][tr], train_y[tr])[0]
            if not model_path.exists(): joblib.dump(model, model_path)
            lab_probabilities[va] = model.predict_proba(lab[training][va])[:, 1]
        lab_model = fit_training_fold(lab[training], train_y)[0]
        joblib.dump(lab_model, lab_dir / 'outer.joblib')
        np.save(lab_dir / 'outer_probabilities.npy', lab_model.predict_proba(lab[validation])[:, 1])
        records['lab_nested'] = {'inner_probabilities': lab_probabilities.tolist(), **select_threshold(train_y, lab_probabilities)}
        records['selected_strategy'] = max(('A', 'B'), key=lambda name: (records['strategies'][name]['inner_ap'], name == 'A'))
        (directory / 'selection.json').write_text(json.dumps(records, indent=2))
        print(json.dumps({'outer_fold': fold, 'complete': True, 'training_selected_strategy': records['selected_strategy']}), flush=True)


def score_registry(output: Path, data: dict[str, Any], historical: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    y, folds = data['local_y'].numpy(), data['local_folds']
    public = np.load(output / 'public_only_probabilities.npy')
    public_rule = json.loads((output / 'public_threshold.json').read_text())['threshold']
    registry = {'exp01_fixed': (historical['exp01'], np.full(len(y), .5)),
                'exp02_fixed': (historical['exp02'], np.full(len(y), .5)),
                'exp02_selected': (historical['exp02'], historical['exp02_thresholds']),
                'public_only_fixed': (public, np.full(len(y), .5)),
                'public_only_public_selected': (public, np.full(len(y), public_rule))}
    scores = {name: np.full(len(y), np.nan) for name in ('A', 'B', 'selected', 'lab_nested')}
    thresholds = {name: np.full(len(y), np.nan) for name in scores}
    for fold in range(1, 6):
        idx = folds == fold; directory = output / f'fold_{fold}'
        record = json.loads((directory / 'selection.json').read_text())
        for name in ('A', 'B'):
            scores[name][idx] = np.load(directory / name / 'outer_probabilities.npy')
            thresholds[name][idx] = record['strategies'][name]['threshold']
        chosen = record['selected_strategy']
        scores['selected'][idx], thresholds['selected'][idx] = scores[chosen][idx], thresholds[chosen][idx]
        scores['lab_nested'][idx] = np.load(directory / 'lab_nested/outer_probabilities.npy')
        thresholds['lab_nested'][idx] = record['lab_nested']['threshold']
    np.testing.assert_allclose(scores['lab_nested'], historical['exp01'], rtol=0, atol=1e-12)
    for name in ('A', 'B', 'selected'):
        registry[f'{name}_fixed'] = (scores[name], np.full(len(y), .5))
        registry[f'{name}_selected'] = (scores[name], thresholds[name])
    registry['lab_nested_selected'] = (scores['lab_nested'], thresholds['lab_nested'])
    return registry


def figures(output: Path, metrics: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['exp01_fixed', 'exp02_selected', 'lab_nested_selected', 'public_only_public_selected', 'A_selected', 'B_selected', 'selected_selected']
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names)); width = .35
    for k, offset, color in [('sensitivity', -width / 2, '#ad4148'), ('specificity', width / 2, '#426a91')]:
        vals = [metrics['models'][n][k] for n in names]
        intervals = [metrics['models'][n]['uncertainty']['intervals'][k] for n in names]
        ax.bar(x + offset, vals, width, label=k, color=color)
        ax.errorbar(x + offset, vals, yerr=np.array([[v - ci[0], ci[1] - v] for v, ci in zip(vals, intervals)]).T, fmt='none', color='black', capsize=3)
    ax.set_xticks(x, ['LAB .5', 'Hybrid selected', 'LAB nested', 'Public only', 'Transfer A', 'Transfer B', 'Inner selected'], rotation=25, ha='right')
    ax.set_ylim(0, 1.05); ax.set_ylabel('Proportion, Wilson 95% interval')
    ax.set_title('Internal AIMA evaluation: 119 subjects, 19 anemia; no external test')
    ax.legend(); fig.tight_layout()
    for suffix in ('png', 'svg'): fig.savefig(output / f'comparison.{suffix}', dpi=180)
    plt.close(fig)


def run(source_root: Path, output: Path, weights: Path) -> dict[str, Any]:
    os.umask(0o077); runtime(SEED)
    output = private_destination(source_root, output)
    if (output / 'checksum_manifest.json').exists():
        return verify(output, source_root)
    if not (output / 'readiness.json').exists() or (output / 'PRETRAINING_ABORTED.json').exists():
        raise ValueError('Require successful independent input preparation')
    previous_unchanged(source_root, output)
    sources_unchanged(source_root, output)
    started = time.time()
    config_path = output / 'experiment_config.json'
    sources = {str(p.resolve()): sha256(p) for p in Path(__file__).parent.glob('*.py')}
    config = {'design': DESIGN, 'preprocessing': PREPROCESSING, 'seed': SEED, 'source_root': str(source_root.resolve()),
              'inputs_sha256': sha256(output / 'inputs.pt'), 'code_sha256': sources, 'weights_path': str(weights.resolve()), 'imagenet_sha256': sha256(weights),
              'outer_folds': 'exact historical Experiment 1 five folds', 'inner_folds': 3,
              'local_holdout': 'none; all119 development as owner instructed', 'training_status': 'configuration_frozen_before_training'}
    if config_path.exists():
        if json.loads(config_path.read_text()) != config: raise ValueError('Frozen run configuration changed')
    else:
        config_path.write_text(json.dumps(config, indent=2))
        implementation = output / 'implementation'; implementation.mkdir()
        for p in Path(__file__).parent.glob('*.py'): shutil.copy2(p, implementation / p.name)
        environment = {'python': platform.python_version(), 'platform': platform.platform(), 'native_threads': 1,
                       'versions': {n: importlib.metadata.version(n) for n in ('torch', 'torchvision', 'numpy', 'Pillow', 'opencv-python', 'scikit-learn', 'joblib', 'scipy', 'lightgbm', 'matplotlib')}}
        (output / 'environment.json').write_text(json.dumps(environment, indent=2))
    data = torch.load(output / 'inputs.pt', weights_only=False, map_location='cpu')
    model = initial_model(weights)
    map_path = output / 'prefix_maps.pt'
    if map_path.exists(): maps = torch.load(map_path, weights_only=True, map_location='cpu')
    else:
        print('Computing fixed ImageNet prefix cache', flush=True)
        maps = {'public': prefix_cache(model, data['public_x']), 'local': prefix_cache(model, data['local_x'])}
        torch.save(maps, map_path)
    cache_hash = sha256(map_path)
    public_y = data['public_y'].numpy()
    public_model, public_info = fit_model(copy.deepcopy(model.state_dict()), maps['public'], public_y,
        data['public_train'], data['public_validation'], data['public_subjects'], 'public', output / 'public', SEED, cache_hash)
    public_val_scores = predict(public_model, maps['public'][data['public_validation']])
    rule = select_threshold(public_y[data['public_validation']], public_val_scores)
    (output / 'public_threshold.json').write_text(json.dumps({'source': 'public validation only; no local labels', **rule}, indent=2))
    (output / 'public_validation_metrics.json').write_text(json.dumps({'fixed': evaluate(public_y[data['public_validation']], public_val_scores, .5),
                'selected': evaluate(public_y[data['public_validation']], public_val_scores, rule['threshold']),
                'limitation': 'Model and threshold selected on this validation set; descriptive selection performance, not independent test'}, indent=2))
    np.save(output / 'public_only_probabilities.npy', predict(public_model, maps['local']))
    print(json.dumps({'public_pretraining': 'complete', 'best_epoch': public_info['best_epoch']}), flush=True)
    train_local(data, maps['local'], copy.deepcopy(public_model.state_dict()), output, cache_hash)
    y, folds, subjects = data['local_y'].numpy(), data['local_folds'], data['local_subjects']
    historical = historical_predictions(source_root, subjects, y, folds)
    registry = score_registry(output, data, historical)
    frame = pd.DataFrame({'subject_id': subjects, 'label': y, 'fold': folds})
    for name, (scores, thresholds) in registry.items():
        frame[f'{name}_probability'] = scores; frame[f'{name}_threshold'] = thresholds
    frame.to_csv(output / 'oof_predictions.csv', index=False)
    frame[['subject_id', 'label', 'fold']].to_csv(output / 'subject_folds.csv', index=False)
    metrics = compare_scores(y, folds, registry)
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2)); figures(output, metrics)
    previous_unchanged(source_root, output)
    result = {'status': 'training_and_evaluation_complete', 'subjects': len(y), 'anemia': int(y.sum()),
              'elapsed_seconds_this_invocation': time.time() - started, 'training_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip(),
              'public_checkpoint': 'public/best.pt', 'outer_neural_checkpoints': 10, 'inner_neural_checkpoints': 30, 'lab_models': 20,
              'previous_experiments_unchanged': True, 'output': str(output.resolve())}
    (output / 'run_manifest.json').write_text(json.dumps(result, indent=2))
    checksum = tree_checksums(output)
    (output / 'checksum_manifest.json').write_text(json.dumps(checksum, indent=2))
    # No additional files are written inside the checksum-covered completed run.
    return {**result, 'verification': verify(output, source_root)}


def verify(output: Path, source_root: Path) -> dict[str, Any]:
    """Verify all files before deserialization, reproduce scores and selections."""
    runtime(SEED)
    expected = json.loads((output / 'checksum_manifest.json').read_text())
    actual = tree_checksums(output); actual.pop('checksum_manifest.json')
    if actual != expected: raise ValueError('Experiment checksum coverage mismatch')
    previous_unchanged(source_root, output)
    sources_unchanged(source_root, output)
    config = json.loads((output / 'experiment_config.json').read_text())
    for path, digest in config['code_sha256'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('Use the recorded implementation commit for reproduction')
    data = torch.load(output / 'inputs.pt', weights_only=False, map_location='cpu')
    maps = torch.load(output / 'prefix_maps.pt', weights_only=True, map_location='cpu')
    y, folds, subjects = data['local_y'].numpy(), data['local_folds'], data['local_subjects']
    public_model = load_checkpoint(output / 'public/best.pt')
    np.testing.assert_array_equal(predict(public_model, maps['local']), np.load(output / 'public_only_probabilities.npy'))
    public_val = predict(public_model, maps['public'][data['public_validation']])
    public_rule = select_threshold(data['public_y'].numpy()[data['public_validation']], public_val)
    assert public_rule['threshold'] == json.loads((output / 'public_threshold.json').read_text())['threshold']
    verified_models = 1
    for fold in range(1, 6):
        tr, va = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        record = json.loads((output / f'fold_{fold}/selection.json').read_text())
        ids = [subjects[i] for i in tr]; train_y = y[tr]
        assert record['training_subjects'] == ids and not set(ids) & set(record['validation_subjects'])
        for strategy in ('A', 'B'):
            inner_scores = np.full(len(tr), np.nan)
            epochs = []
            for partition in record['inner_partitions']:
                number = partition['inner_fold']; ii = np.array(partition['validation_indices']); tt = np.array(partition['train_indices'])
                assert not set(ii) & set(tt) and set(ii) | set(tt) == set(range(len(tr)))
                directory = output / f'fold_{fold}/{strategy}/inner_{number}'
                info = json.loads((directory / 'training.json').read_text())
                assert info['train_subjects'] == [ids[i] for i in tt] and info['validation_subjects'] == [ids[i] for i in ii]
                assert info['pos_weight_training_only'] == np.sum(train_y[tt] == 0) / np.sum(train_y[tt] == 1)
                model = load_checkpoint(directory / 'best.pt'); inner_scores[ii] = predict(model, maps['local'][tr[ii]])
                epochs.append(info['best_epoch']); verified_models += 1
            np.testing.assert_array_equal(inner_scores, record['strategies'][strategy]['inner_probabilities'])
            rule = select_threshold(train_y, inner_scores)
            assert rule['threshold'] == record['strategies'][strategy]['threshold']
            assert float(average_precision_score(train_y, inner_scores)) == record['strategies'][strategy]['inner_ap']
            assert max(1, int(np.floor(np.median(epochs) + .5))) == record['strategies'][strategy]['outer_refit_epochs']
            directory = output / f'fold_{fold}/{strategy}/outer'
            info = json.loads((directory / 'training.json').read_text())
            assert info['train_subjects'] == ids and info['validation_subjects'] == []
            model = load_checkpoint(directory / 'best.pt')
            for key, value in public_model.state_dict().items():
                frozen = (key.startswith("features.") and (strategy == "A" or int(key.split(".")[1]) < len(model.features) - 2)) or "running_" in key or "num_batches_tracked" in key
                if frozen:
                    assert torch.equal(value, model.state_dict()[key])
            with torch.no_grad():
                direct = float(model(data["local_x"][va[:1]]).sigmoid()[0])
            np.testing.assert_allclose(direct, predict(model, maps["local"][va[:1]])[0], rtol=0, atol=1e-7)
            np.testing.assert_array_equal(predict(model, maps['local'][va]), np.load(output / f'fold_{fold}/{strategy}/outer_probabilities.npy'))
            verified_models += 1
        selected = max(('A', 'B'), key=lambda n: (record['strategies'][n]['inner_ap'], n == 'A'))
        assert selected == record['selected_strategy']
        lab_scores = np.full(len(tr), np.nan)
        for partition in record['inner_partitions']:
            ii = np.array(partition['validation_indices']); tt = np.array(partition['train_indices'])
            model = joblib.load(output / f'fold_{fold}/lab_nested/inner_{partition["inner_fold"]}.joblib')
            np.testing.assert_allclose(model['scaler'].mean_, data['local_lab'][tr[tt]].mean(axis=0), rtol=0, atol=1e-14)
            lab_scores[ii] = model.predict_proba(data['local_lab'][tr[ii]])[:, 1]
        np.testing.assert_array_equal(lab_scores, record['lab_nested']['inner_probabilities'])
        assert select_threshold(train_y, lab_scores)['threshold'] == record['lab_nested']['threshold']
        model = joblib.load(output / f'fold_{fold}/lab_nested/outer.joblib')
        np.testing.assert_array_equal(model.predict_proba(data['local_lab'][va])[:, 1], np.load(output / f'fold_{fold}/lab_nested/outer_probabilities.npy'))
    registry = score_registry(output, data, historical_predictions(source_root, subjects, y, folds))
    saved = json.loads((output / 'metrics.json').read_text())
    for name, (probabilities, thresholds) in registry.items():
        measured = evaluate(y, probabilities, thresholds)
        for k, value in measured.items():
            if k == 'research_threshold': continue
            assert saved['models'][name][k] == value
    # Fresh full-network calculation proves cached unflipped prefix maps are correct.
    with torch.no_grad():
        for index in (0, 23, 47, 71, 95):
            direct = float(public_model(data['local_x'][index:index + 1]).sigmoid()[0])
            np.testing.assert_allclose(direct, predict(public_model, maps['local'][index:index + 1])[0], rtol=0, atol=1e-7)
    return {'status': 'reproduced_without_refitting', 'subjects': len(y), 'neural_checkpoints': verified_models,
            'lab_checkpoints': 20, 'checksum_files': len(expected), 'selection_and_threshold_records_verified': True,
            'historical_experiments_unchanged': True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'verify'])
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--public-root', type=Path)
    parser.add_argument('--weights', type=Path)
    args = parser.parse_args(); os.umask(0o077)
    output = private_destination(args.source_root, args.output)
    if args.command == 'prepare':
        output.mkdir(mode=0o700, exist_ok=False)
        prior = {name: tree_checksums(args.source_root / 'data/processed' / name) for name in ['local_eyelid_classification_2026-10-10_v1', 'local_eyelid_readiness_2026-10-10_v2', 'local_eyelid_classification_exp02_hybrid_2026-10-11_v1']}
        (output / 'previous_experiment_checksums.json').write_text(json.dumps(prior, indent=2))
        result = prepare_inputs(args.source_root, args.public_root, output)
        (output / 'readiness.json').write_text(json.dumps(result, indent=2))
    elif args.command == 'run': result = run(args.source_root, output, args.weights)
    else: result = verify(output, args.source_root)
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
