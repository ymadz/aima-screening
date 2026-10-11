"""Independent exact reproduction with matching float32 prefix batch geometry.

Preserves the original recorded training implementation and all sealed artifacts.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from .eyelid_transfer_run import previous_unchanged, sources_unchanged, verify_public_contract, score_registry, historical_predictions
from .eyelid_transfer_neural import runtime, load_checkpoint, predict, initial_model, prefix_cache
from .eyelid_transfer_data import SEED, local_pixels
from .local_eyelid_hybrid_features import validated_inputs, manual_roi_tensor
from .local_eyelid_hybrid_model import select_threshold
from .local_eyelid_hybrid_evaluation import evaluate
from .local_eyelid_validation import sha256
from .preserve_local_experiment import tree_checksums

def reproduce(output: Path, source_root: Path) -> dict[str, Any]:
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
    verify_public_contract(data, json.loads((output / 'public_audit.json').read_text()))
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
    # Use the same deterministic prefix batch geometry as the actual inference path.
    # Different float32 convolution batch sizes can differ by a few units in last place.
    base = initial_model(Path(config["weights_path"]))
    for key, value in base.state_dict().items():
        if key.startswith("features.") and int(key.split(".")[1]) < len(base.features) - 2:
            assert torch.equal(value, public_model.state_dict()[key])
    for domain in ("public", "local"):
        fresh = prefix_cache(public_model, data[f"{domain}_x"])
        np.testing.assert_array_equal(fresh.numpy(), maps[domain].numpy())
    for checkpoint in sorted(output.glob("fold_*/*/*/best.pt")):
        model = load_checkpoint(checkpoint)
        strategy = checkpoint.parts[-3]
        for key, value in public_model.state_dict().items():
            frozen = (key.startswith("features.") and (strategy == "A" or int(key.split(".")[1]) < len(model.features) - 2)) or "running_" in key or "num_batches_tracked" in key
            if frozen: assert torch.equal(value, model.state_dict()[key])
    inputs = validated_inputs(source_root, source_root / "data/processed/local_eyelid_classification_2026-10-10_v1", source_root / "data/processed/local_eyelid_readiness_2026-10-10_v2")
    for index in (0, 23, 47, 71, 95):
        rgb, mask, _ = local_pixels(source_root, inputs["rows"][subjects[index]])
        np.testing.assert_array_equal(manual_roi_tensor(rgb, mask).numpy(), data["local_x"][index].numpy())
    return {'status': 'reproduced_without_refitting', 'subjects': len(y), 'neural_checkpoints': verified_models,
            'lab_checkpoints': 20, 'checksum_files': len(expected), 'selection_and_threshold_records_verified': True,
            'historical_experiments_unchanged': True, 'all_public_local_prefix_maps_fresh_exact': True,
            'all_frozen_parameters_verified': True, 'five_local_source_tensors_fresh_exact': True}



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reproduce(args.output, args.source_root), indent=2))

if __name__ == '__main__': main()
