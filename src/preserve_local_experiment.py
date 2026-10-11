"""Verify saved local CV without refitting and preserve a private local archive."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

from .local_eyelid_classifier import validate_feature_rows
from .local_eyelid_validation import audit_folds, checked_file, require_duplicate_clearance, sha256, strict_csv


def tree_checksums(directory: Path) -> dict[str, str]:
    """Capture content hashes; do not modify the source directory."""
    return {str(path.relative_to(directory)): sha256(path) for path in sorted(directory.rglob('*')) if path.is_file()}


def verify_saved_cv(experiment: Path, bundle: Path, source_root: Path | None = None) -> dict[str, Any]:
    """Reproduce probabilities and metrics from saved fold models, never train."""
    experiment, bundle = experiment.resolve(), bundle.resolve()
    configuration = json.loads((experiment / 'experiment_config.json').read_text())
    metrics = json.loads((experiment / 'metrics.json').read_text())
    if metrics['status'] != 'completed_internal_subject_cv' or configuration['design']['research_probability_threshold'] != 0.5:
        raise ValueError('Expected the completed fixed-threshold Experiment 1')
    if sha256(experiment / 'experiment_config.json') != metrics['configuration_sha256']:
        raise ValueError('Saved configuration hash differs')
    for name, field in [('features.csv', 'features_sha256'), ('subject_folds.csv', 'folds_sha256'), ('verified_eyelid_inputs.csv', 'verified_inputs_sha256'), ('audit.json', 'readiness_audit_sha256')]:
        if sha256(bundle / name) != configuration[field]:
            raise ValueError('Prepared input hash differs')
    final = json.loads((experiment / 'final_verification.json').read_text())
    for name, digest in final['output_sha256'].items():
        checked_file(experiment, name, digest)
    review = strict_csv(experiment / 'duplicate_review_used.csv')
    if sha256(experiment / 'duplicate_review_used.csv') != metrics['duplicate_review_sha256']:
        raise ValueError('Saved human review differs')
    if source_root is not None:
        source_root = source_root.resolve()
        for value, digest in configuration['source_files'].items():
            checked_file(source_root, value, digest)
        for value, digest in configuration['code_sha256'].items():
            checked_file(source_root, value, digest)
        candidates = source_root / 'data/processed/segmentation/manual_eyelid_candidate_review_2026-10-10/candidate_pairs.csv'
        require_duplicate_clearance(strict_csv(candidates), review)
        for row in strict_csv(bundle / 'verified_eyelid_inputs.csv'):
            for path_field, hash_field in [('source_image_path', 'original_source_sha256'), ('annotation_image_path', 'annotation_image_sha256'), ('roi_path', 'roi_sha256'), ('mask_path', 'mask_sha256'), ('source_review_package_path', 'source_review_package_sha256')]:
                checked_file(source_root, row[path_field], row[hash_field])
    inputs = strict_csv(bundle / 'verified_eyelid_inputs.csv')
    labels = {row['subject_id']: int(row['derived_anemia_label']) for row in inputs}
    subjects = sorted(labels)
    if len(subjects) != len(inputs) or len(subjects) != 119 or sum(labels.values()) != 19:
        raise ValueError('Expected original unique 119-subject cohort')
    folds = strict_csv(experiment / 'subject_folds.csv')
    counts = audit_folds(labels, folds)
    fold_by_id = {row['subject_id']: int(row['fold']) for row in folds}
    features = validate_feature_rows(strict_csv(bundle / 'features.csv'), subjects)
    rows = strict_csv(experiment / 'oof_predictions.csv')
    if [row['subject_id'] for row in rows] != subjects or any(int(row['label']) != labels[row['subject_id']] or int(row['fold']) != fold_by_id[row['subject_id']] for row in rows):
        raise ValueError('Saved OOF coverage, labels or folds differ')
    y = np.array([labels[sid] for sid in subjects])
    assignment = np.array([fold_by_id[sid] for sid in subjects])
    for fold in range(1, 6):
        training, validation = assignment != fold, assignment == fold
        for name in ('lab_logistic', 'majority'):
            model = joblib.load(experiment / f'{name}_fold_{fold}.joblib')
            p = model.predict_proba(features[validation])[:, 1]
            expected = np.array([float(row[f'{name}_probability']) for row, valid in zip(rows, validation) if valid])
            np.testing.assert_allclose(p, expected, rtol=0, atol=1e-14)
            if name == 'lab_logistic':
                np.testing.assert_allclose(model.named_steps['scaler'].mean_, features[training].mean(axis=0), rtol=0, atol=1e-12)
                np.testing.assert_allclose(model.named_steps['scaler'].var_, features[training].var(axis=0), rtol=0, atol=1e-12)
    for name in ('lab_logistic', 'majority'):
        p = np.array([float(row[f'{name}_probability']) for row in rows])
        result = metrics['pooled_oof'][name]
        if confusion_matrix(y, p >= 0.5, labels=[0, 1]).tolist() != result['confusion_matrix_labels_0_1']:
            raise ValueError('Saved confusion matrix differs')
        np.testing.assert_allclose([roc_auc_score(y, p), average_precision_score(y, p)], [result['roc_auc'], result['average_precision']], rtol=0, atol=1e-14)
    return {'status': 'reproduced_without_retraining', 'subjects': 119, 'anemia': 19, 'non_anemia': 100, 'fold_models': 10, 'threshold': 0.5, 'fold_class_counts': counts, 'external_source_hashes_checked': source_root is not None, 'pooled_oof': metrics['pooled_oof']}


def copy_verified(source: Path, destination: Path) -> None:
    """Create a private, exact copy without overwriting any archive file."""
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with source.open('rb') as incoming, destination.open('xb') as outgoing:
        while block := incoming.read(1024 * 1024):
            outgoing.write(block)
    destination.chmod(0o600)
    if sha256(source) != sha256(destination):
        raise ValueError('Archived copy checksum differs')


def verify_archive(archive: Path) -> dict[str, Any]:
    """Verify every preserved file before loading trusted saved model bytes."""
    expected = json.loads((archive / 'checksum_manifest.json').read_text())
    actual = tree_checksums(archive)
    actual.pop('checksum_manifest.json', None)
    if actual != expected:
        raise ValueError('Archive file coverage or checksum differs')
    reproduced = verify_saved_cv(archive / 'experiment', archive / 'readiness')
    return {'status': 'verified', 'files': len(expected), 'reproduction': reproduced['status'], 'subjects': reproduced['subjects']}


def create_archive(experiment: Path, bundle: Path, source_root: Path, archive: Path,
                   commit: str, branch: str, tag: str) -> dict[str, Any]:
    """Archive small reproducibility artifacts; retain external raw-data links."""
    source_root, archive = source_root.resolve(), archive.resolve()
    # Avoid any source/output tree, Git directory or standard cloud-sync root.
    if archive.exists() or archive.is_relative_to(source_root) or any(part in ('CloudStorage', 'Mobile Documents', 'Dropbox', 'OneDrive') for part in archive.parts):
        raise ValueError('Archive must be a new private local location outside repository/source trees')
    for ref in (branch, tag):
        actual = subprocess.check_output(['git', 'rev-parse', f'{ref}^{{commit}}'], cwd=source_root, text=True).strip()
        if actual != commit:
            raise ValueError('Preservation ref does not identify the requested milestone')
    before = {name: tree_checksums(path) for name, path in [('experiment', experiment), ('readiness', bundle)]}
    proof = verify_saved_cv(experiment, bundle, source_root)
    archive.mkdir(parents=True, mode=0o700)
    archive.chmod(0o700)
    for name, directory in [('experiment', experiment), ('readiness', bundle)]:
        for value in before[name]:
            copy_verified(directory / value, archive / name / value)
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit], cwd=source_root, text=True).splitlines()
    for value in paths:
        if value.startswith(('src/', 'tests/', 'docs/')) or value in ('README.md', 'AGENTS.md', 'requirements.txt', '.gitignore'):
            data = subprocess.check_output(['git', 'show', f'{commit}:{value}'], cwd=source_root)
            path = archive / 'implementation' / value
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with path.open('xb') as handle:
                handle.write(data)
            path.chmod(0o600)
    metadata_dir = source_root / 'data/raw/local/metadata/frozen_2026-10-07'
    for path in sorted(metadata_dir.iterdir()):
        if path.is_file():
            copy_verified(path, archive / 'metadata_snapshot' / path.name)
    environment = {'python': platform.python_version(), 'platform': platform.platform(), 'packages': sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions() if d.metadata.get('Name')})}
    (archive / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    manifest = {'created_at': datetime.now(ZoneInfo('Asia/Manila')).isoformat(), 'experiment': 'local_eyelid_exp01_cielab_baseline', 'git_commit': commit, 'preservation_branch': branch, 'annotated_tag': tag, 'original_experiment': str(experiment.resolve()), 'original_readiness': str(bundle.resolve()), 'source_root': str(source_root), 'verification': proof, 'original_file_sha256': before, 'privacy': 'Local owner-only archive outside repository and standard cloud-sync locations; no upload. Source images, masks and ROIs are referenced by path/hash without duplicating their large files.', 'source_pixels': 'Unchanged annotation TIFF RGB derived from original DNG/HEIC; not raw sensor pixels', 'predicted_labels': 'Derived from stored OOF scores at fixed threshold 0.5; original CSV stores scores, labels and folds'}
    (archive / 'archive_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    instructions = '''# Experiment 1 private local archive

This archive preserves the verified five-fold manual-eyelid LAB experiment. The original output and readiness directories were not modified. All copied files are covered by checksum_manifest.json (the checksum file excludes itself). Raw source images/ROIs/masks remain external read-only references in readiness/verified_eyelid_inputs.csv; no large source dataset was copied.

From implementation/ with a compatible Python 3.10 environment and the recorded dependencies:

```bash
python -m src.preserve_local_experiment verify-archive --archive ..
python -m src.preserve_local_experiment verify --experiment ../experiment --bundle ../readiness
```

These commands load saved models and cached six-feature rows, check coverage/hashes and reproduce OOF scores and metrics without fitting. For full source-link verification, add --source-root with the recorded authoritative repository directory. To retrain a new reproduction, use the copied classifier with a new versioned readiness/output bundle and a complete distinct-source review; never write inside this archive or the historical output.

environment.json records installed distribution versions; experiment/experiment_config.json records original runtime versions and fixed methods. archive_manifest.json records the Git milestone and private provenance. The Git ref preserves code only; this local archive independently preserves ignored experiment artifacts. Internal CV remains exploratory, with no classifier holdout or clinical validation.
'''
    (archive / 'README.md').write_text(instructions)
    after = {name: tree_checksums(path) for name, path in [('experiment', experiment), ('readiness', bundle)]}
    if after != before:
        raise ValueError('Original experiment changed during preservation')
    # Restrict all descendants, including parents created by copying metadata.
    for path in archive.rglob('*'):
        path.chmod(0o700 if path.is_dir() else 0o600)
    checksums = tree_checksums(archive)
    (archive / 'checksum_manifest.json').write_text(json.dumps(checksums, indent=2) + '\n')
    (archive / 'checksum_manifest.json').chmod(0o600)
    result = verify_archive(archive)
    return {**result, 'archive': str(archive), 'git_commit': commit, 'originals_unchanged': after == before}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    verifier = sub.add_parser('verify')
    verifier.add_argument('--experiment', type=Path, required=True)
    verifier.add_argument('--bundle', type=Path, required=True)
    verifier.add_argument('--source-root', type=Path)
    archived = sub.add_parser('verify-archive')
    archived.add_argument('--archive', type=Path, required=True)
    writer = sub.add_parser('archive')
    for name in ('experiment', 'bundle', 'source-root', 'archive'):
        writer.add_argument(f'--{name}', type=Path, required=True)
    for name in ('commit', 'branch', 'tag'):
        writer.add_argument(f'--{name}', required=True)
    args = parser.parse_args()
    if args.command == 'verify':
        result = verify_saved_cv(args.experiment, args.bundle, args.source_root)
    elif args.command == 'verify-archive':
        result = verify_archive(args.archive)
    else:
        result = create_archive(args.experiment, args.bundle, args.source_root, args.archive, args.commit, args.branch, args.tag)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
