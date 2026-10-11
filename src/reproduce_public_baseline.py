"""Independently rerun the existing public-data LightGBM CV baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from platform import python_version

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .train_classifier import NON_FEATURE_COLUMNS, train_and_evaluate


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/processed/dataset.csv"
OUTPUT = ROOT / "data/processed/public_baseline_reproduction_2026-10-07"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reproduce(dataset_path: Path = DATASET, output_dir: Path = OUTPUT) -> Path:
    """Validate features and save a new subject-level out-of-fold CV run."""
    frame = pd.read_csv(dataset_path)
    features = [column for column in frame if column not in NON_FEATURE_COLUMNS]
    if not features or frame.empty:
        raise ValueError("Public feature table is empty")
    if frame[["country", "subject_id"]].duplicated().any():
        raise ValueError("A public subject appears more than once; row CV would leak")
    if frame["label"].isna().any() or set(frame["label"].unique()) != {0, 1}:
        raise ValueError("Expected confirmed binary public labels")
    numeric_features = frame[features].to_numpy(dtype=np.float32)
    if not np.isfinite(numeric_features).all():
        raise ValueError("Public feature table contains non-finite model features")
    labels = frame["label"].to_numpy(dtype=np.int64)
    class_counts = np.bincount(labels, minlength=2)
    if class_counts.min() < 10:
        raise ValueError("Too few subjects in one class for ten folds")

    output_dir.mkdir(parents=True, exist_ok=False)
    output_oof = output_dir / "out_of_fold_predictions.csv"
    fold_summary = train_and_evaluate(
        dataset_csv_path=dataset_path,
        output_csv_path=output_oof,
        n_splits=10,
        random_state=42,
        num_boost_round=150,
    )
    predictions = pd.read_csv(output_oof)
    if len(predictions) != len(frame):
        raise ValueError("Out-of-fold predictions do not cover every subject")

    fold_ids = np.zeros(len(frame), dtype=np.int64)
    splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for fold, (_, validation_indices) in enumerate(splitter.split(numeric_features, labels), 1):
        fold_ids[validation_indices] = fold
    if (fold_ids == 0).any():
        raise ValueError("Some subjects have no validation fold")
    pd.DataFrame({
        "country": frame["country"], "subject_id": frame["subject_id"],
        "image_id": frame["image_id"], "fold": fold_ids,
        "label": labels, "pred_prob": predictions["pred_prob"],
        "pred_label": predictions["pred_label"],
    }).to_csv(output_dir / "subject_oof_predictions.csv", index=False)

    tn, fp, fn, tp = confusion_matrix(labels, predictions["pred_label"], labels=[0, 1]).ravel()
    pooled = {
        "accuracy": float(accuracy_score(labels, predictions["pred_label"])),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "roc_auc": float(roc_auc_score(labels, predictions["pred_prob"])),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    }
    configuration = {
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "training_script_sha256": _sha256(ROOT / "src/train_classifier.py"),
        "subjects": len(frame), "features": len(features),
        "non_anemic": int(class_counts[0]), "anemic": int(class_counts[1]),
        "cv": "StratifiedKFold(n_splits=10, shuffle=True, random_state=42)",
        "subject_split_check": "one row per country+subject_id; no duplicate subject keys",
        "num_boost_round": 150, "early_stopping_rounds": 20,
        "probability_threshold": 0.5,
        "lightgbm_params": {
            "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
            "learning_rate": 0.05, "num_leaves": 31, "max_depth": -1,
            "feature_fraction": 0.8, "verbosity": -1, "seed": 42,
            "feature_fraction_seed": 42, "bagging_seed": 42,
            "data_random_seed": 42, "num_threads": 1,
        },
        "versions": {"python": python_version(), "lightgbm": lightgbm.__version__,
                     "scikit_learn": sklearn.__version__, "pandas": pd.__version__},
        "evaluation_note": (
            "Out-of-fold cross-validation on public data, "
            "not an independent test set."
        ),
    }
    (output_dir / "configuration.json").write_text(json.dumps(configuration, indent=2) + "\n")
    (output_dir / "metrics.json").write_text(json.dumps({
        "mean_of_fold_metrics": fold_summary, "pooled_out_of_fold_metrics": pooled,
    }, indent=2) + "\n")
    (output_dir / "feature_columns.json").write_text(json.dumps(features, indent=2) + "\n")
    return output_dir


if __name__ == "__main__":
    print(reproduce())
