"""Train and evaluate the fused feature table with stratified LightGBM CV."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold


NON_FEATURE_COLUMNS = {
	"image_id",
	"subject_id",
	"country",
	"gender",
	"age",
	"hgb",
	"label",
}


def train_and_evaluate(
	dataset_csv_path: str | Path = "data/processed/dataset.csv",
	output_csv_path: str | Path = "results/cross_validation_predictions.csv",
	n_splits: int = 10,
	random_state: int = 42,
	num_boost_round: int = 150,
) -> dict[str, float]:
	"""Run stratified LightGBM CV and save one out-of-fold result per subject."""
	dataset_path = Path(dataset_csv_path)
	output_path = Path(output_csv_path)
	frame = pd.read_csv(dataset_path)
	if frame.empty:
		raise ValueError(f"Dataset is empty: {dataset_path}")
	if frame["label"].isna().any():
		raise ValueError("Dataset contains missing labels")

	feature_columns = [
		column for column in frame.columns if column not in NON_FEATURE_COLUMNS
	]
	if not feature_columns:
		raise ValueError("No model features found in the dataset")

	features = frame[feature_columns].to_numpy(dtype=np.float32)
	labels = frame["label"].to_numpy(dtype=np.int64)
	class_counts = np.bincount(labels, minlength=2)
	if class_counts.min() < n_splits:
		raise ValueError("Each class needs at least n_splits samples for CV")

	print(
		f"Loaded dataset: {features.shape[0]} samples, "
		f"{features.shape[1]} input features."
	)
	print(
		f"Class distribution: {class_counts[0]} Non-Anemic (0), "
		f"{class_counts[1]} Anemic (1)"
	)

	params = {
		"objective": "binary",
		"metric": "binary_logloss",
		"boosting_type": "gbdt",
		"learning_rate": 0.05,
		"num_leaves": 31,
		"max_depth": -1,
		"feature_fraction": 0.8,
		"verbosity": -1,
		"seed": random_state,
		"feature_fraction_seed": random_state,
		"bagging_seed": random_state,
		"data_random_seed": random_state,
		"num_threads": 1,
	}

	splitter = StratifiedKFold(
		n_splits=n_splits, shuffle=True, random_state=random_state
	)
	oof_probabilities = np.zeros(len(labels), dtype=np.float64)
	oof_predictions = np.zeros(len(labels), dtype=np.int64)
	fold_metrics: list[dict[str, float]] = []

	print(f"\n--- Starting {n_splits}-Fold Stratified Cross-Validation ---")
	for fold, (train_indices, validation_indices) in enumerate(
		splitter.split(features, labels), 1
	):
		train_data = lgb.Dataset(features[train_indices], label=labels[train_indices])
		validation_data = lgb.Dataset(
			features[validation_indices],
			label=labels[validation_indices],
			reference=train_data,
		)
		model = lgb.train(
			params,
			train_data,
			num_boost_round=num_boost_round,
			valid_sets=[validation_data],
			callbacks=[lgb.early_stopping(20, verbose=False)],
		)

		probabilities = model.predict(
			features[validation_indices], num_iteration=model.best_iteration
		)
		predictions = (probabilities >= 0.5).astype(np.int64)
		oof_probabilities[validation_indices] = probabilities
		oof_predictions[validation_indices] = predictions

		true_negative, false_positive, false_negative, true_positive = confusion_matrix(
			labels[validation_indices], predictions, labels=[0, 1]
		).ravel()
		sensitivity = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
		specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
		accuracy = accuracy_score(labels[validation_indices], predictions)
		auc = roc_auc_score(labels[validation_indices], probabilities)
		metrics = {
			"accuracy": float(accuracy),
			"sensitivity": float(sensitivity),
			"specificity": float(specificity),
			"auc": float(auc),
		}
		fold_metrics.append(metrics)
		print(
			f"Fold {fold:02d} | Acc: {accuracy:.4f} | Se: {sensitivity:.4f} | "
			f"Sp: {specificity:.4f} | AUC: {auc:.4f}"
		)

	frame["pred_prob"] = oof_probabilities
	frame["pred_label"] = oof_predictions
	output_path.parent.mkdir(parents=True, exist_ok=True)
	frame.to_csv(output_path, index=False)

	summary = {
		f"mean_{metric}": float(np.mean([fold[metric] for fold in fold_metrics]))
		for metric in ("accuracy", "sensitivity", "specificity", "auc")
	}
	for metric in ("accuracy", "sensitivity", "specificity", "auc"):
		summary[f"std_{metric}"] = float(
			np.std([fold[metric] for fold in fold_metrics])
		)

	print("\n================ FINAL EVALUATION MATRIX ================")
	print(f"Mean Accuracy:    {summary['mean_accuracy'] * 100:.2f}%")
	print(f"Mean Sensitivity: {summary['mean_sensitivity'] * 100:.2f}%")
	print(f"Mean Specificity: {summary['mean_specificity'] * 100:.2f}%")
	print(f"Mean ROC-AUC:     {summary['mean_auc']:.4f}")
	print("=========================================================")
	print(f"Cross-validation output saved to: {output_path}")
	return summary


if __name__ == "__main__":
	train_and_evaluate()
