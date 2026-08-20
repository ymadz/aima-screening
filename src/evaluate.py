"""Evaluate out-of-fold performance across ITA skin-tone cohorts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix


def categorize_ita(ita_deg: float) -> str:
	"""Return the dermatological ITA category for an angle in degrees."""
	if ita_deg > 55:
		return "Very Light"
	if ita_deg > 41:
		return "Light"
	if ita_deg > 28:
		return "Intermediate"
	if ita_deg > 10:
		return "Tan / Kayumanggi"
	return "Brown / Dark"


def _cohort_metrics(group: pd.DataFrame) -> dict[str, Any]:
	true_negative, false_positive, false_negative, true_positive = confusion_matrix(
		group["label"], group["pred_label"], labels=[0, 1]
	).ravel()
	sensitivity = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
	specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
	return {
		"count": int(len(group)),
		"accuracy": float(accuracy_score(group["label"], group["pred_label"])),
		"sensitivity": float(sensitivity),
		"specificity": float(specificity),
		"mae": float(np.abs(group["label"] - group["pred_prob"]).mean()),
	}


def evaluate_fairness_and_performance(
	oof_csv_path: str | Path = "results/cross_validation_predictions.csv",
	alpha: float = 0.05,
) -> dict[str, Any]:
	"""Report cohort performance and run one-way ANOVA on absolute errors."""
	frame = pd.read_csv(oof_csv_path)
	required_columns = {"ita_angle", "label", "pred_label", "pred_prob"}
	missing_columns = required_columns.difference(frame.columns)
	if missing_columns:
		raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
	if frame[list(required_columns)].isna().any().any():
		raise ValueError("Evaluation input contains missing required values")

	frame["skin_cohort"] = frame["ita_angle"].map(categorize_ita)
	frame["abs_error"] = np.abs(frame["label"] - frame["pred_prob"])
	cohort_order = [
		"Very Light",
		"Light",
		"Intermediate",
		"Tan / Kayumanggi",
		"Brown / Dark",
	]
	cohort_metrics: dict[str, dict[str, Any]] = {}
	error_groups: list[np.ndarray] = []

	print("================ SUBGROUP PERFORMANCE MATRIX ================")
	for cohort in cohort_order:
		group = frame[frame["skin_cohort"] == cohort]
		if len(group) < 2:
			continue
		metrics = _cohort_metrics(group)
		cohort_metrics[cohort] = metrics
		error_groups.append(group["abs_error"].to_numpy())
		print(
			f"Cohort: {cohort:<18} | Count: {metrics['count']:<3} | "
			f"Acc: {metrics['accuracy'] * 100:.2f}% | "
			f"Se: {metrics['sensitivity'] * 100:.2f}% | "
			f"Sp: {metrics['specificity'] * 100:.2f}% | "
			f"MAE: {metrics['mae']:.4f}"
		)
	print("=============================================================\n")

	result: dict[str, Any] = {"cohorts": cohort_metrics}
	if len(error_groups) > 1:
		f_statistic, p_value = stats.f_oneway(*error_groups)
		result.update(
			{
				"anova_f_statistic": float(f_statistic),
				"anova_p_value": float(p_value),
				"alpha": alpha,
				"reject_null": bool(p_value <= alpha),
			}
		)
		print("--- Statistical Fairness Verification (One-Way ANOVA) ---")
		print(f"One-Way ANOVA F-statistic: {f_statistic:.4f}")
		print(f"Calculated p-value:        {p_value:.4f}")
		if p_value <= alpha:
			print(f"Verdict (p <= {alpha}): reject H0; disparity detected.")
		else:
			print(f"Verdict (p > {alpha}): fail to reject H0.")
			print("This result does not prove demographic neutrality.")
	else:
		result.update({"anova_f_statistic": None, "anova_p_value": None})
		print("Insufficient demographic cohorts to execute ANOVA.")

	if len(cohort_metrics) < 3:
		print("Warning: fewer than three usable ITA cohorts; fairness conclusions are underpowered.")
	return result


if __name__ == "__main__":
	evaluate_fairness_and_performance()
