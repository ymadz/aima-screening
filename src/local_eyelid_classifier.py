"""Fixed exploratory classifier using approved manual inner-eyelid masks.

Prepare inputs/features/folds separately from fitting. Fitting fails closed on
unreviewed perceptual candidates or changed provenance. No final holdout is
created or evaluated; all results from this design are internal subject CV.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import cv2
import joblib
import numpy as np
from scipy.stats import norm
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .local_eyelid_readiness import CANDIDATE_DIR, ROOT, require_private_output, write_csv
from .local_eyelid_validation import (
    audit_folds, checked_file, load_manual_pair, require_duplicate_clearance,
    sha256, strict_csv, validate_arrays,
)


FEATURES = ("mean_L", "std_L", "mean_a", "std_a", "mean_b", "std_b")
SEED = 20261010
DESIGN = {
    "scope": "local_eyelid_only_internal_subject_cv",
    "population": "all_119_verified_labeled_subjects",
    "source_pixels": "current_annotation_TIFF_RGB_bytes",
    "color_interpretation": "stored_RGB_assumed_sRGB_for_uncalibrated_OpenCV_uint8_LAB",
    "mask_policy": "approved_manual_anatomical_mask_pixels_only",
    "glare_policy": "retain_all_anatomical_pixels_no_glare_exclusion_or_inpainting",
    "image_transforms": "none_before_RGB_to_LAB_no_resize_enhancement_or_profile_transform",
    "features": list(FEATURES), "standard_deviation_ddof": 0,
    "n_splits": 5, "seed": SEED, "classifier_holdout": "none_owner_selected_all_subjects_for_development",
    "majority_baseline": "DummyClassifier_most_frequent",
    "classifier": "LogisticRegression_L2_C1_lbfgs_max_iter2000_tol0.0001_no_class_weight",
    "learned_preprocessing": "StandardScaler_fit_on_training_fold_only",
    "research_probability_threshold": 0.5, "tuning": "none",
    "intervals": "Wilson_95_percent_for_proportions_and_2000_stratified_subject_bootstraps_for_ROC_AUC_AP_PR_AUC_conditional_on_OOF_predictions",
}
APPROVAL = {
    "date_local": "2026-10-10",
    "input_and_model_choice": "Owner replied 'you decide' to the explicit rendering/LAB/glare and fixed-model questions; this fixed exploratory design was selected under that delegation.",
    "evaluation_population": "Owner selected 'Use all 119 people for development' after explanation of the existing 29-person reservation.",
    "clinical_threshold": "No clinical threshold approved; 0.5 is a fixed research classification rule.",
}


def software_versions() -> dict[str, str]:
    """Record runtime/library versions; do not install or download anything."""
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "Pillow", "opencv-python", "scipy", "scikit-learn", "joblib"):
        versions[name] = importlib.metadata.version(name)
    return versions


def extract_masked_lab(image_rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Compute six statistics on manual anatomy, including glare, without a crop.

    The input policy explicitly interprets stored uint8 RGB as sRGB for OpenCV's
    quantized LAB conversion. This is rendering-dependent, uncalibrated color;
    an sRGB tag does not establish that the stored values were transformed.
    Background is never black-filled or included in the statistics.
    """
    selected = validate_arrays(image_rgb, mask)
    encoded = cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2LAB)
    pixels = encoded[selected].astype(np.float64)
    pixels[:, 0] *= 100.0 / 255.0
    pixels[:, 1:] -= 128.0
    means, deviations = pixels.mean(axis=0), pixels.std(axis=0, ddof=0)
    values = (means[0], deviations[0], means[1], deviations[1], means[2], deviations[2])
    if not np.isfinite(values).all():
        raise ValueError("LAB feature generation produced nonfinite values")
    return dict(zip(FEATURES, map(float, values)))


def make_subject_folds(labels: dict[str, int], n_splits: int = 5, seed: int = SEED) -> list[dict[str, Any]]:
    """Assign unique subjects; repeated-image aggregation needs a new decision."""
    subjects = sorted(labels)
    if set(labels.values()) != {0, 1}:
        raise ValueError("Subject labels must be verified binary values")
    values = np.asarray([labels[sid] for sid in subjects], dtype=np.int64)
    if set(values) != {0, 1} or min(np.bincount(values, minlength=2)) < n_splits:
        raise ValueError("Positive/negative subject support cannot sustain requested folds")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    records = []
    for fold, (_, validation) in enumerate(splitter.split(np.zeros(len(values)), values), 1):
        for index in validation:
            records.append({"subject_id": subjects[index], "derived_anemia_label": int(values[index]), "fold": fold})
    records.sort(key=lambda row: row["subject_id"])
    audit_folds(labels, records)
    return records


def _check_bundle(bundle: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Verify the saved cohort contract and all source bytes before reuse."""
    bundle = bundle.resolve()
    if not bundle.is_relative_to((ROOT / "data/processed").resolve()):
        raise ValueError("Expected an ignored local readiness bundle")
    report = json.loads((bundle / "audit.json").read_text())
    for path, digest in report["source_files"].items():
        checked_file(ROOT, path, digest)
    for filename in ("verified_eyelid_inputs.csv", "all_subject_eligibility.csv"):
        if sha256(bundle / filename) != report["bundle_files"][filename]:
            raise ValueError("Readiness cohort changed since its integrity audit")
    config = json.loads((bundle / "who_label_config.json").read_text())
    original_policy = ROOT / "data/processed/local_cohort_who2024_2026-10-08/who_label_config.json"
    if sha256(bundle / "who_label_config.json") != sha256(original_policy) or config["metadata_source_sha256"] != report["metadata_source"]["sha256"]:
        raise ValueError("Saved label configuration differs from audited policy")
    inputs = strict_csv(bundle / "verified_eyelid_inputs.csv")
    labels = {row["subject_id"]: int(row["derived_anemia_label"]) for row in inputs}
    if len(labels) != len(inputs) or len(labels) != report["labeled_eyelid_subjects"]:
        raise ValueError("Classifier requires one verified eyelid record per subject")
    if report["exact_source_duplicate_groups"] or report["exact_rendered_pixel_duplicate_groups"]:
        raise ValueError("Exact duplicate identity must be resolved before this experiment")
    for record in inputs:
        for field, hash_field in (
            ("source_image_path", "original_source_sha256"),
            ("annotation_image_path", "annotation_image_sha256"),
            ("roi_path", "roi_sha256"), ("mask_path", "mask_sha256"),
            ("source_review_package_path", "source_review_package_sha256"),
        ):
            checked_file(ROOT, record[field], record[hash_field])
    return report, inputs


def prepare(bundle: Path) -> dict[str, Any]:
    """Create approved fixed folds and deterministic features without fitting."""
    report, inputs = _check_bundle(bundle)
    if any((bundle / name).exists() for name in ("features.csv", "subject_folds.csv", "experiment_config.json")):
        raise FileExistsError("Prepared artifacts already exist; create a new versioned bundle")
    if len(inputs) != 119:
        raise ValueError("Approved population differs from current cohort; decision required")
    labels = {row["subject_id"]: int(row["derived_anemia_label"]) for row in inputs}
    folds = make_subject_folds(labels)
    fold_by_id = {row["subject_id"]: row["fold"] for row in folds}
    candidate_rows = strict_csv(CANDIDATE_DIR / "candidate_pairs.csv")
    crossing = sum(row["subject_id_a"] in labels and row["subject_id_b"] in labels and fold_by_id[row["subject_id_a"]] != fold_by_id[row["subject_id_b"]] for row in candidate_rows)
    feature_rows = []
    for record in sorted(inputs, key=lambda row: row["subject_id"]):
        image, mask = load_manual_pair(ROOT / record["annotation_image_path"], ROOT / record["mask_path"])
        feature_rows.append({"subject_id": record["subject_id"], **extract_masked_lab(image, mask)})
    # Recheck sources after decoding; no features are published if lineage moved.
    _check_bundle(bundle)
    write_csv(bundle / "features.csv", feature_rows, ["subject_id", *FEATURES])
    write_csv(bundle / "subject_folds.csv", folds, ["subject_id", "derived_anemia_label", "fold"])
    configuration = {
        "design": DESIGN, "approval_record": APPROVAL, "training_status": "not_run_duplicate_review_pending",
        "fold_class_counts": audit_folds(labels, folds),
        "unadjudicated_candidate_pairs_crossing_new_cv_folds": crossing,
        "software": software_versions(), "readiness_audit_sha256": sha256(bundle / "audit.json"),
        "bundle_path": str(bundle.resolve()), "metadata_source": report["metadata_source"],
        "source_files": report["source_files"],
        "verified_inputs_sha256": sha256(bundle / "verified_eyelid_inputs.csv"),
        "features_sha256": sha256(bundle / "features.csv"), "folds_sha256": sha256(bundle / "subject_folds.csv"),
        "code_sha256": {str(path): sha256(path) for path in (
            Path(__file__), ROOT / "src/local_eyelid_validation.py", ROOT / "src/local_eyelid_readiness.py",
            ROOT / "src/local_who2024_cohort.py",
        )},
        "caveats": report["warnings"] + ["no_classifier_holdout", "no_clinical_operating_threshold", "duplicate_candidates_are_not_confirmed_duplicates"],
    }
    with (bundle / "experiment_config.json").open("x") as handle:
        json.dump(configuration, handle, indent=2)
        handle.write("\n")
    return {"status": configuration["training_status"], "subjects": len(inputs), "features": len(FEATURES),
            "fold_class_counts": configuration["fold_class_counts"], "candidate_pairs_crossing_new_cv_folds": crossing}


def validate_feature_rows(rows: list[dict[str, str]], subjects: list[str]) -> np.ndarray:
    """Require exact feature schema/subject coverage; clinical fields cannot leak."""
    by_id = {row["subject_id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(subjects) or len(subjects) != len(set(subjects)):
        raise ValueError("Feature table lacks unique exact subject coverage")
    if any(set(row) != {"subject_id", *FEATURES} for row in rows):
        raise ValueError("Unexpected feature fields; clinical/demographic columns are forbidden")
    features = np.asarray([[float(by_id[sid][name]) for name in FEATURES] for sid in subjects], dtype=np.float64)
    if features.shape != (len(subjects), len(FEATURES)) or not np.isfinite(features).all():
        raise ValueError("Feature values are missing or nonfinite")
    return features


def fit_training_fold(features: np.ndarray, labels: np.ndarray) -> tuple[Pipeline, DummyClassifier]:
    """Fit the prespecified two models using only one training fold."""
    if (labels.ndim != 1 or set(labels) != {0, 1} or features.shape != (len(labels), len(FEATURES))
            or not np.isfinite(features).all()):
        raise ValueError("Training fold lacks valid features or both classes")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=2000,
                                           tol=0.0001, class_weight=None, random_state=SEED)),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(features, labels)
    majority = DummyClassifier(strategy="most_frequent").fit(features, labels)
    return model, majority


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, *, threshold: float) -> dict[str, Any]:
    """Report fixed-threshold metrics; undefined quantities remain null."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if (labels.ndim != 1 or not len(labels) or probabilities.shape != labels.shape or not np.isin(labels, [0, 1]).all()
            or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1))
            or not 0 <= threshold <= 1):
        raise ValueError("Invalid evaluation labels, probabilities, or threshold")
    prediction = probabilities >= threshold
    tn = int(np.sum((labels == 0) & ~prediction)); fp = int(np.sum((labels == 0) & prediction))
    fn = int(np.sum((labels == 1) & ~prediction)); tp = int(np.sum((labels == 1) & prediction))
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    both = set(labels) == {0, 1}
    precision, recall, _ = precision_recall_curve(labels, probabilities) if both else (None, None, None)
    return {"subjects": len(labels), "anemia": tp + fn, "non_anemia": tn + fp,
            "research_threshold": threshold, "confusion_matrix_labels_0_1": [[tn, fp], [fn, tp]],
            "accuracy": ratio(tn + tp, len(labels)), "sensitivity": ratio(tp, tp + fn),
            "specificity": ratio(tn, tn + fp), "precision": ratio(tp, tp + fp),
            "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "roc_auc": float(roc_auc_score(labels, probabilities)) if both else None,
            "average_precision": float(average_precision_score(labels, probabilities)) if both else None,
            "pr_auc_trapezoidal": float(auc(recall, precision)) if both else None}


def uncertainty(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Conditional OOF intervals; these do not include model-refit uncertainty."""
    metrics = binary_metrics(labels, probabilities, threshold=0.5)
    (tn, fp), (fn, tp) = metrics["confusion_matrix_labels_0_1"]
    z = float(norm.ppf(0.975))
    intervals: dict[str, Any] = {}
    for name, successes, total in (("accuracy", tn + tp, len(labels)), ("sensitivity", tp, tp + fn),
                                    ("specificity", tn, tn + fp), ("precision", tp, tp + fp)):
        if not total:
            intervals[name] = None
            continue
        p = successes / total; denominator = 1 + z * z / total
        center = (p + z * z / (2 * total)) / denominator
        radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
        intervals[name] = [max(0.0, float(center - radius)), min(1.0, float(center + radius))]
    strata = [np.flatnonzero(labels == value) for value in (0, 1)]
    if not all(len(indices) for indices in strata):
        raise ValueError("AUC bootstrap requires both classes")
    rng = np.random.default_rng(SEED)
    aucs = []
    for _ in range(2000):
        index = np.concatenate([rng.choice(indices, len(indices), replace=True) for indices in strata])
        y, p = labels[index], probabilities[index]
        precision, recall, _ = precision_recall_curve(y, p)
        aucs.append([roc_auc_score(y, p), average_precision_score(y, p), auc(recall, precision)])
    for column, name in enumerate(("roc_auc", "average_precision", "pr_auc_trapezoidal")):
        intervals[name] = np.quantile(np.asarray(aucs)[:, column], [0.025, 0.975]).tolist()
    return {"confidence": 0.95, "intervals": intervals, "bootstrap_replicates": 2000, "seed": SEED,
            "limitations": "Conditional on these OOF predictions; stratification fixes prevalence; overlapping CV training sets and model-refit uncertainty are not captured. No external validation."}


def train(bundle: Path, duplicate_review: Path, output: Path) -> dict[str, Any]:
    """Run fixed internal CV only after every provenance and human-review gate."""
    report, inputs = _check_bundle(bundle)
    candidates = strict_csv(CANDIDATE_DIR / "candidate_pairs.csv")
    review_path = checked_file(ROOT, str(duplicate_review))
    require_duplicate_clearance(candidates, strict_csv(review_path))
    configuration = json.loads((bundle / "experiment_config.json").read_text())
    if configuration.get("design") != DESIGN or configuration.get("approval_record") != APPROVAL:
        raise ValueError("Model/input/evaluation design differs from approved fixed plan")
    for name, field in (("audit.json", "readiness_audit_sha256"), ("features.csv", "features_sha256"),
                         ("verified_eyelid_inputs.csv", "verified_inputs_sha256"), ("subject_folds.csv", "folds_sha256")):
        if sha256(bundle / name) != configuration[field]:
            raise ValueError("Prepared experiment artifact differs from frozen configuration")
    for path, digest in configuration["code_sha256"].items():
        checked_file(ROOT, path, digest)
    subjects = sorted(row["subject_id"] for row in inputs)
    labels_by_id = {row["subject_id"]: int(row["derived_anemia_label"]) for row in inputs}
    folds = strict_csv(bundle / "subject_folds.csv")
    audit_folds(labels_by_id, folds)
    if [{**row, "fold": int(row["fold"]), "derived_anemia_label": int(row["derived_anemia_label"])} for row in folds] != make_subject_folds(labels_by_id):
        raise ValueError("Saved folds differ from prespecified subject assignment")
    features = validate_feature_rows(strict_csv(bundle / "features.csv"), subjects)
    labels = np.asarray([labels_by_id[sid] for sid in subjects])
    fold_by_id = {row["subject_id"]: int(row["fold"]) for row in folds}
    assignment = np.asarray([fold_by_id[sid] for sid in subjects])
    output = require_private_output(output)
    output.mkdir(parents=True, exist_ok=False)
    predictions = {name: np.full(len(labels), np.nan) for name in ("lab_logistic", "majority")}
    fold_results = []
    review_hash = sha256(review_path)
    for fold in range(1, 6):
        training, validation = assignment != fold, assignment == fold
        model, majority = fit_training_fold(features[training], labels[training])
        for name, estimator in (("lab_logistic", model), ("majority", majority)):
            probabilities = estimator.predict_proba(features[validation])[:, 1]
            predictions[name][validation] = probabilities
            fold_results.append({"model": name, "fold": fold,
                                 **binary_metrics(labels[validation], probabilities, threshold=0.5)})
            joblib.dump(estimator, output / f"{name}_fold_{fold}.joblib")
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("Out-of-fold predictions do not cover every subject exactly once")
    _check_bundle(bundle)
    if sha256(review_path) != review_hash:
        raise ValueError("Human duplicate adjudication changed during fitting")
    result = {"status": "completed_internal_subject_cv", "design": DESIGN,
              "software": software_versions(), "configuration_sha256": sha256(bundle / "experiment_config.json"),
              "duplicate_review_sha256": review_hash, "fold_metrics": fold_results,
              "pooled_oof": {name: {**binary_metrics(labels, values, threshold=0.5),
                                     "uncertainty": uncertainty(labels, values)} for name, values in predictions.items()},
              "limitations": configuration["caveats"], "created_at": datetime.now(ZoneInfo("Asia/Manila")).isoformat()}
    prediction_rows = [{"subject_id": sid, "fold": int(assignment[index]), "label": int(labels[index]),
                        **{f"{name}_probability": float(values[index]) for name, values in predictions.items()}}
                       for index, sid in enumerate(subjects)]
    write_csv(output / "oof_predictions.csv", prediction_rows, list(prediction_rows[0]))
    (output / "experiment_config.json").write_bytes((bundle / "experiment_config.json").read_bytes())
    (output / "readiness_audit.json").write_bytes((bundle / "audit.json").read_bytes())
    (output / "subject_folds.csv").write_bytes((bundle / "subject_folds.csv").read_bytes())
    (output / "duplicate_review_used.csv").write_bytes(review_path.read_bytes())
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preparation = subparsers.add_parser("prepare", help="Generate approved features/folds; never fit")
    preparation.add_argument("--bundle", type=Path, required=True)
    fitting = subparsers.add_parser("train", help="Fit only after complete human duplicate adjudication")
    fitting.add_argument("--bundle", type=Path, required=True)
    fitting.add_argument("--duplicate-review", type=Path, required=True)
    fitting.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = prepare(args.bundle) if args.command == "prepare" else train(args.bundle, args.duplicate_review, args.output)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, csv.Error, ConvergenceWarning):
        print(json.dumps({"status": "BLOCKED", "reason": "Input, provenance, approved design, or human duplicate-review gate failed; no completed experiment is reported"}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
