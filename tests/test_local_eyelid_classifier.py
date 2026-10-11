"""Synthetic-only local classifier checks; no patient data or downloaded weights."""

import copy
import csv
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.local_eyelid_classifier import (
    FEATURES, binary_metrics, extract_masked_lab, fit_training_fold,
    make_subject_folds, train, uncertainty, validate_feature_rows,
)
from src.local_eyelid_readiness import recompute_labels, require_private_output
from src.local_eyelid_validation import (
    audit_folds, checked_file, load_manual_pair, perceptual_hash,
    require_duplicate_clearance, sha256, source_snapshot, strict_csv,
    validate_arrays, validate_eyelid_link,
)


class SourceValidationTests(unittest.TestCase):
    def test_snapshot_checks_freeze_before_rows_are_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            provenance = Path(directory) / "provenance.json"
            path.write_text("intentionally not parsed here")
            provenance.write_text(json.dumps({"frozen_csv_sha256": sha256(path)}))
            snapshot = source_snapshot(path, provenance)
            self.assertEqual(snapshot["sha256"], sha256(path))
            self.assertIn("+08:00", snapshot["mtime"])
            path.write_text("changed bytes")
            with self.assertRaises(ValueError):
                source_snapshot(path, provenance)

    def test_csv_rejects_blank_duplicate_trailing_and_short_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"
            for text in ("a,\n1,2\n", "a,a\n1,2\n", "a,b\n1,2,3\n", "a,b\n1\n", "a,b\n,\n", "a,b\n\n"):
                with self.subTest(text=text):
                    path.write_text(text)
                    with self.assertRaises(ValueError):
                        strict_csv(path)
            path.write_text("a,b\n1,2\n")
            self.assertEqual(strict_csv(path), [{"a": "1", "b": "2"}])

    def test_paths_require_containment_and_exact_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "synthetic.roi"
            path.write_bytes(b"synthetic ROI fixture")
            self.assertEqual(checked_file(root, path.name, sha256(path)), path.resolve())
            for value, digest in ((path.name, "wrong"), ("missing.roi", None), ("../escape.roi", None), ("", None)):
                with self.assertRaises(ValueError):
                    checked_file(root, value, digest)

    def test_outputs_cannot_escape_private_processed_storage(self) -> None:
        with self.assertRaises(ValueError):
            require_private_output(Path("results/patient-output"))

    def test_existing_labels_preserve_exact_source_hb_and_exclusions(self) -> None:
        config = {"metadata_source_sha256": "synthetic-hash", "name": "WHO_2024_unadjusted_adult_15_65",
                  "thresholds_g_per_dl": {"M": "13.0", "F_nonpregnant": "12.0"}, "age_years_inclusive": [15, 65],
                  "pregnancy_status_source": "study_owner_attestation_outside_frozen_sheet",
                  "pregnancy_attestation": "synthetic attestation fixture"}
        row = {"subject_id": "synthetic-a", "age": "40", "sex": "F", "source": "local",
               "lab_hb_g_per_l": "119.00", "lab_hb_g_per_dl": "11.900", "notes": "",
               "anemia_status": "", "raw_eyelid_file": "synthetic.dng", "raw_nail_file": ""}
        excluded = {**row, "subject_id": "synthetic-b", "lab_hb_g_per_l": "", "lab_hb_g_per_dl": ""}
        eligibility, labels = recompute_labels([row, excluded], config, "synthetic-hash")
        self.assertEqual(labels, {"synthetic-a": 1})
        self.assertEqual(eligibility[0]["primary_recorded_hb_value"], "119.00")
        self.assertEqual(eligibility[0]["lab_hb_g_per_dl"], "11.900")
        self.assertEqual(eligibility[1]["exclusion_reasons"], "missing_laboratory_hb")
        with self.assertRaises(ValueError):
            recompute_labels([row, row], config, "synthetic-hash")
        with self.assertRaises(ValueError):
            recompute_labels([row], {**config, "pregnancy_attestation": ""}, "synthetic-hash")


class LinkageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = {"subject_id": "synthetic-a", "raw_eyelid_file": "synthetic.dng"}
        self.record = {"subject_id": "synthetic-a", "modality": "eyelid", "annotation_status": "approved",
                       "reviewed_status": "approved", "original_image_filename": "synthetic.dng",
                       "source_image_path": "data/synthetic.dng", "original_source_sha256": "source-hash",
                       "annotation_image_path": "data/synthetic.tiff", "annotation_image_sha256": "image-hash",
                       "roi_path": "data/synthetic.roi", "roi_sha256": "roi-hash",
                       "mask_path": "data/synthetic.png", "mask_sha256": "mask-hash",
                       "metadata_source_sha256": "metadata-hash"}
        self.review = {**self.record, "source_image_sha256": "source-hash", "review_status": "approved",
                       "reviewer": "synthetic-reviewer", "review_method": "synthetic visual review",
                       "review_scope": "anatomical_boundary_only", "source_review_package_path": "data/review.csv",
                       "source_review_package_sha256": "review-hash"}

    def test_valid_join(self) -> None:
        validate_eyelid_link(self.record, self.review, self.source, "metadata-hash")

    def test_unapproved_stale_mismatched_and_other_modality_rejected(self) -> None:
        for key, value in (("annotation_status", "pending"), ("subject_id", "other"),
                           ("original_image_filename", "other.dng"), ("mask_sha256", "wrong"),
                           ("metadata_source_sha256", "stale"), ("modality", "nail")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_eyelid_link({**self.record, key: value}, self.review, self.source, "metadata-hash")
        with self.assertRaises(ValueError):
            validate_eyelid_link(self.record, {**self.review, "reviewer": ""}, self.source, "metadata-hash")


class MaskAndFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.full((4, 5, 3), [180, 100, 120], dtype=np.uint8)
        self.mask = np.zeros((4, 5), dtype=np.uint8)
        self.mask[1:3, 1:4] = 255

    def test_invalid_geometry_values_empty_and_full_masks_rejected(self) -> None:
        invalid = [np.zeros((2, 2), dtype=np.uint8), self.mask.astype(np.uint16),
                   np.zeros_like(self.mask), np.full_like(self.mask, 255), np.full_like(self.mask, 1)]
        for mask in invalid:
            with self.assertRaises(ValueError):
                validate_arrays(self.image, mask)

    def test_loading_preserves_pixels_and_rejects_orientation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image, mask = Path(directory) / "synthetic.tiff", Path(directory) / "synthetic.png"
            Image.fromarray(self.image).save(image)
            Image.fromarray(self.mask).save(mask)
            rgb, binary = load_manual_pair(image, mask)
            np.testing.assert_array_equal(rgb, self.image)
            np.testing.assert_array_equal(binary, self.mask)
            Image.fromarray(self.image).save(image, tiffinfo={274: 6})
            with self.assertRaises(ValueError):
                load_manual_pair(image, mask)

    def test_features_exclude_background_and_are_deterministic(self) -> None:
        before = extract_masked_lab(self.image, self.mask)
        altered = self.image.copy()
        altered[self.mask == 0] = [0, 255, 0]
        after = extract_masked_lab(altered, self.mask)
        self.assertEqual(before, after)
        self.assertEqual(tuple(before), FEATURES)
        self.assertEqual(before, extract_masked_lab(self.image, self.mask))
        for key in ("std_L", "std_a", "std_b"):
            self.assertAlmostEqual(before[key], 0, places=12)

    def test_lab_scales_rgb_order_and_glare_retention(self) -> None:
        image = self.image.copy()
        image[1, 1] = [255, 255, 255]
        result = extract_masked_lab(image, self.mask)
        selected = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[self.mask == 255].astype(float)
        self.assertAlmostEqual(result["mean_L"], selected[:, 0].mean() * 100 / 255)
        self.assertAlmostEqual(result["mean_a"], selected[:, 1].mean() - 128)
        self.assertAlmostEqual(result["std_b"], selected[:, 2].std())
        self.assertNotEqual(result, extract_masked_lab(self.image, self.mask))

    def test_full_frame_phash_reproducible(self) -> None:
        np.testing.assert_array_equal(perceptual_hash(self.image), perceptual_hash(self.image.copy()))
        self.assertEqual(perceptual_hash(self.image).shape, (64,))


class SplitAndGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = {f"synthetic-{index:03}": int(index < 10) for index in range(40)}

    def test_subject_folds_deterministic_exclusive_and_supported(self) -> None:
        first = make_subject_folds(self.labels)
        self.assertEqual(first, make_subject_folds(dict(reversed(list(self.labels.items())))))
        self.assertEqual(len(first), len({row["subject_id"] for row in first}))
        counts = audit_folds(self.labels, first)
        self.assertEqual(len(counts), 5)
        self.assertTrue(all(row["anemia"] == 2 and row["non_anemia"] == 6 for row in counts.values()))

    def test_split_leakage_missing_coverage_and_wrong_labels_rejected(self) -> None:
        folds = make_subject_folds(self.labels)
        invalid = [folds + [folds[0]], folds[:-1], [{**row, "fold": 1} for row in folds]]
        wrong = copy.deepcopy(folds); wrong[0]["derived_anemia_label"] = 1 - wrong[0]["derived_anemia_label"]
        invalid.append(wrong)
        for rows in invalid:
            with self.assertRaises(ValueError):
                audit_folds(self.labels, rows)

    def test_absent_class_and_insufficient_positive_support_rejected(self) -> None:
        for labels in ({"synthetic": 0}, {f"synthetic-{i}": int(i < 4) for i in range(20)}, {"a": 1.7, "b": 0}):
            with self.assertRaises(ValueError):
                make_subject_folds(labels)

    def test_duplicate_adjudication_requires_exact_human_evidence(self) -> None:
        candidate = {"candidate_id": "synthetic-pair", "subject_id_a": "synthetic-a", "subject_id_b": "synthetic-b",
                     "original_source_sha256_a": "raw-a", "original_source_sha256_b": "raw-b",
                     "annotation_image_sha256_a": "image-a", "annotation_image_sha256_b": "image-b"}
        review = {**candidate, "decision": "distinct_sources", "reviewer": "synthetic-reviewer",
                  "reviewed_on": "synthetic-date", "evidence_note": "synthetic fixture only"}
        require_duplicate_clearance([candidate], [review])
        for reviews in ([], [review, review], [{**review, "decision": ""}], [{**review, "decision": "same_source_or_duplicate"}],
                        [{**review, "reviewer": ""}], [{**review, "annotation_image_sha256_b": "changed"}]):
            with self.assertRaises(ValueError):
                require_duplicate_clearance([candidate], reviews)

    def test_training_gate_blocks_before_fitting_or_creating_outputs(self) -> None:
        candidate = {"candidate_id": "synthetic-pair"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist"
            with patch("src.local_eyelid_classifier._check_bundle", return_value=({}, [])), \
                    patch("src.local_eyelid_classifier.checked_file", return_value=Path(directory) / "review.csv"), \
                    patch("src.local_eyelid_classifier.strict_csv", side_effect=[[candidate], []]), \
                    patch("src.local_eyelid_classifier.fit_training_fold") as fitting:
                with self.assertRaises(ValueError):
                    train(Path(directory), Path(directory) / "review.csv", output)
                fitting.assert_not_called()
                self.assertFalse(output.exists())

    def test_feature_schema_excludes_clinical_data_and_nonfinite_values(self) -> None:
        row = {"subject_id": "synthetic", **{name: "1" for name in FEATURES}}
        self.assertEqual(validate_feature_rows([row], ["synthetic"]).shape, (1, 6))
        for rows in ([row, row], [{**row, "hb": "synthetic"}], [{**row, "mean_L": "nan"}]):
            with self.assertRaises(ValueError):
                validate_feature_rows(rows, ["synthetic"])


class FixedModelAndMetricTests(unittest.TestCase):
    def test_synthetic_subject_cv_scores_each_subject_once(self) -> None:
        rng = np.random.default_rng(12)
        subjects = [f"synthetic-{index:03}" for index in range(40)]
        labels_by_id = {sid: int(index < 10) for index, sid in enumerate(subjects)}
        labels = np.asarray(list(labels_by_id.values()))
        features = rng.normal(size=(40, 6))
        folds = make_subject_folds(labels_by_id)
        assignments = np.asarray([row["fold"] for row in folds])
        probabilities = np.full(40, np.nan)
        visits = np.zeros(40, dtype=int)
        for fold in range(1, 6):
            train, validation = assignments != fold, assignments == fold
            model, _ = fit_training_fold(features[train], labels[train])
            np.testing.assert_allclose(model.named_steps["scaler"].mean_, features[train].mean(axis=0))
            probabilities[validation] = model.predict_proba(features[validation])[:, 1]
            visits[validation] += 1
        np.testing.assert_array_equal(visits, np.ones(40, dtype=int))
        result = binary_metrics(labels, probabilities, threshold=0.5)
        self.assertEqual(result["subjects"], 40)
        self.assertEqual(result["anemia"], 10)
        self.assertTrue(0 <= result["roc_auc"] <= 1)

    def test_scaler_fits_only_training_fold_and_models_reproduce(self) -> None:
        rng = np.random.default_rng(1)
        features = rng.normal(size=(20, 6))
        labels = np.asarray([0] * 15 + [1] * 5)
        model, majority = fit_training_fold(features, labels)
        again, _ = fit_training_fold(features, labels)
        np.testing.assert_allclose(model.named_steps["scaler"].mean_, features.mean(axis=0))
        validation = features + 100
        before = model.named_steps["scaler"].mean_.copy()
        np.testing.assert_array_equal(model.predict_proba(validation), again.predict_proba(validation))
        np.testing.assert_array_equal(model.named_steps["scaler"].mean_, before)
        np.testing.assert_array_equal(majority.predict(validation), np.zeros(20))

    def test_metric_confusion_matrix_and_undefined_precision(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        result = binary_metrics(labels, np.asarray([0.1, 0.8, 0.2, 0.9]), threshold=0.5)
        self.assertEqual(result["confusion_matrix_labels_0_1"], [[1, 1], [1, 1]])
        for name in ("accuracy", "sensitivity", "specificity", "precision", "f1"):
            self.assertEqual(result[name], 0.5)
        majority = binary_metrics(labels, np.zeros(4), threshold=0.5)
        self.assertIsNone(majority["precision"])
        self.assertEqual(majority["sensitivity"], 0)
        self.assertEqual(majority["roc_auc"], 0.5)
        self.assertEqual(majority["average_precision"], 0.5)

    def test_metrics_reject_bad_probabilities_and_return_null_auc_for_one_class(self) -> None:
        for probabilities in (np.asarray([np.nan, 0.5]), np.asarray([-0.1, 0.1]), np.asarray([1.1, 0.1])):
            with self.assertRaises(ValueError):
                binary_metrics(np.asarray([0, 1]), probabilities, threshold=0.5)
        one_class = binary_metrics(np.asarray([0, 0]), np.zeros(2), threshold=0.5)
        self.assertIsNone(one_class["roc_auc"])
        self.assertIsNone(one_class["sensitivity"])

    def test_conditional_uncertainty_and_undefined_precision_interval(self) -> None:
        labels = np.asarray([0, 0, 0, 1, 1])
        result = uncertainty(labels, np.zeros(5))
        self.assertEqual(result["bootstrap_replicates"], 2000)
        self.assertIsNone(result["intervals"]["precision"])
        self.assertEqual(result["intervals"]["roc_auc"], [0.5, 0.5])
        self.assertEqual(result["intervals"]["average_precision"], [0.4, 0.4])
        for interval in result["intervals"].values():
            if interval is not None:
                self.assertTrue(0 <= interval[0] <= interval[1] <= 1)
        self.assertIn("model-refit uncertainty", result["limitations"])


if __name__ == "__main__":
    unittest.main()
