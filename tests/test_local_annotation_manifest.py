"""Synthetic checks for explicit local annotation approval registration."""

import unittest

from src.local_annotation_manifest import validate_review_register


class ReviewRegisterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.key = ("synthetic-subject", "eyelid")
        self.source_hash = "metadata-hash"
        self.inventory = {
            self.key: {
                "original_filename": "synthetic.dng",
                "original_source_path": "data/source.dng",
                "original_source_sha256": "source-hash",
                "annotation_image_path": "data/annotation.tiff",
                "annotation_image_sha256": "image-hash",
                "anatomical_roi_path": "data/annotation.roi",
                "anatomical_roi_sha256": "roi-hash",
                "anatomical_mask_path": "data/mask.png",
                "anatomical_mask_sha256": "mask-hash",
                "metadata_source_sha256": self.source_hash,
            }
        }
        self.row = {
            "subject_id": self.key[0],
            "modality": self.key[1],
            "original_image_filename": "synthetic.dng",
            "source_image_path": "data/source.dng",
            "source_image_sha256": "source-hash",
            "annotation_image_path": "data/annotation.tiff",
            "annotation_image_sha256": "image-hash",
            "roi_path": "data/annotation.roi",
            "roi_sha256": "roi-hash",
            "mask_path": "data/mask.png",
            "mask_sha256": "mask-hash",
            "review_status": "approved",
            "reviewer": "study_owner",
            "review_method": "contact_sheet_visual_review",
            "review_scope": "anatomical_boundary_only",
            "source_review_package_path": "data/review.csv",
            "source_review_package_sha256": "package-hash",
            "metadata_source_sha256": self.source_hash,
        }

    def test_one_approved_record_is_valid_without_fixed_cohort_size(self) -> None:
        result = validate_review_register(
            [self.row], self.inventory, {self.key}, self.source_hash,
        )
        self.assertEqual(result, {self.key})

    def test_duplicate_or_unexpected_record_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_review_register(
                [self.row, self.row], self.inventory, {self.key}, self.source_hash,
            )

    def test_register_must_match_inventory_hashes(self) -> None:
        mismatched = {**self.row, "mask_sha256": "different-mask-hash"}
        with self.assertRaises(ValueError):
            validate_review_register(
                [mismatched], self.inventory, {self.key}, self.source_hash,
            )

    def test_unapproved_record_is_not_accepted_as_ground_truth(self) -> None:
        pending = {**self.row, "review_status": "pending"}
        with self.assertRaises(ValueError):
            validate_review_register(
                [pending], self.inventory, {self.key}, self.source_hash,
            )


if __name__ == "__main__":
    unittest.main()
