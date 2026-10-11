"""Register human-reviewed local ROI/mask batches without editing source images.

The review package is an evidence snapshot. This command validates its linkage
to frozen metadata and the registered TIFF inventory, checks that its masks
match the canonical ROI-rasterized masks, then updates only the annotation
inventory and the explicit anatomical review register. It never edits the
generated annotation manifest, queue, or audit report.
"""

from __future__ import annotations

import argparse
import csv
import collections
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from .local_annotation_manifest import (
        ANNOTATIONS,
        FROZEN,
        ROOT,
        read_csv,
        repo_path,
        sha256,
        validate_binary_mask,
        verify_metadata,
    )
except ImportError:  # Running as ``python src/register_local_annotation_reviews.py``.
    from local_annotation_manifest import (
        ANNOTATIONS,
        FROZEN,
        ROOT,
        read_csv,
        repo_path,
        sha256,
        validate_binary_mask,
        verify_metadata,
    )


INVENTORY = ANNOTATIONS / "manifests/image_ready_inventory.csv"
MANIFEST = ANNOTATIONS / "manifests/annotation_manifest.csv"
REVIEW_REGISTER = ANNOTATIONS / "manifests/anatomical_review_register.csv"
LEGACY_REVIEW_SOURCE = ROOT / "data/processed/local_cohort_who2024_2026-10-08/reviewed_anatomical_masks.csv"
REVIEW_FIELDS = [
    "subject_id", "modality", "original_image_filename", "source_image_path",
    "source_image_sha256", "annotation_image_path", "annotation_image_sha256",
    "roi_path", "roi_sha256", "mask_path", "mask_sha256", "review_status",
    "reviewer", "reviewed_on", "review_method", "review_scope",
    "source_review_package_path", "source_review_package_sha256",
    "metadata_source_path", "metadata_source_sha256",
]
PACKAGE_FIELDS = {
    "subject_id", "modality", "original_filename", "source_image_path",
    "source_image_sha256", "annotation_image_path", "annotation_image_sha256",
    "roi_path", "roi_sha256",
}


def read_review_package(path: Path) -> list[dict[str, str]]:
    """Read a review snapshot, tolerating only identical duplicate columns.

    The current review CSV contains a duplicate glare-preview header. It is
    outside the required linkage fields; preserve the source file and accept
    it only when both copies agree on every row.
    """
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("Review package is empty") from exc
        if not header or any(not name.strip() for name in header):
            raise ValueError("Review package has a blank header field")
        positions: dict[str, list[int]] = collections.defaultdict(list)
        for index, name in enumerate(header):
            positions[name].append(index)
        if any(len(indexes) > 1 and name in PACKAGE_FIELDS
               for name, indexes in positions.items()):
            raise ValueError("A required review-package field has a duplicate header")
        output: list[dict[str, str]] = []
        for values in reader:
            if len(values) != len(header) or not any(value.strip() for value in values):
                raise ValueError("Review package has a malformed or blank data row")
            record: dict[str, str] = {}
            for name, indexes in positions.items():
                copies = {values[index] for index in indexes}
                if len(copies) != 1:
                    raise ValueError("Duplicate review-package columns contain conflicting values")
                record[name] = values[indexes[0]]
            output.append(record)
        return output


def expected_roi_path(modality: str, filename: str) -> str:
    stem = Path(filename).stem
    target = "palpebral_conjunctiva" if modality == "eyelid" else "nail_plate"
    return (Path("data/raw/local/annotations/completed") / modality / "rois" /
            f"{stem}_{target}.roi").as_posix()


def expected_mask_path(modality: str, filename: str) -> str:
    stem = Path(filename).stem
    target = "palpebral_mask" if modality == "eyelid" else "nail_plate_mask"
    return (Path("data/raw/local/annotations/generated_masks/anatomical") / modality /
            f"{stem}_{target}.png").as_posix()


def binary_mask_bytes(path: Path, image_path: Path) -> tuple[tuple[int, int], bytes]:
    validate_binary_mask(image_path, path)
    with Image.open(image_path) as image, Image.open(path) as mask:
        if mask.histogram()[255] == image.width * image.height:
            raise ValueError("An anatomical mask covers the entire annotation image")
        return mask.size, mask.tobytes()


def make_review_row(
    *,
    subject_id: str,
    modality: str,
    original_filename: str,
    inventory_row: dict[str, str],
    metadata_source_hash: str,
    review_status: str,
    reviewer: str,
    reviewed_on: str,
    review_method: str,
    source_review_package_path: str,
    source_review_package_sha256: str,
) -> dict[str, str]:
    return {
        "subject_id": subject_id,
        "modality": modality,
        "original_image_filename": original_filename,
        "source_image_path": inventory_row["original_source_path"],
        "source_image_sha256": inventory_row["original_source_sha256"],
        "annotation_image_path": inventory_row["annotation_image_path"],
        "annotation_image_sha256": inventory_row["annotation_image_sha256"],
        "roi_path": inventory_row["anatomical_roi_path"],
        "roi_sha256": inventory_row["anatomical_roi_sha256"],
        "mask_path": inventory_row["anatomical_mask_path"],
        "mask_sha256": inventory_row["anatomical_mask_sha256"],
        "review_status": review_status,
        "reviewer": reviewer,
        "reviewed_on": reviewed_on,
        "review_method": review_method,
        "review_scope": "anatomical_boundary_only",
        "source_review_package_path": source_review_package_path,
        "source_review_package_sha256": source_review_package_sha256,
        "metadata_source_path": str(FROZEN),
        "metadata_source_sha256": metadata_source_hash,
    }


def write_staged_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> Path:
    temporary = path.with_name(path.name + ".registration.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def register(
    review_package: Path,
    *,
    modality: str,
    reviewer: str,
    reviewed_on: str,
    review_method: str,
    confirm_all_rows_reviewed: bool,
    require_complete_modality: bool,
    dry_run: bool,
) -> dict[str, Any]:
    if not confirm_all_rows_reviewed:
        raise ValueError("Explicit confirmation is required before recording human approval")
    if modality not in {"eyelid", "nail"}:
        raise ValueError("Modality must be eyelid or nail")
    if not reviewer.strip() or not review_method.strip():
        raise ValueError("Reviewer label and review method are required")
    reviewed_date = date.fromisoformat(reviewed_on).isoformat()
    package_path = review_package.resolve()
    if not package_path.is_relative_to(ROOT.resolve()) or not package_path.is_file():
        raise ValueError("Review package must be an existing file inside the AIMA repository")

    _, metadata, metadata_hash, _ = verify_metadata()
    inventory_rows = read_csv(INVENTORY)
    inventory_by_key = {(r["subject_id"], r["modality"]): r for r in inventory_rows}
    if len(inventory_by_key) != len(inventory_rows):
        raise ValueError("Image inventory contains duplicate subject/modality keys")
    package_rows = read_review_package(package_path)
    if not package_rows or not PACKAGE_FIELDS.issubset(package_rows[0]):
        raise ValueError("Review package does not have the required linkage/hash columns")

    expected_keys = {
        (subject_id, modality)
        for subject_id, row in metadata.items()
        if row[f"raw_{modality}_file"].strip()
    }
    package_by_key: dict[tuple[str, str], dict[str, str]] = {}
    package_hash = sha256(package_path)
    package_rel = package_path.relative_to(ROOT.resolve()).as_posix()
    current_inventory = {key: dict(row) for key, row in inventory_by_key.items()}
    new_rows: dict[tuple[str, str], dict[str, str]] = {}

    for package_row in package_rows:
        subject_id = package_row["subject_id"].strip()
        key = (subject_id, package_row["modality"].strip())
        if key in package_by_key or key[1] != modality or key not in expected_keys:
            raise ValueError("Review package has a duplicate, unexpected, or wrong-modality key")
        package_by_key[key] = package_row
        inventory_row = inventory_by_key.get(key)
        if inventory_row is None:
            raise ValueError("Review package key is missing from the registered TIFF inventory")
        if inventory_row["metadata_source_sha256"] != metadata_hash:
            raise ValueError("Image inventory uses stale frozen-metadata provenance")
        original_filename = metadata[subject_id][f"raw_{modality}_file"].strip()
        if package_row["original_filename"] != original_filename:
            raise ValueError("Review package filename does not match frozen metadata")
        if package_row["source_image_path"] != inventory_row["original_source_path"]:
            raise ValueError("Review package original-image path differs from the inventory")
        if package_row["source_image_sha256"] != inventory_row["original_source_sha256"]:
            raise ValueError("Review package original-image hash differs from the inventory")
        if package_row["annotation_image_path"] != inventory_row["annotation_image_path"]:
            raise ValueError("Review package annotation TIFF path differs from the inventory")
        if package_row["annotation_image_sha256"] != inventory_row["annotation_image_sha256"]:
            raise ValueError("Review package annotation TIFF hash differs from the inventory")

        source_path = repo_path(package_row["source_image_path"])
        image_path = repo_path(package_row["annotation_image_path"])
        roi_rel = package_row["roi_path"]
        mask_rel = expected_mask_path(modality, original_filename)
        if roi_rel != expected_roi_path(modality, original_filename):
            raise ValueError("ROI filename or modality folder does not follow the annotation contract")
        roi_path = repo_path(roi_rel)
        mask_path = repo_path(mask_rel)
        if sha256(source_path) != package_row["source_image_sha256"]:
            raise ValueError("Original image changed after the review package was prepared")
        if sha256(image_path) != package_row["annotation_image_sha256"]:
            raise ValueError("Annotation TIFF changed after the review package was prepared")
        if sha256(roi_path) != package_row["roi_sha256"]:
            raise ValueError("Manual Fiji ROI changed after the review package was prepared")

        candidate_mask_rel = package_row.get("mask_path", "").strip() or package_row.get(
            "candidate_mask_path", ""
        ).strip()
        candidate_mask_sha = package_row.get("mask_sha256", "").strip() or package_row.get(
            "candidate_mask_sha256", ""
        ).strip()
        if not candidate_mask_rel or not candidate_mask_sha:
            raise ValueError("Review package lacks a hash-verified mask used in the reviewed preview")
        candidate_mask_path = repo_path(candidate_mask_rel)
        if sha256(candidate_mask_path) != candidate_mask_sha:
            raise ValueError("Reviewed preview mask changed after the review package was prepared")
        if not mask_path.is_file():
            raise FileNotFoundError("Canonical ROI-rasterized mask has not been created")
        candidate_size, candidate_bytes = binary_mask_bytes(candidate_mask_path, image_path)
        canonical_size, canonical_bytes = binary_mask_bytes(mask_path, image_path)
        if candidate_size != canonical_size or candidate_bytes != canonical_bytes:
            raise ValueError("Canonical rasterized mask differs from the mask shown for review")

        roi_hash = sha256(roi_path)
        mask_hash = sha256(mask_path)
        if inventory_row["anatomical_roi_path"] and inventory_row["anatomical_roi_path"] != roi_rel:
            raise ValueError("Refusing to replace a different registered ROI")
        if inventory_row["anatomical_roi_sha256"] and inventory_row["anatomical_roi_sha256"] != roi_hash:
            raise ValueError("Refusing to replace a different registered ROI hash")
        if inventory_row["anatomical_mask_path"] and inventory_row["anatomical_mask_path"] != mask_rel:
            raise ValueError("Refusing to replace a different registered mask")
        if inventory_row["anatomical_mask_sha256"] and inventory_row["anatomical_mask_sha256"] != mask_hash:
            raise ValueError("Refusing to replace a different registered mask hash")

        updated_inventory_row = current_inventory[key]
        updated_inventory_row["anatomical_roi_path"] = roi_rel
        updated_inventory_row["anatomical_roi_sha256"] = roi_hash
        updated_inventory_row["anatomical_mask_path"] = mask_rel
        updated_inventory_row["anatomical_mask_sha256"] = mask_hash
        updated_inventory_row["anatomical_annotation_status"] = "valid_anatomical_roi_and_binary_mask"
        updated_inventory_row["mask_integrity_issue"] = ""
        new_rows[key] = make_review_row(
            subject_id=subject_id,
            modality=modality,
            original_filename=original_filename,
            inventory_row=updated_inventory_row,
            metadata_source_hash=metadata_hash,
            review_status="approved",
            reviewer=reviewer.strip(),
            reviewed_on=reviewed_date,
            review_method=review_method.strip(),
            source_review_package_path=package_rel,
            source_review_package_sha256=package_hash,
        )

    if require_complete_modality and set(package_by_key) != expected_keys:
        raise ValueError("Review package does not cover every frozen-metadata image for this modality")
    if not package_by_key:
        raise ValueError("Review package contains no images")

    existing_ledger: list[dict[str, str]] = []
    if REVIEW_REGISTER.is_file():
        existing_ledger = read_csv(REVIEW_REGISTER)
        existing_keys = [(r["subject_id"], r["modality"]) for r in existing_ledger]
        if len(existing_keys) != len(set(existing_keys)):
            raise ValueError("Existing anatomical review register has duplicate keys")
        ledger_by_key = {(r["subject_id"], r["modality"]): r for r in existing_ledger}
    else:
        prior_manifest = read_csv(MANIFEST)
        legacy_package_hash = sha256(LEGACY_REVIEW_SOURCE)
        legacy_package_rel = LEGACY_REVIEW_SOURCE.relative_to(ROOT).as_posix()
        ledger_by_key = {}
        for row in prior_manifest:
            if row["annotation_status"] != "approved":
                continue
            key = (row["subject_id"], row["modality"])
            inventory_row = inventory_by_key.get(key)
            if inventory_row is None:
                raise ValueError("A pre-existing approved annotation is absent from the inventory")
            legacy_row = make_review_row(
                subject_id=row["subject_id"],
                modality=row["modality"],
                original_filename=row["original_image_filename"],
                inventory_row=inventory_row,
                metadata_source_hash=metadata_hash,
                review_status="approved",
                reviewer="preserved_prior_manifest_approval",
                reviewed_on="",
                review_method="preexisting_approved_status; reviewer/date not recorded",
                source_review_package_path=legacy_package_rel,
                source_review_package_sha256=legacy_package_hash,
            )
            ledger_by_key[key] = legacy_row

    for key, new_row in new_rows.items():
        prior = ledger_by_key.get(key)
        if prior is not None:
            identity_fields = (
                "original_image_filename", "annotation_image_path", "annotation_image_sha256",
                "roi_path", "roi_sha256", "mask_path", "mask_sha256",
            )
            if any(prior.get(field, "") != new_row[field] for field in identity_fields):
                raise ValueError("New approval conflicts with a prior review record for this image")
        ledger_by_key[key] = new_row

    ledger_rows = sorted(ledger_by_key.values(), key=lambda r: (r["subject_id"], r["modality"]))
    report = {
        "status": "validated_only" if dry_run else "registered",
        "modality": modality,
        "reviewed_images": len(package_by_key),
        "complete_modality_coverage": set(package_by_key) == expected_keys,
        "review_register_total": len(ledger_rows),
        "review_register_by_modality": {
            name: sum(row["modality"] == name for row in ledger_rows)
            for name in ("eyelid", "nail")
        },
        "dry_run": dry_run,
    }
    if dry_run:
        return report

    inventory_temp = write_staged_csv(INVENTORY, list(current_inventory.values()), list(inventory_rows[0]))
    ledger_temp = write_staged_csv(REVIEW_REGISTER, ledger_rows, REVIEW_FIELDS)
    try:
        os.replace(inventory_temp, INVENTORY)
        os.replace(ledger_temp, REVIEW_REGISTER)
    finally:
        inventory_temp.unlink(missing_ok=True)
        ledger_temp.unlink(missing_ok=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-package", required=True, type=Path)
    parser.add_argument("--modality", required=True, choices=("eyelid", "nail"))
    parser.add_argument("--reviewer", required=True, help="Non-identifying reviewer label")
    parser.add_argument("--reviewed-on", required=True, help="Human review date in YYYY-MM-DD form")
    parser.add_argument("--review-method", required=True)
    parser.add_argument(
        "--confirm-all-rows-reviewed", action="store_true",
        help="Required assertion that every row in the review package was human-reviewed and approved",
    )
    parser.add_argument("--require-complete-modality", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(register(
        args.review_package,
        modality=args.modality,
        reviewer=args.reviewer,
        reviewed_on=args.reviewed_on,
        review_method=args.review_method,
        confirm_all_rows_reviewed=args.confirm_all_rows_reviewed,
        require_complete_modality=args.require_complete_modality,
        dry_run=args.dry_run,
    ), indent=2))


if __name__ == "__main__":
    main()
