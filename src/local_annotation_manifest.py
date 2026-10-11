"""Build the canonical, provenance-linked local annotation manifest and queue.

This script inventories existing manual ROIs/masks only. It never creates,
changes, or guesses an annotation. Run from the repository root after the
local cohort output has been built from the frozen metadata snapshot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/local"
ANNOTATIONS = RAW / "annotations"
FROZEN = RAW / "metadata/frozen_2026-10-07/AIMA Patients Metadata - Sheet1 values.csv"
PROVENANCE = FROZEN.parent / "provenance.json"
PRIOR_MANIFEST = RAW / "manifest/frozen_2026-10-07/local_subject_manifest.csv"
DEFAULT_COHORT = ROOT / "data/processed/local_cohort_who2024_2026-10-08"
REVIEW_REGISTER = ANNOTATIONS / "manifests/anatomical_review_register.csv"
EXPECTED_HEADER = [
    "subject_id", "age", "sex", "lab_hb_g_per_l", "lab_hb_g_per_dl",
    "anemia_status", "raw_eyelid_file", "raw_nail_file", "notes", "source",
]
VALID_STATUSES = {"pending", "annotated", "needs_review", "approved", "rejected"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Missing or duplicate CSV header: {path}")
        rows = list(reader)
        if any(None in row or None in row.values() for row in rows):
            raise ValueError(f"Extra or missing CSV cells: {path}")
        if any(not any(value.strip() for value in row.values()) for row in rows):
            raise ValueError(f"Blank data row: {path}")
        return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def repo_path(value: str, *, required: bool = True) -> Path | None:
    if not value:
        if required:
            raise ValueError("Required repository path is blank")
        return None
    resolved = (ROOT / value).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Path escapes repository: {value}")
    if required and not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def validate_binary_mask(image_path: Path, mask_path: Path) -> None:
    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        if image.mode != "RGB" or mask.mode != "L" or image.size != mask.size:
            raise ValueError(f"Image/mask dimensions or modes differ: {image_path.name}")
        histogram = mask.histogram()
        if sum(histogram[1:255]) or not histogram[255]:
            raise ValueError(f"Mask is not a nonempty binary 0/255 PNG: {mask_path}")


def verify_metadata() -> tuple[list[dict[str, str]], dict[str, dict[str, str]], str, str]:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    source_hash = sha256(FROZEN)
    if source_hash != provenance.get("frozen_csv_sha256"):
        raise ValueError("Canonical frozen metadata hash differs from its freeze provenance")
    modified = datetime.fromtimestamp(FROZEN.stat().st_mtime).astimezone().isoformat()
    with FROZEN.open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle))
    if header != EXPECTED_HEADER or provenance.get("nonempty_cells_beyond_column_j") != 0:
        raise ValueError("Canonical metadata schema or trailing-column check failed")
    metadata_rows = read_csv(FROZEN)
    metadata = {row["subject_id"]: row for row in metadata_rows}
    if len(metadata) != len(metadata_rows) or len(metadata) != 140:
        raise ValueError("Canonical metadata contains duplicate IDs or an unexpected row count")
    prior_rows = read_csv(PRIOR_MANIFEST)
    prior = {row["subject_id"]: row for row in prior_rows}
    if len(prior) != len(prior_rows) or set(prior) != set(metadata):
        raise ValueError("Frozen metadata and prior manifest subject sets differ")
    for subject_id, row in metadata.items():
        if row["sex"] not in {"F", "M"} or not row["age"].isdigit():
            raise ValueError(f"Invalid sex or age in canonical metadata: {subject_id}")
        if any(row[field] != prior[subject_id][field] for field in EXPECTED_HEADER):
            raise ValueError(f"Canonical metadata differs from frozen prior manifest: {subject_id}")
        if prior[subject_id]["metadata_source_sha256"] != source_hash:
            raise ValueError(f"Prior manifest source hash mismatch: {subject_id}")
    return metadata_rows, metadata, source_hash, modified


def read_cohort(cohort_dir: Path, source_hash: str) -> tuple[
    dict[str, dict[str, str]], dict[str, str],
]:
    summary = json.loads((cohort_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("metadata_source_sha256") != source_hash:
        raise ValueError("Cohort output was not built from the current frozen metadata")
    eligibility_rows = read_csv(cohort_dir / "all_subject_eligibility.csv")
    eligibility = {row["subject_id"]: row for row in eligibility_rows}
    fold_rows = read_csv(cohort_dir / "subject_folds.csv")
    folds = {row["subject_id"]: row["fold"] for row in fold_rows}
    if len(eligibility) != len(eligibility_rows) or len(eligibility) != 140:
        raise ValueError("Cohort eligibility table does not contain one row per frozen subject")
    if len(folds) != len(fold_rows):
        raise ValueError("Duplicate subject in cohort fold table")
    for row in eligibility_rows:
        if row["metadata_source_sha256"] != source_hash:
            raise ValueError("Cohort eligibility row has stale metadata provenance")
    for row in fold_rows:
        if row["metadata_source_sha256"] != source_hash:
            raise ValueError("Cohort fold row has stale metadata provenance")
    return eligibility, folds


def validate_review_register(
    rows: list[dict[str, str]],
    inventory_by_key: dict[tuple[str, str], dict[str, str]],
    expected_refs: set[tuple[str, str]],
    source_hash: str,
) -> set[tuple[str, str]]:
    """Validate explicit human approval records against the registered files."""
    approved: set[tuple[str, str]] = set()
    field_pairs = (
        ("original_image_filename", "original_filename"),
        ("source_image_path", "original_source_path"),
        ("source_image_sha256", "original_source_sha256"),
        ("annotation_image_path", "annotation_image_path"),
        ("annotation_image_sha256", "annotation_image_sha256"),
        ("roi_path", "anatomical_roi_path"),
        ("roi_sha256", "anatomical_roi_sha256"),
        ("mask_path", "anatomical_mask_path"),
        ("mask_sha256", "anatomical_mask_sha256"),
    )
    for row in rows:
        key = (row.get("subject_id", ""), row.get("modality", ""))
        if key in approved or key not in expected_refs:
            raise ValueError("Review register has a duplicate or unexpected image key")
        inventory_row = inventory_by_key.get(key)
        if inventory_row is None:
            raise ValueError("Approved annotation is absent from the registered image inventory")
        if row.get("review_status") != "approved":
            raise ValueError("Review register may contain approved records only")
        if not all(row.get(field, "").strip() for field in (
            "reviewer", "review_method", "review_scope", "metadata_source_sha256",
        )):
            raise ValueError("Review register is missing approval provenance")
        if row["metadata_source_sha256"] != source_hash:
            raise ValueError("Review register uses stale frozen-metadata provenance")
        if inventory_row.get("metadata_source_sha256") != source_hash:
            raise ValueError("Image inventory uses stale frozen-metadata provenance")
        for review_field, inventory_field in field_pairs:
            if row.get(review_field, "") != inventory_row.get(inventory_field, ""):
                raise ValueError("Review register does not match the registered image/ROI/mask")
        if not all(row.get(field, "").strip() for field in (
            "source_image_path", "source_image_sha256", "annotation_image_path",
            "roi_path", "mask_path", "roi_sha256", "mask_sha256",
            "source_review_package_path", "source_review_package_sha256",
        )):
            raise ValueError("Approved review record lacks an image, ROI, or mask reference")
        approved.add(key)
    return approved


def verify_review_package_hashes(rows: list[dict[str, str]]) -> None:
    """Ensure the recorded approval evidence files are unchanged."""
    verified: dict[str, str] = {}
    for row in rows:
        relative_path = row["source_review_package_path"]
        if relative_path not in verified:
            review_package = repo_path(relative_path)
            verified[relative_path] = sha256(review_package)
        if verified[relative_path] != row["source_review_package_sha256"]:
            raise ValueError("Source review package hash differs from the human approval register")


def numeric_subject_key(subject_id: str) -> tuple[int, str]:
    try:
        return int(subject_id.rsplit("-", 1)[1]), subject_id
    except (IndexError, ValueError):
        return 10**9, subject_id


def build(cohort_dir: Path = DEFAULT_COHORT) -> dict[str, Any]:
    metadata_rows, metadata, source_hash, source_mtime = verify_metadata()
    eligibility, folds = read_cohort(cohort_dir, source_hash)
    expected_refs = {(sid, modality) for sid, row in metadata.items()
                     for modality in ("eyelid", "nail") if row[f"raw_{modality}_file"].strip()}
    inventory_path = ANNOTATIONS / "manifests/image_ready_inventory.csv"
    inventory = read_csv(inventory_path)
    by_key = {(row["subject_id"], row["modality"]): row for row in inventory}
    if len(by_key) != len(inventory) or not set(by_key).issubset(expected_refs):
        raise ValueError("Consolidated image inventory has duplicate or unreferenced keys")
    review_rows = read_csv(REVIEW_REGISTER)
    reviewed_keys = validate_review_register(review_rows, by_key, expected_refs, source_hash)
    verify_review_package_hashes(review_rows)
    annotators = {
        (row["subject_id"], "eyelid"): row.get("annotator", "").strip() or "not_recorded"
        for row in read_csv(ANNOTATIONS / "manifests/pilot1.csv")
    }
    for row in read_csv(ANNOTATIONS / "manifests/paired_pilot10.csv"):
        annotators[(row["subject_id"], row["modality"])] = row.get("annotator", "").strip() or "not_recorded"

    incoming = RAW / "incoming"
    incoming_counts = Counter(path.name for path in incoming.iterdir() if path.is_file())
    metadata_ref_counts: Counter[tuple[str, str]] = Counter()
    records: list[dict[str, Any]] = []
    for subject_id, source_row in metadata.items():
        for modality in ("eyelid", "nail"):
            original = source_row[f"raw_{modality}_file"].strip()
            if not original:
                continue
            key = (subject_id, modality)
            metadata_ref_counts[key] += 1
            if Path(original).name != original or incoming_counts[original] != 1:
                raise ValueError(f"Metadata source image is missing or ambiguous: {subject_id}/{original}")
            source_path = incoming / original
            source_rel = source_path.relative_to(ROOT).as_posix()
            if eligibility[subject_id][f"{modality}_reference_status"] != "exact_unique_match":
                raise ValueError(f"Cohort reference status disagrees with metadata check: {subject_id}/{modality}")
            cohort_row = eligibility[subject_id]
            label = cohort_row["derived_anemia_label"] if cohort_row["eligibility_status"] == "eligible" else ""
            inventory_row = by_key.get(key)
            image_path = ""
            roi_path = ""
            mask_path = ""
            image_sha = ""
            roi_sha = ""
            mask_sha = ""
            source_batch = ""
            glare_path = ""
            glare_status = "not_applicable"
            glare_candidate_status = "not_applicable"
            glare_rejection_reason = ""
            glare_roi_path = ""
            glare_roi_expected = ""
            glare_roi_status = "not_applicable"
            annotator = annotators.get(key, "not_recorded")
            notes: list[str] = []
            image_readiness = "raw_only_needs_standardized_TIFF"
            if inventory_row:
                if inventory_row["original_filename"] != original:
                    raise ValueError(f"Image inventory filename does not match metadata: {subject_id}/{modality}")
                image_path = inventory_row["annotation_image_path"]
                image_file = repo_path(image_path)
                image_sha = sha256(image_file)
                if image_sha != inventory_row["annotation_image_sha256"]:
                    raise ValueError(f"Annotation TIFF hash mismatch: {subject_id}/{modality}")
                image_readiness = "standardized_TIFF_ready"
                source_batch = inventory_row["source_batch"]
                roi_path = inventory_row["anatomical_roi_path"]
                mask_path = inventory_row["anatomical_mask_path"]
                roi_sha = inventory_row["anatomical_roi_sha256"]
                mask_sha = inventory_row["anatomical_mask_sha256"]
                glare_path = inventory_row["glare_candidate_or_mask_path"]
                glare_status = inventory_row["glare_status"]
                glare_roi_path = inventory_row["glare_roi_path"]
                glare_roi_expected = inventory_row.get("glare_roi_expected_path", "")
                glare_roi_status = inventory_row["glare_roi_status"]
                if roi_path:
                    roi_file = repo_path(roi_path)
                    if sha256(roi_file) != roi_sha:
                        raise ValueError(f"Fiji ROI hash mismatch: {subject_id}/{modality}")
                if mask_path:
                    mask_file = repo_path(mask_path)
                    if sha256(mask_file) != mask_sha:
                        raise ValueError(f"Generated anatomical mask hash mismatch: {subject_id}/{modality}")
                    validate_binary_mask(image_file, mask_file)
                source_status = inventory_row["anatomical_annotation_status"]
                if source_status == "valid_anatomical_roi_and_binary_mask" and key in reviewed_keys:
                    status, review_status = "approved", "approved"
                elif source_status == "potentially_mismatched_annotation" or roi_path or mask_path:
                    status, review_status = "needs_review", "needs_review"
                    notes.append("Existing annotation artifacts need integrity or human review.")
                elif source_status == "missing_anatomical_annotation":
                    status, review_status = "pending", "not_reviewed"
                else:
                    raise ValueError(f"Unrecognized inventory annotation status: {subject_id}/{modality}")
                if status == "approved" and not roi_path:
                    raise ValueError(f"Approved annotation has no Fiji ROI: {subject_id}/{modality}")
                if inventory_row["image_review_status"] == "image_qc_not_recorded_in_paired_manifest":
                    notes.append("Source paired-pilot manifest did not record image QC status; anatomical mask is listed as manually reviewed in the frozen cohort audit.")
                if glare_status == "threshold_candidate_rejected_no_usable_glare_mask":
                    glare_candidate_status = "rejected"
                    glare_rejection_reason = (
                        "Naive HSV threshold was overinclusive (31.8%-82.0% of the nail-plate mask); "
                        "it cannot isolate specular glare from naturally pale nail surface. Do not use as glare exclusion."
                    )
                    notes.append("Nail glare threshold candidate was rejected; it is not an accepted glare mask.")
                elif glare_status == "candidate_mask_present_visual_review_pending_glare_roi_missing":
                    glare_candidate_status = "needs_review"
                elif glare_status == "batch1_rule_mask_present_rule_not_clinically_validated":
                    glare_candidate_status = "pilot_rule_not_clinically_validated"
            else:
                if modality != "nail":
                    raise ValueError(f"Metadata eyelid image lacks a standardized annotation TIFF: {subject_id}")
                status, review_status = "pending", "not_reviewed"
                notes.append("Original exists in incoming, but no standardized annotation TIFF is staged yet.")
            if status not in VALID_STATUSES:
                raise ValueError(f"Invalid annotation status: {status}")
            fold = folds.get(subject_id, "")
            has_pair = bool(source_row["raw_eyelid_file"].strip() and source_row["raw_nail_file"].strip())
            records.append({
                "subject_id": subject_id,
                "modality": modality,
                "original_image_filename": original,
                "source_image_path": source_rel,
                "annotation_image_path": image_path,
                "image_readiness_status": image_readiness,
                "annotation_source_file": roi_path,
                "roi_filename": Path(roi_path).name if roi_path else "",
                "roi_path": roi_path,
                "roi_sha256": roi_sha,
                "mask_filename": Path(mask_path).name if mask_path else "",
                "mask_path": mask_path,
                "mask_sha256": mask_sha,
                "annotation_status": status,
                "reviewed_status": review_status,
                "annotator": annotator,
                "exclusion_rejection_reason": "",
                "source_batch": source_batch,
                "paired_subject": str(has_pair).lower(),
                "cohort_eligibility_status": cohort_row["eligibility_status"],
                "cohort_exclusion_reasons": cohort_row["exclusion_reasons"],
                "derived_anemia_label": label,
                "cv_fold": fold,
                "glare_candidate_or_mask_path": glare_path,
                "glare_candidate_status": glare_candidate_status,
                "glare_rejection_reason": glare_rejection_reason,
                "glare_status": glare_status,
                "glare_roi_path": glare_roi_path,
                "glare_roi_expected_path": glare_roi_expected,
                "glare_roi_status": glare_roi_status,
                "notes": " ".join(notes),
                "source_manifest_path": inventory_row["source_manifest_path"] if inventory_row else "",
                "metadata_source_path": str(FROZEN),
                "metadata_source_sha256": source_hash,
                "metadata_source_mtime": source_mtime,
                "original_source_sha256": sha256(source_path),
                "annotation_image_sha256": image_sha,
            })

    if {(r["subject_id"], r["modality"]) for r in records} != expected_refs:
        raise ValueError("Canonical annotation manifest does not cover every metadata image reference")
    approved_keys = {(r["subject_id"], r["modality"]) for r in records
                     if r["annotation_status"] == "approved"}
    if approved_keys != reviewed_keys:
        raise ValueError("Approved annotation set differs from the human review register")
    if len(records) != 275:
        raise ValueError(f"Expected 275 referenced images, found {len(records)}")

    # Select one positive eyelid per preallocated fold that lacks an approved positive mask.
    pending = [row for row in records if row["annotation_status"] == "pending"]
    pending_by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pending:
        pending_by_subject[str(row["subject_id"])].append(row)
    fold_map = {row["subject_id"]: row["fold"] for row in read_csv(cohort_dir / "subject_folds.csv")}
    covered_positive_folds = {
        int(fold_map[sid]) for sid, modality in approved_keys
        if modality == "eyelid" and eligibility[sid]["derived_anemia_label"] == "1"
        and sid in fold_map
    }
    selected_cv_subjects: dict[str, int] = {}
    for fold in sorted(set(int(value) for value in fold_map.values())):
        if fold in covered_positive_folds:
            continue
        candidates = [row for row in pending if row["modality"] == "eyelid"
                      and row["derived_anemia_label"] == "1" and row["cv_fold"] == str(fold)]
        candidates.sort(key=lambda row: (
            0 if len(pending_by_subject[str(row["subject_id"])]) > 1 else 1,
            0 if row["image_readiness_status"] == "standardized_TIFF_ready" else 1,
            numeric_subject_key(str(row["subject_id"])),
        ))
        if not candidates:
            raise ValueError(f"No pending anemia-positive eyelid candidate in CV fold {fold}")
        selected_cv_subjects[str(candidates[0]["subject_id"])] = fold

    queue: list[dict[str, Any]] = []
    for subject_id, subject_rows in pending_by_subject.items():
        label_values = {str(row["derived_anemia_label"]) for row in subject_rows}
        has_pair_pending = {str(row["modality"]) for row in subject_rows} == {"eyelid", "nail"}
        eligible = all(row["cohort_eligibility_status"] == "eligible" for row in subject_rows)
        selected_fold = selected_cv_subjects.get(subject_id)
        if selected_fold is not None:
            tier = 1
        elif "1" in label_values:
            tier = 2
        elif eligible and has_pair_pending:
            tier = 3
        elif eligible:
            tier = 4
        else:
            tier = 5
        for row in subject_rows:
            modality = str(row["modality"])
            if selected_fold is not None:
                reason = (f"cv_fold_{selected_fold}_anemia_positive_coverage" if modality == "eyelid"
                          else "paired_nail_for_cv_priority_subject")
            elif row["derived_anemia_label"] == "1":
                reason = "remaining_anemia_positive_subject"
            elif tier == 3:
                reason = "labeled_paired_subject_complete_both_modalities"
            elif tier == 4:
                reason = "remaining_WHO_labeled_image"
            else:
                reason = "segmentation_research_only_no_WHO_eligible_label"
            queue.append({
                **row,
                "priority_tier": tier,
                "priority_reason": reason,
                "annotation_ready_now": str(row["image_readiness_status"] == "standardized_TIFF_ready").lower(),
                "next_action": "manual_annotate_in_Fiji" if row["image_readiness_status"] == "standardized_TIFF_ready"
                else "render_standardized_RGB_TIFF_then_manual_annotate",
            })
    queue.sort(key=lambda row: (
        int(row["priority_tier"]),
        int(row["cv_fold"]) if row["cv_fold"] else 99,
        numeric_subject_key(str(row["subject_id"])),
        0 if row["modality"] == "eyelid" else 1,
    ))
    for rank, row in enumerate(queue, start=1):
        row["priority_rank"] = rank

    fields = list(records[0])
    queue_fields = list(queue[0])
    write_csv(ANNOTATIONS / "manifests/annotation_manifest.csv", records, fields)
    write_csv(ANNOTATIONS / "manifests/pending_annotation_queue.csv", queue, queue_fields)

    status_counts = Counter(row["annotation_status"] for row in records)
    modality_counts = {
        modality: {
            "metadata_references": sum(row["modality"] == modality for row in records),
            "standardized_TIFFs": sum(row["modality"] == modality and bool(row["annotation_image_path"]) for row in records),
            "approved": sum(row["modality"] == modality and row["annotation_status"] == "approved" for row in records),
            "pending": sum(row["modality"] == modality and row["annotation_status"] == "pending" for row in records),
            "needs_review": sum(row["modality"] == modality and row["annotation_status"] == "needs_review" for row in records),
            "rejected": sum(row["modality"] == modality and row["annotation_status"] == "rejected" for row in records),
        } for modality in ("eyelid", "nail")
    }
    queue_label_counts = Counter(
        "anemia_positive" if row["derived_anemia_label"] == "1" else
        "non_anemia" if row["derived_anemia_label"] == "0" else "no_WHO_eligible_label"
        for row in queue
    )
    cohort_counts = {
        modality: sum(row["eligibility_status"] == "eligible" and row[f"{modality}_reference_status"] == "exact_unique_match"
                      for row in eligibility.values())
        for modality in ("eyelid", "nail")
    }
    if status_counts["needs_review"]:
        audit_status = "PASS_WITH_ANNOTATION_REVIEW_REQUIRED"
    elif status_counts["pending"]:
        audit_status = "PASS_WITH_ANNOTATION_PENDING"
    elif any(row["glare_roi_expected_path"] and not row["glare_roi_path"] for row in records):
        audit_status = "PASS_WITH_GLARE_REVIEW_PENDING"
    else:
        audit_status = "PASS"
    report = {
        "status": audit_status,
        "metadata_source_path": str(FROZEN),
        "metadata_source_sha256": source_hash,
        "metadata_source_mtime": source_mtime,
        "anatomical_review_register_path": str(REVIEW_REGISTER.relative_to(ROOT)),
        "subjects": len(metadata_rows),
        "metadata_image_references": len(records),
        "annotation_ready_TIFFs": sum(bool(row["annotation_image_path"]) for row in records),
        "annotation_ready_TIFFs_by_modality": {modality: counts["standardized_TIFFs"] for modality, counts in modality_counts.items()},
        "WHO_labeled_usable_image_references_by_modality": cohort_counts,
        "manual_ROI_mask_pairs": len(approved_keys),
        "approved_masks_by_modality": {modality: counts["approved"] for modality, counts in modality_counts.items()},
        "annotation_status_counts": dict(status_counts),
        "annotation_status_counts_by_modality": modality_counts,
        "pending_queue_images": len(queue),
        "pending_queue_by_modality": dict(Counter(row["modality"] for row in queue)),
        "pending_queue_label_counts": dict(queue_label_counts),
        "pending_queue_positive_images": queue_label_counts["anemia_positive"],
        "pending_queue_non_anemia_images": queue_label_counts["non_anemia"],
        "pending_queue_without_WHO_eligible_label_images": queue_label_counts["no_WHO_eligible_label"],
        "CV_positive_fold_priority_subjects": len(selected_cv_subjects),
        "CV_positive_fold_priority_subject_ids": sorted(selected_cv_subjects, key=numeric_subject_key),
        "glare_ROIs_missing_for_paired_pilot_images": sum(
            bool(row["glare_roi_expected_path"]) and not row["glare_roi_path"] for row in records
        ),
        "rejected_nail_glare_threshold_candidates": sum(
            row["glare_candidate_status"] == "rejected" for row in records
        ),
        "notes": [
            "Annotation status is about manual anatomical annotation. Glare processing status is a separate field.",
            (
                f"{sum(row['modality'] == 'nail' and not row['annotation_image_path'] for row in records)} "
                "metadata-referenced nail originals remain without a standardized annotation TIFF."
            ),
            "Five subjects have no nail reference in metadata and are not queue items.",
            f"{len(approved_keys)} anatomical ROI/mask pairs have explicit human approval records in the local review register.",
            "Glare review is tracked separately and is not implied by anatomical mask approval.",
        ],
    }
    (ANNOTATIONS / "manifests/annotation_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, default=DEFAULT_COHORT)
    args = parser.parse_args()
    print(json.dumps(build(args.cohort_dir), indent=2))
