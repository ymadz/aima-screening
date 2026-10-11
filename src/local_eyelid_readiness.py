"""Refresh a private eyelid-only cohort/readiness bundle without training.

The label policy is reused verbatim from the versioned WHO cohort. Historical
cohorts and annotation manifests are never overwritten. Patient rows are saved
only below ignored data/processed paths; stdout contains aggregate results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageCms

from . import local_annotation_manifest as annotations
from .local_eyelid_validation import (
    audit_folds, checked_file, load_manual_pair, perceptual_hash, sha256,
    source_snapshot, strict_csv, validate_eyelid_link,
)
from .local_who2024_cohort import normalized_hb, who_label
from .register_local_annotation_reviews import read_review_package


ROOT = Path(__file__).resolve().parents[1]
OLD_COHORT = ROOT / "data/processed/local_cohort_who2024_2026-10-08"
CANDIDATE_DIR = ROOT / "data/processed/segmentation/manual_eyelid_candidate_review_2026-10-10"
SEGMENTATION_SPLITS = ROOT / "data/processed/segmentation/local_unet_dev_eval_fullframe_2026-10-09/subject_split.json"
DEFAULT_OUTPUT = ROOT / "data/processed/local_eyelid_readiness_2026-10-10_v1"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write a new derived file, including an explicit header for empty tables."""
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def require_private_output(path: Path) -> Path:
    """Keep patient-derived artifacts inside Git-ignored processed storage."""
    path = path.resolve()
    if not path.is_relative_to((ROOT / "data/processed").resolve()):
        raise ValueError("Patient artifacts must stay below data/processed")
    check = subprocess.run(
        ["git", "check-ignore", "-q", "--", str(path / "private_artifact.csv")],
        cwd=ROOT, capture_output=True, check=False,
    )
    if check.returncode != 0:
        raise ValueError("Patient artifact destination is not Git-ignored")
    if path.exists():
        raise FileExistsError("Refusing to overwrite a historical derived bundle")
    return path


def recompute_labels(
    rows: list[dict[str, str]], config: dict[str, Any], source_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Apply the existing policy and preserve both exact source Hb strings."""
    if (config.get("metadata_source_sha256") != source_hash
            or config.get("name") != "WHO_2024_unadjusted_adult_15_65"
            or config.get("thresholds_g_per_dl") != {"M": "13.0", "F_nonpregnant": "12.0"}
            or config.get("age_years_inclusive") != [15, 65]
            or config.get("pregnancy_status_source") != "study_owner_attestation_outside_frozen_sheet"
            or not config.get("pregnancy_attestation")):
        raise ValueError("Existing label policy or owner attestation is unresolved")
    subjects = [row["subject_id"] for row in rows]
    if not subjects or any(not value.strip() for value in subjects) or len(set(subjects)) != len(subjects):
        raise ValueError("Metadata subject IDs are blank or duplicated")
    output, labels = [], {}
    for row in rows:
        if not row["age"].isdigit() or row["sex"] not in {"F", "M"} or row["source"] != "local":
            raise ValueError("Metadata age, sex, or source is invalid")
        if row["anemia_status"].strip():
            raise ValueError("Unexpected populated source anemia-status field")
        hb, column, _, unit = normalized_hb(row)
        if row["notes"].strip().lower() == "not_cbc" and hb is not None:
            raise ValueError("Non-CBC note conflicts with numeric laboratory Hb")
        reasons = []
        if hb is None:
            reasons.append("missing_laboratory_hb")
        if not 15 <= int(row["age"]) <= 65:
            reasons.append("outside_who_15_65")
        if not reasons:
            labels[row["subject_id"]] = who_label(hb, row["sex"], int(row["age"]), True)
        output.append({
            **row, "normalized_hb_g_per_dl": str(hb) if hb is not None else "",
            "primary_recorded_hb_column": column,
            "primary_recorded_hb_value": row[column] if column else "",
            "primary_recorded_hb_unit": unit,
            "original_laboratory_reporting_unit": "not_recorded",
            "pregnancy_status_source": config["pregnancy_status_source"],
            "derived_anemia_label": labels.get(row["subject_id"], ""),
            "eligibility_status": "excluded" if reasons else "eligible",
            "exclusion_reasons": ";".join(reasons),
            "metadata_source_path": str(annotations.FROZEN),
            "metadata_source_sha256": source_hash,
        })
    return output, labels


def audit() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    """Recheck current lineage, reviewed mask pixels, labels and historical folds."""
    snapshot = source_snapshot(annotations.FROZEN, annotations.PROVENANCE)
    source_hash = snapshot["sha256"]
    drive = annotations.FROZEN.with_name("AIMA Patients Metadata - original Drive export.csv")
    config_path = OLD_COHORT / "who_label_config.json"
    manifest_path = annotations.ANNOTATIONS / "manifests/annotation_manifest.csv"
    inventory_path = annotations.ANNOTATIONS / "manifests/image_ready_inventory.csv"
    sources = [annotations.FROZEN, annotations.PROVENANCE, drive, annotations.PRIOR_MANIFEST,
               config_path, manifest_path, inventory_path, annotations.REVIEW_REGISTER,
               OLD_COHORT / "all_subject_eligibility.csv", OLD_COHORT / "subject_folds.csv",
               SEGMENTATION_SPLITS, CANDIDATE_DIR / "candidate_pairs.csv",
               annotations.ANNOTATIONS / "manifests/pilot1.csv", annotations.ANNOTATIONS / "manifests/batch2.csv"]
    initial_hashes = {str(path): sha256(path) for path in sources}
    if initial_hashes[str(annotations.FROZEN)] != source_hash:
        raise ValueError("Canonical metadata changed after source verification")
    rows = strict_csv(annotations.FROZEN)
    with annotations.FROZEN.open(newline="", encoding="utf-8-sig") as handle:
        if next(csv.reader(handle)) != annotations.EXPECTED_HEADER:
            raise ValueError("Canonical metadata schema differs from approved source")
    freeze = json.loads(annotations.PROVENANCE.read_text())
    if freeze.get("nonempty_cells_beyond_column_j") != 0 or freeze.get("data_rows") != len(rows):
        raise ValueError("Freeze row count or trailing-column evidence differs")
    if sha256(drive) != freeze.get("original_drive_export_sha256"):
        raise ValueError("Preserved original Drive export differs from freeze evidence")
    if strict_csv(drive) != rows:
        raise ValueError("Parsed Drive export differs from frozen values")
    subjects = {row["subject_id"]: row for row in rows}
    previous = strict_csv(annotations.PRIOR_MANIFEST)
    previous_by_id = {row["subject_id"]: row for row in previous}
    if len(previous_by_id) != len(previous) or set(previous_by_id) != set(subjects):
        raise ValueError("Prior manifest subject coverage differs")
    if any(any(row[field] != previous_by_id[row["subject_id"]][field]
               for field in annotations.EXPECTED_HEADER)
           or previous_by_id[row["subject_id"]]["metadata_source_sha256"] != source_hash for row in rows):
        raise ValueError("Prior manifest values or source lineage differ")
    config = json.loads(config_path.read_text())
    eligibility, labels = recompute_labels(rows, config, source_hash)
    cached = strict_csv(OLD_COHORT / "all_subject_eligibility.csv")
    if ({row["subject_id"] for row in cached} != set(subjects) or len(cached) != len(rows)
            or any(row["derived_anemia_label"] != str(labels.get(row["subject_id"], ""))
                   or row["metadata_source_sha256"] != source_hash for row in cached)):
        raise ValueError("Historical cohort differs from recomputed labels")
    incoming_names = Counter(path.name for path in (ROOT / "data/raw/local/incoming").iterdir() if path.is_file())
    expected = {(sid, modality) for sid, row in subjects.items() for modality in ("eyelid", "nail")
                if row[f"raw_{modality}_file"]}
    for sid, modality in expected:
        filename = subjects[sid][f"raw_{modality}_file"]
        if Path(filename).name != filename or incoming_names[filename] != 1:
            raise ValueError("An original image reference is missing or ambiguous")
    manifest, inventory = strict_csv(manifest_path), strict_csv(inventory_path)
    review_rows = strict_csv(annotations.REVIEW_REGISTER)
    manifest_keys = {(row["subject_id"], row["modality"]) for row in manifest}
    if len(manifest_keys) != len(manifest) or manifest_keys != expected:
        raise ValueError("Annotation manifest contains duplicate or unresolved keys")
    inventory_keys = {(row["subject_id"], row["modality"]): row for row in inventory}
    if len(inventory_keys) != len(inventory) or set(inventory_keys) != expected:
        raise ValueError("Registered image inventory does not cover source references")
    approved = annotations.validate_review_register(review_rows, inventory_keys, expected, source_hash)
    annotations.verify_review_package_hashes(review_rows)
    review_by_key = {(row["subject_id"], row["modality"]): row for row in review_rows}
    packages = {}
    for value in {row["source_review_package_path"] for row in review_rows if row["modality"] == "eyelid"}:
        for row in read_review_package(checked_file(ROOT, value)):
            packages[(row["subject_id"], row["modality"])] = row
    pilot = {row["subject_id"]: row for row in strict_csv(annotations.ANNOTATIONS / "manifests/pilot1.csv")}
    batch = {row["subject_id"]: row for row in strict_csv(annotations.ANNOTATIONS / "manifests/batch2.csv")}
    verified, fingerprints, pixel_hashes, profiles, methods = [], {}, [], Counter(), Counter()
    for record in (row for row in manifest if row["modality"] == "eyelid"):
        sid = record["subject_id"]
        key = (sid, "eyelid")
        if key not in approved:
            raise ValueError("An eyelid lacks explicit human anatomical review")
        validate_eyelid_link(record, review_by_key[key], subjects[sid], source_hash)
        if record["derived_anemia_label"] != str(labels.get(sid, "")):
            raise ValueError("Annotation label differs from current metadata policy")
        paths = {field: checked_file(ROOT, record[field], record[hash_field]) for field, hash_field in (
            ("source_image_path", "original_source_sha256"),
            ("annotation_image_path", "annotation_image_sha256"),
            ("roi_path", "roi_sha256"), ("mask_path", "mask_sha256"),
        )}
        image, mask = load_manual_pair(paths["annotation_image_path"], paths["mask_path"])
        package = packages[key]
        if (package["original_filename"] != record["original_image_filename"]
                or package["annotation_image_sha256"] != record["annotation_image_sha256"]
                or package["roi_sha256"] != record["roi_sha256"]):
            raise ValueError("Reviewed preview package has different image/ROI lineage")
        reviewed_path = checked_file(ROOT, package["mask_path"], package["mask_sha256"])
        with Image.open(reviewed_path) as reviewed:
            if reviewed.mode != "L" or not np.array_equal(np.asarray(reviewed), mask):
                raise ValueError("Canonical mask pixels differ from reviewed preview")
        with Image.open(paths["annotation_image_path"]) as tiff:
            profile = tiff.info.get("icc_profile", b"")
            description = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(profile))).strip() if profile else "absent"
        profile_hash = hashlib.sha256(profile).hexdigest()
        profiles[(description, profile_hash)] += 1
        fingerprints[sid] = perceptual_hash(image)
        pixel_hash = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        pixel_hashes.append(pixel_hash)
        history = pilot.get(sid, batch.get(sid))
        if (not history or history["annotation_image_sha256"] != record["annotation_image_sha256"]
                or history["original_filename"] != record["original_image_filename"]
                or history["source_sha256"] != record["original_source_sha256"]):
            raise ValueError("Rendering history does not match annotation bytes")
        if history.get("width") and history.get("height"):
            if sorted((int(history["width"]), int(history["height"]))) != sorted(image.shape[:2]):
                raise ValueError("Recorded source dimensions differ from annotation geometry")
        method = history["render_tool"] or "not_recorded"
        methods[method] += 1
        verified.append({
            **record, "derived_anemia_label": labels.get(sid, ""),
            "render_method_recorded": method, "render_settings_recorded": history["render_settings"],
            "render_history_path": str(annotations.ANNOTATIONS / ("manifests/pilot1.csv" if sid in pilot else "manifests/batch2.csv")),
            "width_px": image.shape[1], "height_px": image.shape[0],
            "stored_icc_description": description, "stored_icc_sha256": profile_hash,
            "decoded_rgb_sha256": pixel_hash,
            "source_review_package_path": review_by_key[key]["source_review_package_path"],
            "source_review_package_sha256": review_by_key[key]["source_review_package_sha256"],
        })
    original_duplicates = sum(count > 1 for count in Counter(row["original_source_sha256"] for row in verified).values())
    pixel_duplicates = sum(count > 1 for count in Counter(pixel_hashes).values())
    existing_candidates = strict_csv(CANDIDATE_DIR / "candidate_pairs.csv")
    recomputed = set()
    sids = sorted(fingerprints)
    for index, left in enumerate(sids):
        for right in sids[index + 1:]:
            distance = int(np.count_nonzero(fingerprints[left] != fingerprints[right]))
            if distance <= 6:
                recomputed.add((left, right, distance))
    if recomputed != {(*sorted((row["subject_id_a"], row["subject_id_b"])), int(row["phash_hamming_distance_64"])) for row in existing_candidates}:
        raise ValueError("Perceptual candidate set differs from current review artifact")
    verified_by_id = {row["subject_id"]: row for row in verified}
    for candidate in existing_candidates:
        for side in ("a", "b"):
            record = verified_by_id[candidate[f"subject_id_{side}"]]
            if any(candidate[f"{field}_{side}"] != record[field] for field in (
                "original_source_sha256", "annotation_image_sha256", "roi_sha256", "mask_sha256",
            )):
                raise ValueError("Duplicate candidate report refers to different image bytes")
    old_folds = strict_csv(OLD_COHORT / "subject_folds.csv")
    if any(row["metadata_source_sha256"] != source_hash for row in old_folds):
        raise ValueError("Historical folds use stale metadata")
    fold_counts = audit_folds(labels, old_folds)
    fold_map = {row["subject_id"]: row["fold"] for row in old_folds}
    splits = json.loads(SEGMENTATION_SPLITS.read_text())
    split_map = {row["subject_id"]: row["split"] for row in splits}
    if len(split_map) != len(splits) or set(split_map) != set(subjects) or set(split_map.values()) != {"train", "validation", "test"}:
        raise ValueError("Segmentation split ledger lacks exclusive subject coverage")
    eligible_inputs = [row for row in verified if row["subject_id"] in labels]
    if len(eligible_inputs) != len(labels):
        raise ValueError("Eligible eyelid coverage is incomplete or repeated")
    counts = Counter(labels.values())
    summary = {
        "created_at": datetime.now(ZoneInfo("Asia/Manila")).isoformat(),
        "readiness": "PARTIALLY_READY", "training_status": "not_run",
        "metadata_source": snapshot, "source_files": initial_hashes,
        "subjects": len(rows), "subjects_with_numeric_hb": sum(bool(row["normalized_hb_g_per_dl"]) for row in eligibility),
        "labeled_eyelid_subjects": len(labels), "anemia": counts[1], "non_anemia": counts[0],
        "excluded_subjects": len(rows) - len(labels),
        "exclusion_counts": dict(Counter(reason for row in eligibility for reason in row["exclusion_reasons"].split(";") if reason)),
        "resolved_metadata_image_references": len(expected), "approved_eyelids": len(verified),
        "approved_labeled_eyelids": len(eligible_inputs), "file_hash_checks": len(verified) * 4,
        "review_preview_mask_pixel_matches": len(verified), "prior_metadata_and_label_discrepancies": 0,
        "exact_source_duplicate_groups": original_duplicates, "exact_rendered_pixel_duplicate_groups": pixel_duplicates,
        "perceptual_candidate_pairs": len(existing_candidates),
        "perceptual_screen_rule": "Full TIFF RGB to grayscale, Lanczos 32x32, scipy.fftpack two-axis DCT, 8x8 median bits, Hamming <=6; review candidates only",
        "candidate_comparison_pdf": str(CANDIDATE_DIR / "manual_eyelid_candidate_comparisons.pdf"),
        "candidates_crossing_old_cv_folds": sum(row["subject_id_a"] in labels and row["subject_id_b"] in labels and fold_map[row["subject_id_a"]] != fold_map[row["subject_id_b"]] for row in existing_candidates),
        "candidates_crossing_segmentation_partitions": sum(split_map[row["subject_id_a"]] != split_map[row["subject_id_b"]] for row in existing_candidates),
        "historical_prepared_fold_counts": fold_counts,
        "segmentation_split_counts": {split: {
            "all_subjects": sum(value == split for value in split_map.values()),
            "labeled_anemia": sum(labels[sid] == 1 and split_map[sid] == split for sid in labels),
            "labeled_non_anemia": sum(labels[sid] == 0 and split_map[sid] == split for sid in labels),
        } for split in ("train", "validation", "test")},
        "render_method_counts": dict(methods),
        "stored_profile_groups": [{"description": key[0], "sha256": key[1], "count": count} for key, count in profiles.items()],
        "blockers": ["human_duplicate_adjudication_pending"],
        "warnings": ["mixed_rendering_histories", "pilot_render_settings_unrecorded", "anatomical_review_is_not_clinician_validation", "small_imbalanced_internal_cohort"],
        "historical_artifact_note": "Old mask counts/training wording are stale. Historical ten-fold and segmentation partitions are audited references, not the current classifier design.",
    }
    if original_duplicates or pixel_duplicates:
        summary["blockers"].append("exact_duplicate_identity_decision_required")
    # Ensure no file recorded as a source changed during this read-only audit.
    if any(sha256(Path(path)) != digest for path, digest in summary["source_files"].items()):
        raise ValueError("A source manifest changed during readiness auditing")
    return summary, eligibility, eligible_inputs, existing_candidates


def refresh(output: Path) -> dict[str, Any]:
    """Save a new ignored bundle with an empty human duplicate-review form."""
    output = require_private_output(output)
    summary, eligibility, inputs, candidates = audit()
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "all_subject_eligibility.csv", eligibility, list(eligibility[0]))
    write_csv(output / "verified_eyelid_inputs.csv", inputs, list(inputs[0]))
    review_fields = list(candidates[0]) + ["decision", "reviewer", "reviewed_on", "evidence_note"]
    write_csv(output / "duplicate_review.csv", candidates, review_fields)
    summary["bundle_files"] = {path.name: sha256(path) for path in output.glob("*.csv")}
    summary["code_sha256"] = {path.name: sha256(path) for path in (
        Path(__file__), ROOT / "src/local_eyelid_validation.py", ROOT / "src/local_who2024_cohort.py",
    )}
    (output / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "who_label_config.json").write_bytes((OLD_COHORT / "who_label_config.json").read_bytes())
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Create a new versioned private bundle; omit for check-only")
    args = parser.parse_args()
    try:
        summary = refresh(args.output) if args.output else audit()[0]
        print(json.dumps(summary, indent=2))
    except (ValueError, OSError, KeyError, csv.Error):
        print(json.dumps({"readiness": "BLOCKED", "error": "Integrity audit failed; no training is permitted"}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
