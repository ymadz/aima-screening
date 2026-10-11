"""Fail-closed validation for local manual-eyelid classification.

These helpers never approve annotations, change labels, assign partitions, or
resolve duplicate identity. Error messages contain no patient identifiers.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image
from scipy.fftpack import dct


def sha256(path: Path) -> str:
    """Hash file bytes without decoding or modifying the source."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_csv(path: Path) -> list[dict[str, str]]:
    """Reject blank/duplicate headers, blank rows, and extra/missing cells."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or any(not field.strip() for field in header):
            raise ValueError("CSV header is missing or blank")
        if len(header) != len(set(header)):
            raise ValueError("CSV header contains duplicate fields")
        values = list(reader)
    if any(len(row) != len(header) for row in values):
        raise ValueError("CSV contains extra or missing cells")
    rows = [dict(zip(header, row)) for row in values]
    if any(not any(value.strip() for value in row.values()) for row in rows):
        raise ValueError("CSV contains a blank data row")
    return rows


def source_snapshot(path: Path, provenance_path: Path) -> dict[str, Any]:
    """Capture mtime/size/hash before any metadata-row read, then verify freeze."""
    if not path.is_file() or not provenance_path.is_file():
        raise ValueError("Canonical source or freeze provenance is missing")
    stat = path.stat()
    snapshot = {
        "path": str(path.resolve()),
        "mtime": datetime.fromtimestamp(stat.st_mtime, ZoneInfo("Asia/Manila")).isoformat(),
        "size_bytes": stat.st_size,
        "sha256": sha256(path),
        "provenance_path": str(provenance_path.resolve()),
        "provenance_sha256": sha256(provenance_path),
    }
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if snapshot["sha256"] != provenance.get("frozen_csv_sha256"):
        raise ValueError("Canonical source differs from recorded freeze hash")
    return snapshot


def checked_file(root: Path, value: str, expected_hash: str | None = None) -> Path:
    """Resolve only existing files inside the repository and verify their bytes."""
    if not value.strip():
        raise ValueError("Required artifact path is blank")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Artifact is missing or escapes repository")
    if expected_hash is not None and (not expected_hash or sha256(path) != expected_hash):
        raise ValueError("Artifact hash differs from recorded provenance")
    return path


def validate_arrays(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Require full-size uint8 RGB and a nonempty, non-full binary 0/255 mask."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
        raise ValueError("Input must be an 8-bit RGB array")
    if mask.ndim != 2 or mask.shape != image_rgb.shape[:2] or mask.dtype != np.uint8:
        raise ValueError("Mask geometry or type differs from image")
    if not np.isin(mask, [0, 255]).all():
        raise ValueError("Anatomical mask contains nonbinary values")
    selected = mask == 255
    if not selected.any() or selected.all():
        raise ValueError("Anatomical mask is empty or covers the entire image")
    return selected


def load_manual_pair(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Decode without recoloring, resizing, EXIF transforms, or mask changes."""
    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        if image.format != "TIFF" or image.mode != "RGB" or mask.mode != "L":
            raise ValueError("Expected RGB TIFF and grayscale anatomical mask")
        if image.tag_v2.get(274, 1) != 1:
            raise ValueError("Annotation TIFF has unresolved orientation")
        if getattr(image, "n_frames", 1) != 1 or getattr(mask, "n_frames", 1) != 1:
            raise ValueError("Multi-frame inputs require an explicit selection policy")
        image_array, mask_array = np.asarray(image), np.asarray(mask)
    validate_arrays(image_array, mask_array)
    return image_array, mask_array


def validate_eyelid_link(
    record: dict[str, str], review: dict[str, str], source: dict[str, str], source_hash: str,
) -> None:
    """Join exact subject, original filename, paths and hashes to human approval."""
    if record.get("modality") != "eyelid" or review.get("modality") != "eyelid":
        raise ValueError("Only eyelid records belong in this classifier")
    if record.get("annotation_status") != "approved" or record.get("reviewed_status") != "approved":
        raise ValueError("Anatomical input is not approved")
    if review.get("review_status") != "approved":
        raise ValueError("Human anatomical approval is absent")
    if not all(review.get(key, "").strip() for key in (
        "reviewer", "review_method", "review_scope", "source_review_package_path",
        "source_review_package_sha256",
    )):
        raise ValueError("Human approval evidence is incomplete")
    if record.get("subject_id") != source.get("subject_id") or review.get("subject_id") != source.get("subject_id"):
        raise ValueError("Subject linkage differs across sources")
    if record.get("original_image_filename") != source.get("raw_eyelid_file"):
        raise ValueError("Exact original filename differs from metadata")
    if Path(record.get("source_image_path", "")).name != source.get("raw_eyelid_file"):
        raise ValueError("Original path does not resolve the metadata filename")
    pairs = [(key, key) for key in (
        "original_image_filename", "source_image_path", "annotation_image_path",
        "annotation_image_sha256", "roi_path", "roi_sha256", "mask_path", "mask_sha256",
    )] + [("original_source_sha256", "source_image_sha256")]
    if any(not record.get(left) or record.get(left) != review.get(right) for left, right in pairs):
        raise ValueError("Image/ROI/mask lineage differs from human review")
    if any(row.get("metadata_source_sha256") != source_hash for row in (record, review)):
        raise ValueError("Annotation or review uses stale metadata")


def perceptual_hash(image_rgb: np.ndarray) -> np.ndarray:
    """Reproduce the existing full-frame 64-bit pHash review screen.

    This is a review aid, never an identity or duplicate determination.
    """
    gray = Image.fromarray(image_rgb).convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    coefficients = dct(dct(np.asarray(gray, dtype=np.float64), axis=0), axis=1)[:8, :8]
    return (coefficients > np.median(coefficients)).ravel()


def audit_folds(labels: dict[str, int], assignments: list[dict[str, str]]) -> dict[str, Any]:
    """Validate an existing fold table; never create or approve a split."""
    if not labels or set(labels.values()) != {0, 1}:
        raise ValueError("Evaluation cohort must contain both classes")
    groups: dict[str, str] = {}
    for row in assignments:
        subject, fold = row["subject_id"], row["fold"]
        if subject in groups or subject not in labels or not fold:
            raise ValueError("Fold table duplicates, omits, or adds subjects")
        if int(row["derived_anemia_label"]) != labels[subject]:
            raise ValueError("Fold label differs from verified source label")
        groups[subject] = fold
    if set(groups) != set(labels) or len(set(groups.values())) < 2:
        raise ValueError("Fold table lacks exact cohort coverage or multiple folds")
    report = {}
    for fold in sorted(set(groups.values()), key=lambda value: int(value)):
        counts = Counter(labels[sid] for sid in groups if groups[sid] == fold)
        training = Counter(labels[sid] for sid in groups if groups[sid] != fold)
        if not all(counts[label] and training[label] for label in (0, 1)):
            raise ValueError("A training or evaluation fold lacks a required class")
        report[fold] = {"subjects": sum(counts.values()), "anemia": counts[1], "non_anemia": counts[0],
                        "training_anemia": training[1], "training_non_anemia": training[0]}
    return report


def require_duplicate_clearance(
    candidates: list[dict[str, str]], reviews: list[dict[str, str]],
) -> None:
    """Require explicit human distinct-source adjudication of every candidate.

    Confirmed/uncertain duplicates require a new coordinator decision; they
    cannot be automatically deleted, grouped, or reassigned here.
    """
    expected = {row["candidate_id"]: row for row in candidates}
    actual = {row.get("candidate_id", ""): row for row in reviews}
    if len(expected) != len(candidates) or len(actual) != len(reviews) or set(actual) != set(expected):
        raise ValueError("Duplicate review lacks exact candidate coverage")
    for candidate_id, candidate in expected.items():
        review = actual[candidate_id]
        if review.get("decision") != "distinct_sources":
            raise ValueError("Duplicate candidate is unreviewed, uncertain, or confirmed")
        if not all(review.get(key, "").strip() for key in ("reviewer", "reviewed_on", "evidence_note")):
            raise ValueError("Duplicate adjudication lacks human review evidence")
        for side in ("a", "b"):
            for key in ("subject_id", "original_source_sha256", "annotation_image_sha256"):
                field = f"{key}_{side}"
                if not candidate.get(field) or review.get(field) != candidate[field]:
                    raise ValueError("Duplicate adjudication refers to different image bytes")
