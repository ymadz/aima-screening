"""Build the frozen local WHO-2024 cohort; never train on unreviewed masks.

Outputs are ignored patient-level research artifacts. No source files are changed.
Run from the repository root with ``--nonpregnant-attested`` only after the
study owner has explicitly confirmed pregnancy status outside the Sheet.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import median

from PIL import Image
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "data/raw/local/metadata/frozen_2026-10-07/AIMA Patients Metadata - Sheet1 values.csv"
PROVENANCE = FROZEN.parent / "provenance.json"
PRIOR_MANIFEST = ROOT / "data/raw/local/manifest/frozen_2026-10-07/local_subject_manifest.csv"
INCOMING = ROOT / "data/raw/local/incoming"
ANNOTATIONS = ROOT / "data/raw/local/annotations"
ANNOTATION_MANIFEST = ANNOTATIONS / "manifests/annotation_manifest.csv"
OUTPUT = ROOT / "data/processed/local_cohort_who2024_2026-10-08"
WHO_URL = "https://iris.who.int/bitstream/handle/10665/376196/9789240088542-eng.pdf"
EXPECTED_HEADER = [
    "subject_id", "age", "sex", "lab_hb_g_per_l", "lab_hb_g_per_dl",
    "anemia_status", "raw_eyelid_file", "raw_nail_file", "notes", "source",
]
FROZEN_FIELDS = [
    "subject_id", "age", "sex", "lab_hb_g_per_l", "lab_hb_g_per_dl",
    "anemia_status", "raw_eyelid_file", "raw_nail_file", "notes", "source",
]
THRESHOLDS_G_DL = {"M": Decimal("13.0"), "F": Decimal("12.0")}
SEED = 20261007


def sha256(path: Path) -> str:
    """Return the SHA-256 of a source or annotation file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a strict CSV, rejecting trailing columns and malformed rows."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Missing or duplicate CSV header: {path}")
        rows = list(reader)
        if any(None in row or None in row.values() for row in rows):
            raise ValueError(f"Trailing or missing CSV columns: {path}")
        if any(not any(value.strip() for value in row.values()) for row in rows):
            raise ValueError(f"Blank data row: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    """Write a private derived table with an explicit stable schema."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def decimal_or_none(value: str, field: str, subject_id: str) -> Decimal | None:
    """Parse a recorded Hb cell without conflating empty and zero."""
    if not value.strip():
        return None
    try:
        result = Decimal(value.strip())
    except InvalidOperation as exc:
        raise ValueError(f"Invalid {field} for {subject_id}") from exc
    upper_bound = Decimal("300") if field.endswith("g_per_l") else Decimal("30")
    if not result.is_finite() or not Decimal("0") < result < upper_bound:
        raise ValueError(f"Out-of-range {field} for {subject_id}")
    return result


def normalized_hb(row: dict[str, str]) -> tuple[Decimal | None, str, str, str]:
    """Return Hb in g/dL and the exact recorded column/value/unit used.

    The Sheet does not identify the original laboratory reporting unit. The
    returned provenance describes a recorded Sheet column, not the lab source.
    """
    subject_id = row["subject_id"]
    gl = decimal_or_none(row["lab_hb_g_per_l"], "lab_hb_g_per_l", subject_id)
    gdl = decimal_or_none(row["lab_hb_g_per_dl"], "lab_hb_g_per_dl", subject_id)
    if gl is not None and gdl is not None and gl / 10 != gdl:
        raise ValueError(f"Hb g/L and g/dL disagree for {subject_id}")
    if gl is not None:
        return gl / 10, "lab_hb_g_per_l", str(gl), "g/L"
    if gdl is not None:
        return gdl, "lab_hb_g_per_dl", str(gdl), "g/dL"
    return None, "", "", ""


def who_label(hb_g_dl: Decimal, sex: str, age: int,
              nonpregnant_attested: bool) -> int:
    """Apply unadjusted WHO 2024 adult anaemia cutoffs (strictly below)."""
    if not 15 <= age <= 65 or sex not in THRESHOLDS_G_DL:
        raise ValueError("WHO 15-65 threshold cannot be safely applied")
    if sex == "F" and not nonpregnant_attested:
        raise ValueError("Non-pregnant status must be independently attested")
    return int(hb_g_dl < THRESHOLDS_G_DL[sex])


def checked_relative(base: Path, relative: str) -> Path:
    """Resolve a manifest path only when it remains inside its source folder."""
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f"Manifest path escapes source folder: {relative}")
    if not path.is_file():
        raise ValueError(f"Missing annotation artifact: {path}")
    return path


def validate_mask(mask_path: Path, image_path: Path) -> None:
    """Check reviewed binary PNG geometry against its annotation TIFF."""
    with Image.open(image_path) as source, Image.open(mask_path) as mask:
        if source.size != mask.size or source.mode != "RGB":
            raise ValueError(f"Image/mask dimensions or image mode differ: {mask_path}")
        if mask.mode != "L":
            raise ValueError(f"Mask must be 8-bit grayscale: {mask_path}")
        histogram = mask.histogram()
        if sum(histogram[1:255]) or not histogram[255]:
            raise ValueError(f"Mask is not a nonempty 0/255 binary PNG: {mask_path}")


def load_masks(subjects: dict[str, dict[str, str]],
               source_hash: str) -> list[dict[str, str]]:
    """Load only approved manual anatomical ROIs/masks, never proposals."""
    masks: list[dict[str, str]] = []
    for row in read_csv(ANNOTATION_MANIFEST):
        if row["annotation_status"] != "approved":
            continue
        subject_id, modality = row["subject_id"], row["modality"]
        if subject_id not in subjects or modality not in ("eyelid", "nail"):
            raise ValueError(f"Unexpected annotated subject/modality: {subject_id}/{modality}")
        expected = subjects[subject_id][f"raw_{modality}_file"]
        if row["original_image_filename"] != expected:
            raise ValueError(f"Annotation/source filename mismatch: {subject_id}/{modality}")
        if row["reviewed_status"] != "approved" or row["metadata_source_sha256"] != source_hash:
            raise ValueError(f"Annotation is not approved or has stale metadata provenance: {subject_id}/{modality}")
        source = checked_relative(ROOT, row["annotation_image_path"])
        mask = checked_relative(ROOT, row["mask_path"])
        roi = checked_relative(ROOT, row["roi_path"])
        if mask.suffix.lower() != ".png" or source.suffix.lower() not in (".tif", ".tiff"):
            raise ValueError(f"Unexpected annotation format: {mask}")
        if sha256(mask) != row["mask_sha256"]:
            raise ValueError(f"Annotation mask hash mismatch: {mask}")
        if sha256(source) != row["annotation_image_sha256"]:
            raise ValueError(f"Annotation image hash mismatch: {source}")
        validate_mask(mask, source)
        masks.append({
            "subject_id": subject_id, "modality": modality,
            "original_filename": expected, "annotation_image_path": str(source),
            "anatomical_roi_path": str(roi), "reviewed_mask_path": str(mask),
            "reviewed_mask_sha256": row["mask_sha256"],
            "anatomical_mask_status": "manually_annotated_reviewed",
            "glare_status": row["glare_status"],
            "annotation_manifest_path": str(ANNOTATION_MANIFEST),
            "metadata_source_path": str(FROZEN), "metadata_source_sha256": source_hash,
        })
    keys = [(m["subject_id"], m["modality"]) for m in masks]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate reviewed mask per subject/modality")
    return masks


def hb_summary(values: list[Decimal]) -> dict[str, str]:
    """Describe normalized g/dL Hb without dropping the continuous variable."""
    ordered = sorted(values)
    return {
        "n": str(len(ordered)), "min_g_per_dl": str(ordered[0]),
        "median_g_per_dl": str(median(ordered)), "max_g_per_dl": str(ordered[-1]),
    }


def annotation_queue(cohort: list[dict[str, object]],
                     masks_by_key: dict[tuple[str, str], dict[str, str]],
                     fold_for: dict[str, int], source_hash: str) -> list[dict[str, str]]:
    """Choose the minimum 9 additional positive eyelids for 10-fold feasibility.

    Select one positive in every fold without a reviewed positive. This is an
    annotation-priority list, not a population-representative sample.
    """
    candidates = [row for row in cohort if row["derived_anemia_label"] == 1
                  and (str(row["subject_id"]), "eyelid") not in masks_by_key]
    covered_folds = {fold_for[str(row["subject_id"])] for row in cohort
                     if row["derived_anemia_label"] == 1
                     and (str(row["subject_id"]), "eyelid") in masks_by_key}
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        grouped[fold_for[str(row["subject_id"])]].append(row)
    rng = random.Random(SEED)
    annotation_manifest = {
        (row["subject_id"], row["modality"]): row
        for row in read_csv(ANNOTATION_MANIFEST)
    }
    chosen: list[dict[str, object]] = []
    for fold in range(1, 11):
        if fold in covered_folds:
            continue
        pool = grouped[fold]
        if not pool:
            raise ValueError(f"No anemia-positive eyelid candidate for fold {fold}")
        chosen.append(rng.choice(sorted(pool, key=lambda r: str(r["subject_id"]))))
    queue: list[dict[str, str]] = []
    for row in sorted(chosen, key=lambda r: str(r["subject_id"])):
        original = str(row["raw_eyelid_file"])
        annotation_row = annotation_manifest.get((str(row["subject_id"]), "eyelid"))
        if annotation_row is None or annotation_row["image_readiness_status"] != "standardized_TIFF_ready":
            raise ValueError(f"No standardized annotation TIFF: {row['subject_id']}")
        tiff = checked_relative(ROOT, annotation_row["annotation_image_path"])
        if tiff.suffix.lower() not in (".tif", ".tiff"):
            raise ValueError(f"Unexpected annotation TIFF path: {tiff}")
        queue.append({
            "subject_id": str(row["subject_id"]), "sex": str(row["sex"]),
            "fold": str(fold_for[str(row["subject_id"])]),
            "original_eyelid_filename": original,
            "annotation_tiff_path": str(tiff),
            "preparation_status": "ready_to_annotate_manual_primary",
            "selection_note": "one anemia-positive eyelid in each uncovered exploratory CV fold",
            "metadata_source_path": str(FROZEN), "metadata_source_sha256": source_hash,
        })
    return queue


def build(nonpregnant_attested: bool, attestation_source: str,
          output_dir: Path = OUTPUT) -> dict[str, object]:
    """Validate the frozen Sheet and emit cohort, masks, folds and blockers."""
    if not nonpregnant_attested or not attestation_source.strip():
        raise ValueError("Explicit non-pregnancy attestation and its source are required")
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    source_hash = sha256(FROZEN)
    if source_hash != provenance["frozen_csv_sha256"]:
        raise ValueError("Frozen Sheet SHA-256 differs from freeze provenance")
    with FROZEN.open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle))
    if header != EXPECTED_HEADER or provenance["nonempty_cells_beyond_column_j"]:
        raise ValueError("Unexpected frozen metadata header or trailing nonblank columns")
    source_rows = read_csv(FROZEN)
    if len(source_rows) != provenance["data_rows"]:
        raise ValueError("Frozen Sheet count differs from provenance")
    subjects = {r["subject_id"]: r for r in source_rows}
    if len(subjects) != len(source_rows) or any(not sid for sid in subjects):
        raise ValueError("Blank or duplicate subject ID")
    prior = {r["subject_id"]: r for r in read_csv(PRIOR_MANIFEST)}
    if set(prior) != set(subjects) or len(prior) != len(source_rows):
        raise ValueError("Frozen Sheet/previous manifest subject mismatch")
    for subject_id, row in subjects.items():
        if any(row[field] != prior[subject_id][field] for field in FROZEN_FIELDS):
            raise ValueError(f"Frozen Sheet/previous manifest values differ: {subject_id}")
        if prior[subject_id]["metadata_source_sha256"] != source_hash:
            raise ValueError(f"Prior manifest source hash mismatch: {subject_id}")

    incoming_names = Counter(path.name for path in INCOMING.iterdir() if path.is_file())
    masks = load_masks(subjects, source_hash)
    masks_by_key = {(m["subject_id"], m["modality"]): m for m in masks}
    all_rows: list[dict[str, object]] = []
    for row in source_rows:
        subject_id = row["subject_id"]
        reasons: list[str] = []
        try:
            age = int(row["age"])
        except ValueError:
            age = -1
            reasons.append("invalid_age")
        if age < 0 or not 15 <= age <= 65:
            reasons.append("outside_who_15_65" if age >= 0 else "unusable_age")
        sex = row["sex"].strip().upper()
        if sex not in THRESHOLDS_G_DL:
            reasons.append("sex_not_supported_by_rule")
        hb_g_dl, source_col, source_value, source_unit = normalized_hb(row)
        if hb_g_dl is None:
            reasons.append("missing_laboratory_hb")
        if row["notes"].strip().lower() == "not_cbc" and hb_g_dl is not None:
            raise ValueError(f"Numeric Hb conflicts with not_cbc note: {subject_id}")
        if row["anemia_status"].strip():
            raise ValueError(f"Unexpected nonblank source anemia_status: {subject_id}")
        image_status: dict[str, str] = {}
        for modality in ("eyelid", "nail"):
            filename = row[f"raw_{modality}_file"].strip()
            if not filename:
                image_status[modality] = "blank_reference"
            elif Path(filename).name != filename or incoming_names[filename] != 1:
                image_status[modality] = "missing_or_ambiguous"
            else:
                image_status[modality] = "exact_unique_match"
            expected_prior = prior[subject_id][f"{modality}_reference_status"]
            if image_status[modality] != expected_prior:
                raise ValueError(f"Image resolution differs from frozen manifest: {subject_id}/{modality}")
        if all(status != "exact_unique_match" for status in image_status.values()):
            reasons.append("no_resolved_image_modality")
        eligible = not reasons
        label = who_label(hb_g_dl, sex, age, nonpregnant_attested) if eligible else ""
        all_rows.append({
            **row, "sex": sex, "primary_recorded_hb_value": source_value,
            "primary_recorded_hb_unit": source_unit, "primary_recorded_hb_column": source_col,
            "original_laboratory_reporting_unit": "not_recorded",
            "normalized_hb_g_per_dl": str(hb_g_dl) if hb_g_dl is not None else "",
            "derived_anemia_label": label,
            "derived_anemia_definition": "WHO_2024_unadjusted_adult_15_65" if eligible else "",
            "eligibility_status": "eligible" if eligible else "excluded",
            "exclusion_reasons": ";".join(dict.fromkeys(reasons)),
            "eyelid_reference_status": image_status["eyelid"],
            "nail_reference_status": image_status["nail"],
            "eyelid_reviewed_mask_path": masks_by_key.get((subject_id, "eyelid"), {}).get("reviewed_mask_path", ""),
            "nail_reviewed_mask_path": masks_by_key.get((subject_id, "nail"), {}).get("reviewed_mask_path", ""),
            "metadata_source_path": str(FROZEN), "metadata_source_sha256": source_hash,
        })
    cohort = [r for r in all_rows if r["eligibility_status"] == "eligible"]
    if len(cohort) != len({r["subject_id"] for r in cohort}):
        raise ValueError("Cohort has duplicated subjects")
    if Counter(r["derived_anemia_label"] for r in cohort)[1] < 10:
        raise ValueError("Full labeled cohort lacks ten anemia-positive subjects")
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    ordered = sorted(cohort, key=lambda r: str(r["subject_id"]))
    fold_rows: list[dict[str, object]] = []
    fold_for: dict[str, int] = {}
    for fold_index, (_, validation_indices) in enumerate(
        folds.split(ordered, [r["derived_anemia_label"] for r in ordered]), start=1
    ):
        for index in validation_indices:
            row = ordered[index]
            fold_for[str(row["subject_id"])] = fold_index
    for row in ordered:
        fold_rows.append({
            "subject_id": row["subject_id"], "derived_anemia_label": row["derived_anemia_label"],
            "fold": fold_for[str(row["subject_id"])],
            "metadata_source_path": str(FROZEN), "metadata_source_sha256": source_hash,
        })
    if len(fold_for) != len(cohort):
        raise ValueError("Not every eligible subject has exactly one fold")
    modality_rows = {
        "eyelid": [r for r in cohort if r["eyelid_reference_status"] == "exact_unique_match"],
        "nail": [r for r in cohort if r["nail_reference_status"] == "exact_unique_match"],
        "paired": [r for r in cohort if r["eyelid_reference_status"] == "exact_unique_match"
                   and r["nail_reference_status"] == "exact_unique_match"],
    }
    reviewed_counts: dict[str, dict[str, int]] = {}
    for modality in ("eyelid", "nail"):
        mask_field = f"{modality}_reviewed_mask_path"
        subset = [r for r in modality_rows[modality] if r[mask_field]]
        reviewed_counts[modality] = {
            "subjects": len(subset),
            "anemia": sum(int(r["derived_anemia_label"]) for r in subset),
            "non_anemia": sum(1 - int(r["derived_anemia_label"]) for r in subset),
        }
    paired_masked = [r for r in modality_rows["paired"] if r["eyelid_reviewed_mask_path"]
                     and r["nail_reviewed_mask_path"]]
    reviewed_counts["paired"] = {
        "subjects": len(paired_masked),
        "anemia": sum(int(r["derived_anemia_label"]) for r in paired_masked),
        "non_anemia": sum(1 - int(r["derived_anemia_label"]) for r in paired_masked),
    }
    queue = annotation_queue(cohort, masks_by_key, fold_for, source_hash)
    label_counts = Counter(int(r["derived_anemia_label"]) for r in cohort)
    excluded = [r for r in all_rows if r["eligibility_status"] == "excluded"]
    reasons_count = Counter(reason for row in excluded for reason in str(row["exclusion_reasons"]).split(";"))
    summary = {
        "metadata_source_path": str(FROZEN), "metadata_source_sha256": source_hash,
        "metadata_source_mtime_local": datetime.fromtimestamp(FROZEN.stat().st_mtime).astimezone().isoformat(),
        "total_subjects": len(all_rows),
        "subjects_with_usable_hb": sum(bool(r["normalized_hb_g_per_dl"]) for r in all_rows),
        "who_labeled_subjects": len(cohort), "anemia": label_counts[1], "non_anemia": label_counts[0],
        "anemia_fraction": round(label_counts[1] / len(cohort), 4),
        "non_anemia_to_anemia_ratio": round(label_counts[0] / label_counts[1], 2),
        "excluded_subjects": len(excluded), "exclusion_reason_counts": dict(reasons_count),
        "sex_distribution_labeled": dict(Counter(str(r["sex"]) for r in cohort)),
        "sex_by_label": {sex: dict(Counter(int(r["derived_anemia_label"]) for r in cohort
                                          if r["sex"] == sex)) for sex in ("F", "M")},
        "hb_distribution_all_labeled": hb_summary([Decimal(str(r["normalized_hb_g_per_dl"])) for r in cohort]),
        "hb_distribution_by_label": {
            str(label): hb_summary([Decimal(str(r["normalized_hb_g_per_dl"])) for r in cohort
                                    if r["derived_anemia_label"] == label]) for label in (0, 1)
        },
        "modality_metadata_eligible": {key: len(value) for key, value in modality_rows.items()},
        "reviewed_anatomical_mask_counts": reviewed_counts,
        "ten_fold_seed": SEED,
        "fold_class_counts": {
            str(fold): {
                "anemia": sum(r["derived_anemia_label"] == 1 and r["fold"] == fold
                              for r in fold_rows),
                "non_anemia": sum(r["derived_anemia_label"] == 0 and r["fold"] == fold
                                  for r in fold_rows),
            } for fold in range(1, 11)
        },
        "training_status": "not_run_insufficient_reviewed_positive_masks",
    }
    label_config = {
        "name": "WHO_2024_unadjusted_adult_15_65", "who_primary_source": WHO_URL,
        "threshold_rule": "anemia iff Hb strictly below sex-specific threshold",
        "thresholds_g_per_dl": {"M": "13.0", "F_nonpregnant": "12.0"},
        "age_years_inclusive": [15, 65],
        "pregnancy_status_source": "study_owner_attestation_outside_frozen_sheet",
        "pregnancy_attestation": attestation_source,
        "source_anemia_status_field": "blank_in_frozen_sheet",
        "original_lab_reporting_unit": "not_recorded; both g/L and g/dL Sheet fields retained",
        "hb_normalization": "g/L divided by 10, exact decimal equality checked against g/dL column",
        "smoking_elevation_adjustment": "not_applied; data_unavailable",
        "metadata_source_path": str(FROZEN), "metadata_source_sha256": source_hash,
    }
    training_config = {
        "status": "not_run", "modality_priority": "eyelid_and_nail_manual_ROIs",
        "segmentation_ground_truth_strategy": "manual anatomical ROIs and reviewed masks are primary; automated segmentation is optional future assistance only",
        "reason": "Only one anemia-positive eyelid mask and zero anemia-positive nail/paired masks are currently approved for the WHO-labeled cohort",
        "evaluation_plan": "preallocated subject-level stratified 10-fold CV, seed 20261007; no training run yet",
        "fold_assignment_scope": "119 metadata-eligible subjects; subset class counts must be rechecked before any run",
        "next_manual_annotation_priority": f"{len(queue)} anemia-positive eyelid images for exploratory fold coverage; full queue is under data/raw/local/annotations/manifests/pending_annotation_queue.csv",
        "performance_scope_warning": "Small enriched pilot cannot estimate population screening performance",
        "glare_policy": "Do not use batch-2 unreviewed glare candidates; settle eyelid glare QC before training",
        "metrics": "none; no model trained",
        "metadata_source_path": str(FROZEN), "metadata_source_sha256": source_hash,
    }
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True)
    fields = list(all_rows[0])
    write_csv(output_dir / "all_subject_eligibility.csv", all_rows, fields)
    write_csv(output_dir / "labeled_cohort.csv", cohort, fields)
    write_csv(output_dir / "exclusions.csv", excluded, fields)
    for modality, rows in modality_rows.items():
        write_csv(output_dir / f"{modality}_metadata_eligible.csv", rows, fields)
    write_csv(output_dir / "reviewed_anatomical_masks.csv", masks, list(masks[0]))
    write_csv(output_dir / "subject_folds.csv", fold_rows, list(fold_rows[0]))
    write_csv(output_dir / "cv_balance_shortlist.csv", queue, list(queue[0]))
    for filename, content in (("summary.json", summary), ("who_label_config.json", label_config),
                              ("training_config.json", training_config)):
        (output_dir / filename).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nonpregnant-attested", action="store_true")
    parser.add_argument("--attestation-source", required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.nonpregnant_attested, args.attestation_source,
                           args.output_dir), indent=2))
