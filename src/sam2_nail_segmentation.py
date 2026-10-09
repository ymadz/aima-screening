"""Local SAM 2.1 nail-plate segmentation proposal and leave-one-subject-out audit.

The current ten reviewed nail masks train a small pixel model that supplies an
automatic box prompt. SAM 2.1 refines that prompt. All outputs are predictions
for human review and are kept outside the authoritative annotation manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import torch

from . import segmentation_prototype as prototype


ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "data/raw/local/annotations"
MANIFEST = ANNOTATIONS / "manifests/annotation_manifest.csv"
FROZEN_METADATA = ROOT / "data/raw/local/metadata/frozen_2026-10-07/AIMA Patients Metadata - Sheet1 values.csv"
SAM2_SOURCE = ROOT / "data/processed/segmentation/sam2_source"
SAM2_CHECKPOINT = SAM2_SOURCE / "checkpoints/sam2.1_hiera_tiny.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
DEFAULT_OUTPUT = ROOT / "data/processed/segmentation/sam2_nail_lopo_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bbox_from_mask(mask: np.ndarray, padding_fraction: float = 0.15) -> np.ndarray:
    """Return a padded XYXY box around a nonempty coarse binary mask."""
    if mask.ndim != 2 or not mask.any():
        raise ValueError("Cannot create a SAM prompt box from an empty mask")
    if not 0 <= padding_fraction <= 1:
        raise ValueError("padding_fraction must be between zero and one")
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad_x = max(2, round((x1 - x0) * padding_fraction))
    pad_y = max(2, round((y1 - y0) * padding_fraction))
    return np.array(
        [max(0, x0 - pad_x), max(0, y0 - pad_y),
         min(width, x1 + pad_x), min(height, y1 + pad_y)],
        dtype=np.float32,
    )


def points_from_mask(mask: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Make one interior positive and four nearby exterior negative prompts."""
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2 or not binary.any():
        raise ValueError("Cannot create point prompts from an empty mask")
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    positive_y, positive_x = np.unravel_index(int(distance.argmax()), distance.shape)
    height, width = binary.shape
    x0, y0, x1, y1 = [int(round(value)) for value in box]
    desired = [
        (max(0, min(width - 1, (x0 + x1) // 2)), max(0, min(height - 1, y0))),
        (max(0, min(width - 1, (x0 + x1) // 2)), max(0, min(height - 1, y1 - 1))),
        (max(0, min(width - 1, x0)), max(0, min(height - 1, (y0 + y1) // 2))),
        (max(0, min(width - 1, x1 - 1)), max(0, min(height - 1, (y0 + y1) // 2))),
    ]
    negatives: list[tuple[int, int]] = []
    for target_x, target_y in desired:
        candidates = sorted(
            ((x, y) for y in range(max(0, target_y - 24), min(height, target_y + 25))
             for x in range(max(0, target_x - 24), min(width, target_x + 25))
             if binary[y, x] == 0),
            key=lambda point: (point[0] - target_x) ** 2 + (point[1] - target_y) ** 2,
        )
        if not candidates:
            raise ValueError("Could not find a negative point near the predicted mask")
        negatives.append(candidates[0])
    coords = np.array([(positive_x, positive_y), *negatives], dtype=np.float32)
    labels = np.array([1, 0, 0, 0, 0], dtype=np.int32)
    return coords, labels


def binary_scores(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted = prediction.astype(bool)
    reference = target.astype(bool)
    if predicted.shape != reference.shape:
        raise ValueError("Prediction and reference mask shapes differ")
    tp = int(np.count_nonzero(predicted & reference))
    fp = int(np.count_nonzero(predicted & ~reference))
    fn = int(np.count_nonzero(~predicted & reference))
    union = tp + fp + fn
    denominator = 2 * tp + fp + fn
    return {
        "dice": 2 * tp / denominator if denominator else 1.0,
        "iou": tp / union if union else 1.0,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
    }


def largest_component(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Keep the largest 8-connected region and report the raw component count."""
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2:
        raise ValueError("A segmentation mask must be two-dimensional")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return np.zeros_like(binary, dtype=bool), 0
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_label, count - 1


def _load_sam_predictor(checkpoint: Path, device_name: str) -> tuple[Any, str]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM 2.1 checkpoint missing: {checkpoint}")
    device = (
        "mps" if device_name == "auto" and torch.backends.mps.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(
        SAM2_CONFIG, str(checkpoint), device=device, mode="eval",
        apply_postprocessing=True,
    )
    return SAM2ImagePredictor(model), device


def _infer_box_prompt(
    predictor: Any, image_rgb: np.ndarray, coarse: np.ndarray,
    device: str, prompt_mode: str,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    box = bbox_from_mask(coarse)
    if prompt_mode not in {"box_only", "box_plus_points"}:
        raise ValueError(f"Unsupported prompt mode: {prompt_mode}")
    points, labels = points_from_mask(coarse, box) if prompt_mode == "box_plus_points" else (None, None)
    if device == "mps":
        torch.mps.synchronize()
    started = time.perf_counter()
    predictor.set_image(image_rgb)
    masks, quality, _ = predictor.predict(
        box=box, point_coords=points, point_labels=labels,
        multimask_output=False, return_logits=False,
    )
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    prediction = np.asarray(masks[0], dtype=bool)
    if prediction.shape != image_rgb.shape[:2]:
        raise ValueError("SAM 2.1 returned an unexpected mask geometry")
    return prediction, elapsed, box, float(np.asarray(quality).reshape(-1)[0])


def _read_manifest() -> list[dict[str, str]]:
    source_hash = sha256(FROZEN_METADATA)
    with MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(row["metadata_source_sha256"] != source_hash for row in rows):
        raise ValueError("Annotation manifest does not match frozen metadata")
    return rows


def _approved_training_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    approved = [row for row in rows if row["modality"] == "nail"
                and row["annotation_status"] == "approved"]
    if len(approved) != 10 or len({row["subject_id"] for row in approved}) != 10:
        raise ValueError("Expected exactly ten unique reviewed nail references")
    for row in approved:
        image = ROOT / row["annotation_image_path"]
        roi = ROOT / row["roi_path"]
        mask = ROOT / row["mask_path"]
        if any(not path.is_file() for path in (image, roi, mask)):
            raise FileNotFoundError("A reviewed nail image/ROI/mask reference is missing")
        if (sha256(image) != row["annotation_image_sha256"]
                or sha256(roi) != row["roi_sha256"]
                or sha256(mask) != row["mask_sha256"]):
            raise ValueError("A reviewed nail image/ROI/mask hash differs from manifest")
    return sorted(approved, key=lambda row: row["subject_id"])


def _load_training_cache(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        row["subject_id"]: prototype._load_example({
            **row,
            "annotation_image": row["annotation_image_path"],
            "annotation_mask_path": row["mask_path"],
        })
        for row in rows
    }


def _reference_area_range(
    subject_ids: list[str], cache: dict[str, Any],
) -> tuple[float, float]:
    fractions = [float(cache[subject_id][1].mean()) for subject_id in subject_ids]
    if not fractions or min(fractions) <= 0:
        raise ValueError("Reviewed nail masks do not provide a valid area range")
    return min(fractions), max(fractions)


def _color_delta(image_bgr: np.ndarray, first: np.ndarray, second: np.ndarray) -> float | None:
    if not first.any() or not second.any():
        return None
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean_a = lab[first].mean(axis=0)
    mean_b = lab[second].mean(axis=0)
    return float(np.linalg.norm(mean_a - mean_b))


def _save_preview(
    image_bgr: np.ndarray, prediction: np.ndarray, path: Path,
    reference: np.ndarray | None = None,
) -> None:
    height, width = image_bgr.shape[:2]
    scale = min(1.0, 1200 / max(height, width))
    size = (round(width * scale), round(height * scale))
    preview = cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA)
    pred_small = cv2.resize(prediction.astype(np.uint8), size,
                            interpolation=cv2.INTER_NEAREST).astype(bool)
    overlay = preview.copy()
    overlay[pred_small] = (40, 200, 40)
    preview = cv2.addWeighted(preview, 0.62, overlay, 0.38, 0)
    if reference is not None:
        ref_small = cv2.resize(reference.astype(np.uint8), size,
                                interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(ref_small, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview, contours, -1, (255, 0, 255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), preview, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError(f"Unable to write preview: {path}")


def _save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (mask.astype(np.uint8) * 255)
    if not cv2.imwrite(str(path), encoded):
        raise OSError(f"Unable to write candidate mask: {path}")


def _record_prediction(
    row: dict[str, str], image_bgr: np.ndarray, coarse: np.ndarray,
    sam_mask: np.ndarray, raw_sam_mask: np.ndarray, raw_component_count: int,
    seconds: float, box: np.ndarray, reference_area_range: tuple[float, float],
    score: float, prompt_mode: str, output: Path, reference: np.ndarray | None,
    reference_path: Path | None,
) -> dict[str, Any]:
    stem = Path(row["original_image_filename"]).stem
    mask_path = output / "predicted_masks" / f"{stem}_sam2_nail_candidate.png"
    raw_mask_path = output / "predicted_masks" / f"{stem}_sam2_raw.png"
    preview_path = output / "previews" / f"{stem}_sam2_nail_overlay.jpg"
    _save_mask(sam_mask, mask_path)
    _save_mask(raw_sam_mask, raw_mask_path)
    _save_preview(image_bgr, sam_mask, preview_path, reference)
    components, labels, stats, _ = cv2.connectedComponentsWithStats(
        sam_mask.astype(np.uint8), 8
    )
    area = float(sam_mask.mean())
    coarse_area = float(coarse.mean())
    flags: list[str] = []
    if area < 0.001:
        flags.append("empty_or_tiny")
    if area > 0.15:
        flags.append("oversized")
    reference_min, reference_max = reference_area_range
    if area < reference_min * 0.5:
        flags.append("below_reviewed_area_range")
    if area > reference_max * 1.5:
        flags.append("above_reviewed_area_range")
    if components > 2:
        flags.append("fragmented")
    if (sam_mask[0].any() or sam_mask[-1].any() or
            sam_mask[:, 0].any() or sam_mask[:, -1].any()):
        flags.append("touches_frame_border")
    ratio = area / coarse_area if coarse_area else float("inf")
    if ratio < 0.25 or ratio > 4.0:
        flags.append("area_differs_from_prompt_mask")
    metrics = binary_scores(sam_mask, reference) if reference is not None else {}
    row_record: dict[str, Any] = {
        "subject_id": row["subject_id"],
        "modality": "nail",
        "original_image_filename": row["original_image_filename"],
        "source_image_path": row["source_image_path"],
        "original_source_sha256": row.get("original_source_sha256", ""),
        "annotation_image_path": row["annotation_image_path"],
        "annotation_image_sha256": row["annotation_image_sha256"],
        "prompt_method": f"fold_specific_extratrees_mask_bbox_15pct_padding_{prompt_mode}",
        "prompt_box_xyxy": json.dumps([float(v) for v in box]),
        "extratrees_prompt_area_fraction": coarse_area,
        "reviewed_reference_area_min_fraction": reference_area_range[0],
        "reviewed_reference_area_max_fraction": reference_area_range[1],
        "sam2_predicted_quality": score,
        "sam2_raw_component_count": raw_component_count,
        "postprocessing": "retain_largest_8_connected_component",
        "predicted_mask_path": str(mask_path.relative_to(ROOT)),
        "predicted_mask_sha256": sha256(mask_path),
        "raw_predicted_mask_path": str(raw_mask_path.relative_to(ROOT)),
        "raw_predicted_mask_sha256": sha256(raw_mask_path),
        "preview_path": str(preview_path.relative_to(ROOT)),
        "preview_sha256": sha256(preview_path),
        "predicted_area_fraction": area,
        "inference_seconds": seconds,
        "qc_flags": "|".join(flags),
        "status": "predicted_unverified",
        "reference_mask_path": str(reference_path.relative_to(ROOT)) if reference_path else "",
        "cie_lab_mean_delta_e76": _color_delta(image_bgr, sam_mask, reference)
        if reference is not None else "",
        **metrics,
    }
    return row_record


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty prediction table")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_run_metadata(
    output: Path, mode: str, checkpoint: Path, device: str,
    training_rows: list[dict[str, str]],
) -> dict[str, Any]:
    try:
        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=SAM2_SOURCE, check=True,
            text=True, capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = "unavailable"
    value = {
        "mode": mode,
        "status": "candidate_proposals_only",
        "device": device,
        "sam2_source_commit": source_commit,
        "sam2_checkpoint_path": str(checkpoint.relative_to(ROOT)),
        "sam2_checkpoint_sha256": sha256(checkpoint),
        "sam2_config": SAM2_CONFIG,
        "prompt_model": "separate ExtraTrees pixel classifier trained on approved nail masks",
        "approved_reference_pool_subject_ids": [row["subject_id"] for row in training_rows],
        "approved_reference_pool_mask_sha256": [row["mask_sha256"] for row in training_rows],
        "prediction_specific_training_subjects": (
            "leave-one-subject-out; the target subject is excluded from the ExtraTrees prompt model"
            if mode == "leave_one_subject_out_evaluation"
            else "all ten approved nail subjects"
        ),
        "sam2_nail_source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            text=True, capture_output=True,
        ).stdout.strip(),
        "sam2_nail_script_sha256": sha256(Path(__file__)),
        "mask_policy": "binary 0/255 candidate stored in ignored prediction folder; no manifest status changes",
        "area_qc_rule": "flag below half or above 1.5x the training-mask min/max area range; attention flag only",
        "color_caveat": "OpenCV LAB means on existing TIFF pixels; not calibrated across source rendering profiles",
    }
    (output / "run_metadata.json").write_text(json.dumps(value, indent=2) + "\n")
    return value


def evaluate_leave_one_out(
    output: Path = DEFAULT_OUTPUT, checkpoint: Path = SAM2_CHECKPOINT,
    device_name: str = "auto", prompt_mode: str = "box_plus_points",
) -> dict[str, Any]:
    """Compare fold-specific ExtraTrees masks and SAM 2.1 box refinements."""
    output = output.resolve()
    checkpoint = checkpoint.resolve()
    if not output.resolve().is_relative_to((ROOT / "data/processed").resolve()):
        raise ValueError("Local prediction output must stay under ignored data/processed")
    rows = _approved_training_rows(_read_manifest())
    cache = _load_training_cache(rows)
    predictor, device = _load_sam_predictor(checkpoint, device_name)
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for index, heldout in enumerate(rows):
        training = [row for row in rows if row["subject_id"] != heldout["subject_id"]]
        model = prototype._fit(training, cache)
        features, target, _ = cache[heldout["subject_id"]]
        started = time.perf_counter()
        coarse_small = prototype._predict(model, features, target.shape).astype(bool)
        annotation_path = ROOT / heldout["annotation_image_path"]
        image_bgr = prototype._read_image(annotation_path)
        full_size = image_bgr.shape[:2]
        coarse = cv2.resize(coarse_small.astype(np.uint8),
                            (full_size[1], full_size[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        reference = cv2.imread(str(ROOT / heldout["mask_path"]), cv2.IMREAD_GRAYSCALE) > 0
        if reference.shape != full_size:
            raise ValueError("Held-out reference mask does not align with TIFF")
        color_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        raw_prediction, sam_seconds, box, sam_quality = _infer_box_prompt(
            predictor, color_rgb, coarse, device, prompt_mode,
        )
        predicted, raw_component_count = largest_component(raw_prediction)
        et_scores = binary_scores(coarse, reference)
        raw_sam_scores = binary_scores(raw_prediction, reference)
        record = _record_prediction(
            heldout, image_bgr, coarse, predicted, raw_prediction,
            raw_component_count, sam_seconds, box,
            _reference_area_range([row["subject_id"] for row in training], cache),
            sam_quality,
            prompt_mode, output, reference, ROOT / heldout["mask_path"],
        )
        record.update({f"raw_sam_{key}": value
                       for key, value in raw_sam_scores.items()})
        record["prompt_training_subject_ids"] = json.dumps(
            [row["subject_id"] for row in training]
        )
        record["prompt_training_mask_sha256"] = json.dumps(
            [row["mask_sha256"] for row in training]
        )
        record.update({f"extratrees_{key}": value for key, value in et_scores.items()})
        record["sam2_inference_seconds"] = sam_seconds
        record["total_inference_seconds"] = time.perf_counter() - started
        records.append(record)
        if (index + 1) % 2 == 0:
            print(f"Evaluated {index + 1}/{len(rows)} held-out nail subjects", flush=True)
    _write_csv(output / "evaluation.csv", records)
    means = {
        f"mean_{key}": statistics.mean(float(row[key]) for row in records)
        for key in ("dice", "iou", "precision", "recall")
    }
    et_means = {
        f"extratrees_mean_{key}": statistics.mean(
            float(row[f"extratrees_{key}"]) for row in records
        ) for key in ("dice", "iou", "precision", "recall")
    }
    raw_means = {
        f"raw_sam_mean_{key}": statistics.mean(
            float(row[f"raw_sam_{key}"]) for row in records
        ) for key in ("dice", "iou", "precision", "recall")
    }
    lab_deltas = [
        float(row["cie_lab_mean_delta_e76"])
        for row in records
        if row.get("cie_lab_mean_delta_e76") not in (None, "")
    ]
    metadata = _save_run_metadata(output, "leave_one_subject_out_evaluation",
                                  checkpoint, device, rows)
    report = {
        **means,
        **et_means,
        **raw_means,
        "mean_cie_lab_delta_e76": (
            statistics.mean(lab_deltas) if lab_deltas else None
        ),
        "cie_lab_delta_scope": (
            "OpenCV 8-bit LAB means on existing TIFF pixels; uncalibrated and "
            "not comparable across source rendering profiles"
        ),
        "evaluated_unique_subjects": len(records),
        "sam2_mean_inference_seconds": statistics.mean(
            float(row["sam2_inference_seconds"]) for row in records
        ),
        "total_pipeline_mean_inference_seconds": statistics.mean(
            float(row["total_inference_seconds"]) for row in records
        ),
        "qc_flagged_count": sum(bool(row["qc_flags"]) for row in records),
        "sam2_checkpoint_sha256": metadata["sam2_checkpoint_sha256"],
        "status": "small_pilot_not_acceptance_validation",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _unannotated_nail_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    saved_rois: set[str] = set()
    for roi in (ANNOTATIONS / "completed/nail/rois").glob("*.roi"):
        if roi.name.endswith("_nail_plate.roi"):
            saved_rois.add(roi.name.removesuffix("_nail_plate.roi"))
        elif roi.name.endswith("_nail_plate_mask.roi"):
            saved_rois.add(roi.name.removesuffix("_nail_plate_mask.roi"))
    candidates = [row for row in rows if row["modality"] == "nail"
                  and row["annotation_status"] == "pending"
                  and row["annotation_image_path"]
                  and Path(row["annotation_image_path"]).stem not in saved_rois]
    for row in candidates:
        image = ROOT / row["annotation_image_path"]
        if not image.is_file() or sha256(image) != row["annotation_image_sha256"]:
            raise ValueError("Pending nail annotation image is missing or changed")
    return sorted(candidates, key=lambda row: row["subject_id"])


def generate_candidate_pilot(
    output: Path, checkpoint: Path = SAM2_CHECKPOINT,
    device_name: str = "auto", limit: int = 6,
    prompt_mode: str = "box_plus_points",
) -> dict[str, Any]:
    """Generate a small unverified nail proposal batch from unlabeled images."""
    output = output.resolve()
    checkpoint = checkpoint.resolve()
    if not output.resolve().is_relative_to((ROOT / "data/processed").resolve()):
        raise ValueError("Local prediction output must stay under ignored data/processed")
    if limit <= 0:
        raise ValueError("limit must be positive")
    rows = _read_manifest()
    training_rows = _approved_training_rows(rows)
    candidates = _unannotated_nail_rows(rows)
    if len(candidates) < limit:
        raise ValueError(f"Only {len(candidates)} unannotated nail images are available")
    # Spread the small review pilot across the sorted queue; do not inspect labels.
    indices = np.linspace(0, len(candidates) - 1, limit, dtype=int)
    selected = [candidates[index] for index in indices]
    cache = _load_training_cache(training_rows)
    model = prototype._fit(training_rows, cache)
    area_range = _reference_area_range(
        [row["subject_id"] for row in training_rows], cache,
    )
    predictor, device = _load_sam_predictor(checkpoint, device_name)
    output.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, output / "extratrees_prompt_model.joblib")
    records: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        image_path = ROOT / row["annotation_image_path"]
        image_bgr = prototype._read_image(image_path)
        small = prototype._small_image(image_bgr)
        coarse_small = prototype._predict(model, prototype._features(small), small.shape[:2]).astype(bool)
        coarse = cv2.resize(coarse_small.astype(np.uint8),
                            (image_bgr.shape[1], image_bgr.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        if not coarse.any():
            raise ValueError("Prompt model returned an empty nail candidate; stop for manual review")
        raw_prediction, seconds, box, sam_quality = _infer_box_prompt(
            predictor, cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), coarse,
            device, prompt_mode,
        )
        predicted, raw_component_count = largest_component(raw_prediction)
        blank = {**row, "source_image_path": row["source_image_path"]}
        record = _record_prediction(blank, image_bgr, coarse, predicted,
                                    raw_prediction, raw_component_count,
                                    seconds, box, area_range, sam_quality, prompt_mode,
                                    output, None, None)
        record["original_source_sha256"] = row["original_source_sha256"]
        record["prediction_model_sha256"] = sha256(output / "extratrees_prompt_model.joblib")
        records.append(record)
        if (index + 1) % 2 == 0:
            print(f"Generated {index + 1}/{len(selected)} unverified nail proposals", flush=True)
    _write_csv(output / "predictions.csv", records)
    metadata = _save_run_metadata(output, "unannotated_candidate_pilot",
                                  checkpoint, device, training_rows)
    summary = {
        "candidate_count": len(records),
        "pending_unannotated_nail_images": len(candidates),
        "model_trained_on_approved_nail_subjects": len(training_rows),
        "mean_predicted_area_fraction": statistics.mean(
            float(row["predicted_area_fraction"]) for row in records
        ),
        "qc_flagged_count": sum(bool(row["qc_flags"]) for row in records),
        "sam2_checkpoint_sha256": metadata["sam2_checkpoint_sha256"],
        "status": "prediction_only_not_verified",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("evaluate", "predict"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=SAM2_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--prompt-mode", choices=("box_only", "box_plus_points"),
                        default="box_plus_points")
    args = parser.parse_args()
    if args.mode == "evaluate":
        result = evaluate_leave_one_out(args.output, args.checkpoint,
                                        args.device, args.prompt_mode)
    else:
        result = generate_candidate_pilot(args.output, args.checkpoint,
                                          args.device, args.limit, args.prompt_mode)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
