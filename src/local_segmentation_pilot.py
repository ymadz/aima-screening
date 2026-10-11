"""Prediction-only public-U-Net pilot on approved local eyelid references.

The script freezes a subject-disjoint local evaluation partition before any
fine-tuning. Its default pilot excludes the untouched test partition and never
updates the authoritative annotation manifest or approved masks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from .public_segmentation_data import sha256
from .segmentation_model import MobileNetSmallUNet
from .train_public_segmentation import IMAGENET_MEAN, IMAGENET_STD, binary_metrics
from .validate_public_segmentation import write_if_same_or_new


SPLIT_SALT = "20261008"


def local_subject_split(subject_id: str) -> str:
    """Stable 60/20/20 train/validation/test assignment without image labels."""
    bucket = hashlib.sha256(f"{SPLIT_SALT}:{subject_id}".encode()).digest()[0] % 10
    return "test" if bucket < 2 else "validation" if bucket < 4 else "train"


def _select_pilot(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    """Select development examples deterministically; never peek at test masks."""
    train = sorted(
        (row for row in rows if local_subject_split(row["subject_id"]) == "train"),
        key=lambda row: row["subject_id"],
    )
    validation = sorted(
        (row for row in rows if local_subject_split(row["subject_id"]) == "validation"),
        key=lambda row: row["subject_id"],
    )
    selected: list[dict[str, str]] = []
    while len(selected) < limit and (train or validation):
        for queue in (train, validation):
            if queue and len(selected) < limit:
                selected.append(queue.pop(0))
    return selected


def _letterbox_image(image_rgb: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = image_rgb.shape[:2]
    scale = min(size / width, size / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    x0 = (size - new_width) // 2
    y0 = (size - new_height) // 2
    square = np.zeros((size, size, 3), dtype=np.uint8)
    square[y0:y0 + new_height, x0:x0 + new_width] = cv2.resize(
        image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA
    )
    return square, (x0, y0, new_width, new_height)


def predict_local_mask(
    model: MobileNetSmallUNet, image_bgr: np.ndarray,
    size: int, device: torch.device,
    crop: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, float]:
    """Predict a full-size binary mask using only the image and saved model."""
    full_height, full_width = image_bgr.shape[:2]
    if crop is None:
        left, top, right, bottom = 0, 0, full_width, full_height
    else:
        x0, y0, x1, y1 = crop
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("Crop must be four ordered fractions within [0,1]")
        left = round(x0 * full_width)
        top = round(y0 * full_height)
        right = round(x1 * full_width)
        bottom = round(y1 * full_height)
    source = image_bgr[top:bottom, left:right]
    image_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    square, (x0, y0, width, height) = _letterbox_image(image_rgb, size)
    normalized = (square.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1).copy())[None].to(device)
    model.eval()
    with torch.inference_mode():
        if device.type == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()
        logits = model(tensor)
        if device.type == "mps":
            torch.mps.synchronize()
        seconds = time.perf_counter() - start
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
    content = probability[y0:y0 + height, x0:x0 + width]
    resized = cv2.resize(
        content, (source.shape[1], source.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    full_mask = np.zeros((full_height, full_width), dtype=bool)
    full_mask[top:bottom, left:right] = resized >= 0.5
    return full_mask, seconds


def _lab_mean(image_bgr: np.ndarray, mask: np.ndarray) -> list[float] | None:
    if not mask.any():
        return None
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    means = lab[mask].astype(np.float32).mean(axis=0)
    return [float(means[0] * (100.0 / 255.0)),
            float(means[1] - 128.0), float(means[2] - 128.0)]


def _preview(image_bgr: np.ndarray, candidate: np.ndarray, reference: np.ndarray) -> bytes:
    width = 450
    height = round(image_bgr.shape[0] * width / image_bgr.shape[1])
    original = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)
    candidate_small = cv2.resize(
        candidate.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    ) > 0
    reference_small = cv2.resize(
        reference.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    ) > 0
    panels = [original.copy() for _ in range(3)]
    panels[1][candidate_small] = (
        0.55 * panels[1][candidate_small] + 0.45 * np.array([0, 0, 255])
    ).astype(np.uint8)
    panels[2][reference_small] = (
        0.55 * panels[2][reference_small] + 0.45 * np.array([0, 255, 0])
    ).astype(np.uint8)
    headers = []
    for title, panel in zip(("Original", "Predicted — unverified", "Manual reference"), panels):
        header = np.full((32, width, 3), 255, dtype=np.uint8)
        cv2.putText(header, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)
        headers.append(np.vstack((header, panel)))
    success, encoded = cv2.imencode(
        ".jpg", np.hstack(headers), [cv2.IMWRITE_JPEG_QUALITY, 88]
    )
    if not success:
        raise RuntimeError("Unable to encode local preview")
    return encoded.tobytes()


def run_local_pilot(
    manifest: Path, metadata: Path, checkpoint: Path, output: Path,
    limit: int, requested_device: str,
    crop: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Evaluate a small development-only pilot, preserving local test subjects."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Pilot output is not empty: {output}")
    private_root = (Path.cwd() / "data/processed").resolve()
    if not output.resolve().is_relative_to(private_root):
        raise ValueError("Local prediction output must remain under ignored data/processed")
    if limit <= 0:
        raise ValueError("Pilot limit must be positive")
    metadata_hash = sha256(metadata)
    with manifest.open(newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    if not all_rows or any(row["metadata_source_sha256"] != metadata_hash for row in all_rows):
        raise ValueError("Manifest does not match frozen local metadata hash")
    all_subjects = sorted({row["subject_id"] for row in all_rows})
    if len(all_subjects) != 140:
        raise ValueError("Unexpected number of local subjects; review split scope")
    split_rows = [
        {"subject_id": subject, "split": local_subject_split(subject)}
        for subject in all_subjects
    ]
    eligible = [
        row for row in all_rows
        if row["modality"] == "eyelid" and row["annotation_status"] == "approved"
    ]
    selected = _select_pilot(eligible, limit)
    if len(selected) != limit or any(
        local_subject_split(row["subject_id"]) == "test" for row in selected
    ):
        raise ValueError("Insufficient development references or test leakage")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    size = int(saved["config"]["image_size"])
    model = MobileNetSmallUNet(pretrained_encoder=False)
    model.load_state_dict(saved["model_state"])
    device = torch.device(
        "mps" if requested_device == "auto" and torch.backends.mps.is_available()
        else "cpu" if requested_device == "auto" else requested_device
    )
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested MPS unavailable")
    model.to(device).eval()
    checkpoint_sha256 = sha256(checkpoint)
    inference_script_sha256 = sha256(Path(__file__).resolve())
    output.mkdir(parents=True, exist_ok=True)
    write_if_same_or_new(
        output / "subject_split.json", (json.dumps(split_rows, indent=2) + "\n").encode()
    )
    records: list[dict[str, Any]] = []
    for row in selected:
        image_path = Path(row["annotation_image_path"])
        reference_path = Path(row["mask_path"])
        roi_path = Path(row["roi_path"])
        if any(not path.is_file() for path in (image_path, reference_path, roi_path)):
            raise FileNotFoundError("Approved local image/ROI/mask is missing")
        if (sha256(image_path) != row["annotation_image_sha256"] or
                sha256(reference_path) != row["mask_sha256"] or
                sha256(roi_path) != row["roi_sha256"]):
            raise ValueError("Approved local image/ROI/mask hash mismatch")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        with Image.open(reference_path) as source:
            if source.mode != "L":
                raise ValueError("Manual reference mask is not 8-bit grayscale")
            reference = np.array(source)
        if image is None or reference.shape != image.shape[:2] or set(np.unique(reference)) != {0, 255}:
            raise ValueError("Approved local image/mask alignment or values are invalid")
        candidate, seconds = predict_local_mask(model, image, size, device, crop)
        mask_name = image_path.stem + "_predicted_palpebral.png"
        success, encoded = cv2.imencode(
            ".png", np.where(candidate, 255, 0).astype(np.uint8)
        )
        if not success:
            raise RuntimeError("Unable to encode predicted local mask")
        mask_path = output / "predicted_masks" / mask_name
        write_if_same_or_new(mask_path, encoded.tobytes())
        preview_path = output / "previews" / (image_path.stem + "_comparison.jpg")
        write_if_same_or_new(preview_path, _preview(image, candidate, reference > 0))
        metrics = binary_metrics(candidate, reference > 0)
        predicted_lab = _lab_mean(image, candidate)
        reference_lab = _lab_mean(image, reference > 0)
        lab_delta = (
            float(np.linalg.norm(np.array(predicted_lab) - np.array(reference_lab)))
            if predicted_lab is not None and reference_lab is not None else None
        )
        components, _, _, _ = cv2.connectedComponentsWithStats(
            candidate.astype(np.uint8), 8
        )
        flags = []
        area = float(candidate.mean())
        if area < 0.0005:
            flags.append("empty_or_tiny")
        if area > 0.03:
            flags.append("oversized")
        if components - 1 > 5:
            flags.append("fragmented")
        if candidate[0].any() or candidate[-1].any() or candidate[:, 0].any() or candidate[:, -1].any():
            flags.append("touches_frame_border")
        records.append({
            "subject_id": row["subject_id"],
            "original_image_filename": row["original_image_filename"],
            "annotation_image_path": str(image_path),
            "annotation_image_sha256": row["annotation_image_sha256"],
            "manual_mask_path": str(reference_path),
            "manual_mask_sha256": row["mask_sha256"],
            "predicted_mask_path": str(mask_path),
            "predicted_mask_sha256": sha256(mask_path),
            "preview_path": str(preview_path),
            "split": local_subject_split(row["subject_id"]),
            "status": "predicted_unverified",
            "prediction_method": "public_only_zero_shot_local_inference",
            "model_architecture": "MobileNetV3-Small encoder U-Net",
            "model_checkpoint_path": str(checkpoint.resolve()),
            "model_checkpoint_sha256": checkpoint_sha256,
            "inference_script_sha256": inference_script_sha256,
            "inference_device": device.type,
            "input_size": size,
            "crop": json.dumps(crop) if crop is not None else "full_frame",
            "probability_threshold": 0.5,
            "input_preprocessing": (
                "OpenCV BGR decode, channel reorder to RGB, aspect-preserving zero-pad "
                "to square, ImageNet mean/std; no ICC transform"
            ),
            "inference_seconds": seconds,
            "predicted_area_fraction": area,
            "reference_area_fraction": float((reference > 0).mean()),
            "qc_flags": "|".join(flags),
            "cie_lab_delta_e76_mean": lab_delta,
            **metrics,
        })
    with (output / "predictions.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    result = {
        "status": "prediction_only_not_verified",
        "pilot_count": len(records),
        "local_test_subjects_reserved": sum(
            row["split"] == "test" for row in split_rows
        ),
        "approved_test_references_not_evaluated": sum(
            local_subject_split(row["subject_id"]) == "test" for row in eligible
        ),
        "by_development_split": dict(Counter(row["split"] for row in records)),
        "mean_dice": statistics.mean(row["dice"] for row in records),
        "mean_iou": statistics.mean(row["iou"] for row in records),
        "mean_precision": statistics.mean(row["precision"] for row in records),
        "mean_recall": statistics.mean(row["recall"] for row in records),
        "mean_inference_seconds": statistics.mean(row["inference_seconds"] for row in records),
        "mean_lab_delta_e76": statistics.mean(
            row["cie_lab_delta_e76_mean"] for row in records
            if row["cie_lab_delta_e76_mean"] is not None
        ) if any(row["cie_lab_delta_e76_mean"] is not None for row in records) else None,
        "qc_flagged_count": sum(bool(row["qc_flags"]) for row in records),
        "metadata_source_sha256": metadata_hash,
        "model_architecture": "MobileNetV3-Small encoder U-Net",
        "model_checkpoint_path": str(checkpoint.resolve()),
        "model_checkpoint_sha256": checkpoint_sha256,
        "inference_script_sha256": inference_script_sha256,
        "inference_device": device.type,
        "input_size": size,
        "probability_threshold": 0.5,
        "input_preprocessing": (
            "OpenCV BGR decode, channel reorder to RGB, aspect-preserving zero-pad "
            "to square, ImageNet mean/std; no ICC transform"
        ),
        "evaluation_split": "local train+validation development subjects only",
        "local_split_salt": SPLIT_SALT,
        "crop": list(crop) if crop is not None else "full_frame",
        "color_caveat": "OpenCV 8-bit LAB on annotation TIFF pixels; not a calibrated color comparison",
        "note": "Development pilot only; test subjects remain untouched; no mask is approved automatically.",
    }
    write_if_same_or_new(output / "summary.json", (json.dumps(result, indent=2) + "\n").encode())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--crop", type=float, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    args = parser.parse_args()
    result = run_local_pilot(
        args.manifest, args.metadata, args.checkpoint,
        args.output, args.limit, args.device,
        tuple(args.crop) if args.crop is not None else None,
    )
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("metadata_source_sha256", "model_checkpoint_sha256")},
                     indent=2))


if __name__ == "__main__":
    main()
