"""Optional small-data segmentation assistance experiment.

Manual anatomical ROIs and reviewed masks remain the primary ground truth.
Outputs are proposals for human review, never replacements. The rejected nail
glare candidates and all glare masks are intentionally unused.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import tifffile
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "data/raw/local/annotations"
PILOT_MANIFEST = ANNOTATIONS / "manifests/paired_pilot10.csv"
ANNOTATION_MANIFEST = ANNOTATIONS / "manifests/annotation_manifest.csv"
SUBJECT_MANIFEST = ROOT / "data/raw/local/manifest/frozen_2026-10-07/local_subject_manifest.csv"
INCOMING = ROOT / "data/raw/local/incoming"
DEFAULT_OUTPUT = ROOT / "data/processed/segmentation_prototype_2026-10-07"
MAX_SIDE = 768
SEED = 42


def _load_records(modality: str) -> list[dict[str, str]]:
    with PILOT_MANIFEST.open(newline="") as handle:
        pilot_subjects = {row["subject_id"] for row in csv.DictReader(handle)
                          if row["modality"] == modality}
    with ANNOTATION_MANIFEST.open(newline="") as handle:
        source_rows = [row for row in csv.DictReader(handle)
                       if row["modality"] == modality
                       and row["subject_id"] in pilot_subjects
                       and row["annotation_status"] == "approved"]
    records = [{**row, "annotation_image": row["annotation_image_path"],
                "annotation_mask_path": row["mask_path"],
                "original_filename": row["original_image_filename"]}
               for row in source_rows]
    if len(records) != 10 or len({row["subject_id"] for row in records}) != 10:
        raise ValueError(f"Expected 10 unique manual {modality} examples")
    for row in records:
        if not (ROOT / row["annotation_image"]).is_file():
            raise FileNotFoundError(row["annotation_image"])
        if not (ROOT / row["annotation_mask_path"]).is_file():
            raise FileNotFoundError(row["annotation_mask_path"])
    return sorted(records, key=lambda row: row["subject_id"])


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode annotation image: {path}")
    return image


def _read_dng_preview(path: Path) -> np.ndarray:
    """Decode the JPEG preview stored in Apple DNG, applying TIFF orientation.

    On pilot IMG_0655.DNG this is pixel-identical to the Photos-exported TIFF.
    The raw sensor mosaic is not demosaiced or altered.
    """
    with tifffile.TiffFile(path) as tiff:
        page = tiff.pages[0]
        if len(page.dataoffsets) != 1 or str(page.compression) != "COMPRESSION.JPEG":
            raise ValueError(f"No single JPEG preview in {path}")
        offset, length = page.dataoffsets[0], page.databytecounts[0]
        orientation = int(page.tags["Orientation"].value)
    with path.open("rb") as handle:
        handle.seek(offset)
        jpeg = handle.read(length)
    rgb = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    rotations = {1: None, 3: cv2.ROTATE_180,
                 6: cv2.ROTATE_90_CLOCKWISE, 8: cv2.ROTATE_90_COUNTERCLOCKWISE}
    if orientation not in rotations:
        raise ValueError(f"Unsupported DNG preview orientation {orientation}: {path}")
    rotation = rotations[orientation]
    return cv2.rotate(image, rotation) if rotation is not None else image


def _small_image(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = MAX_SIDE / max(height, width)
    return cv2.resize(
        image, (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _features(image: np.ndarray) -> np.ndarray:
    """RGB, HSV, Lab, local color, and relative position at 768px max side."""
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    smooth = cv2.GaussianBlur(rgb, (0, 0), 7.0)
    yy, xx = np.mgrid[:height, :width]
    position = np.stack(
        (xx / max(width - 1, 1), yy / max(height - 1, 1)), axis=-1,
    ).astype(np.float32)
    return np.concatenate((rgb, hsv, lab, smooth, position), axis=-1).reshape(-1, 14)


def _load_example(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    image = _read_image(ROOT / row["annotation_image"])
    mask = cv2.imread(str(ROOT / row["annotation_mask_path"]), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != image.shape[:2]:
        raise ValueError(f"Mask dimensions do not match image: {row['annotation_mask_path']}")
    small = _small_image(image)
    small_mask = cv2.resize(
        mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST,
    ) > 0
    if not small_mask.any() or small_mask.all():
        raise ValueError(f"Invalid manual mask: {row['annotation_mask_path']}")
    return _features(small), small_mask, image.shape[:2]


def _fit(
    rows: list[dict[str, str]],
    cache: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int]]],
) -> ExtraTreesClassifier:
    rng = np.random.default_rng(SEED)
    sampled_features: list[np.ndarray] = []
    sampled_labels: list[np.ndarray] = []
    for row in rows:
        features, mask, _ = cache[row["subject_id"]]
        flat_mask = mask.ravel()
        positives = np.flatnonzero(flat_mask)
        negatives = np.flatnonzero(~flat_mask)
        pos = rng.choice(positives, size=min(3000, len(positives)), replace=False)
        neg = rng.choice(negatives, size=min(6000, len(negatives)), replace=False)
        selected = np.concatenate((pos, neg))
        sampled_features.append(features[selected])
        sampled_labels.append(flat_mask[selected].astype(np.uint8))
    model = ExtraTreesClassifier(
        n_estimators=40,
        max_depth=18,
        min_samples_leaf=5,
        max_features=0.8,
        n_jobs=4,
        random_state=SEED,
    )
    model.fit(np.concatenate(sampled_features), np.concatenate(sampled_labels))
    return model


def _predict(
    model: ExtraTreesClassifier, features: np.ndarray, shape: tuple[int, int],
) -> np.ndarray:
    probabilities = model.predict_proba(features)[:, 1].reshape(shape)
    binary = (probabilities >= 0.5).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return np.zeros(shape, dtype=np.uint8)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    binary = (components == largest).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return binary


def _scores(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    predicted = prediction.astype(bool)
    manual = target.astype(bool)
    intersection = np.count_nonzero(predicted & manual)
    denominator = np.count_nonzero(predicted) + np.count_nonzero(manual)
    union = np.count_nonzero(predicted | manual)
    return (2 * intersection / denominator if denominator else 1.0,
            intersection / union if union else 1.0)


def _save_proposal(
    out: Path, row: dict[str, str], original: np.ndarray, small_mask: np.ndarray,
    manual_mask_path: Path | None = None,
) -> tuple[Path, Path]:
    height, width = original.shape[:2]
    full_mask = cv2.resize(small_mask, (width, height), interpolation=cv2.INTER_NEAREST) * 255
    stem = f"{row['subject_id']}__{Path(row['original_filename']).stem}"
    mask_path = out / f"{stem}_proposed_mask.png"
    overlay_path = out / f"{stem}_overlay.jpg"
    if not cv2.imwrite(str(mask_path), full_mask):
        raise OSError(mask_path)

    scale = min(1.0, 1400 / max(height, width))
    preview = cv2.resize(
        original, (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    proposed_preview = cv2.resize(
        full_mask, (preview.shape[1], preview.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    tint = preview.copy()
    tint[proposed_preview] = (45, 190, 45)  # green is the model proposal
    preview = cv2.addWeighted(preview, 0.65, tint, 0.35, 0)
    if manual_mask_path is not None:
        manual = cv2.imread(str(manual_mask_path), cv2.IMREAD_GRAYSCALE)
        manual_preview = cv2.resize(
            manual, (preview.shape[1], preview.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        contours, _ = cv2.findContours(manual_preview, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview, contours, -1, (255, 0, 255), 2)  # magenta manual edge
    cv2.putText(preview, stem, (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    cv2.putText(preview, stem, (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    if not cv2.imwrite(str(overlay_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError(overlay_path)
    return mask_path, overlay_path


def _unseen_rows(modality: str, training_subjects: set[str]) -> list[dict[str, str]]:
    column = "raw_eyelid_file" if modality == "eyelid" else "raw_nail_file"
    staged = {
        (row["subject_id"], row["modality"]): row
        for row in csv.DictReader(ANNOTATION_MANIFEST.open(newline=""))
        if row["image_readiness_status"] == "standardized_TIFF_ready"
    }
    with SUBJECT_MANIFEST.open(newline="") as handle:
        candidates = [row for row in csv.DictReader(handle)
                      if row["subject_id"] not in training_subjects
                      and row[column]
                      and (((candidate := staged.get((row["subject_id"], modality))) is not None
                            and (ROOT / candidate["annotation_image_path"]).is_file())
                      if modality == "eyelid" else (INCOMING / row[column]).is_file())]
    candidates.sort(key=lambda row: int(row["subject_id"].split("-")[-1]))
    if len(candidates) < 3:
        raise ValueError(f"Only {len(candidates)} unseen {modality} candidates")
    positions = np.linspace(0, len(candidates) - 1, 5, dtype=int)[1:4]
    result: list[dict[str, str]] = []
    for index in positions:
        candidate = candidates[index]
        ready = staged.get((candidate["subject_id"], modality))
        annotation_image = (str(ROOT / ready["annotation_image_path"])
                            if ready else str(INCOMING / candidate[column]))
        result.append({"subject_id": candidate["subject_id"],
                       "original_filename": candidate[column],
                       "annotation_image": annotation_image})
    return result


def run(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Run five-fold subject-held-out validation and three unseen proposals per task."""
    output_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, object] = {"model": "ExtraTrees pixel classifier", "max_side": MAX_SIDE,
                                 "seed": SEED, "folds": 5, "modalities": {}}
    for modality in ("eyelid", "nail"):
        rows = _load_records(modality)
        cache = {row["subject_id"]: _load_example(row) for row in rows}
        validation_dir = output_dir / modality / "validation"
        unseen_dir = output_dir / modality / "unseen_proposals"
        validation_dir.mkdir(parents=True)
        unseen_dir.mkdir(parents=True)
        score_rows: list[dict[str, object]] = []
        splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)
        for fold, (train_indices, validation_indices) in enumerate(splitter.split(rows), 1):
            model = _fit([rows[index] for index in train_indices], cache)
            for index in validation_indices:
                row = rows[index]
                features, target, _ = cache[row["subject_id"]]
                prediction = _predict(model, features, target.shape)
                dice, iou = _scores(prediction, target)
                mask_path, overlay_path = _save_proposal(
                    validation_dir, row, _read_image(ROOT / row["annotation_image"]), prediction,
                    ROOT / row["annotation_mask_path"],
                )
                score_rows.append({"fold": fold, "subject_id": row["subject_id"],
                                   "original_filename": row["original_filename"],
                                   "dice": dice, "iou": iou,
                                   "mask_area_fraction": float(prediction.mean()),
                                   "ground_truth_area_fraction": float(target.mean()),
                                   "proposed_mask": str(mask_path), "overlay": str(overlay_path)})
        with (output_dir / modality / "validation_scores.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
            writer.writeheader()
            writer.writerows(score_rows)

        final_model = _fit(rows, cache)
        joblib.dump(final_model, output_dir / modality / "model.joblib")
        unseen_report: list[dict[str, object]] = []
        for row in _unseen_rows(modality, {item["subject_id"] for item in rows}):
            image_path = Path(row["annotation_image"])
            image = (
                _read_image(image_path) if image_path.suffix.lower() in {".tif", ".tiff"}
                else _read_dng_preview(image_path)
            )
            small = _small_image(image)
            prediction = _predict(final_model, _features(small), small.shape[:2])
            mask_path, overlay_path = _save_proposal(unseen_dir, row, image, prediction)
            unseen_report.append({"subject_id": row["subject_id"],
                                  "original_filename": row["original_filename"],
                                  "annotation_image": str(image_path),
                                  "mask_area_fraction": float(prediction.mean()),
                                  "proposed_mask": str(mask_path), "overlay": str(overlay_path)})
        dice_values = [float(row["dice"]) for row in score_rows]
        iou_values = [float(row["iou"]) for row in score_rows]
        report["modalities"][modality] = {
            "training_examples_per_fold": 8, "validation_examples_per_fold": 2,
            "unique_validation_examples": len(score_rows),
            "mean_dice": float(np.mean(dice_values)), "median_dice": float(np.median(dice_values)),
            "min_dice": float(np.min(dice_values)), "mean_iou": float(np.mean(iou_values)),
            "unseen_proposals": unseen_report,
        }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output_dir), indent=2))
