"""Build a multimodal feature table from the downloaded eye-image dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from .feature_extractor import MobileNetFeatureExtractor
    from .preprocess import (
        bgr_to_cielab,
        calculate_ita,
        extract_tissue_color_statistics,
    )
except ImportError:
    from feature_extractor import MobileNetFeatureExtractor
    from preprocess import bgr_to_cielab, calculate_ita, extract_tissue_color_statistics


def extract_periocular_skin_ita(image_bgr: np.ndarray) -> float:
    """Calculate ITA from a cutaneous margin in a full facial image."""
    height, width = image_bgr.shape[:2]
    skin_patch = image_bgr[int(height * 0.85) :, : int(width * 0.15)]
    if skin_patch.size == 0:
        return 30.0

    patch_lab = bgr_to_cielab(skin_patch)
    l_channel, _, b_channel = cv2.split(patch_lab)
    mean_l = float(np.mean(l_channel)) * (100.0 / 255.0)
    mean_b = float(np.mean(b_channel)) - 128.0
    return calculate_ita(mean_l, mean_b)


def _metadata_by_subject(raw_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    metadata: dict[tuple[str, int], dict[str, Any]] = {}
    for workbook_path in raw_dir.rglob("*.xlsx"):
        country = workbook_path.parent.name
        frame = pd.read_excel(workbook_path)
        required_columns = {"Number", "Hgb", "Gender", "Age"}
        if not required_columns.issubset(frame.columns):
            continue
        for row in frame.to_dict("records"):
            if pd.isna(row["Number"]) or pd.isna(row["Hgb"]):
                continue
            subject_id = int(row["Number"])
            gender = str(row["Gender"]).strip().upper()
            try:
                hgb = float(str(row["Hgb"]).replace(",", "."))
            except ValueError:
                continue
            threshold = 12.0 if gender == "F" else 13.0
            metadata[(country, subject_id)] = {
                "hgb": hgb,
                "gender": gender,
                "age": float(row["Age"]) if not pd.isna(row["Age"]) else np.nan,
                "label": int(hgb < threshold),
            }
    return metadata


def _find_full_image(roi_path: Path) -> Path | None:
    image_id = roi_path.name.removesuffix("_palpebral.png")
    full_image = roi_path.with_name(f"{image_id}.jpg")
    return full_image if full_image.exists() else None


def build_dataset(
    raw_dir: str | Path = "data/raw",
    output_csv_path: str | Path = "data/processed/dataset.csv",
    pretrained: bool = True,
    max_images: int | None = None,
) -> pd.DataFrame:
    """Extract and save one fused feature row per segmented palpebral ROI."""
    raw_path = Path(raw_dir)
    output_path = Path(output_csv_path)
    metadata = _metadata_by_subject(raw_path)
    roi_paths = sorted(
        path
        for path in raw_path.rglob("*_palpebral.png")
        if "_forniceal_" not in path.name
    )
    if max_images is not None:
        roi_paths = roi_paths[:max_images]

    extractor = MobileNetFeatureExtractor(pretrained=pretrained)
    records: list[dict[str, Any]] = []
    for roi_path in tqdm(roi_paths, desc="Extracting multimodal features"):
        subject_dir = roi_path.parent
        country = subject_dir.parent.name
        try:
            subject_id = int(subject_dir.name)
        except ValueError:
            continue
        subject_metadata = metadata.get((country, subject_id))
        if subject_metadata is None:
            continue

        roi_bgr = cv2.imread(str(roi_path), cv2.IMREAD_COLOR)
        full_path = _find_full_image(roi_path)
        full_bgr = cv2.imread(str(full_path), cv2.IMREAD_COLOR) if full_path else None
        if roi_bgr is None or full_bgr is None:
            continue

        color_stats = extract_tissue_color_statistics(bgr_to_cielab(roi_bgr))
        embedding = extractor.extract(roi_path)
        row: dict[str, Any] = {
            "image_id": roi_path.name,
            "country": country,
            "subject_id": subject_id,
            "label": subject_metadata["label"],
            "hgb": subject_metadata["hgb"],
            "gender": subject_metadata["gender"],
            "age": subject_metadata["age"],
            "ita_angle": extract_periocular_skin_ita(full_bgr),
            **color_stats,
        }
        row.update({f"emb_{index}": float(value) for index, value in enumerate(embedding)})
        records.append(row)

    frame = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(f"Dataset compiled successfully: {frame.shape[0]} samples x {frame.shape[1]} features.")
    print(f"Saved to: {output_path}")
    return frame


if __name__ == "__main__":
    build_dataset()