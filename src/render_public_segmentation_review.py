"""Render contact sheets for human QC of validated public palpebral masks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .validate_public_segmentation import write_if_same_or_new


def render_review_sheets(
    validated_csv: Path, output: Path, columns: int = 4, rows: int = 5
) -> dict[str, list[str]]:
    """Make labeled overlay sheets without modifying source or binary masks."""
    with validated_csv.open(newline="") as stream:
        records = list(csv.DictReader(stream))
    if not records or columns < 1 or rows < 1:
        raise ValueError("No validated pairs or invalid sheet geometry")
    tile_width, tile_height = 200, 295
    capacity = columns * rows
    index: dict[str, list[str]] = {}
    for start in range(0, len(records), capacity):
        sheet = np.full(
            (rows * tile_height, columns * tile_width, 3), 255, dtype=np.uint8
        )
        page = records[start:start + capacity]
        for index_in_page, record in enumerate(page):
            image = cv2.imread(str(record["original_path"]), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(record["mask_path"]), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None or set(np.unique(mask)) != {0, 255}:
                raise ValueError(f"Unable to review validated pair {record['key']}")
            image = cv2.resize(image, (tile_width, 267), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (tile_width, 267), interpolation=cv2.INTER_NEAREST) > 0
            overlay = image.copy()
            overlay[mask] = (
                0.55 * image[mask] + 0.45 * np.array([0, 0, 255])
            ).astype(np.uint8)
            x0 = (index_in_page % columns) * tile_width
            y0 = (index_in_page // columns) * tile_height
            sheet[y0 + 28:y0 + tile_height, x0:x0 + tile_width] = overlay
            cv2.putText(
                sheet, str(record["key"]), (x0 + 5, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA
            )
        name = f"sheet_{start // capacity + 1:02d}.jpg"
        success, encoded = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not success:
            raise RuntimeError("Unable to encode contact sheet")
        write_if_same_or_new(output / name, encoded.tobytes())
        index[name] = [str(record["key"]) for record in page]
    write_if_same_or_new(
        output / "index.json", (json.dumps(index, indent=2) + "\n").encode()
    )
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validated-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    index = render_review_sheets(args.validated_csv, args.output)
    print(f"Rendered {len(index)} public mask-review sheets")


if __name__ == "__main__":
    main()
