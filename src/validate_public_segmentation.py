"""Validate and materialize public palpebral binary masks in ignored storage.

Run from the repository root, for example::

    python -m src.validate_public_segmentation \
      --source 'data/raw/dataset anemia' \
      --output data/processed/segmentation/public_palpebral_v1

The source JPEGs and segmented color PNGs are only read, never changed.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .public_segmentation_data import (
    discover_public_pairs,
    find_similar_pairs,
    load_public_pair,
    sha256,
)


def write_if_same_or_new(path: Path, content: bytes) -> None:
    """Write once, or verify that an existing artifact is byte-identical."""
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Existing validation output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


def encode_png(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Unable to encode PNG")
    return encoded.tobytes()


def validate_public_segmentation(source: Path, output: Path) -> dict[str, object]:
    """Validate all pairs before writing binary masks and a reproducible ledger."""
    pairs = discover_public_pairs(source)
    records: list[dict[str, object]] = []
    for pair in pairs:
        _, mask, evidence = load_public_pair(pair)
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), 8
        )
        records.append({
            "key": pair.key,
            "country": pair.country,
            "subject_id": pair.subject_id,
            "original_path": str(pair.image_path.resolve()),
            "original_sha256": sha256(pair.image_path),
            "segmented_path": str(pair.segmentation_path.resolve()),
            "segmented_sha256": sha256(pair.segmentation_path),
            "mask_path": str((output / "masks" / pair.country /
                              f"{pair.subject_id}_palpebral_mask.png").resolve()),
            "mask_height": mask.shape[0],
            "mask_width": mask.shape[1],
            "original_height": evidence["original_shape"][0],
            "original_width": evidence["original_shape"][1],
            "encoding": evidence["encoding"],
            "foreground_fraction": round(float(mask.mean()), 8),
            "largest_component_fraction": round(
                float(stats[1:, cv2.CC_STAT_AREA].max() / mask.sum()), 8
            ),
            "component_count": count - 1,
            "foreground_rgb_error": evidence["foreground_rgb_error"],
        })
    similar_pairs = find_similar_pairs(pairs)

    # The first pass must succeed in full before any derived output is created.
    for pair, record in zip(pairs, records):
        _, mask, _ = load_public_pair(pair)
        binary = np.where(mask, 255, 0).astype(np.uint8)
        content = encode_png(binary)
        write_if_same_or_new(Path(str(record["mask_path"])), content)
        record["mask_sha256"] = sha256(Path(str(record["mask_path"])))

    fieldnames = list(records[0])
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)
    write_if_same_or_new(output / "validated_pairs.csv", csv_buffer.getvalue().encode())
    summary = {
        "source": str(source.resolve()),
        "pair_count": len(records),
        "by_country": dict(Counter(str(row["country"]) for row in records)),
        "mask_encoding": dict(Counter(str(row["encoding"]) for row in records)),
        "multiple_component_count": sum(int(row["component_count"]) > 1 for row in records),
        "near_duplicate_candidates": [
            {"first": first, "second": second, "hamming_distance": distance}
            for first, second, distance in similar_pairs
        ],
        "mask_policy": "alpha>0 only when alpha contains zero; otherwise RGB!=exact white",
        "status": "file_and_alignment_validated_not_clinically_reviewed",
    }
    write_if_same_or_new(
        output / "summary.json", (json.dumps(summary, indent=2) + "\n").encode()
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate_public_segmentation(args.source, args.output)
    print(json.dumps({key: value for key, value in result.items()
                      if key != "near_duplicate_candidates"}, indent=2))


if __name__ == "__main__":
    main()
