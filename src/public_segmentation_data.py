"""Validated Eyes Defy Anemia palpebral image/mask pairs.

The public PNGs are masked RGB(A) images, not ordinary binary masks. This
module never modifies them and never reads forniceal or combined variants.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


PALPEBRAL_SUFFIXES = ("_palpebral", "_papebral", "_palplebral")


@dataclass(frozen=True)
class PublicPair:
    country: str
    subject_id: str
    image_path: Path
    segmentation_path: Path

    @property
    def key(self) -> str:
        return f"{self.country}/{self.subject_id}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_public_pairs(root: Path) -> list[PublicPair]:
    """Find one original and one primary palpebral PNG per numbered subject.

    Exact ``(1)`` copies of a primary PNG are allowed but never counted as
    extra examples. Distinct alternatives or missing sources fail closed.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    pairs: list[PublicPair] = []
    for country_dir in sorted(root.iterdir()):
        if not country_dir.is_dir():
            continue
        subject_dirs = sorted(
            (path for path in country_dir.iterdir() if path.is_dir()),
            key=lambda path: (not path.name.isdecimal(), path.name),
        )
        if not subject_dirs:
            continue
        for subject_dir in subject_dirs:
            if not subject_dir.name.isdecimal():
                raise ValueError(f"Non-numeric public subject directory: {subject_dir}")
            originals = sorted(
                path for path in subject_dir.iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg"}
            )
            if len(originals) != 1:
                raise ValueError(
                    f"Expected one original JPEG for {country_dir.name}/{subject_dir.name}; "
                    f"found {len(originals)}"
                )
            pngs = sorted(subject_dir.glob("*.png"))
            candidates = [
                path for path in pngs
                if path.stem.endswith(PALPEBRAL_SUFFIXES)
                and "forniceal" not in path.stem
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"Expected one primary palpebral PNG for "
                    f"{country_dir.name}/{subject_dir.name}; found {len(candidates)}"
                )
            primary = candidates[0]
            for extra in pngs:
                if not extra.name.endswith("(1).png"):
                    continue
                base = extra.with_name(extra.name.replace("(1).png", ".png"))
                if base.exists() and sha256(extra) != sha256(base):
                    raise ValueError(f"Non-identical public duplicate variant: {extra}")
            pairs.append(
                PublicPair(country_dir.name, subject_dir.name, originals[0], primary)
            )
    keys = [pair.key for pair in pairs]
    if not pairs or len(keys) != len(set(keys)):
        raise ValueError("No public pairs found or duplicate composite subject keys")
    return pairs


def reconstruct_mask(segmented: np.ndarray) -> tuple[np.ndarray, str]:
    """Return a boolean foreground mask and the evidence-based mask encoding."""
    if segmented.ndim != 3 or segmented.shape[2] not in (3, 4):
        raise ValueError("Expected an RGB or RGBA masked PNG")
    if segmented.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"Unsupported masked PNG dtype: {segmented.dtype}")
    if segmented.shape[2] == 4 and np.any(segmented[:, :, 3] == 0):
        mask = segmented[:, :, 3] > 0
        encoding = "alpha"
    else:
        rgb = segmented[:, :, :3]
        white = np.iinfo(segmented.dtype).max
        exact_white = np.all(rgb == white, axis=2)
        if float(exact_white.mean()) < 0.70:
            raise ValueError("Opaque PNG has no dominant exact-white background")
        mask = ~exact_white
        encoding = "exact_white_background"
    foreground_fraction = float(mask.mean())
    if not 0.005 <= foreground_fraction <= 0.30:
        raise ValueError(f"Implausible palpebral mask area: {foreground_fraction:.4f}")
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count < 2 or float(stats[1:, cv2.CC_STAT_AREA].max() / mask.sum()) < 0.98:
        raise ValueError("Palpebral mask is empty or too fragmented")
    return mask, encoding


def load_public_pair(pair: PublicPair) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Load oriented RGB original and aligned boolean mask at PNG resolution."""
    original_bgr = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)
    segmented = cv2.imread(str(pair.segmentation_path), cv2.IMREAD_UNCHANGED)
    if original_bgr is None or segmented is None:
        raise ValueError(f"Unable to decode public pair {pair.key}")
    mask, encoding = reconstruct_mask(segmented)
    image_aspect = original_bgr.shape[1] / original_bgr.shape[0]
    mask_aspect = mask.shape[1] / mask.shape[0]
    if abs(image_aspect / mask_aspect - 1.0) > 0.01:
        raise ValueError(f"Image/mask orientation mismatch for {pair.key}")
    target_size = (mask.shape[1], mask.shape[0])
    resized_bgr = cv2.resize(original_bgr, target_size, interpolation=cv2.INTER_AREA)
    if segmented.dtype == np.uint8:
        error = cv2.absdiff(resized_bgr, segmented[:, :, :3]).mean(axis=2)
        foreground_rgb_error = float(error[mask].mean())
        if foreground_rgb_error > 15.0:
            raise ValueError(
                f"Masked RGB/original registration error for {pair.key}: "
                f"{foreground_rgb_error:.2f}"
            )
    else:
        foreground_rgb_error = None
    return (
        cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB),
        mask,
        {
            "encoding": encoding,
            "original_shape": list(original_bgr.shape[:2]),
            "mask_shape": list(mask.shape),
            "foreground_fraction": float(mask.mean()),
            "foreground_rgb_error": foreground_rgb_error,
        },
    )


def letterbox_pair(
    image_rgb: np.ndarray, mask: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Preserve aspect ratio while resizing aligned RGB and mask to a square."""
    if size <= 0 or image_rgb.shape[:2] != mask.shape:
        raise ValueError("Invalid letterbox size or misaligned image/mask")
    height, width = mask.shape
    scale = min(size / width, size / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    x0 = (size - new_width) // 2
    y0 = (size - new_height) // 2
    image_out = np.zeros((size, size, 3), dtype=np.uint8)
    mask_out = np.zeros((size, size), dtype=np.uint8)
    image_out[y0:y0 + new_height, x0:x0 + new_width] = cv2.resize(
        image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA
    )
    mask_out[y0:y0 + new_height, x0:x0 + new_width] = cv2.resize(
        mask.astype(np.uint8),
        (new_width, new_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return image_out, mask_out, (x0, y0, new_width, new_height)


def perceptual_hash(image_path: Path) -> np.ndarray:
    """Compute a small grayscale DCT fingerprint for conservative split grouping."""
    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError(f"Unable to decode {image_path}")
    resized = cv2.resize(original, (32, 32), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    coefficients = cv2.dct(gray.astype(np.float32))[:8, :8]
    # Match the audited fingerprint rule: exclude the full first DCT row.
    threshold = np.median(coefficients[1:])
    return (coefficients > threshold).reshape(-1)


def find_similar_pairs(
    pairs: Iterable[PublicPair], maximum_hamming_distance: int = 6
) -> list[tuple[str, str, int]]:
    """Flag visually similar cross-key originals; this does not prove identity."""
    pairs = list(pairs)
    fingerprints = {pair.key: perceptual_hash(pair.image_path) for pair in pairs}
    similar: list[tuple[str, str, int]] = []
    for index, first in enumerate(pairs):
        for second in pairs[index + 1:]:
            distance = int(np.count_nonzero(
                fingerprints[first.key] ^ fingerprints[second.key]
            ))
            if distance <= maximum_hamming_distance:
                similar.append((first.key, second.key, distance))
    return similar


def assign_subject_splits(
    pairs: Iterable[PublicPair],
    similar_pairs: Iterable[tuple[str, str, int]],
    seed: int,
) -> dict[str, str]:
    """Assign subject-disjoint 70/15/15 splits, grouping similarity candidates."""
    pairs = list(pairs)
    keys = {pair.key for pair in pairs}
    parent = {key: key for key in keys}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for first, second, _ in similar_pairs:
        if first not in keys or second not in keys:
            raise ValueError("Similarity pair does not belong to public inventory")
        parent[find(first)] = find(second)
    clusters: dict[str, list[str]] = {}
    for key in sorted(keys):
        clusters.setdefault(find(key), []).append(key)
    groups = sorted(clusters.values(), key=lambda members: members[0])
    random.Random(seed).shuffle(groups)
    targets = {"train": round(len(keys) * 0.70), "validation": round(len(keys) * 0.15)}
    targets["test"] = len(keys) - targets["train"] - targets["validation"]
    counts = {split: 0 for split in targets}
    assignment: dict[str, str] = {}
    for members in groups:
        remaining = {split: targets[split] - counts[split] for split in targets}
        split = max(("train", "validation", "test"), key=lambda name: remaining[name])
        for key in members:
            assignment[key] = split
        counts[split] += len(members)
    if set(assignment) != keys or any(count == 0 for count in counts.values()):
        raise ValueError("Failed to create nonempty, complete public splits")
    return assignment
