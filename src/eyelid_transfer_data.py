"""Read-only, provenance-checked inputs for independent Experiment 3."""
from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import tifffile
import torch
from PIL import Image, ImageCms, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

from .local_eyelid_hybrid_features import manual_roi_tensor, validated_inputs
from .local_eyelid_validation import load_manual_pair, perceptual_hash, sha256
from .public_segmentation_data import discover_public_pairs, load_public_pair, find_similar_pairs

SEED = 20261011
PREPROCESSING = {
    "version": "profile_managed_mask_bbox_224_v1",
    "color": "ICC to sRGB relative colorimetric; untagged public JPEG explicitly assumed sRGB; verified pixel-identical DNG preview profile supersedes misleading TIFF tag",
    "geometry": "Public original EXIF-oriented once to match validated mask geometry, aligned to mask by area resize; local TIFF and mask native aligned raster",
    "roi": "Approved anatomy only, zero-margin tight bounding box, outside-mask ImageNet mean before bilinear antialiased resize, aspect-preserving mean letterbox 224",
    "normalization": "RGB/255; fixed ImageNet mean [0.485,0.456,0.406], std [0.229,0.224,0.225]",
    "glare": "Retain all anatomical pixels; no inpainting or glare removal",
    "enhancement": "None; no white balance, histogram equalization, color jitter or learned color normalization",
    "augmentation": "Training-only horizontal flip, probability 0.5; validation/inference unflipped",
    "historical_lab_comparator": "Reuse unchanged Experiment 1 full-resolution stored-RGB LAB features separately from color-managed neural inputs",
}


def snapshot(path: Path) -> dict[str, Any]:
    """Record source identity before reading rows or pixels."""
    return {"path": str(path.resolve()), "mtime_ns": path.stat().st_mtime_ns,
            "bytes": path.stat().st_size, "sha256": sha256(path)}


def harmonized_label(age: float, sex: str, hb: float) -> tuple[int | None, str]:
    """Unadjusted WHO adult labels; exclude pregnancy-dependent public labels."""
    if not np.isfinite(age) or not 15 <= age <= 65:
        return None, "age_outside_15_65_or_missing"
    if sex not in {"M", "F"}:
        raise ValueError("Unrecognized public sex; no inferred cutoff")
    if not np.isfinite(hb):
        return None, "missing_hb"
    if not 0 < hb < 30:
        raise ValueError("Public Hb is inconsistent with documented g/dL units")
    if sex == "F" and 10.5 <= hb < 12:
        return None, "pregnancy_dependent_label_unknown_status"
    return int(hb < (13 if sex == "M" else 12)), "eligible"


def public_metadata(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Read workbooks strictly, preserving source strings and ancillary cells."""
    records, sources, ancillary = {}, [], {}
    for country in ("India", "Italy"):
        path = root / country / f"{country}.xlsx"
        sources.append(snapshot(path))
        frame = pd.read_excel(path, dtype=object)
        if list(frame.columns[:5]) != ["Number", "Hgb", "Gender", "Age", "Note"]:
            raise ValueError("Public workbook schema changed")
        if any(not str(c).startswith("Unnamed:") for c in frame.columns[5:]):
            raise ValueError("Unrecognized extra public metadata fields")
        ancillary[country] = {str(c): frame[c].dropna().astype(str).tolist() for c in frame.columns[4:]}
        for row in frame.to_dict("records"):
            number = float(row["Number"])
            if not number.is_integer() or number <= 0:
                raise ValueError("Invalid public patient number")
            key = f"{country}/{int(number)}"
            if key in records:
                raise ValueError("Duplicate public subject metadata")
            age = float(str(row["Age"]).replace(",", "."))
            raw_hb = str(row["Hgb"])
            hb = float(raw_hb.replace(",", ".")) if raw_hb not in {"nan", "---", "_"} else np.nan
            sex = str(row["Gender"]).strip().upper()
            label, reason = harmonized_label(age, sex, hb)
            records[key] = {"subject": key, "country": country, "age": age, "sex": sex,
                            "source_hb": raw_hb, "hb_g_dl": hb if np.isfinite(hb) else None,
                            "label": label, "eligibility": reason}
        if sha256(path) != sources[-1]["sha256"]:
            raise ValueError("Public workbook changed during audit")
    return records, {"sources": sources, "ancillary_workbook_cells_private": ancillary,
                     "exclusions": dict(Counter(r["eligibility"] for r in records.values() if r["label"] is None))}


def convert_srgb(image: Image.Image, profile: bytes | None) -> np.ndarray:
    """Apply an actual ICC transform when available, never merely relabel pixels."""
    rgb = image.convert("RGB")
    if profile:
        rgb = ImageCms.profileToProfile(rgb, ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                                       ImageCms.createProfile("sRGB"), renderingIntent=1, outputMode="RGB")
    return np.asarray(rgb, dtype=np.uint8).copy()


def dng_preview(source: Path) -> tuple[np.ndarray, bytes]:
    with tifffile.TiffFile(source) as dng:
        page = dng.pages[0]
        if len(page.dataoffsets) != 1 or str(page.compression) != "COMPRESSION.JPEG":
            raise ValueError("DNG lacks validated single JPEG preview")
        offset, length = page.dataoffsets[0], page.databytecounts[0]
        orientation = int(page.tags["Orientation"].value)
    with source.open("rb") as handle:
        handle.seek(offset)
        jpeg = handle.read(length)
    with Image.open(io.BytesIO(jpeg)) as image:
        profile = image.info.get("icc_profile")
        if not profile or orientation not in {1, 3, 6, 8}:
            raise ValueError("DNG preview color/orientation unknown")
        rgb = image.convert("RGB")
        rotations = {3: Image.Transpose.ROTATE_180, 6: Image.Transpose.ROTATE_270, 8: Image.Transpose.ROTATE_90}
        if orientation in rotations:
            rgb = rgb.transpose(rotations[orientation])
        return np.asarray(rgb).copy(), profile


def local_pixels(root: Path, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    image_path, mask_path = root / row["annotation_image_path"], root / row["mask_path"]
    pixels, mask = load_manual_pair(image_path, mask_path)
    with Image.open(image_path) as image:
        profile = image.info.get("icc_profile")
    if not profile:
        raise ValueError("Local annotation lacks recorded ICC profile")
    policy = "stored_annotation_icc_to_srgb"
    source = root / row["source_image_path"]
    if source.suffix.lower() == ".dng":
        preview, preview_profile = dng_preview(source)
        if np.array_equal(pixels, preview):
            profile = preview_profile
            policy = "pixel_identical_original_dng_preview_icc_to_srgb"
    description = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(profile))).strip()
    managed = convert_srgb(Image.fromarray(pixels), profile)
    return managed, mask, {"color_policy": policy, "input_profile": description,
                           "effective_profile_sha256": hashlib.sha256(profile).hexdigest(),
                           "color_changed_pixels": int(np.any(managed != pixels, axis=2).sum())}


def public_split(subjects: list[str], labels: np.ndarray, countries: list[str],
                 near_pairs: list[tuple[str, str, int]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Keep conservative near-duplicate groups together; stratify country+label."""
    parent = {s: s for s in subjects}
    def find(s: str) -> str:
        while parent[s] != s:
            s = parent[s]
        return s
    for a, b, _ in near_pairs:
        if a in parent and b in parent:
            parent[find(b)] = find(a)
    groups = [find(s) for s in subjects]
    strata = np.array([f"{c}_{y}" for c, y in zip(countries, labels)])
    train, validation = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(subjects, strata, groups))
    if set(np.asarray(groups)[train]) & set(np.asarray(groups)[validation]) or set(labels[train]) != {0, 1} or set(labels[validation]) != {0, 1}:
        raise ValueError("Public split leakage/class support failure")
    return train, validation, groups


def prepare_inputs(source_root: Path, public_root: Path, output: Path) -> dict[str, Any]:
    """Audit both domains, fail on cross-domain similarity, save private tensors."""
    exp1 = source_root / "data/processed/local_eyelid_classification_2026-10-10_v1"
    bundle = source_root / "data/processed/local_eyelid_readiness_2026-10-10_v2"
    local = validated_inputs(source_root, exp1, bundle)
    metadata, audit = public_metadata(public_root)
    pairs = discover_public_pairs(public_root)
    if {p.key for p in pairs} != set(metadata):
        raise ValueError("Public source/metadata coverage mismatch")
    fingerprints, pixel_hashes, source_hashes, details = {}, {}, {}, []
    tensors, subjects, labels, countries = [], [], [], []
    for index, pair in enumerate(pairs):
        pixels, mask, evidence = load_public_pair(pair)
        with Image.open(pair.image_path) as original:
            icc = original.info.get("icc_profile")
            rgb = convert_srgb(ImageOps.exif_transpose(original), icc)
            exif = original.getexif()
            device = [str(exif.get(271, "unknown")), str(exif.get(272, "unknown"))]
        if list(rgb.shape[:2]) != evidence["original_shape"]:
            raise ValueError("Public PIL/OpenCV orientation disagrees")
        rgb = cv2.resize(rgb, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
        if not icc and np.abs(rgb.astype(float) - pixels.astype(float)).mean() > 1.0:
            raise ValueError("Public managed decode/registered pixels disagree")
        fingerprints[pair.key] = perceptual_hash(pixels)
        pixel_hashes[pair.key] = hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()
        source_hashes[pair.key] = sha256(pair.image_path)
        record = {**metadata[pair.key], **evidence, "image": snapshot(pair.image_path),
                  "mask": snapshot(pair.segmentation_path), "device": device,
                  "icc": "embedded_converted" if icc else "absent_assumed_srgb", "decoded_sha256": pixel_hashes[pair.key]}
        details.append(record)
        if record["label"] is not None:
            tensors.append(manual_roi_tensor(rgb, mask.astype(np.uint8) * 255))
            subjects.append(pair.key); labels.append(record["label"]); countries.append(pair.country)
        if (index + 1) % 25 == 0:
            print(json.dumps({"public_audit_images": index + 1, "total": len(pairs)}), flush=True)
    if len(set(source_hashes.values())) != len(pairs) or len(set(pixel_hashes.values())) != len(pairs):
        raise ValueError("Public exact duplicates require identity resolution")
    near = []
    keys = sorted(fingerprints)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            distance = int(np.count_nonzero(fingerprints[a] != fingerprints[b]))
            if distance <= 6:
                near.append((a, b, distance))
    aligned_near = list(near)
    original_near = find_similar_pairs(pairs)
    near = sorted(set(near) | set(original_near))
    local_tensors, local_details, cross = [], [], []
    for subject in local["subjects"]:
        row = local["rows"][subject]
        raw, _ = load_manual_pair(source_root / row["annotation_image_path"], source_root / row["mask_path"])
        fp = perceptual_hash(raw)
        cross.extend((s, subject, int(np.count_nonzero(fp != other))) for s, other in fingerprints.items() if np.count_nonzero(fp != other) <= 6)
        if row["original_source_sha256"] in source_hashes.values():
            raise ValueError("Cross-domain exact source duplicate")
        rgb, mask, detail = local_pixels(source_root, row)
        local_details.append({"subject": subject, **detail})
        local_tensors.append(manual_roi_tensor(rgb, mask))
    if cross:
        (output / "cross_domain_similarity_candidates.json").write_text(json.dumps(cross, indent=2))
        raise ValueError("Cross-domain similarity candidates require identity resolution")
    public_y = np.array(labels, dtype=int)
    train, validation, groups = public_split(subjects, public_y, countries, near)
    audit.update(subjects=len(pairs), eligible=len(subjects), class_counts=np.bincount(public_y, minlength=2).tolist(),
                 exact_source_duplicates=0, exact_decoded_duplicates=0, near_pairs=near, aligned_phash_candidates=aligned_near, original_public_phash_candidates=original_near, cross_domain_candidates=cross,
                 training_subjects=[subjects[i] for i in train], validation_subjects=[subjects[i] for i in validation],
                 training_class_counts=np.bincount(public_y[train], minlength=2).tolist(), validation_class_counts=np.bincount(public_y[validation], minlength=2).tolist(),
                 records_private=details, groups=groups,
                 label_policy="WHO unadjusted adult15-65 M<13,F<12 g/dL; exclude women10.5<=Hb<12 with unknown pregnancy. Original paper <=10.5 labels deliberately not reused.")
    (output / "public_audit.json").write_text(json.dumps(audit, indent=2))
    (output / "local_color_audit.json").write_text(json.dumps(local_details, indent=2))
    cache = {"public_x": torch.stack(tensors), "public_y": torch.tensor(public_y), "public_subjects": subjects,
             "public_train": train, "public_validation": validation, "local_x": torch.stack(local_tensors),
             "local_y": torch.tensor(local["labels"]), "local_subjects": local["subjects"], "local_folds": local["folds"], "local_lab": local["lab"]}
    torch.save(cache, output / "inputs.pt")
    return {"public": {k: v for k, v in audit.items() if k not in {"records_private", "ancillary_workbook_cells_private", "training_subjects", "validation_subjects", "groups", "near_pairs"}},
            "local_subjects": len(local["subjects"]), "local_class_counts": np.bincount(local["labels"], minlength=2).tolist(),
            "local_color_policies": dict(Counter(d["color_policy"] for d in local_details)),
            "local_profiles": dict(Counter(d["input_profile"] for d in local_details)),
            "local_source_verification": local["baseline_verification"], "preprocessing": PREPROCESSING}
