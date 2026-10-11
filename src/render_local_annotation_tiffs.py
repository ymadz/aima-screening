"""Render missing local nail annotation TIFFs from the frozen image references.

The DNG path reproduces the RGB pixels in the ten existing pilot nail TIFFs:
decode the full-resolution embedded JPEG preview and apply its orientation.
Those pilot TIFFs carry an sRGB profile although the embedded JPEG declares
Display P3. We reuse the pilot TIFF profile for annotation consistency, without
claiming calibrated sRGB color. HEIC files are rendered to sRGB by macOS sips.
The raw originals and approved TIFFs are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import tifffile
from PIL import Image, ImageChops

from local_annotation_manifest import (
    ANNOTATIONS,
    FROZEN,
    PRIOR_MANIFEST,
    ROOT,
    read_csv,
    sha256,
    verify_metadata,
    write_csv,
)


INCOMING = ROOT / "data/raw/local/incoming"
MANIFEST = ANNOTATIONS / "manifests/annotation_manifest.csv"
INVENTORY = ANNOTATIONS / "manifests/image_ready_inventory.csv"
REPORT = ANNOTATIONS / "manifests/tiff_render_2026-10-08.json"
SYSTEM_SRGB = Path("/System/Library/ColorSync/Profiles/sRGB Profile.icc")
MIN_FREE_BYTES = 700 * 1024 * 1024
TIFF_COMPRESSION = "tiff_adobe_deflate"
TIFF_PREDICTOR = 2


def validate_hb_fields(metadata_rows: list[dict[str, str]]) -> None:
    """Require the two recorded Hb units to agree without deriving a label."""
    for row in metadata_rows:
        grams_per_liter = row["lab_hb_g_per_l"].strip()
        grams_per_deciliter = row["lab_hb_g_per_dl"].strip()
        if bool(grams_per_liter) != bool(grams_per_deciliter):
            raise ValueError("Frozen metadata has an incomplete Hb unit pair")
        if grams_per_liter:
            try:
                if Decimal(grams_per_liter) != 10 * Decimal(grams_per_deciliter):
                    raise ValueError("Frozen metadata Hb units disagree")
            except InvalidOperation as exc:
                raise ValueError("Frozen metadata has a nonnumeric Hb value") from exc


def read_dng_preview(source: Path) -> tuple[Image.Image, str]:
    """Return the oriented full-size JPEG preview pixels and input ICC hash."""
    with tifffile.TiffFile(source) as dng:
        page = dng.pages[0]
        if len(page.dataoffsets) != 1 or str(page.compression) != "COMPRESSION.JPEG":
            raise ValueError("DNG has no single embedded JPEG preview")
        orientation = int(page.tags["Orientation"].value)
        offset, length = page.dataoffsets[0], page.databytecounts[0]
    if orientation not in {1, 3, 6, 8}:
        raise ValueError("Unsupported DNG preview orientation")
    with source.open("rb") as handle:
        handle.seek(offset)
        jpeg = handle.read(length)
    with Image.open(io.BytesIO(jpeg)) as preview:
        preview.load()
        icc = preview.info.get("icc_profile", b"")
        if not icc:
            raise ValueError("DNG JPEG preview has no color profile")
        rgb = preview.convert("RGB")
    rotations = {
        1: None,
        3: Image.Transpose.ROTATE_180,
        6: Image.Transpose.ROTATE_270,
        8: Image.Transpose.ROTATE_90,
    }
    if rotations[orientation] is not None:
        rgb = rgb.transpose(rotations[orientation])
    return rgb, hashlib.sha256(icc).hexdigest()


def pilot_nail_profile(inventory: list[dict[str, str]]) -> tuple[bytes, str]:
    """Reuse the ICC profile of every existing approved pilot nail TIFF."""
    pilot = [row for row in inventory if row["modality"] == "nail"
             and row["anatomical_annotation_status"] == "valid_anatomical_roi_and_binary_mask"]
    if len(pilot) != 10:
        raise ValueError("Expected ten approved pilot nail TIFFs to establish the profile")
    profiles: set[bytes] = set()
    dng_profiles: set[str] = set()
    for row in pilot:
        image_path = ROOT / row["annotation_image_path"]
        if sha256(image_path) != row["annotation_image_sha256"]:
            raise ValueError("A pilot nail TIFF changed after inventory registration")
        with Image.open(image_path) as image:
            if image.format != "TIFF" or image.mode != "RGB":
                raise ValueError("Pilot nail image is not an RGB TIFF")
            profiles.add(image.info.get("icc_profile", b""))
        if Path(row["original_filename"]).suffix.lower() == ".dng":
            preview, dng_profile = read_dng_preview(ROOT / row["original_source_path"])
            preview.close()
            dng_profiles.add(dng_profile)
    if len(profiles) != 1 or not next(iter(profiles)) or len(dng_profiles) != 1:
        raise ValueError("Pilot nail source or TIFF color profiles disagree")
    return next(iter(profiles)), next(iter(dng_profiles))


def render_heic(source: Path, work_dir: Path) -> tuple[Image.Image, str]:
    """Render an oriented HEIC to RGB using macOS ColorSync's sRGB profile."""
    temporary_tiff = work_dir / "heic_render.tiff"
    result = subprocess.run(
        ["sips", "-m", str(SYSTEM_SRGB), "-s", "format", "tiff",
         str(source), "--out", str(temporary_tiff)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode or not temporary_tiff.is_file():
        raise ValueError("macOS could not render a referenced HEIC image")
    with Image.open(temporary_tiff) as rendered:
        rendered.load()
        if rendered.format != "TIFF" or rendered.mode != "RGB":
            raise ValueError("macOS HEIC render is not an RGB TIFF")
        icc = rendered.info.get("icc_profile", b"")
        if not icc:
            raise ValueError("macOS HEIC render has no sRGB profile")
        return rendered.copy(), hashlib.sha256(icc).hexdigest()


def verify_tiff(path: Path, intended: Image.Image, expected_icc: bytes) -> None:
    """Verify the lossless TIFF's pixels, profile, dimensions and TIFF tags."""
    with Image.open(path) as image:
        if image.format != "TIFF" or image.mode != "RGB" or image.size != intended.size:
            raise ValueError("Rendered TIFF mode or geometry differs from its RGB source")
        if image.info.get("icc_profile", b"") != expected_icc:
            raise ValueError("Rendered TIFF profile differs from the pilot nail profile")
        if image.tag_v2.get(259) != 8 or image.tag_v2.get(317) != TIFF_PREDICTOR:
            raise ValueError("Rendered TIFF lacks lossless Deflate/predictor encoding")
        if ImageChops.difference(image, intended).getbbox() is not None:
            raise ValueError("Rendered TIFF pixels differ from the intended RGB image")


def planned_images() -> tuple[list[dict[str, str]], list[dict[str, str]], str, str]:
    """Validate frozen metadata, source links, inventory and pending outputs."""
    metadata_rows, _, source_hash, source_mtime = verify_metadata()
    validate_hb_fields(metadata_rows)
    manifest = read_csv(MANIFEST)
    inventory = read_csv(INVENTORY)
    references = {
        (row["subject_id"], modality): row[f"raw_{modality}_file"].strip()
        for row in metadata_rows for modality in ("eyelid", "nail")
        if row[f"raw_{modality}_file"].strip()
    }
    registered = {(row["subject_id"], row["modality"]): row for row in inventory}
    manifest_by_key = {(row["subject_id"], row["modality"]): row for row in manifest}
    if len(references) != 275 or len(manifest_by_key) != len(manifest) or set(manifest_by_key) != set(references):
        raise ValueError("Frozen metadata and annotation manifest image references disagree")
    if len(registered) != len(inventory) or not set(registered).issubset(references):
        raise ValueError("Registered image inventory has duplicate or unreferenced images")
    incoming_names = Counter(path.name for path in INCOMING.iterdir() if path.is_file())
    stems: set[tuple[str, str]] = set()
    for key, original in references.items():
        if Path(original).name != original or incoming_names[original] != 1:
            raise ValueError("A frozen image reference does not resolve uniquely")
        stem_key = (key[1], Path(original).stem.casefold())
        if stem_key in stems:
            raise ValueError("Two referenced images would share an annotation TIFF name")
        stems.add(stem_key)
        row = manifest_by_key[key]
        if row["original_image_filename"] != original or row["metadata_source_sha256"] != source_hash:
            raise ValueError("Annotation manifest differs from the frozen metadata")
        source = INCOMING / original
        if row["source_image_path"] != source.relative_to(ROOT).as_posix():
            raise ValueError("Annotation manifest source path differs from the exact original")
        if sha256(source) != row["original_source_sha256"]:
            raise ValueError("An original image changed after its source hash was recorded")
    missing = [manifest_by_key[key] for key in references if key not in registered]
    if any(row["modality"] != "nail" or row["annotation_status"] != "pending" for row in missing):
        raise ValueError("Only pending nail TIFFs have an established render method")
    if any(Path(row["original_image_filename"]).suffix.lower() not in {".dng", ".heic"}
           for row in missing):
        raise ValueError("A pending nail source uses an unverified image format")
    return missing, inventory, source_hash, source_mtime


def inventory_row(
    source_row: dict[str, str], destination: Path, output_sha: str,
    source_hash: str, source_mtime: str, fields: list[str],
) -> dict[str, str]:
    """Register a TIFF without claiming an anatomical annotation exists."""
    row = {field: "" for field in fields}
    row.update({
        "subject_id": source_row["subject_id"],
        "modality": source_row["modality"],
        "original_filename": source_row["original_image_filename"],
        "original_source_path": source_row["source_image_path"],
        "original_source_exists": "true",
        "original_source_sha256": source_row["original_source_sha256"],
        "annotation_image_path": destination.relative_to(ROOT).as_posix(),
        "annotation_image_sha256": output_sha,
        "source_batch": "tiff_completion_2026-10-08",
        "source_manifest_path": PRIOR_MANIFEST.relative_to(ROOT).as_posix(),
        "anatomical_annotation_status": "missing_anatomical_annotation",
        "manifest_annotation_status": "pending",
        "image_review_status": "not_reviewed",
        "glare_roi_status": "not_created",
        "glare_status": "not_created",
        "metadata_source_path": str(FROZEN),
        "metadata_source_sha256": source_hash,
        "metadata_source_mtime": source_mtime,
    })
    return row


def run(*, dry_run: bool) -> dict[str, Any]:
    missing, inventory, source_hash, source_mtime = planned_images()
    formats = Counter(Path(row["original_image_filename"]).suffix.lower() for row in missing)
    if dry_run or not missing:
        return {"dry_run": dry_run, "frozen_subjects": 140,
                "metadata_image_references": 275, "registered_TIFFs": len(inventory),
                "TIFFs_to_create": len(missing), "formats": dict(formats),
                "available_bytes": shutil.disk_usage(ROOT).free}
    if formats[".heic"] and (not shutil.which("sips") or not SYSTEM_SRGB.is_file()):
        raise RuntimeError("macOS sips or the system sRGB profile is unavailable")
    icc_profile, expected_dng_icc_sha = pilot_nail_profile(inventory)
    inventory_sha_before = sha256(INVENTORY)
    new_rows: list[dict[str, str]] = []
    output_records: list[dict[str, Any]] = []
    created = 0
    reused = 0
    for source_row in missing:
        if shutil.disk_usage(ROOT).free < MIN_FREE_BYTES:
            raise OSError("Disk free space fell below the safe conversion reserve")
        source = INCOMING / source_row["original_image_filename"]
        destination = ANNOTATIONS / "pending" / "nail" / "images" / (source.stem + ".tiff")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".tiff-render-", dir=destination.parent) as work:
            work_dir = Path(work)
            if source.suffix.lower() == ".dng":
                rgb, decoded_icc_sha = read_dng_preview(source)
                if decoded_icc_sha != expected_dng_icc_sha:
                    raise ValueError("A DNG preview color profile differs from the pilot nail sources")
                method = "oriented_full_resolution_DNG_JPEG_preview_pixels"
            else:
                rgb, decoded_icc_sha = render_heic(source, work_dir)
                method = "macOS_sips_ColorSync_sRGB_HEIC_render"
            if destination.exists():
                verify_tiff(destination, rgb, icc_profile)
                reused += 1
            else:
                temporary_output = work_dir / "new_annotation.tiff"
                rgb.save(temporary_output, format="TIFF", compression=TIFF_COMPRESSION,
                         tiffinfo={317: TIFF_PREDICTOR}, icc_profile=icc_profile)
                verify_tiff(temporary_output, rgb, icc_profile)
                os.link(temporary_output, destination)
                created += 1
        output_sha = sha256(destination)
        new_rows.append(inventory_row(source_row, destination, output_sha,
                                      source_hash, source_mtime, list(inventory[0])))
        output_records.append({
            "subject_id": source_row["subject_id"],
            "modality": "nail",
            "original_filename": source_row["original_image_filename"],
            "source_path": source_row["source_image_path"],
            "source_sha256": source_row["original_source_sha256"],
            "annotation_image_path": destination.relative_to(ROOT).as_posix(),
            "annotation_image_sha256": output_sha,
            "method": method,
            "decoded_input_icc_sha256": decoded_icc_sha,
            "width_px": rgb.width,
            "height_px": rgb.height,
            "output_bytes": destination.stat().st_size,
        })
        rgb.close()
        if len(new_rows) % 25 == 0 or len(new_rows) == len(missing):
            free_gib = shutil.disk_usage(ROOT).free / (1024 ** 3)
            print(f"TIFFs prepared: {len(new_rows)}/{len(missing)}; free disk: {free_gib:.2f} GiB", flush=True)
    if sha256(INVENTORY) != inventory_sha_before:
        raise RuntimeError("Image inventory changed during TIFF rendering; outputs need reconciliation")
    write_csv(INVENTORY, inventory + new_rows, list(inventory[0]))
    report = {
        "rendered_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_metadata_path": str(FROZEN),
        "frozen_metadata_sha256": source_hash,
        "frozen_metadata_mtime": source_mtime,
        "registered_before": len(inventory),
        "registered_after": len(inventory) + len(new_rows),
        "new_TIFFs_created": created,
        "existing_matching_TIFFs_registered": reused,
        "source_format_counts": dict(formats),
        "TIFF_encoding": "RGB 8-bit, lossless Adobe Deflate, horizontal predictor 2",
        "pilot_nail_output_icc_sha256": hashlib.sha256(icc_profile).hexdigest(),
        "DNG_color_note": (
            "The embedded DNG JPEG declares Display P3. Its RGB pixels are copied without "
            "a profile transform to match the ten existing pilot nail TIFFs, and carry their "
            "sRGB built-in ICC tag. These annotation copies are not a calibrated color input."
        ),
        "HEIC_color_note": (
            "macOS sips renders HEIC through ColorSync to system sRGB before saving "
            "with the existing pilot nail sRGB ICC tag."
        ),
        "outputs": output_records,
    }
    temporary_report = REPORT.with_name(REPORT.name + ".tmp")
    temporary_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_report, REPORT)
    return {"dry_run": False, "frozen_subjects": 140,
            "metadata_image_references": 275, "registered_TIFFs": len(inventory) + len(new_rows),
            "TIFFs_created": created, "existing_matching_TIFFs_registered": reused,
            "formats": dict(formats), "output_bytes": sum(row["output_bytes"] for row in output_records),
            "available_bytes": shutil.disk_usage(ROOT).free,
            "provenance_report": REPORT.relative_to(ROOT).as_posix()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate and count missing TIFFs only")
    args = parser.parse_args()
    print(json.dumps(run(dry_run=args.dry_run), indent=2))
