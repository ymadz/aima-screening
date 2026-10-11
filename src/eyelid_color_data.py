"""Deterministic native-resolution color features from approved manual anatomy."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import cv2
import numpy as np
from .eyelid_transfer_data import local_pixels
from .local_eyelid_classifier import FEATURES, extract_masked_lab
from .local_eyelid_validation import load_manual_pair, validate_arrays

ROBUST_NAMES = (*FEATURES, 'median_L', 'median_a', 'median_b', 'iqr_L', 'iqr_a', 'iqr_b',
                'p10_L', 'p10_a', 'p10_b', 'p90_a', 'median_log_red_green', 'iqr_log_red_green')
REGIONAL_NAMES = (*ROBUST_NAMES, 'right_minus_left_median_L', 'right_minus_left_median_a',
                  'right_minus_left_median_b')
FEATURE_SETS = {'original': FEATURES, 'robust': ROBUST_NAMES, 'regional': REGIONAL_NAMES}
PREPROCESSING = {
    'version': 'native_manual_mask_original_vs_icc_srgb_v1',
    'A': 'Identical Experiment 1 TIFF RGB uint8, stored values assumed sRGB; no ICC transform',
    'B': 'Pillow RGB decode and actual ICC-to-sRGB relative-colorimetric transform; verified identical DNG preview ICC overrides misleading TIFF tag; all other aligned TIFFs use their embedded ICC',
    'lab': 'OpenCV RGB2LAB uint8; L*=encoded_L*100/255; a*=encoded_a-128; b*=encoded_b-128; same quantization in both pipelines',
    'geometry': 'Native aligned TIFF raster and identical approved manual mask, no resizing/crop/interpolation/background pixels',
    'glare': 'All anatomical pixels retained; diagnostic proxies never used as exclusions',
    'enhancement': 'None; no histogram equalization, white balance, color normalization or augmentation',
    'calibration_limit': 'An ICC conversion is not camera calibration. Photos/Preview/unknown export tone curves cannot be retrospectively inverted. No color-chart evidence.',
    'unknown_profiles': 'Local TIFF without ICC blocks processing. Original DNG/HEIC metadata are audited; a source with differing rendered pixels does not override an export profile merely by filename.',
}
FEATURE_DEFINITIONS = {
    'original': 'Six mean/population std(ddof=0) full-resolution masked quantized CIELAB statistics in historical order.',
    'robust': 'Original six plus L/a/b medians and IQRs(q75-q25), L/a/b q10, a q90, median and IQR of log((R+1)/(G+1)). NumPy linear percentile interpolation. RGB is gamma-encoded uint8; +1 avoids zero; log is natural. Red/green is an uncalibrated tissue redness descriptor, not Hb.',
    'regional': 'Robust18 plus median L/a/b right-minus-left differences. Regions divide the occupied mask x-bounding range at (xmin+xmax)/2; equal weighting by region median. Empty side gives 0 and is recorded by geometry tests. Image-coordinate halves are not anatomically homologous across captures.',
    'motivation': 'Robust location/spread reduce dependence on extreme pixels without removing glare; tails describe color heterogeneity; three regional differences test a limited spatial extension. No Hb/sex/age/quality/source identifiers enter classification features.',
}


def color_features(rgb: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Return finite deterministic six/eighteen/twenty-one-value configurations."""
    selected = validate_arrays(rgb, mask)
    original = np.array(list(extract_masked_lab(rgb, mask).values()), dtype=np.float64)
    encoded = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2LAB)
    lab = encoded[selected].astype(np.float64)
    lab[:, 0] *= 100 / 255
    lab[:, 1:] -= 128
    q10, q25, q50, q75, q90 = np.percentile(lab, [10, 25, 50, 75, 90], axis=0, method='linear')
    pix = rgb[selected].astype(np.float64)
    redness = np.log((pix[:, 0] + 1) / (pix[:, 1] + 1))
    redq = np.percentile(redness, [25, 50, 75], method='linear')
    robust = np.r_[original, q50, q75-q25, q10, q90[1], redq[1], redq[2]-redq[0]]
    xx = np.where(selected)[1]
    left = xx <= (xx.min() + xx.max()) / 2
    delta = np.median(lab[~left], axis=0)-np.median(lab[left], axis=0) if left.any() and (~left).any() else np.zeros(3)
    result = {'original': original, 'robust': robust, 'regional': np.r_[robust, delta]}
    for name, values in result.items():
        if values.shape != (len(FEATURE_SETS[name]),) or not np.isfinite(values).all():
            raise ValueError('Nonfinite or incorrectly ordered color features')
    return result


def render_pair(source_root: Path, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Reuse verified DNG preview handling while preserving annotation coordinates."""
    raw, mask = load_manual_pair(source_root / row['annotation_image_path'], source_root / row['mask_path'])
    managed, second_mask, audit = local_pixels(source_root, row)
    validate_arrays(managed, mask)
    np.testing.assert_array_equal(mask, second_mask)
    if raw.shape != managed.shape:
        raise ValueError('Color management changed mask alignment')
    return raw, managed, mask, {**audit, 'raw_decoded_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
                               'managed_decoded_sha256': hashlib.sha256(managed.tobytes()).hexdigest(),
                               'height': raw.shape[0], 'width': raw.shape[1], 'dtype': str(raw.dtype),
                               'mask_pixels': int(np.count_nonzero(mask)), 'alignment': 'same raster, identical mask'}


def extract_all(source_root: Path, inputs: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Extract label-independent candidates; verify original features exactly."""
    collected = {f'{p}_{f}': [] for p in ('A', 'B') for f in FEATURE_SETS}
    audits = []
    for i, sid in enumerate(inputs['subjects']):
        raw, managed, mask, audit = render_pair(source_root, inputs['rows'][sid])
        a, b = color_features(raw, mask), color_features(managed, mask)
        np.testing.assert_allclose(a['original'], inputs['lab'][i], rtol=0, atol=1e-14)
        for p, values in [('A', a), ('B', b)]:
            for name, vector in values.items(): collected[f'{p}_{name}'].append(vector)
        audits.append({'subject_id': sid, **audit})
        if (i+1) % 20 == 0: print(f'Extracted {i+1} color inputs', flush=True)
    return {k: np.array(v, dtype=np.float64) for k, v in collected.items()}, audits
