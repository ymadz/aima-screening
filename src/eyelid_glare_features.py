"""Controlled glare diagnostics and orientation-invariant ROI-relative regions."""
from __future__ import annotations

import cv2
import numpy as np
from .eyelid_color_data import color_features, ROBUST_NAMES
from .eyelid_glare_masks import validate_glare
from .local_eyelid_validation import validate_arrays

REGIONAL_NAMES = (*ROBUST_NAMES, 'core_minus_rim_median_L', 'core_minus_rim_median_a',
                  'core_minus_rim_median_b', 'core_minus_rim_median_log_red_green')
FEATURES = {'robust': ROBUST_NAMES, 'regional': REGIONAL_NAMES}
REGIONS = {'version': 'manual_roi_distance_core_rim_v1',
           'definition': 'Euclidean distance to original manual ROI boundary; core distance >= median positive distance, rim below median. Geometry fixed before glare exclusion. Medians produce four core-minus-rim contrasts.',
           'orientation': 'Invariant to image-axis flips/90-degree rotations; no left/right anatomical correspondence is asserted. Core/rim are ROI-relative sampling regions, not new anatomical annotations.',
           'features': list(REGIONAL_NAMES), 'degenerate_regions': 'Block extraction if either region is empty.',
           'motivation': 'Four bounded geometry-relative contrasts in lightness, a/b and natural-log R/G. No clinical or source-history predictors.'}


def region_masks(anatomy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute native geometry once from original anatomy, including boundary."""
    sel = anatomy == 255
    padded = np.pad(sel.astype(np.uint8), 1)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    cutoff = np.median(distance[sel])
    core, rim = sel & (distance >= cutoff), sel & (distance < cutoff)
    if not core.any() or not rim.any():
        raise ValueError('Manual ROI cannot support two distance regions')
    return core, rim


def extract(rgb: np.ndarray, anatomy: np.ndarray, glare: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Original/excluded use identical pixels, conversion and fixed geometry."""
    validate_arrays(rgb, anatomy)
    if glare is None: glare = np.zeros_like(anatomy)
    validate_glare(rgb, anatomy, glare)
    retained = (anatomy == 255) & (glare == 0)
    robust = color_features(rgb, retained.astype(np.uint8)*255)['robust']
    core, rim = region_masks(anatomy)
    core &= retained; rim &= retained
    if not core.any() or not rim.any():
        raise ValueError('Glare exclusion empties a sampling region')
    lab = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2LAB).astype(np.float64)
    lab[:,:,0] *= 100/255; lab[:,:,1:] -= 128
    delta = np.median(lab[core], axis=0)-np.median(lab[rim], axis=0)
    redness = np.log((rgb[:,:,0].astype(float)+1)/(rgb[:,:,1].astype(float)+1))
    regional = np.r_[robust, delta, np.median(redness[core])-np.median(redness[rim])]
    if not np.isfinite(regional).all(): raise ValueError('Nonfinite regional features')
    return {'robust': robust, 'regional': regional}


def require_verified_review(records: list[dict]) -> None:
    """Never promote candidates/empty proposals into validated exclusion inputs."""
    if not records or any(r.get('status') not in {'APPROVED_GLARE','REVIEWED_NO_GLARE'} or
                          not r.get('reviewer') or not r.get('reviewed_mask_sha256') for r in records):
        raise ValueError('Glare review evidence is incomplete; exclusion modeling blocked')
