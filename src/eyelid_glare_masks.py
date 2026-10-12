"""Conservative specular proposals and private, de-identified review panels.

No function in this module records human approval or modifies source masks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .local_eyelid_validation import validate_arrays

STATUSES = ('APPROVED_GLARE', 'REVIEWED_NO_GLARE', 'AUTO_GENERATED_PENDING_REVIEW',
            'UNCERTAIN', 'INVALID_MASK')
DETECTOR = {
    'version': 'achromatic_seed_local_contrast_v1',
    'candidate_min_channel': 180, 'candidate_max_channel_range': 65,
    'seed_min_channel': 220, 'seed_max_channel_range': 35,
    'minimum_local_contrast': 15, 'sigma_fraction_sqrt_roi_area': .015,
    'minimum_component_pixels': 3, 'maximum_component_roi_fraction': .05,
    'review_removed_fraction': .03, 'review_broad_bright_fraction': .05,
    'definition': 'Compact bright near-neutral components with a whiter core and positive mask-normalized local minimum-channel contrast. Candidate pixels must lie inside approved anatomy.',
    'limitations': 'Unvalidated heuristic; pale tissue and chromatic/wet reflections can be confused or missed. No learned confidence. Red-channel clipping and low redness alone never trigger exclusion. Empty proposals require review.',
    'use': 'Diagnostics and review proposals only until explicit human acceptance.'}


def validate_glare(rgb: np.ndarray, anatomy: np.ndarray, glare: np.ndarray) -> dict[str, Any]:
    """Allow empty glare masks; reject any nonbinary, misaligned/outside pixels."""
    selected = validate_arrays(rgb, anatomy)
    if glare.dtype != np.uint8 or glare.shape != anatomy.shape or not np.isin(glare, [0, 255]).all():
        raise ValueError('Invalid glare mask geometry/type/values')
    marked = glare == 255
    if np.any(marked & ~selected):
        raise ValueError('Glare proposal escapes approved anatomy')
    retained = selected & ~marked
    if not retained.any():
        raise ValueError('Glare removes all anatomical tissue')
    return {'roi_pixels': int(selected.sum()), 'glare_pixels': int(marked.sum()),
            'retained_pixels': int(retained.sum()), 'removed_fraction': float(marked.sum()/selected.sum()),
            'outside_roi_pixels': 0, 'binary_valid': True, 'alignment_valid': True,
            'empty_proposal': not bool(marked.any())}


def propose_glare(rgb: np.ndarray, anatomy: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate native raster proposals from appearance, without clinical labels."""
    sel = validate_arrays(rgb, anatomy)
    yy, xx = np.where(sel)
    y0, y1, x0, x1 = yy.min(), yy.max()+1, xx.min(), xx.max()+1
    roi = sel[y0:y1, x0:x1]
    pixels = rgb[y0:y1, x0:x1].astype(np.float32)
    minimum = pixels.min(axis=2); span = np.ptp(pixels, axis=2)
    sigma = max(1., float(np.sqrt(sel.sum())*DETECTOR['sigma_fraction_sqrt_roi_area']))
    weight = cv2.GaussianBlur(roi.astype(np.float32), (0, 0), sigma)
    surrounding = cv2.GaussianBlur(minimum*roi, (0, 0), sigma)/np.maximum(weight, 1e-6)
    contrast = minimum-surrounding
    bright = roi & (minimum >= DETECTOR['candidate_min_channel']) & (span <= DETECTOR['candidate_max_channel_range'])
    candidate = bright & (contrast >= DETECTOR['minimum_local_contrast'])
    seed = candidate & (minimum >= DETECTOR['seed_min_channel']) & (span <= DETECTOR['seed_max_channel_range'])
    count, components, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
    accepted = np.zeros(roi.shape, bool); oversized = 0; selected_components = 0
    for component in range(1, count):
        region = components == component
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area > DETECTOR['maximum_component_roi_fraction']*sel.sum():
            oversized += 1
        elif area >= DETECTOR['minimum_component_pixels'] and np.any(seed & region):
            accepted |= region; selected_components += 1
    glare = np.zeros(anatomy.shape, np.uint8)
    glare[y0:y1, x0:x1] = accepted.astype(np.uint8)*255
    quality = validate_glare(rgb, anatomy, glare)
    reasons = []
    if oversized: reasons.append('oversized_bright_component_not_excluded')
    if bright.sum()/sel.sum() > DETECTOR['review_broad_bright_fraction']:
        reasons.append('broad_bright_tissue_requires_discrimination')
    if quality['removed_fraction'] > DETECTOR['review_removed_fraction']:
        reasons.append('substantial_removal_requires_review')
    if candidate.any() and not accepted.any():
        reasons.append('bright_contrast_without_conservative_core')
    status = 'UNCERTAIN' if reasons else 'AUTO_GENERATED_PENDING_REVIEW'
    group = 'uncertain_high_priority' if reasons else ('likely_glare_present' if accepted.any() else 'empty_proposal_pending_review')
    return glare, {**quality, 'status': status, 'review_group': group,
                   'review_reasons': ';'.join(reasons) or 'human_glare_review_absent',
                   'selected_components': selected_components, 'oversized_components': oversized,
                   'bright_candidate_fraction': float(bright.sum()/sel.sum()),
                   'confidence': 'unvalidated_rule_no_probability', 'verified_no_glare': False}


def panel(rgb: np.ndarray, anatomy: np.ndarray, glare: np.ndarray, record: dict[str, Any], path: Path) -> None:
    """Show cropped source pixels only, without a full face or original filename."""
    validate_glare(rgb, anatomy, glare)
    yy, xx = np.where(anatomy>0); pad = 12
    y0, y1 = max(0, yy.min()-pad), min(rgb.shape[0], yy.max()+pad+1)
    x0, x1 = max(0, xx.min()-pad), min(rgb.shape[1], xx.max()+pad+1)
    raw = rgb[y0:y1, x0:x1].copy(); roi = anatomy[y0:y1, x0:x1]>0; marked = glare[y0:y1, x0:x1]>0
    outlined = raw.copy()
    boundary = roi & ~(cv2.erode(roi.astype(np.uint8), np.ones((3,3), np.uint8))>0)
    outlined[boundary] = [0,255,0]
    overlay = outlined.copy(); overlay[marked] = (.35*raw[marked]+.65*np.array([0,220,255])).astype(np.uint8)
    retained = raw.copy(); retained[~roi | marked] = 32
    canvas = Image.new('RGB', (1440, 390), 'white'); draw = ImageDraw.Draw(canvas)
    try: font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 20)
    except OSError: font = ImageFont.load_default()
    draw.text((12, 8), f"{record['review_id']} | {record['status']} | removed {record['removed_fraction']:.2%}", fill='black', font=font)
    captions = ['Source eyelid crop', 'Approved anatomy: green edge', 'Proposed glare: cyan', 'Proposed retained tissue']
    for i, (values, caption) in enumerate(zip([raw, outlined, overlay, retained], captions)):
        thumb = Image.fromarray(values); thumb.thumbnail((350, 255), Image.Resampling.LANCZOS)
        canvas.paste(thumb, (i*360+(360-thumb.width)//2, 72+(255-thumb.height)//2))
        draw.text((i*360+8, 45), caption, fill='black', font=font)
    draw.text((12, 339), record['review_reasons'], fill='black', font=font)
    draw.text((12, 367), 'Proposal only. Empty mask is not confirmed no glare. No Hb/label shown.', fill='black', font=font)
    canvas.save(path)
