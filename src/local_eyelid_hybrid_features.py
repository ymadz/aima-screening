"""Private, read-only source access and frozen manual-eyelid hybrid features."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode

from .local_eyelid_classifier import FEATURES, extract_masked_lab, validate_feature_rows
from .local_eyelid_validation import audit_folds, load_manual_pair, sha256, strict_csv, validate_arrays
from .preserve_local_experiment import verify_saved_cv

WEIGHTS_SHA256 = '047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f'
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PREPROCESSING = {
    'version': 'manual_roi_bbox_mean_letterbox_224_v1',
    'pixels': 'Unchanged stored annotation TIFF RGB assumed sRGB; mixed historical rendering',
    'lab': 'Reuse Experiment 1 extract_masked_lab on full-resolution approved mask; six means/population stds',
    'embedding': 'Mask outside tissue with ImageNet mean BEFORE resize; tight manual-mask bbox; aspect-preserving bilinear antialiased resize to fit 224; center letterbox with mean; float RGB/255 and ImageNet mean/std normalization',
    'size': 224, 'mean': list(MEAN), 'std': list(STD),
    'geometry_note': 'Explicit ROI-specific geometry replaces official 256-resize/224-center-crop to retain the entire annotated tissue',
    'glare': 'Retained', 'enhancement': 'None', 'augmentation': 'None',
    'backbone': 'MobileNetV3-Small ImageNet1K_V1 features+avgpool, frozen eval CPU, 576 dimensions',
}


def manual_roi_tensor(image: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    """Exclude outside pixels before interpolation; never crop annotated tissue."""
    validate_arrays(image, mask)
    selected = mask > 0
    yy, xx = np.where(selected)
    crop = image[yy.min():yy.max()+1, xx.min():xx.max()+1].astype(np.float32) / 255.0
    crop_mask = selected[yy.min():yy.max()+1, xx.min():xx.max()+1]
    crop[~crop_mask] = MEAN
    tensor = torch.from_numpy(crop).permute(2, 0, 1)
    height, width = tensor.shape[1:]
    scale = 224 / max(height, width)
    shape = [max(1, round(height * scale)), max(1, round(width * scale))]
    tissue = resize(tensor, shape, interpolation=InterpolationMode.BILINEAR, antialias=True)
    mean = torch.tensor(MEAN, dtype=torch.float32)[:, None, None]
    canvas = mean.expand(3, 224, 224).clone()
    top, left = (224-shape[0])//2, (224-shape[1])//2
    canvas[:, top:top+shape[0], left:left+shape[1]] = tissue
    return (canvas - mean) / torch.tensor(STD, dtype=torch.float32)[:, None, None]


class FrozenMobileNet:
    """Load exact cached public ImageNet weights; no network or label fitting."""
    def __init__(self, weights_path: Path):
        if not weights_path.is_file() or sha256(weights_path) != WEIGHTS_SHA256:
            raise ValueError('Expected verified cached ImageNet1K_V1 weights')
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self.model = mobilenet_v3_small(weights=None).cpu().eval()
        self.model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
        self.model.requires_grad_(False)
        self.metadata = {'variant': 'MobileNet_V3_Small_Weights.IMAGENET1K_V1',
                         'url': MobileNet_V3_Small_Weights.IMAGENET1K_V1.url,
                         'file': str(weights_path.resolve()), 'sha256': sha256(weights_path),
                         'dimensions': 576, 'frozen': True, 'device': 'cpu'}

    def extract(self, tensor: torch.Tensor) -> np.ndarray:
        if tensor.shape != (3, 224, 224) or not torch.isfinite(tensor).all():
            raise ValueError('Invalid normalized ROI tensor')
        with torch.inference_mode():
            embedding = self.model.avgpool(self.model.features(tensor.unsqueeze(0))).flatten(1)[0].numpy().copy()
        if embedding.shape != (576,) or not np.isfinite(embedding).all():
            raise ValueError('Invalid frozen embedding')
        return embedding


def private_output_path(source_root: Path, output: Path) -> Path:
    """Only new, explicitly ignored Experiment 2 outputs may be created."""
    root, output = source_root.resolve(), output.resolve()
    if not output.is_relative_to(root / 'data/processed') or not output.name.startswith('local_eyelid_classification_exp02_hybrid_'):
        raise ValueError('Require distinct versioned private Experiment 2 output')
    if output.exists():
        raise FileExistsError('Never overwrite an experiment')
    ignored = subprocess.run(['git', 'check-ignore', '-q', str(output)], cwd=root)
    if ignored.returncode != 0:
        raise ValueError('Output must be Git-ignored')
    return output


def validated_inputs(source_root: Path, experiment1: Path, bundle: Path) -> dict[str, Any]:
    """Recheck the immutable baseline and source links before extracting features."""
    proof = verify_saved_cv(experiment1, bundle, source_root)
    rows = strict_csv(bundle / 'verified_eyelid_inputs.csv')
    by_id = {r['subject_id']: r for r in rows}
    subjects = sorted(by_id)
    labels = np.array([int(by_id[s]['derived_anemia_label']) for s in subjects])
    folds = strict_csv(experiment1 / 'subject_folds.csv')
    audit_folds(dict(zip(subjects, labels)), folds)
    assignments = {r['subject_id']: int(r['fold']) for r in folds}
    lab = validate_feature_rows(strict_csv(bundle / 'features.csv'), subjects)
    return {'subjects': subjects, 'labels': labels, 'folds': np.array([assignments[s] for s in subjects]),
            'lab': lab, 'rows': by_id, 'baseline_verification': proof}


def extract_features(source_root: Path, inputs: dict[str, Any], backbone: FrozenMobileNet) -> np.ndarray:
    """Recompute LAB exactly and extract label-independent frozen embeddings."""
    embeddings = []
    for index, subject in enumerate(inputs['subjects']):
        row = inputs['rows'][subject]
        image, mask = load_manual_pair(source_root / row['annotation_image_path'], source_root / row['mask_path'])
        features = extract_masked_lab(image, mask)
        np.testing.assert_allclose([features[name] for name in FEATURES], inputs['lab'][index], rtol=0, atol=1e-14)
        embeddings.append(backbone.extract(manual_roi_tensor(image, mask)))
        if (index + 1) % 10 == 0 or index + 1 == len(inputs['subjects']):
            print(json.dumps({'feature_extraction_completed': index+1, 'total': len(inputs['subjects'])}), flush=True)
    return np.concatenate([inputs['lab'], np.array(embeddings, dtype=np.float64)], axis=1)
