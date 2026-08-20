# Model Card: AIMA Screening Baseline

## Model Summary

This repository contains a research baseline for anemia screening from eye-region imagery. It uses feature-level multimodal fusion:

1. MobileNetV3-Small extracts a 576-value visual embedding from a canonical palpebral ROI.
2. OpenCV converts the ROI to CIE L*a*b* and computes six tissue color statistics.
3. A cutaneous margin from the full JPG supplies one ITA skin-tone angle.
4. LightGBM consumes the resulting 583 numeric values for binary classification.

The model is a research artifact and is not cleared for diagnosis, triage, or autonomous clinical decisions.

## Architecture

### Visual feature extractor

- Backbone: torchvision MobileNetV3-Small.
- Pretrained weights: ImageNet weights by default.
- Classifier head: removed.
- Input: RGB image resized to `224 x 224`.
- Preprocessing: ImageNet mean/std normalization.
- Output: adaptive-average-pooled `576`-element `float32` embedding.
- Runtime: evaluation mode with gradients disabled.

### Tabular fusion

The LightGBM input contains:

- `ita_angle`: 1 feature.
- `mean_L`, `std_L`, `mean_a`, `std_a`, `mean_b`, `std_b`: 6 features.
- `emb_0` through `emb_575`: 576 features.
- Total: 583 numeric inputs.

The following fields are retained in CSV outputs but excluded from `X`: `image_id`, `subject_id`, `country`, `gender`, `age`, `hgb`, and `label`.

### Classifier

- Library: LightGBM 4.7.0.
- Objective: binary classification.
- Boosting: GBDT with leaf-wise tree growth.
- Learning rate: `0.05`.
- Number of leaves: `31`.
- Maximum depth: `-1`.
- Feature fraction: `0.8`.
- Maximum boosting rounds: `150`.
- Early stopping: `20` rounds on each validation fold.
- Random seed: `42` for model and fold-related seeds.
- Threads: `1` for reproducible local execution.
- Decision threshold: probability `>= 0.5` maps to class `1`.

## Training and Evaluation

The baseline uses `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`. Each subject appears in one validation fold, and out-of-fold probabilities are saved to `results/cross_validation_predictions.csv`.

Verified mean cross-validation results:

| Metric | Result |
| --- | ---: |
| Accuracy | 80.41% |
| Sensitivity | 76.90% |
| Specificity | 82.73% |
| ROC-AUC | 0.8723 |

The dataset contains 184 unique country-subject pairs, with 115 non-anemic and 69 anemic labels. These results are not external validation results.

## Intended Use

- Research benchmarking of colorimetric and visual feature fusion.
- Reproducible experimentation with the downloaded Kaggle dataset.
- Investigation of subgroup error patterns and feature contributions.

## Out-of-Scope Use

- Clinical diagnosis or exclusion of anemia.
- Patient-level treatment decisions.
- Deployment without external validation, calibration, prospective testing, and appropriate clinical governance.
- Interpreting feature gain as evidence that a feature is biologically causal.

## Limitations and Risks

- The dataset is small and class-imbalanced.
- Skin-tone coverage is severely imbalanced: 182 subjects are Brown/Dark and only 2 are Very Light; the other ITA bands are absent.
- The ANOVA fairness result is therefore underpowered and cannot establish demographic neutrality.
- Images come from country-specific acquisition settings and may not represent other devices, lighting, populations, or clinical workflows.
- Metadata labels are derived from workbook hemoglobin values and sex-specific thresholds; rows with unavailable hemoglobin are excluded.
- The ITA implementation uses a fixed image-margin patch as a proxy for cutaneous skin. It is not a validated dermatological measurement protocol.
- The reported OOF metrics can still be optimistic relative to a genuinely external test set.

## Quantization and Edge Deployment

The MobileNet backbone is compact and suitable for investigation on edge hardware, but this repository does not perform quantization, pruning, ONNX conversion, or device benchmarking. FP16 readiness is not evidence of validated FP16 behavior here. The current pipeline runs CPU-oriented PyTorch inference and LightGBM; deployment requires:

1. A fixed, validated image acquisition and ROI protocol.
2. Export and numerical-equivalence tests for the MobileNet feature extractor.
3. Explicit FP32/FP16 or INT8 accuracy and latency measurements on the target device.
4. Packaging of the LightGBM model and exact feature-column order.
5. External clinical validation and monitoring for subgroup drift.

## Explainability

The repository records aggregate LightGBM gain importance in `results/feature_gain_importance.csv`. Gain ranks indicate how often split gain was accumulated during training; they do not provide per-patient explanations and should not be interpreted as causal physiology.

## Reproduction

```bash
source aima_env/bin/activate
python src/dataset_builder.py
python src/train_classifier.py
python src/evaluate.py
```
