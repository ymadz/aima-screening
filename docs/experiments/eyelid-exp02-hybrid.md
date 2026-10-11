# Experiment 2: local manual-eyelid hybrid

The owner authorized this isolated experiment on 2026-10-11, with the same 119 subjects and five saved outer folds as Experiment 1. The source dataset and baseline outputs are opened read-only through an explicit `--source-root`; they are not copied into the native worktree. Outputs must be a new Git-ignored directory named `data/processed/local_eyelid_classification_exp02_hybrid_*` under that source root. Existing outputs are never overwritten.

Experiment 1 preservation succeeded before this worktree was created: branch `archive/eyelid-exp01-cielab-baseline`, annotated tag `eyelid-exp01-cielab-baseline-2026-10-11`, commit `4bf6e2577d02b5e3338e1937140b7eaff31369b4`. The owner-only archive at `~/Library/Application Support/AIMA/experiment-archives/eyelid-exp01-cielab-baseline_2026-10-11_v1` verified all 97 copied files and reproduced all ten saved models without fitting. Original output/readiness files remain unchanged.

## Prespecified input and modeling design

- Use the approved manual anatomical masks and unchanged stored annotation TIFF RGB. Retain glare. No enhancement, automatic segmentation, nails, demographics or Hb predictors; Hb only supplies verified labels.
- Recompute the exact six Experiment 1 full-resolution masked LAB statistics and require equality with its saved feature cache.
- Extract 576 frozen MobileNetV3-Small ImageNet1K_V1 features on CPU in eval/inference mode. Load the existing public weight file offline, checking its exact SHA-256. No neural parameters or batch-normalization statistics are fitted locally.
- Before interpolation, replace every outside-mask pixel with ImageNet channel means. Crop to the manual-mask bounding box, bilinearly resize with antialiasing to fit 224×224 while preserving aspect ratio, and center on a mean-filled canvas. Normalize float RGB/255 with means [0.485, 0.456, 0.406] and standard deviations [0.229, 0.224, 0.225]. This explicit ROI geometry replaces the official resize-256/center-crop-224 so the annotated tissue remains inside the canvas. LAB extraction stays unchanged.
- Fit embedding StandardScaler and PCA with eight components inside every training fold, concatenate these components with the six original LAB values, then fit a conservative LightGBM classifier. PCA/scaling are never fitted globally.
- Use the unchanged five outer folds; within each outer training partition, run three stratified subject folds with seed 20261011. Compare exactly two candidates differing only in no class weighting versus balanced weighting computed on each fit's training labels. Choose highest pooled inner OOF average precision; ties favor no weighting.
- Fixed tree settings: 100 estimators, learning rate 0.03, depth 2, three leaves, minimum 10 child samples, L2 penalty 5, minimum split gain 0.05, max_bin 31; deterministic single-thread CPU, no early stopping, augmentation or oversampling. No further tuning against outer results.
- In the selected candidate's inner OOF scores, maximize sensitivity subject to specificity ≥0.80; ties favor greater specificity then the higher threshold. This research objective is not a clinical target or guarantee on outer subjects. Fit the selected pipeline on the complete outer training partition. Score outer subjects once, at fixed 0.5 and this training-selected threshold.

## Planned evaluation and limits

Report accuracy, sensitivity, specificity, balanced accuracy, precision, F1, ROC-AUC, average precision and confusion matrices. Preserve all inner assignments, candidate predictions/parameters, selected thresholds, fitted transformers and inner/outer models privately. Paired stratified subject bootstrap intervals compare the same OOF subjects against Experiment 1; proportion intervals use Wilson intervals. All intervals are conditional on the observed OOF scores, fix prevalence and omit refit/overlapping-training uncertainty. Inner scores support selection only; they are not reported as independent performance.

The mixed rendering histories, uncalibrated color, only 19 positive cases, mask-shape/background cues and reuse of the same development cohort limit interpretation. Class weighting can distort probability calibration. This is neither an external validation nor public-anemia transfer experiment. Numerical gains alone cannot establish superiority or clinical screening readiness.

Official technical references: [Torchvision MobileNetV3-Small weights and normalization](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html), [LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html). The run records the installed versions rather than assuming the latest documentation version.

Current implementation milestone: deterministic masked feature extraction and source/output guards are implemented; hybrid fitting and evaluation follow in a separate feature commit.
