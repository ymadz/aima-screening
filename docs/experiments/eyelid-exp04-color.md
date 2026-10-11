# Experiment 4 — local manual-eyelid color optimization

Owner-authorized exploratory experiment, 2026-10-11. Worktree `../aima-screening-exp04-color`, branch `experiment/eyelid-exp04-color-optimization`, verified reusable base `66afa8e`. Experiments 1–3, their ignored outputs, models, branches and preservation tag stay immutable. Source access is read-only through the original repository; no dataset copy or new annotations.

## Diagnostic and input gate

First reproduce saved models/metrics and checksum the original experiments, then freshly audit canonical metadata, current source/image/ROI/mask hashes and owner review. The original LAB nested thresholds identify 12 FN and 7 TP. Private reports use separate pseudonyms and identity keys. Hb severity is descriptive unadjusted [WHO 2024 adult severity](https://www.ncbi.nlm.nih.gov/books/NBK602183/table/fm-ch1.tab2/): mild >=11 g/dL below the sex-specific cutoff, moderate 8–<11, severe <8. No labels are changed. Non-pregnancy is an owner attestation outside the Sheet.

Exposure, Laplacian focus proxies, bright-neutral highlight proxies, profile/export histories and mask geometry are descriptive. Proxies are not accepted glare masks, quality exclusions, proof of white-balance error or evidence that poor quality caused a miss. The 12 missed cases cannot be a new independent tuning set.

## Rendering and features

`src/eyelid_color_data.py` implements full-resolution, mask-only features with identical geometry and uint8 RGB/LAB scaling. A reproduces the historical stored-TIFF RGB policy. B applies real embedded-profile-to-sRGB conversion; a pixel-identical oriented DNG preview supplies its genuine ICC in place of the misleading TIFF tag. Differently rendered Photos/Preview images retain their export profile; their original camera tone mapping is unknown. No image-level normalization, histogram equalization, white-balance correction, resizing, glare exclusion or learned enhancement is applied. The absence of color-reference capture prevents retrospective camera calibration.

Feature definitions are exported in the run configuration:

- `original`: historical six means/population standard deviations, order mean_L/std_L/mean_a/std_a/mean_b/std_b.
- `robust`: 18 total; six original, three medians, three IQRs, L/a/b 10th percentiles, a 90th percentile, median and IQR of natural log((R+1)/(G+1)). Percentiles use NumPy linear interpolation; the +1 handles zero channels. These retain every anatomical pixel, including glare.
- `regional`: 21 total; robust18 plus right-minus-left median L/a/b in two mask x-bounding-range halves. Image-coordinate regions are not anatomically homologous across captures. A missing side gives zero; no pixel or subject is excluded.

No age, sex, laboratory Hb, quality score, subject identifier or export method is a classification predictor. Features are deterministic; scaling is learned only from training subjects.

## Frozen candidate and evaluation contract

Exact original 119 eligible subjects, 19 anemia/100 non-anemia and five outer subject folds, seed 20261010. All are owner-authorized classifier development subjects; there is no classifier test reservation. The separate segmentation reservation is preserved. Three inner stratified subject folds use seed 20261011 and the same partitions for every candidate.

The bounded search contains 36 candidates: two renderings × three feature sets × six classifier settings. Logistic regression uses L2/lbfgs/max_iter=2000, training-only StandardScaler, C in {1, 0.1}, class weight in {none, balanced}. LightGBM has the existing conservative settings (100 trees, learning_rate=0.03, max_depth=2, num_leaves=3, min_child_samples=10, reg_lambda=5, min_split_gain=0.05, max_bin=31, deterministic single-thread), with none/balanced weights. No other hyperparameter search, oversampling, augmentation, calibration or feature selection.

Primary configuration selection maximizes pooled inner OOF average precision, ties follow the documented candidate order (A before B, original before robust before regional, C=1/unweighted before weighting or stronger regularization, logistic before LightGBM). This keeps configuration selection comparable with Experiments 2–3. Threshold selection maximizes inner OOF sensitivity subject to specificity >=0.80, ties higher specificity then higher cutoff, including an all-negative fallback. This is a research constraint, not an approved clinical threshold. Also predefine 0.70 and 0.90 constraints as secondary tradeoffs; do not choose a rule using outer results. Report fixed 0.5 and selected cutoffs separately.

Report render-specific inner-selected models and a joint inner-selected model as predefined policies; outer results cannot select a winner. Save every candidate's inner/outer models and decisions to support controlled matched rendering/feature/weighting comparisons. Outer labels are absent from model-selection APIs. All configurations and source/code hashes must be frozen before full training.

Metrics: all standard class metrics and counts, per-fold results, Wilson proportion intervals and 2,000 paired stratified subject bootstrap draws (seed 20261011). These condition on saved scores, fix prevalence, omit model-refit/overlapping-training dependence, and do not correct for repeated experiments. Since prior results informed this design on the same subjects, nested CV remains exploratory development evidence. No clinical or independent-confirmation claim.

After classification completes, a fixed Ridge(alpha=10) model on managed robust18 features and a training-mean Hb baseline may be assessed separately. Report g/dL MAE, RMSE and R²; anemia information uses the existing sex-specific cutoffs on predicted Hb. No Hb regression output becomes a clinical measurement.

## Artifact contract

Only new, exclusive, owner-private ignored `data/processed/local_eyelid_classification_exp04_color_optimization_<date>_v<version>/` outputs. Save fresh input/provenance/rendering audits, diagnostic findings, feature definitions/cache, frozen config, folds, all fitted models/thresholds/inner decisions, OOF rows, metrics and paired differences, aggregate ROC/PR/confusion/tradeoff figures, environment/code copies, manifest, checksums and saved-model reproduction instructions. Patient-level rows, source linkage and fitted models never enter Git. Validate mask alignment, profile handling, RGB order, LAB units, pixel ranges, finite features, training-only fitting and reproducibility before reporting completion.
