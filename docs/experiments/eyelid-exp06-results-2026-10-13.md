# Experiment 6 verified results — 2026-10-13

The simple ensembles did not provide a convincing benefit over the predefined robust CIELAB comparison. Equal averaging at the primary training-selected specificity constraint detected **11/19 at 79% specificity**, versus robust LAB **10/19 at 84%**: one additional detection and five additional false positives. The training-selected weighted policy detected **10/19 at 79%**, with no net detection gain and five additional false positives. Retain robust color-only classification as the practical exploratory candidate for independent confirmation. No clinical superiority or external validation is established.

## Preservation and alignment

Branch `experiment/eyelid-exp06-ensemble`; worktree `/Users/madz/Documents/aima-screening-exp06-ensemble`; implementation base `0943363b874c612be6ac241de8bd3819c55ccc50`, with protocol/helper commit `bc116af`, evaluator commit `f501039` and image-audit commit `23b04d6`. No merges, earlier reference changes, pushes or alterations of original labels/masks/images. Main was clean at `717da2c`; all earlier experiment worktrees were clean. The detached historical worktree retained its four unrelated untracked documentation files. Main’s centralized records had not incorporated Experiment 5; the new branch inherits its completed central records from the Experiment 5 handoff and adds Experiment 6 to the same six documents. Main and previous worktrees stay unchanged.

Canonical frozen CSV: mtime `2026-10-07T06:55:39.254357+00:00`, SHA-256 `1e704fb8d575757f16fc4f690cdc3315bc4063a5cf258d6c2e1b5f784d19596d`; protocol PDF: mtime `2026-09-22T10:39:41.367195+00:00`, SHA-256 `1b575406391f017d246d505704d3e077745c526f59ffef4b266b02ad026f61d0`. Recorded before source reading. Fresh metadata audit checked schema/trailing fields, unique IDs, both Hb fields/units, age/sex, all 275 source references and prior manifests; no discrepancy. There are 140 subjects, 119 eligible eyelids (19 anemia/100 non-anemia), and 21 historical exclusions. Non-pregnancy is an owner attestation outside the Sheet.

Fresh source checks verified all 140 reviewed eyelids, 560 source/TIFF/ROI/mask hashes, review-preview pixel equality, no exact source/pixel duplicates and the existing 46 owner-adjudicated distinct-source similarity pairs. Classifier folds are unchanged: folds 1–4 contain 4/20 anemia/non-anemia and fold 5 contains 3/20. The segmentation reservation is separate and preserved. There is no untouched classifier holdout.

Experiment 4 independently reproduced 720 classification and ten regression pipelines, 810 checksum-covered files and all source features. Experiment 5 independently reproduced 122 neural and 60 classical models, 637 files, all 370 original input tensors/both-view prefix maps, selections, thresholds, probabilities and metrics. These checks fitted no model and changed no prior artifact.

## Historical model distinction

The compared LAB model is original-render robust18, unweighted logistic C=1, a predefined candidate comparison rather than Experiment 4’s joint primary policy. Its historical cutoff policy detects 10/19, with 16 false positives, ROC-AUC 0.709474 and AP 0.385148. Experiment 4’s primary joint policy remains separate: 7/19 at 83%. The CP model is Experiment 5 classification candidate C: frozen CP backbone features + robust18, training-only embedding PCA8 and fused scaling, balanced logistic C=0.1. It detects 12/19, with 22 false positives, ROC-AUC 0.734211 and AP 0.372166. Experiment 5’s primary joint policy detects 11/19 at 78% and is not substituted for candidate C.

## Prediction complementarity

Anemia overlap: **8 shared detections, 2 LAB-only, 4 CP-only, 5 shared misses**. Non-anemia overlap: **78 shared clearances, 16 shared false positives, zero LAB-only false positives, 6 CP-only false positives**. The descriptive union would identify 14/19 with 22 false positives, but it was not implemented or selected as an ensemble policy. Differing individual results alone do not establish added information.

Pearson correlation 0.801841 (conditional 95% bootstrap interval 0.750307–0.849239); Spearman 0.823465 (0.746212–0.880460). Binary agreement 107/119 = 89.915966% (Wilson 83.198484–94.136970%); disagreements 12/119 = 10.084034% (5.863030–16.801516%). Anemia disagreement is 6/19; non-anemia disagreement is 6/100. CP is uniquely correct on four subjects and LAB on eight; descriptive exact McNemar p=0.387695. These conditional statistics neither account for model refitting nor establish independent superiority.

The two LAB-only and four CP-only positive detections supported a small feasibility test despite high correlation. The gate and base-model choice followed earlier outer results, so this is exploratory research selection. The protocol in [eyelid-exp06-ensemble.md](eyelid-exp06-ensemble.md) was frozen before any new ensemble outer outcomes.

## Methods actually tested

Equal raw-probability average and three LAB weights (0.50, 0.75, 0.25) were evaluated. The weighted policy chooses highest inner OOF AP, with ties in that frozen order; chosen LAB weights by outer fold were **0.50, 0.50, 0.25, 0.25, 0.25**. No stacking or calibration was fitted. The small positive support and strong score correlation favored testing simpler combinations first. No neural model or base classifier was refitted.

Both bases have identical outer-training subject order, labels and three inner partitions (seed 20261011). Forty saved inner/outer pipelines were reproduced with verified `classes_=[0,1]` and training-only scaler/PCA statistics. Inner scores arise entirely inside each outer training fold. Weights and cutoffs use those matched scores, never full-cohort OOF predictions. The fitting/selection API accepts no held-out labels.

Thresholds maximize inner sensitivity at specificity ≥0.80 (primary), ≥0.85 or ≥0.90 (secondary), breaking ties by specificity and higher cutoff. Fixed 0.5 is reported separately. Base models were evaluated under these same target policies; historical primary policies retain their own previous thresholds. The inner constraints do not guarantee their numerical targets on held-out subjects. LAB is unweighted while CP is class-balanced, so raw scales differ and neither is an adult calibrated risk.

## Exact pooled outer OOF outcomes

All rules below are internal five-fold estimates on the same reused 119 subjects. Secondary candidate rows are descriptive, not outer-selected winners. Counts are TP/FN/TN/FP; confusion matrices are `[[TN,FP],[FN,TP]]`.

| Model / rule | Accuracy | Sensitivity | Specificity | Balanced accuracy | Precision | F1 | ROC-AUC | AP | TP/FN/TN/FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exp04_primary_historical | 0.756303 | 0.368421 | 0.830000 | 0.599211 | 0.291667 | 0.325581 | 0.656842 | 0.352228 | 7/12/83/17 |
| exp05_primary_historical | 0.747899 | 0.578947 | 0.780000 | 0.679474 | 0.333333 | 0.423077 | 0.734211 | 0.372546 | 11/8/78/22 |
| exp04_robust_historical | 0.789916 | 0.526316 | 0.840000 | 0.683158 | 0.384615 | 0.444444 | 0.709474 | 0.385148 | 10/9/84/16 |
| exp05_cp_C_historical | 0.756303 | 0.631579 | 0.780000 | 0.705789 | 0.352941 | 0.452830 | 0.734211 | 0.372166 | 12/7/78/22 |
| lab_fixed | 0.823529 | 0.105263 | 0.960000 | 0.532632 | 0.333333 | 0.160000 | 0.709474 | 0.385148 | 2/17/96/4 |
| lab_spec0.8 | 0.789916 | 0.526316 | 0.840000 | 0.683158 | 0.384615 | 0.444444 | 0.709474 | 0.385148 | 10/9/84/16 |
| lab_spec0.85 | 0.798319 | 0.263158 | 0.900000 | 0.581579 | 0.333333 | 0.294118 | 0.709474 | 0.385148 | 5/14/90/10 |
| lab_spec0.9 | 0.806723 | 0.210526 | 0.920000 | 0.565263 | 0.333333 | 0.258065 | 0.709474 | 0.385148 | 4/15/92/8 |
| cp_fixed | 0.722689 | 0.631579 | 0.740000 | 0.685789 | 0.315789 | 0.421053 | 0.734211 | 0.372166 | 12/7/74/26 |
| cp_spec0.8 | 0.756303 | 0.631579 | 0.780000 | 0.705789 | 0.352941 | 0.452830 | 0.734211 | 0.372166 | 12/7/78/22 |
| cp_spec0.85 | 0.739496 | 0.315789 | 0.820000 | 0.567895 | 0.250000 | 0.279070 | 0.734211 | 0.372166 | 6/13/82/18 |
| cp_spec0.9 | 0.781513 | 0.210526 | 0.890000 | 0.550263 | 0.266667 | 0.235294 | 0.734211 | 0.372166 | 4/15/89/11 |
| equal_fixed | 0.789916 | 0.263158 | 0.890000 | 0.576579 | 0.312500 | 0.285714 | 0.737895 | 0.379070 | 5/14/89/11 |
| equal_spec0.8 | 0.756303 | 0.578947 | 0.790000 | 0.684474 | 0.343750 | 0.431373 | 0.737895 | 0.379070 | 11/8/79/21 |
| equal_spec0.85 | 0.773109 | 0.368421 | 0.850000 | 0.609211 | 0.318182 | 0.341463 | 0.737895 | 0.379070 | 7/12/85/15 |
| equal_spec0.9 | 0.789916 | 0.157895 | 0.910000 | 0.533947 | 0.250000 | 0.193548 | 0.737895 | 0.379070 | 3/16/91/9 |
| lab75_fixed | 0.831933 | 0.210526 | 0.950000 | 0.580263 | 0.444444 | 0.285714 | 0.741053 | 0.394325 | 4/15/95/5 |
| lab75_spec0.8 | 0.781513 | 0.526316 | 0.830000 | 0.678158 | 0.370370 | 0.434783 | 0.741053 | 0.394325 | 10/9/83/17 |
| lab75_spec0.85 | 0.789916 | 0.473684 | 0.850000 | 0.661842 | 0.375000 | 0.418605 | 0.741053 | 0.394325 | 9/10/85/15 |
| lab75_spec0.9 | 0.815126 | 0.210526 | 0.930000 | 0.570263 | 0.363636 | 0.266667 | 0.741053 | 0.394325 | 4/15/93/7 |
| lab25_fixed | 0.756303 | 0.421053 | 0.820000 | 0.620526 | 0.307692 | 0.355556 | 0.738947 | 0.379968 | 8/11/82/18 |
| lab25_spec0.8 | 0.739496 | 0.473684 | 0.790000 | 0.631842 | 0.300000 | 0.367347 | 0.738947 | 0.379968 | 9/10/79/21 |
| lab25_spec0.85 | 0.764706 | 0.421053 | 0.830000 | 0.625526 | 0.320000 | 0.363636 | 0.738947 | 0.379968 | 8/11/83/17 |
| lab25_spec0.9 | 0.789916 | 0.210526 | 0.900000 | 0.555263 | 0.285714 | 0.242424 | 0.738947 | 0.379968 | 4/15/90/10 |
| weighted_fixed | 0.764706 | 0.315789 | 0.850000 | 0.582895 | 0.285714 | 0.300000 | 0.731579 | 0.371157 | 6/13/85/15 |
| weighted_spec0.8 | 0.747899 | 0.526316 | 0.790000 | 0.658158 | 0.322581 | 0.400000 | 0.731579 | 0.371157 | 10/9/79/21 |
| weighted_spec0.85 | 0.764706 | 0.421053 | 0.830000 | 0.625526 | 0.320000 | 0.363636 | 0.731579 | 0.371157 | 8/11/83/17 |
| weighted_spec0.9 | 0.781513 | 0.210526 | 0.890000 | 0.550263 | 0.266667 | 0.235294 | 0.731579 | 0.371157 | 4/15/89/11 |

## Uncertainty and stability

Wilson intervals describe class proportions. Ranking metrics, F1, balanced accuracy and paired differences use 2,000 shared stratified subject bootstrap draws, seed 20261011. They condition on saved OOF predictions, fix prevalence and omit refit, model-selection and overlapping-training uncertainty. No multiple-comparison correction or confirmatory hypothesis test is claimed.

| Model / primary rule | Sensitivity 95% | Specificity 95% | ROC-AUC 95% | AP 95% |
| --- | --- | --- | --- | --- |
| exp04_robust_historical | 0.317078–0.726702 | 0.755797–0.899047 | 0.564724–0.835803 | 0.239103–0.585147 |
| exp05_cp_C_historical | 0.410395–0.808505 | 0.689296–0.849987 | 0.600500–0.840013 | 0.235087–0.557620 |
| equal_spec0.8 | 0.362759–0.768581 | 0.700200–0.858343 | 0.605776–0.843184 | 0.238237–0.569472 |
| weighted_spec0.8 | 0.317078–0.726702 | 0.700200–0.858343 | 0.598934–0.838421 | 0.235234–0.554108 |

| Ensemble minus robust LAB | Δ sensitivity [95%] | Δ specificity [95%] | Δ ROC-AUC [95%] | Δ AP [95%] |
| --- | --- | --- | --- | --- |
| equal_spec0.8 | 0.052632 [0.000000, 0.157895] | -0.050000 [-0.100000, -0.010000] | 0.028421 [-0.038947, 0.099474] | -0.006078 [-0.068915, 0.044386] |
| weighted_spec0.8 | 0.000000 [-0.157895, 0.157895] | -0.050000 [-0.100000, -0.010000] | 0.022105 [-0.053684, 0.104737] | -0.013991 [-0.089601, 0.046200] |

The equal-average sensitivity difference interval includes zero; its specificity difference is −5 points [−10, −1]. Its AUC difference (+0.028421) and AP difference (−0.006078) both have intervals spanning zero. Weighted averaging adds no net detection and lowers specificity. Relative to CP candidate C, equal averaging loses one anemia detection and clears one more non-anemia subject. Numerical ranking gains do not justify superiority.

| Fold | Robust LAB sensitivity / specificity | CP C sensitivity / specificity | Equal sensitivity / specificity | Weighted sensitivity / specificity |
| --- | --- | --- | --- | --- |
| 1 | 0.250000 / 0.600000 | 0.750000 / 0.500000 | 0.250000 / 0.500000 | 0.250000 / 0.500000 |
| 2 | 1.000000 / 0.950000 | 0.750000 / 0.900000 | 1.000000 / 0.950000 | 1.000000 / 0.950000 |
| 3 | 0.750000 / 0.850000 | 0.750000 / 0.850000 | 0.750000 / 0.850000 | 0.500000 / 0.850000 |
| 4 | 0.500000 / 1.000000 | 0.500000 / 0.900000 | 0.500000 / 0.900000 | 0.500000 / 0.900000 |
| 5 | 0.000000 / 0.800000 | 0.333333 / 0.750000 | 0.333333 / 0.750000 | 0.333333 / 0.750000 |

The equal-average fold sensitivity ranges 0.25–1.00 and specificity 0.50–0.95; ranking remains unstable (fold AUC 0.466667–0.962500). The poor first-fold specificity is not repaired by the training specificity constraint. At the secondary 0.85 target the fixed 75% LAB candidate yielded 9/19 at 85%, compared with LAB 5/19 at 90%, CP 6/19 at 82% and equal 7/19 at 85%. This interesting exploratory operating-point difference was not used to replace the weighted primary rule or declare a winner. It requires prespecification on new subjects.

## Targeted image-information audit

All 119 approved model masks are single-component, nonempty binary masks; none touches the image edge. Five private context sheets covering all 119 were visually inspected. Outlines broadly follow exposed lower palpebral conjunctiva, with no obvious gross sclera/iris inclusion at thumbnail resolution. Visible tissue extent, slanted orientation, boundary shape and treatment of margin/overlying hair vary. Some interiors contain eyelashes/hair while others use boundary notches. This is a descriptive finding, not clinician validation, a new anatomical approval or proof that every boundary is correct. Mask area ranges 12,886–244,923 native pixels and cannot alone measure tissue equivalence. No approved mask changed.

The 429-file original archive still matches its registered file hashes. Exactly 140 originals have verified eyelid linkage, one per subject; no subject has multiple linked eyelid originals. The 154 unassigned files lack role/identity evidence for retake analysis. Therefore repeated-image prediction consistency cannot be estimated safely. Additional images would require provenance and approved corresponding anatomy, retaining the subject fold; no patient count would increase.

Using verified Hb and [WHO2024 adult15–65 severity definitions](https://www.ncbi.nlm.nih.gov/books/NBK602185/table/ch5.tab3/), the 19 anemia subjects comprise 11 mild, six moderate and two severe. LAB’s nine misses comprise six mild/three moderate; CP’s seven comprise five mild/two moderate. Five shared misses comprise three mild/two moderate: they are not exclusively near-threshold cases. Their median distance below the sex-specific threshold is 1.0 g/dL versus 1.6 in shared detections. Equal averaging’s eight misses comprise six mild/two moderate, weighted averaging’s nine seven mild/two moderate. This is descriptive, without measurement uncertainty, relabeling or held-out-case optimization.

Historical rendering remains mixed among the 119: 81 Photos exports, 17 Preview/ColorSync, 11 embedded DNG previews and ten unrecorded pilot exports. Brightness and redness vary; median any-channel clipping is 34.5%, while the bright-neutral proxy median is about 0.0054%. Neither is a validated glare measure. Missing standardized acquisition/color references and tone-processing differences plausibly limit predictions but are not established causes. Existing Experiment 4B reviewed glare exclusions detected the same ten robust-LAB positives with no recall gain; no new glare or regional feature search was run.

## Evidence, verification and recommendation

Private immutable bundle: `data/processed/local_eyelid_classification_exp06_ensemble_v1/`, with source/previous-worktree provenance, alignment, feasibility, separate identity key, patient agreement, inner scores, fold model references, weights/cutoffs, 28 OOF policy outputs, metrics/confusions/intervals, aggregate PNG/SVG overlap and ROC/PR figures, image audit, dependency versions, copied implementation, execution/test logs and checksum manifest. `REPRODUCTION.md` documents exact inference without retraining. Independent proof is under `data/processed/exp06_verification_2026-10-13_v1/`. Patient data/models/context sheets remain ignored and are not committed.

All 136 synthetic/project tests completed: 133 passed, three legacy-source tests skipped; compileall and whitespace passed; central-documentation validation is recorded separately at handoff. Independent checksum-first verification reproduces all 40 relevant base pipelines, identity/inner membership, all weight/cutoff decisions, outer scores and all pooled/fold metrics and conditional intervals without fitting. Prior experiments and references are unchanged.

Ensemble inference requires a frozen CNN plus both classical pipelines, compared with LAB statistics/scaling/logistic alone; averaging arithmetic is negligible, but extracting the CP representation increases compute, checkpoint storage and complexity. Both predictors already share robust18 features, reducing mechanistic independence. Interpretation and calibration become harder; this cohort does not show enough benefit to justify expanding training. Preserve the 0.85-target candidate finding as a future hypothesis, not a replacement final model. Favor fixed acquisition, explicit human anatomy/occlusion policy and more independently collected positive/borderline subjects before a prespecified comparison.
