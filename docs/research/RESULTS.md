# AIMA verified results

This document contains aggregate results that were checked against the saved
experiment reports and private metric manifests. It does not reproduce
patient-level predictions, identifiers, images, checkpoints, or row-level Hb
values. All local classification rows are internal development estimates on
119 subjects (19 anemia, 100 non-anemia), not independent or clinical results.

## Sources and reporting rules

- Experiment 1: `data/processed/local_eyelid_classification_2026-10-10_v1/report.md`
  and `metrics.json`.
- Experiment 2: `data/processed/local_eyelid_classification_exp02_hybrid_2026-10-11_v1/report.md`
  and `metrics.json`.
- Experiment 3: `data/processed/local_eyelid_classification_exp03_transfer_report_2026-10-11_v1/eyelid-exp03-results-2026-10-11.md`
  and the accepted v3 `metrics.json`.
- Experiment 4: `data/processed/local_eyelid_classification_exp04_color_optimization_2026-10-12_v1/metrics.json`,
  `candidate_metrics.json`, `regression/metrics.json`, and the Experiment 4
  report in the color worktree.
- Public baseline: `data/processed/public_baseline_reproduction_2026-10-07/metrics.json`.

Rows labeled **selected** use inner training information. Rows labeled
**predefined comparison** were reported because they were specified in the
experiment design; they were not chosen from outer performance.

## Public baseline

The public baseline contains 184 country-subject keys (115 non-anemic, 69
anemic) and uses ten-fold stratified CV. The fold-mean and pooled OOF numbers
are different aggregations of the same reproduction.

| Aggregation | Accuracy | Sensitivity | Specificity | ROC-AUC | Separate test set |
| --- | ---: | ---: | ---: | ---: | --- |
| Mean of ten folds | 0.8041 | 0.7690 | 0.8273 | 0.8723 | None |
| Pooled OOF | 0.8043 | 0.7681 | 0.8261 | 0.8692 | None |

These are public-dataset internal CV results. They are not local-cohort
performance, external validation, or clinical validation.

## Local classification comparison: Experiments 1–4

| Experiment / model | Threshold policy | Accuracy | Sensitivity | Specificity | Balanced accuracy | Precision | F1 | ROC-AUC | AP | Confusion counts TP/FN/TN/FP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Exp1 six-LAB logistic | fixed 0.5 | 0.848739 | 0.105263 | 0.990000 | 0.547632 | 0.666667 | 0.181818 | 0.719474 | 0.387866 | 2/17/99/1 |
| Exp1 majority baseline | fixed majority | 0.840336 | 0.000000 | 1.000000 | 0.500000 | undefined | 0.000000 | 0.500000 | 0.159664 | 0/19/100/0 |
| Exp2 hybrid | fixed 0.5 | 0.756303 | 0.263158 | 0.850000 | 0.556579 | 0.250000 | 0.256410 | 0.575789 | 0.247161 | 5/14/85/15 |
| Exp2 hybrid | inner selected | 0.739496 | 0.368421 | 0.810000 | 0.589211 | 0.269231 | 0.311111 | 0.575789 | 0.247161 | 7/12/81/19 |
| Exp3 public-only local | fixed 0.5 | 0.781513 | 0.000000 | 0.930000 | 0.465000 | 0.000000 | 0.000000 | 0.451053 | 0.140775 | 0/19/93/7 |
| Exp3 public-only local | public selected | 0.840336 | 0.000000 | 1.000000 | 0.500000 | undefined | 0.000000 | 0.451053 | 0/19/100/0 |
| Exp3 transfer A | fixed 0.5 | 0.529412 | 0.210526 | 0.590000 | 0.400263 | 0.088889 | 0.125000 | 0.475263 | 0.146584 | 4/15/59/41 |
| Exp3 transfer A | inner selected | 0.697479 | 0.052632 | 0.820000 | 0.436316 | 0.052632 | 0.052632 | 0.475263 | 0.146584 | 1/18/82/18 |
| Exp3 transfer B | fixed 0.5 | 0.521008 | 0.210526 | 0.580000 | 0.395263 | 0.086957 | 0.123077 | 0.445789 | 0.140270 | 4/15/58/42 |
| Exp3 transfer B | inner selected | 0.714286 | 0.052632 | 0.840000 | 0.446316 | 0.058824 | 0.055556 | 0.445789 | 0.140270 | 1/18/84/16 |
| Exp3 inner-selected A/B | inner selected | 0.714286 | 0.052632 | 0.840000 | 0.446316 | 0.058824 | 0.055556 | 0.463684 | 0.143881 | 1/18/84/16 |
| Exp3 nested LAB comparator | inner selected | 0.764706 | 0.368421 | 0.840000 | 0.604211 | 0.304348 | 0.333333 | 0.719474 | 0.387866 | 7/12/84/16 |
| Exp4 A policy | inner specificity ≥0.80 | 0.764706 | 0.368421 | 0.840000 | 0.604211 | 0.304348 | 0.333333 | 0.655263 | 0.351693 | 7/12/84/16 |
| Exp4 B policy | inner specificity ≥0.80 | 0.756303 | 0.368421 | 0.830000 | 0.599211 | 0.291667 | 0.325581 | 0.653684 | 0.352379 | 7/12/83/17 |
| Exp4 joint primary | inner specificity ≥0.80 | 0.756303 | 0.368421 | 0.830000 | 0.599211 | 0.291667 | 0.325581 | 0.656842 | 0.352228 | 7/12/83/17 |
| Exp4 joint primary | inner specificity ≥0.70 | 0.705882 | 0.631579 | 0.720000 | 0.675789 | 0.300000 | 0.406780 | 0.656842 | 0.352228 | 12/7/72/28 |
| Exp4 joint primary | inner specificity ≥0.90 | 0.798319 | 0.263158 | 0.900000 | 0.581579 | 0.333333 | 0.294118 | 0.656842 | 0.352228 | 5/14/90/10 |
| Exp4 A robust18 logistic | predefined comparison | 0.789916 | 0.526316 | 0.840000 | 0.683158 | 0.384615 | 0.444444 | 0.709474 | 0.385148 | 10/9/84/16 |
| Exp4 B robust18 logistic | predefined comparison | 0.781513 | 0.368421 | 0.860000 | 0.614211 | 0.333333 | 0.350000 | 0.696842 | 0.365129 | 7/12/86/14 |

The Exp4 robust18 row is not the automatically selected primary model. It is a
predefined, matched exploratory comparison; its apparent sensitivity gain must
not be reported as independently validated improvement.

## Confusion matrices and detected/missed cases

For local matrices, rows are true `[non-anemia, anemia]` and columns are
predicted `[non-anemia, anemia]`.

| Model/rule | Matrix | Anemia detected | Anemia missed |
| --- | --- | ---: | ---: |
| Exp1 LAB / fixed 0.5 | `[[99, 1], [17, 2]]` | 2/19 | 17/19 |
| Exp1 majority | `[[100, 0], [19, 0]]` | 0/19 | 19/19 |
| Exp2 hybrid / fixed 0.5 | `[[85, 15], [14, 5]]` | 5/19 | 14/19 |
| Exp2 hybrid / selected | `[[81, 19], [12, 7]]` | 7/19 | 12/19 |
| Exp3 transfer A/B selected | `[[84, 16], [18, 1]]` | 1/19 | 18/19 |
| Exp3 nested LAB selected | `[[84, 16], [12, 7]]` | 7/19 | 12/19 |
| Exp4 joint primary selected | `[[83, 17], [12, 7]]` | 7/19 | 12/19 |
| Exp4 joint / inner 0.70 | `[[72, 28], [7, 12]]` | 12/19 | 7/19 |
| Exp4 A robust18 comparison | `[[84, 16], [9, 10]]` | 10/19 | 9/19 |

## Statistical uncertainty

Intervals below are conditional on saved OOF scores. Wilson intervals are used
for fixed-threshold proportions; ranking and paired intervals use the recorded
2,000 stratified subject bootstrap draws. They omit model-refit uncertainty and
overlap dependence across CV training folds.

| Result | Sensitivity 95% interval | Specificity 95% interval | ROC-AUC 95% interval | AP 95% interval | Source |
| --- | --- | --- | --- | --- | --- |
| Exp1 LAB / fixed 0.5 | 0.029359–0.313941 | 0.946–0.998 | 0.575789–0.838961 | 0.248723–0.595255 | Exp1 `metrics.json` `uncertainty` |
| Exp2 hybrid / fixed 0.5 | 0.118064–0.487916 | 0.767–0.907 | 0.424184–0.720796 | 0.163599–0.447485 | Exp2 `report.md` “Paired uncertainty” |
| Exp2 hybrid / inner selected | 0.191495–0.589605 | 0.722–0.875 | 0.424184–0.720796 | 0.163599–0.447485 | Exp2 `report.md` “Paired uncertainty” |
| Exp3 transfer selected | 0.009–0.246 | 0.756–0.899 | 0.347–0.578 | reported in report | Exp3 results “Uncertainty and interpretation” |
| Exp3 nested LAB selected | 0.191–0.590 | 0.756–0.899 | 0.582–0.842 | reported in report | Exp3 results “Uncertainty and interpretation” |
| Exp4 joint primary | 0.191–0.590 | 0.745–0.891 | 0.522–0.786 | 0.213–0.550 | Exp4 results “Controlled interpretation and uncertainty” |
| Exp4 A robust18 comparison | 0.317–0.727 | reported in report | reported in report | reported in report | Exp4 results “Controlled interpretation and uncertainty” |

No interval in this table is an external validation interval or a clinical
performance guarantee.

## Segmentation results kept separate from anemia classification

| Study | Data/protocol | Result | Interpretation |
| --- | --- | --- | --- |
| Local ExtraTrees proposal pilot | Ten held-out subjects per modality; manual masks as truth | Eyelid Dice 0.694 / IoU 0.555; nail Dice 0.746 / IoU 0.602 | Proposal-only; observed adjacent-tissue failures and no metric for six unannotated proposals. |
| Public U-Net local development | 111 local references; public-only model, full-frame inference | Dice 0.0027 / IoU 0.0014 / precision 0.0021 / recall 0.0075; 110/111 QC flags | Poor local transfer; not anemia performance; manual masks unchanged. |
| SAM 2.1 nail assistance | Ten approved masks, leave-one-subject-out; 116 queue proposals | Dice 0.794 / IoU 0.669 / precision 0.680 / recall 0.987; 21 area flags | Unverified annotation assistance; not ground truth. |

Sources: `data/processed/segmentation_prototype_2026-10-07/report.json`,
`data/processed/segmentation/local_unet_dev_eval_fullframe_2026-10-09/summary.json`,
`data/processed/segmentation/sam2_nail_lopo_areaqc_v1/summary.json`, and the
corresponding dated experiment/audit documents.

## Hb regression feasibility output

Experiment 4 assessed the continuous-Hb question only after classification.
The managed robust18 Ridge model (`alpha=10`) used training-only scaling and
reported:

| Model | MAE (g/dL) | RMSE (g/dL) | R² |
| --- | ---: | ---: | ---: |
| Robust18 Ridge | 1.3274 | 1.7022 | 0.1341 |
| Training-mean baseline | 1.3750 | 1.8403 | −0.0121 |

The 0.0476 g/dL MAE difference is a feasibility observation on the reused local
development cohort. It is not a clinical Hb estimate, calibration result, or
replacement for laboratory measurement. Source: private Exp4
`regression/metrics.json` and the report section “Optional Hb regression”.

## Verified findings

1. **Accuracy is misleading in the local cohort.** The Exp1 model had 84.9%
   accuracy but detected only 2/19 anemia subjects; the majority baseline had
   84.0% accuracy while detecting none. Sensitivity and confusion counts are
   the more informative description for this prevalence.
2. **Frozen hybrid image features did not improve ranking.** Exp2 increased
   detections at some thresholds but decreased specificity and ROC-AUC/AP
   relative to Exp1 on the same development subjects.
3. **Public-to-local transfer was negative in this design.** Public-only
   inference detected no local anemia subjects, and the selected transfer model
   detected 1/19 with ROC-AUC 0.464. The different neural preprocessing and
   domains prevent attributing the result solely to pretraining.
4. **Training-selected thresholds change the operating trade-off.** The
   nested LAB analysis increased detections from 2/19 to 7/19 while increasing
   false positives from 1 to 16. This is an exploratory decision-rule result,
   not a clinical threshold recommendation.
5. **Experiment 4’s primary and robust comparison are different claims.** The
   predefined joint primary policy detected 7/19; the predefined original-
   rendering robust18 comparison detected 10/19. The latter has an interval
   including zero paired sensitivity improvement and its color-managed analogue
   detected 7/19, so it is a hypothesis for confirmation, not the selected
   primary model.
6. **Rendering is a real methodological limitation.** The audit found mixed
   local rendering histories and a Display-P3-preview/sRGB-tag mismatch. The
   implemented profile-aware correction changes the declared color path but
   does not establish camera calibration.
7. **Hb regression did not establish measurement validity.** The regression
   output is a small feasibility result and must remain separate from binary
   anemia classification.

## Figures and tables already preserved

Existing aggregate figures were inspected and are retained in ignored private
experiment directories; they were not regenerated or copied into Git:

- Exp1: `data/processed/local_eyelid_classification_2026-10-10_v1/internal_cv_summary.png` and `.svg`.
- Exp2: `data/processed/local_eyelid_classification_exp02_hybrid_2026-10-11_v1/evaluation.png` and `.svg`.
- Exp3: `data/processed/local_eyelid_classification_exp03_transfer_2026-10-11_v3/comparison.png` and `.svg`.
- Exp4: `data/processed/local_eyelid_classification_exp04_color_optimization_2026-10-12_v1/evaluation.png`,
  `tradeoffs.png`, and corresponding SVGs; verified comparison figures are in
  `data/processed/exp04_verification_2026-10-12_v1/`.

These files may contain patient-derived visual information and must remain
private. The thesis handoff identifies them as source figures rather than
publishing them automatically.

## 2026-10-12 - Verified Experiment4B and Experiment5 initialization

The existing research records from committed main717da2c were incorporated into the isolated5 worktree without merging or modifying earlier worktrees. Reconciled their provisional4B/5 placeholders with the owner-approved actual experiment IDs. Authoritative metrics/checksums confirm4B160 models/222 files, weighted masked robust10/19 at84%, AUC0.720526/AP0.392086; full-selected9/19 at78%. Baselines1-4 and4B are preserved, with no independent validation claim. Experiment5 source/public availability audits are in progress; the existing V2 folder has251 unique RGBA images,110/141 labels, Hb4.2-15.0 and no cutoff mismatch; official release authenticity/grouping remains under review before training.

## Experiment 5 — completed verified CP transfer

CP-AnemiC pretraining did not establish an improvement that justifies its added complexity on AIMA. The predefined joint policy detected **11/19 anemia cases (57.9% sensitivity) at 78% specificity**, compared with **10/19 (52.6%) at 84% specificity** for the predefined Experiment 4 robust color comparison. One additional detection came with six additional false positives. Balanced accuracy was slightly lower (67.95% versus 68.32%), ROC-AUC was 0.734211 versus 0.709474, and AP was 0.372546 versus 0.385148. The paired sensitivity difference had a conditional 95% interval of −21.1 to +31.6 percentage points; the specificity difference was −6.0 points [−11.0, −2.0]. This does not establish independent or clinically acceptable improvement.

The predefined CP-classification frozen-feature plus color candidate C detected **12/19 (63.2%) at 78% specificity**, with ROC-AUC 0.734211 and AP 0.372166. Its sensitivity 95% Wilson interval was 41.0–80.9%; specificity was 68.9–85.0%. Against the robust color comparator, its +10.5-point sensitivity difference had an interval of [−10.5, +36.8], while specificity fell by 6.0 points [−11.0, −2.0]. This is a predefined exploratory candidate comparison, not a replacement primary policy chosen from outer results. CP classification's actual inner-selected A/B/C policy detected 10/19 at 77% specificity; it selected C in four folds and B in one.

CP Hb-regression C and its inner-selected policy detected **11/19 at 79% specificity** (ROC-AUC 0.730000; AP 0.370429), with exactly the same binary OOF decisions as matched ImageNet C (11/19 at 79%; ROC-AUC 0.730526; AP 0.368331). CP classification C added one detection and one false positive versus matched ImageNet C; its paired ROC-AUC and AP intervals included zero. The frozen-feature performance is therefore compatible with ImageNet features plus robust color statistics without useful added CP pretraining. Head-only and partial CP adaptations with training-selected thresholds detected only 2/19 and 0/19 for classification, or 4/19 and 0/19 for regression. These outer results did not guide any redesign.

The public-only pediatric classifier detected **0/19 at 96% specificity** (ROC-AUC 0.459474; AP 0.150111). Public classification validation was weak (ROC-AUC 0.478125), and both bounded public objectives selected epoch 1. Public-only Hb prediction on AIMA had MAE 2.225671 g/dL, bias −1.905735 g/dL and R² −1.069180. Although its MAE was below the public-training-mean baseline of 2.801160, substantial adult underprediction and negative R² provide no evidence of a usable Hb estimator. Weak public fitting and observed acquisition/label differences are plausible explanations, not proven causes.

Retain robust color-only classification as the practical candidate for independent confirmation, with a prespecified acquisition, model and operating-rule policy and genuinely new positive and borderline cases. Frozen embeddings can remain a predefined comparator in that study. These repeated-cohort results do not justify additional CP fine-tuning complexity or choosing a new winner by outer sensitivity. The inner specificity constraint of 0.80 was a research threshold policy and did not guarantee 0.80 outer specificity. No clinical acceptability threshold or independent validation has been established.


| Model / rule | Accuracy | Sensitivity | Specificity | Balanced accuracy | Precision | F1 | ROC-AUC | AP | TP/FN/TN/FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exp01_fixed | 0.848739 | 0.105263 | 0.990000 | 0.547632 | 0.666667 | 0.181818 | 0.719474 | 0.387866 | 2/17/99/1 |
| lab_nested_selected | 0.764706 | 0.368421 | 0.840000 | 0.604211 | 0.304348 | 0.333333 | 0.719474 | 0.387866 | 7/12/84/16 |
| exp02_selected | 0.739496 | 0.368421 | 0.810000 | 0.589211 | 0.269231 | 0.311111 | 0.575789 | 0.247161 | 7/12/81/19 |
| exp03_selected | 0.714286 | 0.052632 | 0.840000 | 0.446316 | 0.058824 | 0.055556 | 0.463684 | 0.143881 | 1/18/84/16 |
| exp04_joint_selected | 0.756303 | 0.368421 | 0.830000 | 0.599211 | 0.291667 | 0.325581 | 0.656842 | 0.352228 | 7/12/83/17 |
| exp04_robust_selected | 0.789916 | 0.526316 | 0.840000 | 0.683158 | 0.384615 | 0.444444 | 0.709474 | 0.385148 | 10/9/84/16 |
| exp04b_glare_weighted_selected | 0.789916 | 0.526316 | 0.840000 | 0.683158 | 0.384615 | 0.444444 | 0.720526 | 0.392086 | 10/9/84/16 |
| cp_public_only_fixed | 0.806723 | 0.000000 | 0.960000 | 0.480000 | 0.000000 | 0.000000 | 0.459474 | 0.150111 | 0/19/96/4 |
| cp_classification_A_fixed | 0.588235 | 0.526316 | 0.600000 | 0.563158 | 0.200000 | 0.289855 | 0.557368 | 0.202268 | 10/9/60/40 |
| cp_classification_A_threshold | 0.672269 | 0.105263 | 0.780000 | 0.442632 | 0.083333 | 0.093023 | 0.557368 | 0.202268 | 2/17/78/22 |
| cp_classification_B_fixed | 0.630252 | 0.157895 | 0.720000 | 0.438947 | 0.096774 | 0.120000 | 0.466842 | 0.163613 | 3/16/72/28 |
| cp_classification_B_threshold | 0.739496 | 0.000000 | 0.880000 | 0.440000 | 0.000000 | 0.000000 | 0.466842 | 0.163613 | 0/19/88/12 |
| cp_classification_C_fixed | 0.722689 | 0.631579 | 0.740000 | 0.685789 | 0.315789 | 0.421053 | 0.734211 | 0.372166 | 12/7/74/26 |
| cp_classification_C_threshold | 0.756303 | 0.631579 | 0.780000 | 0.705789 | 0.352941 | 0.452830 | 0.734211 | 0.372166 | 12/7/78/22 |
| cp_classification_selected_fixed | 0.655462 | 0.526316 | 0.680000 | 0.603158 | 0.238095 | 0.327869 | 0.695789 | 0.305089 | 10/9/68/32 |
| cp_classification_selected_threshold | 0.731092 | 0.526316 | 0.770000 | 0.648158 | 0.303030 | 0.384615 | 0.695789 | 0.305089 | 10/9/77/23 |
| cp_regression_A_fixed | 0.512605 | 0.631579 | 0.490000 | 0.560789 | 0.190476 | 0.292683 | 0.540526 | 0.193936 | 12/7/49/51 |
| cp_regression_A_threshold | 0.655462 | 0.210526 | 0.740000 | 0.475263 | 0.133333 | 0.163265 | 0.540526 | 0.193936 | 4/15/74/26 |
| cp_regression_B_fixed | 0.672269 | 0.210526 | 0.760000 | 0.485263 | 0.142857 | 0.170213 | 0.445263 | 0.164125 | 4/15/76/24 |
| cp_regression_B_threshold | 0.731092 | 0.000000 | 0.870000 | 0.435000 | 0.000000 | 0.000000 | 0.445263 | 0.164125 | 0/19/87/13 |
| cp_regression_C_fixed | 0.714286 | 0.631579 | 0.730000 | 0.680789 | 0.307692 | 0.413793 | 0.730000 | 0.370429 | 12/7/73/27 |
| cp_regression_C_threshold | 0.756303 | 0.578947 | 0.790000 | 0.684474 | 0.343750 | 0.431373 | 0.730000 | 0.370429 | 11/8/79/21 |
| cp_regression_selected_fixed | 0.714286 | 0.631579 | 0.730000 | 0.680789 | 0.307692 | 0.413793 | 0.730000 | 0.370429 | 12/7/73/27 |
| cp_regression_selected_threshold | 0.756303 | 0.578947 | 0.790000 | 0.684474 | 0.343750 | 0.431373 | 0.730000 | 0.370429 | 11/8/79/21 |
| imagenet_control_A_fixed | 0.512605 | 0.473684 | 0.520000 | 0.496842 | 0.157895 | 0.236842 | 0.518421 | 0.177612 | 9/10/52/48 |
| imagenet_control_A_threshold | 0.722689 | 0.210526 | 0.820000 | 0.515263 | 0.181818 | 0.195122 | 0.518421 | 0.177612 | 4/15/82/18 |
| imagenet_control_B_fixed | 0.680672 | 0.157895 | 0.780000 | 0.468947 | 0.120000 | 0.136364 | 0.426842 | 0.157141 | 3/16/78/22 |
| imagenet_control_B_threshold | 0.747899 | 0.000000 | 0.890000 | 0.445000 | 0.000000 | 0.000000 | 0.426842 | 0.157141 | 0/19/89/11 |
| imagenet_control_C_fixed | 0.731092 | 0.631579 | 0.750000 | 0.690789 | 0.324324 | 0.428571 | 0.730526 | 0.368331 | 12/7/75/25 |
| imagenet_control_C_threshold | 0.756303 | 0.578947 | 0.790000 | 0.684474 | 0.343750 | 0.431373 | 0.730526 | 0.368331 | 11/8/79/21 |
| imagenet_control_selected_fixed | 0.731092 | 0.631579 | 0.750000 | 0.690789 | 0.324324 | 0.428571 | 0.730526 | 0.368331 | 12/7/75/25 |
| imagenet_control_selected_threshold | 0.756303 | 0.578947 | 0.790000 | 0.684474 | 0.343750 | 0.431373 | 0.730526 | 0.368331 | 11/8/79/21 |
| joint_selected_fixed | 0.722689 | 0.631579 | 0.740000 | 0.685789 | 0.315789 | 0.421053 | 0.734211 | 0.372546 | 12/7/74/26 |
| joint_selected_threshold | 0.747899 | 0.578947 | 0.780000 | 0.679474 | 0.333333 | 0.423077 | 0.734211 | 0.372546 | 11/8/78/22 |

| Model/rule | Sensitivity 95% | Specificity 95% | ROC-AUC 95% | AP 95% |
| --- | --- | --- | --- | --- |
| cp_public_only_fixed | 0.000000–0.168179 | 0.901629–0.984337 | 0.325776–0.596842 | 0.125566–0.231330 |
| cp_classification_A_threshold | 0.029359–0.313941 | 0.689296–0.849987 | 0.418947–0.693197 | 0.147932–0.353104 |
| cp_classification_B_threshold | 0.000000–0.168179 | 0.801879–0.930006 | 0.318421–0.613711 | 0.126624–0.291019 |
| cp_classification_C_threshold | 0.410395–0.808505 | 0.689296–0.849987 | 0.600500–0.840013 | 0.235087–0.557620 |
| cp_classification_selected_threshold | 0.317078–0.726702 | 0.678456–0.841567 | 0.567368–0.806842 | 0.201042–0.478447 |
| cp_regression_A_threshold | 0.085077–0.433343 | 0.646290–0.815953 | 0.397368–0.669474 | 0.141784–0.341638 |
| cp_regression_B_threshold | 0.000000–0.168179 | 0.790196–0.922428 | 0.292066–0.597368 | 0.122743–0.304773 |
| cp_regression_C_threshold | 0.362759–0.768581 | 0.700200–0.858343 | 0.593671–0.838421 | 0.231820–0.553327 |
| cp_regression_selected_threshold | 0.362759–0.768581 | 0.700200–0.858343 | 0.593671–0.838421 | 0.231820–0.553327 |
| imagenet_control_selected_threshold | 0.362759–0.768581 | 0.700200–0.858343 | 0.594197–0.838434 | 0.231911–0.553629 |
| joint_selected_threshold | 0.362759–0.768581 | 0.689296–0.849987 | 0.598408–0.841079 | 0.235130–0.558791 |

| Evaluation | MAE g/dL [95%] | RMSE g/dL | Bias predicted−lab g/dL [95%] | R² |
| --- | --- | ---: | --- | ---: |
| Public CP regression validation | 1.699905 [1.324415, 2.107379] | 2.251749 | 0.435266 [-0.141504, 1.059850] | -0.001030 |
| Public validation training-mean baseline | 1.802457 [1.452457, 2.173687] | 2.251072 | -0.046542 [-0.621735, 0.603506] | -0.000428 |
| Public-only regression on AIMA | 2.225671 [1.965814, 2.498436] | 2.631343 | -1.905735 [-2.215274, -1.578677] | -1.069180 |
| Public-training mean on AIMA | 2.801160 [2.545484, 3.072611] | 3.154166 | -2.569538 [-2.882164, -2.241765] | -1.973120 |

All exact paired differences, fixed/selected tables and provenance: [verified Experiment5 report](../experiments/eyelid-exp05-results-2026-10-13.md), private `metrics.json`, `public_validation/metrics.json`, `public_only/metrics.json` in the E5 bundle. Conditional intervals omit refitting/selection uncertainty. Predefined E4 robust comparison remains separate from its primary policy.

## Experiment 6 — verified exploratory ensemble outcomes

Partial complementarity justified a bounded test; equal averaging did not preserve robust LAB specificity and the training-selected weighted policy added no net anemia detection. Historical robust and CP C comparisons are distinct from their primary selected policies. Exact28-rule/fixed/secondary, per-fold, uncertainty and paired tables are in [Experiment 6 report](../experiments/eyelid-exp06-results-2026-10-13.md). All rates are pooled internal five-fold OOF on119 reused subjects 19/100.

| Model / rule | Accuracy | Sensitivity | Specificity | Balanced accuracy | Precision | F1 | ROC-AUC | AP | TP/FN/TN/FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exp04_primary_historical | 0.756303 | 0.368421 | 0.830000 | 0.599211 | 0.291667 | 0.325581 | 0.656842 | 0.352228 | 7/12/83/17 |
| exp04_robust_historical | 0.789916 | 0.526316 | 0.840000 | 0.683158 | 0.384615 | 0.444444 | 0.709474 | 0.385148 | 10/9/84/16 |
| exp05_cp_C_historical | 0.756303 | 0.631579 | 0.780000 | 0.705789 | 0.352941 | 0.452830 | 0.734211 | 0.372166 | 12/7/78/22 |
| exp05_primary_historical | 0.747899 | 0.578947 | 0.780000 | 0.679474 | 0.333333 | 0.423077 | 0.734211 | 0.372546 | 11/8/78/22 |
| equal_fixed | 0.789916 | 0.263158 | 0.890000 | 0.576579 | 0.312500 | 0.285714 | 0.737895 | 0.379070 | 5/14/89/11 |
| equal_spec0.8 | 0.756303 | 0.578947 | 0.790000 | 0.684474 | 0.343750 | 0.431373 | 0.737895 | 0.379070 | 11/8/79/21 |
| weighted_fixed | 0.764706 | 0.315789 | 0.850000 | 0.582895 | 0.285714 | 0.300000 | 0.731579 | 0.371157 | 6/13/85/15 |
| weighted_spec0.8 | 0.747899 | 0.526316 | 0.790000 | 0.658158 | 0.322581 | 0.400000 | 0.731579 | 0.371157 | 10/9/79/21 |
| equal_spec0.85 | 0.773109 | 0.368421 | 0.850000 | 0.609211 | 0.318182 | 0.341463 | 0.737895 | 0.379070 | 7/12/85/15 |
| equal_spec0.9 | 0.789916 | 0.157895 | 0.910000 | 0.533947 | 0.250000 | 0.193548 | 0.737895 | 0.379070 | 3/16/91/9 |
| weighted_spec0.85 | 0.764706 | 0.421053 | 0.830000 | 0.625526 | 0.320000 | 0.363636 | 0.731579 | 0.371157 | 8/11/83/17 |
| weighted_spec0.9 | 0.781513 | 0.210526 | 0.890000 | 0.550263 | 0.266667 | 0.235294 | 0.731579 | 0.371157 | 4/15/89/11 |

Eight shared detections, two LAB-only, four CP-only and five shared misses; 78 shared clearances,16 shared false positives and six CP-only false positives. Pearson0.801841/Spearman0.823465; agreement107/119. Equal0.80 gains1 detection and 5 false alarms versus robust; weighted gains0 and 5. Equal sensitivity Wilson95%36.3–76.9%, specificity 70.0–85.8%. Paired equal-minus-LAB sensitivity+5.263 points [0,15.789], specificity−5 [−10,−1], AUC+0.028421 [−0.038947,0.099474] and AP−0.006078 [−0.068915,0.044386]. These conditional intervals omit refit/selection uncertainty; differences do not establish superiority.

The secondary75%-LAB0.85 candidate detected 9/19 at 85% specificity but was not selected as primary from outer outcomes. Training-selected weights are0.50,0.50,0.25,0.25,0.25. Fixed0.5 and training-selected rules differ; achieved specificity is not the requested inner target. No stacking, calibration or base/neural retraining occurred.

Sources: `data/processed/local_eyelid_classification_exp06_ensemble_v1/``metrics.json`, `error_overlap.json`, fold decisions and checksum manifest; `data/processed/exp06_verification_2026-10-13_v1/reproduction.json`. See [Experiment 6 report](../experiments/eyelid-exp06-results-2026-10-13.md). Favor robust color-only for independent confirmation; no external/clinical performance claim.


## Experiment 7 — verified negative normal-only/transfer result (2026-10-14)

Primary target is training specificity.85; realized outer specificity below
is measured on100 non-anemic subjects. A is normal-only cutoff calibration;
B uses inner training labels only for decisions. Both fit normal examples
only. Counts are TP/FP/TN/FN; confusion matrices are[[TN,FP],[FN,TP]]. Ranking
uses predefined normal cross-fit standardization. All87 outputs appear in
[the full report](../experiments/eyelid-exp07-results.md); no outer winner is
selected.

| Policy | TP/FP/TN/FN | Sensitivity | Specificity | ROC-AUC | AP |
|---|---|---|---|---|---|
| Mahalanobis A85 | 4/13/87/15 | 21.0526% | 87% | 0.537368 | 0.258772 |
| SVM0.10/scale A85 | 4/12/88/15 | 21.0526% | 88% | 0.531579 | 0.344039 |
| Isolation Forest A85 | 4/15/85/15 | 21.0526% | 85% | 0.498421 | 0.246435 |
| Secondary nested B85 | 3/8/92/16 | 15.7895% | 92% | 0.514737 | 0.252140 |
| B1 Eyes direct A85 | 19/100/0/0 | 100% | 0% | 0.550526 | 0.184737 |
| B2 Eyes normal adaptation A85 | 5/12/88/14 | 26.3158% | 88% | 0.541053 | 0.295872 |
| B3 Eyes/local pooled A85 | 4/17/83/15 | 21.0526% | 83% | 0.519474 | 0.302284 |
| C1 CP direct A85 | 12/77/23/7 | 63.1579% | 23% | 0.378421 | 0.129228 |
| C2 CP normal adaptation A85 | 4/12/88/15 | 21.0526% | 88% | 0.500526 | 0.261827 |
| C3 CP/local pooled A85 | 3/15/85/16 | 15.7895% | 85% | 0.496842 | 0.202217 |
| D1 both public A85 | 16/75/25/3 | 84.2105% | 25% | 0.468947 | 0.150368 |
| D2 both normal adaptation A85 | 4/13/87/15 | 21.0526% | 87% | 0.482632 | 0.216244 |
| Preserved Exp4 robust18 comparator | 10/16/84/9 | 52.6316% | 84% | 0.709474 | 0.385148 |

All four local SVMs and Mahalanobis detect4/19 under A85; Forest also4/19.
At B85 Mahalanobis/Forest detect3/19 at92%/88%, all SVMs4/19 at93%.
Public B85 TP/FP: B1 2/8, B2 5/12, B3 3/2, C1 0/7, C2 2/5, C3 1/11,
D1 1/12, D2 5/12. Direct public calibration failure must not be presented as
a sensitivity gain.

Against preserved robust18, Mahalanobis A85 loses six detections and saves
three FP; sensitivity difference−31.6pp [−57.9,−5.3], specificity+3pp
[−7,+13], AUC−0.172105 [−0.361053,+0.021066], AP−0.126376
[−0.271858,+0.030073]. Shared/anomaly-only/robust-only/shared-missed anemia
counts are3/1/7/8. B2 loses five detections/saves four FP; paired sensitivity
−26.3pp [−52.6,+5.3]. Nested B loses seven/saves eight; sensitivity−36.8pp
[−63.2,−10.5]. These conditional fixed-OOF intervals do not establish
independent clinical differences. One positive is5.26pp; no multiplicity
correction or refitting bootstrap was performed.

B2 versus local Mahalanobis A85 adds one detection/removes one FP, but its
paired AUC+0.003684 interval[−0.038947,+0.046329] and AP+0.037100
[−0.019739,+0.088899] span zero. Public normals did not establish useful
improvement. All prior metrics remain unchanged. Source commit6513c9c;
289 models/87 outputs independently reproduced,148 tests passed/three skips.
