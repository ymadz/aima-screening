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
