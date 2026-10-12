# Experiment 4B results and glare-review handoff - 2026-10-12

Neither the regional nor owner-reviewed glare-exclusion comparison improved anemia detection over the unchanged Experiment 4 robust comparator. All 119 glare proposals were subsequently accepted by the study owner, enabling a separate controlled glare-excluded comparison. This is exploratory repeated-development evidence, with no untouched classifier holdout or external validation.

At training-selected cutoffs (inner specificity constraint0.80):

| Comparison | Sensitivity | Specificity | ROC-AUC | AP | TP/FN/TN/FP |
| --- | ---: | ---: | ---: | ---: | --- |
| Preserved robust18 | 52.6% | 84.0% | 0.709 | 0.385 | 10/9/84/16 |
| Glare-excluded robust18 | 52.6% | 83.0% | 0.710 | 0.392 | 10/9/83/17 |
| Glare-excluded robust18 weighted | 52.6% | 84.0% | 0.721 | 0.392 | 10/9/84/16 |
| Glare-excluded regional22 | 47.4% | 79.0% | 0.701 | 0.362 | 9/10/79/21 |
| Full-eight nested policy | 47.4% | 78.0% | 0.667 | 0.356 | 9/10/78/22 |

## 1. Experiment 4 preservation

Experiment 4 branch `experiment/eyelid-exp04-color-optimization` remains at `0b5050436862df240fb60e1e93113e5c2388de98`, in `/Users/madz/Documents/aima-screening-exp04-color`. Its 720 classification pipelines, 10 regression pipelines, 810 covered files, all decisions/metrics and freshly extracted source features reproduced without refitting before this follow-up. The reported primary sensitivity 36.8%, specificity 83%, AUC 0.656842 and AP 0.352228 match saved artifacts. Robust18 original/C1 unweighted detects 10/19 at 84%, AUC 0.709474/AP 0.385148. Exact binary predictions and all five JSON decision rules match the follow-up comparator. Original labels, manual anatomy, original images, prior experiment outputs and preservation references are unchanged.

## 2. Branch and worktree

New branch `experiment/eyelid-exp04b-glare-regional`, linked worktree `/Users/madz/Documents/aima-screening-exp04b-glare`, based on verified Experiment 4 commit `0b50504`. Main remains at its newer documentation consolidation `717da2c`, separate from this experiment. No merges, historical rewrites, pushes or unrelated edits.

## 3-6. Annotation coverage and review status

| Quantity | All manually approved eyelids | Eligible classifier eyelids |
| --- | ---: | ---: |
| Approved anatomy masks | 140 | 119 |
| Existing accepted glare masks | 0 | 0 |
| Existing unvalidated/unreviewed glare candidates | 22 | 20 |
| Without an existing glare mask | 118 | 99 |
| Verified no-glare before this review | 0 | 0 |
| Newly generated proposals | - | 119 |
| Nonempty generated proposals | - | 96 |
| Empty generated proposals | - | 23 |
| AUTO_GENERATED_PENDING_REVIEW | - | 105 |
| UNCERTAIN | - | 14 |
| INVALID_MASK | - | 0 |
| Initial human reviews required | - | 119 |
| Current APPROVED_GLARE after owner review | - | 96 |
| Current REVIEWED_NO_GLARE after owner review | - | 23 |
| Current unresolved glare reviews | - | 0 |

Before review, missing masks meant not reviewed/missing annotation and empty proposals were unreviewed. The owner then answered “they all look good to me.” to the all-119 review question. A separate register accepts 96 nonempty masks and 23 empty masks as reviewed no-glare, bound to the exact proposal/PDF hashes; original generated statuses and all 14 historical uncertainty flags remain unchanged. The 22 old candidates have valid binary geometry and no pixels outside anatomy, but no accepted glare ground truth. Existing candidate coverage is 1/19 anemia versus 19/100 non-anemia; interpreting missing masks as no glare would introduce a differential annotation gap. New nonempty proposals are 15/19 and 81/100, respectively; these are appearance-rule outputs, not verified glare prevalence.

## 7. Private review materials

Preserved anatomy-only output root (also supplies unchanged masks/review sheets):

`/Users/madz/Documents/aima-screening/data/processed/local_eyelid_exp04b_glare_regional_2026-10-12_v2/`

- `review_sheets/glare_review_all_119.pdf`: 40 pages; pages 1-5 uncertain/high priority (14 cases), 6-36 likely reflection (93), 37-40 empty pending proposals (12). Eleven additional empty proposals and three nonempty proposals are in the uncertain group.
- `review_sheets/page-01.png` through `page-40.png`: rendered sheets.
- `review_panels/`: 119 individual four-view panels with source eyelid crop, green anatomical edge, cyan proposal and retained-tissue preview.
- `review_queue_private.csv`: the preserved initial119-entry blank review queue. Current decisions are in the separate accepted review register; do not treat this historical queue as outstanding work.
- `glare_masks/`: 119 native same-size binary proposals. `glare_coverage_private.csv` records quality/status/hash; `identity_key_private.csv` keeps the source-subject mapping separate. Accepted review evidence is in `/Users/madz/Documents/aima-screening/data/processed/exp04b_glare_reviewed_2026-10-12_v1/`.

Source filenames, Hb, labels and direct identifiers are absent from sheets. Only tissue-region crops are displayed, not full faces. Scientific features retain native pixels; thumbnails are resized only for display. All 40 pages rendered; representative pages and the full layout montage were visually inspected. Layout QC is not human approval of glare boundaries.

## 8-9. Automatic quality and readiness

Frozen near-neutral bright-core/local-contrast rule in [methods](eyelid-exp04b-methods.md); it does not use Hb/labels or low redness alone. All 119 generated masks pass type, binary values, dimensions, ROI containment and nonempty retained-tissue checks. No outside-ROI pixel; no full-tissue removal. Removed ROI fraction: median 0.306%, maximum 1.885%, aggregate 0.428% of tissue pixels. No removal exceeds the predefined 3% review threshold; 14 cases are uncertain because of broad brightness or contrast lacking a conservative white core.

There is no independent reference glare annotation with which to measure detector precision/recall/Dice. Old candidate IoU is saved as a diagnostic only, never as accuracy. Visual examples show localized bright reflections and potential missed chromatic/wet reflections; naturally pale tissue remains a potential ambiguity. Structural validity and small removal do not prove correct tissue-vs-reflection discrimination. Following owner review of all panels, the exact accepted masks are suitable for this exploratory research comparison. Owner visual QC does not establish independent clinical detector accuracy.

Initially, the verified glare subset was empty and exclusion modeling was blocked. Owner acceptance cleared this gate: the strictly owner-reviewed subset equals the full 119-subject cohort (19/100); a single controlled run covers both scopes and is not two independent analyses. The approved-anatomy regional result remains preserved separately. Accepted-mask feature perturbations are descriptive, and classification uses the unchanged source/rendering/regions.

## 10. Feature stability and missed-case diagnostics

Original robust18 is exactly equal to Experiment 4's A cache. Median absolute proposed-exclusion changes in mean L/a/b are 0.076/0.097/0.043 CIELAB units; median L/a/b and most percentile/IQR changes have median zero under the existing quantization. Median changes in L/a standard deviations are 0.135/0.216; median natural-log R/G changes by 0.000379. These describe the subsequently owner-accepted masks; feature changes alone do not demonstrate classification benefit.

Unweighted robust coefficients have the same sign across all five training folds for 12/18 features, and at least four folds for 15/18. Unweighted regional: 11/22 across all five, 17/22 across at least four. The new core-minus-rim a contrast has a consistent sign in five folds and appears in the top five absolute standardized coefficients in three; b contrast changes sign in two folds. Correlated predictors, overlapping training sets and source rendering prevent causal or external-stability claims. No feature was selected from these summaries.

The nine unchanged robust misses comprise six mild and three moderate anemia records; median Hb 11.7 g/dL, median distance below the sex-specific cutoff 0.7 g/dL. Ten detected records comprise five mild/three moderate/two severe; Hb median 10.9, distance median 1.1. All nine misses have Photos TIFF rendering history, which also occurs among detected cases; this is not a causal export effect. Proposal removal medians are 0.176% among misses versus 0.404% among detected cases. Masks subsequently received owner visual acceptance, but no independent reference or causal design establishes glare as the cause of errors. In the preserved anatomy-only comparison, fold 5 misses all three anemia subjects under every training-selected candidate/policy. Individual Hb, prediction, geometry, rendering and regional rows remain private.

The robust comparator recovers three historical nested-LAB misses and loses none of its seven detections. Relative to Experiment 1's original fixed 0.5 rule, it recovers eight misses and retains both original detections; these comparisons use different operating rules and do not isolate a feature benefit.

## 11. Regional design

Four bounded median contrasts add L/a/b and natural-log R/G from a boundary-distance core/rim partition of the original manual ROI. Core is at/above median occupied distance, rim below. Original anatomy fixes regions before proposed exclusion. Geometry is invariant to axis flips/90-degree rotations; these are ROI-relative sampling regions, not anatomical landmarks or new segmentation labels. No demographic/clinical/capture-history predictor, normalization, augmentation, calibration or anatomical segmentation training. Rendering A is identical stored TIFF RGB assumed sRGB, retaining mixed export-history/calibration limits.

## 12-13. Controlled model comparison and sensitivity tradeoff

Same 119 subjects (19/100), exact five outer folds. Three inner folds, seed 20261011. Four fixed robust18/regional22 C1 logistic candidates, none/balanced weighting, unchanged Experiment 4 training-only StandardScaler/pipeline. Selection maximizes inner OOF AP. Selected thresholds maximize inner sensitivity subject to specificity >=0.80, with fixed ties. Outer labels do not enter fitting APIs. That research constraint does not guarantee outer specificity or clinical acceptability.

| Configuration/rule | Sensitivity | Specificity | Balanced accuracy | Accuracy | Precision | F1 | ROC-AUC | AP | TP/FN/TN/FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| robust_none_fixed | 10.5% | 96.0% | 53.3% | 82.4% | 33.3% | 16.0% | 0.709 | 0.385 | 2/17/96/4 |
| robust_none_selected | 52.6% | 84.0% | 68.3% | 79.0% | 38.5% | 44.4% | 0.709 | 0.385 | 10/9/84/16 |
| robust_balanced_fixed | 63.2% | 75.0% | 69.1% | 73.1% | 32.4% | 42.9% | 0.713 | 0.380 | 12/7/75/25 |
| robust_balanced_selected | 52.6% | 84.0% | 68.3% | 79.0% | 38.5% | 44.4% | 0.713 | 0.380 | 10/9/84/16 |
| regional_none_fixed | 10.5% | 96.0% | 53.3% | 82.4% | 33.3% | 16.0% | 0.702 | 0.362 | 2/17/96/4 |
| regional_none_selected | 47.4% | 82.0% | 64.7% | 76.5% | 33.3% | 39.1% | 0.702 | 0.362 | 9/10/82/18 |
| regional_balanced_fixed | 57.9% | 71.0% | 64.4% | 68.9% | 27.5% | 37.3% | 0.713 | 0.386 | 11/8/71/29 |
| regional_balanced_selected | 42.1% | 83.0% | 62.6% | 76.5% | 32.0% | 36.4% | 0.713 | 0.386 | 8/11/83/17 |
| nested_selected_fixed | 52.6% | 83.0% | 67.8% | 78.2% | 37.0% | 43.5% | 0.694 | 0.452 | 10/9/83/17 |
| nested_selected | 47.4% | 81.0% | 64.2% | 75.6% | 32.1% | 38.3% | 0.694 | 0.452 | 9/10/81/19 |
| exp4_joint_primary | 36.8% | 83.0% | 59.9% | 75.6% | 29.2% | 32.6% | 0.657 | 0.352 | 7/12/83/17 |

No selected regional configuration improves sensitivity or specificity over robust18's 10/19 at84%. Nested selection produces9/19 at81% (AUC0.694/AP0.452); the higher pooled AP does not translate to improved sensitivity at the chosen rule. Pooled weighted/unweighted scores are uncalibrated and mix scales across folds. Nested sensitivity delta versus robust is-5.3 percentage points (conditional95% interval-15.8 to0); specificity-3.0 points(-7 to0). Regional unweighted sensitivity delta-5.3(-15.8 to0), weighted-10.5(-26.3 to0). Robust sensitivity Wilson95% interval31.7-72.7%; regional unweighted27.3-68.3%; nested27.3-68.3%. Ranking/paired intervals are also broad.

Balanced robust at fixed0.5 detects12/19 but specificity drops to75% (25 false positives versus16), below the predefined80% research constraint. Balanced regional at0.5 detects11/19 at71% (29 false positives). These operating-point changes do not establish acceptable clinical performance or a sensitivity gain at matched specificity. No sensitivity improvement without a specificity cost was demonstrated.

Full confusion matrices, per-fold results and all eight requested metric intervals are saved in metrics.json. Intervals use Wilson proportions and shared2000 stratified conditional subject-score bootstrap draws, seed20261011. They omit model-refit uncertainty and dependence from overlapping folds; repeated experiments and19 positives limit inference.

## 14. Blocker and exact next actions

All119 proposals have owner acceptance, including14 previously uncertain cases; no glare-review blocker remains. The private register retains review scope/text, date, reviewer and exact native-mask/PDF hashes. The original anatomy and pending proposal records remain immutable. Strict reviewed subset and full cohort coincide, with both classes in the unchanged folds. Further clinical confirmation requires independent subjects; no retuning of these repeatedly used cases is independent validation.

## 15. Artifacts, reproduction and Git

Acceptedv2 stores detector/region/preprocessing/source configs, fresh audit/labels/inputs, native masks, coverage/queue/panels/sheets, original and proposed-exclusion robust/regional caches,80 classifier checkpoints, full outer/inner subject assignments, candidate/selected OOF probabilities/cutoffs, metrics/intervals, coefficient stability, reports, PNG/SVG plots, implementation copies, training commit and reproduction guide. The anatomy-only output contains no glare-exclusion model; the separate owner-reviewed comparison contains160 original/excluded pipelines. Private file/directory modes are600/700.

Stoppedv1 remains sealed as STOPPED_BEFORE_EVALUATION_CSV_ROUNDING_CHECK, with its80 fitted models retained but not accepted as results. The authoritative comparator was already exact; the fix replaces a CSV-roundtrip assertion only. Acceptedv2 reused295 checksum-identical deterministic preparation assets and refitted all models under corrected commit350b6c9. The initial typed-label comparison failure stopped before creating a preparation directory and its log remains preserved in exp04b_preflight_2026-10-12_v1.

Reproduce with the project Python3.10 environment from the Experiment4B worktree, one CPU thread and writable PYTHONPYCACHEPREFIX:

```bash
python -m src.eyelid_glare_reproduce --source-root /Users/madz/Documents/aima-screening --output /Users/madz/Documents/aima-screening/data/processed/local_eyelid_exp04b_glare_regional_2026-10-12_v2 --fresh-features
```

Checksums are verified before model loading; recorded Git/source hashes,80 checkpoints, all inner/outer decisions, metrics/intervals/stability and119 freshly regenerated masks/features are checked without fitting. Independent sklearn metrics are checked separately. Verification proof/logs and final Git/file handoff stay in exp04b_verification_2026-10-12_v2.

Feature commits:

- 4b8d0fe feat(annotations): add conservative eyelid glare proposals and review panels - detector, panels, PDF renderer, synthetic tests, frozen methods/core docs.
- a70350a feat(preprocessing): add controlled glare diagnostics and ROI-relative color features - exact features, regions, audit/preparation, privacy/readiness tests/core docs.
- 85cbb9a fix(preprocessing): compare fresh audit labels across serialized types - label representation regression fix/test and log.
- 859ffa9 feat(evaluation): add bounded nested regional comparison and reproduction - four candidates, saved models/conditional evaluation/stability, independent verifier and tests/core docs.
- 350b6c9 fix(evaluation): compare authoritative experiment thresholds without CSV rounding - exact comparator regression fix/test and log.

Final results documentation commit and all full hashes/file lists are recorded in the private final_git_handoff.json. New commits contain source/tests/aggregate documentation only. No patient image, annotation/mask, clinical row, prediction or trained model is committed. Legacy tracked public result files remain untouched. Other worktrees and their unrelated files remain untouched.

## Owner-reviewed full-cohort extension

Final reviewed comparison output:

`/Users/madz/Documents/aima-screening/data/processed/local_eyelid_exp04b_reviewed_comparison_2026-10-12_v1/`

The owner-approved protocol adds only the four exact glare-excluded counterparts to the four original candidates, keeping all parameters, regions, subjects, outer/inner folds and cutoff rules. Original, excluded and full-eight policies select by inner AP. Every accepted mask and every original/excluded feature was freshly validated before fitting; no source/anatomical mask changed. Original robust binary predictions and JSON thresholds remain exact.

Additional feature commit:

- 1388ec7 feat(evaluation): add owner-reviewed full-cohort glare comparison - reviewed hash/status gate, fresh source/mask/feature checks, eight-candidate comparison/160 checkpoints, reproduction, review tests and approved method/core docs.

| Reviewed configuration/rule | Sensitivity | Specificity | Balanced accuracy | Accuracy | Precision | F1 | ROC-AUC | AP | TP/FN/TN/FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| original_robust_none_fixed | 10.5% | 96.0% | 53.3% | 82.4% | 33.3% | 16.0% | 0.709 | 0.385 | 2/17/96/4 |
| original_robust_none_selected | 52.6% | 84.0% | 68.3% | 79.0% | 38.5% | 44.4% | 0.709 | 0.385 | 10/9/84/16 |
| original_robust_balanced_fixed | 63.2% | 75.0% | 69.1% | 73.1% | 32.4% | 42.9% | 0.713 | 0.380 | 12/7/75/25 |
| original_robust_balanced_selected | 52.6% | 84.0% | 68.3% | 79.0% | 38.5% | 44.4% | 0.713 | 0.380 | 10/9/84/16 |
| original_regional_none_fixed | 10.5% | 96.0% | 53.3% | 82.4% | 33.3% | 16.0% | 0.702 | 0.362 | 2/17/96/4 |
| original_regional_none_selected | 47.4% | 82.0% | 64.7% | 76.5% | 33.3% | 39.1% | 0.702 | 0.362 | 9/10/82/18 |
| original_regional_balanced_fixed | 57.9% | 71.0% | 64.4% | 68.9% | 27.5% | 37.3% | 0.713 | 0.386 | 11/8/71/29 |
| original_regional_balanced_selected | 42.1% | 83.0% | 62.6% | 76.5% | 32.0% | 36.4% | 0.713 | 0.386 | 8/11/83/17 |
| glare_excluded_robust_none_fixed | 10.5% | 96.0% | 53.3% | 82.4% | 33.3% | 16.0% | 0.710 | 0.392 | 2/17/96/4 |
| glare_excluded_robust_none_selected | 52.6% | 83.0% | 67.8% | 78.2% | 37.0% | 43.5% | 0.710 | 0.392 | 10/9/83/17 |
| glare_excluded_robust_balanced_fixed | 63.2% | 73.0% | 68.1% | 71.4% | 30.8% | 41.4% | 0.721 | 0.392 | 12/7/73/27 |
| glare_excluded_robust_balanced_selected | 52.6% | 84.0% | 68.3% | 79.0% | 38.5% | 44.4% | 0.721 | 0.392 | 10/9/84/16 |
| glare_excluded_regional_none_fixed | 10.5% | 96.0% | 53.3% | 82.4% | 33.3% | 16.0% | 0.701 | 0.362 | 2/17/96/4 |
| glare_excluded_regional_none_selected | 47.4% | 79.0% | 63.2% | 73.9% | 30.0% | 36.7% | 0.701 | 0.362 | 9/10/79/21 |
| glare_excluded_regional_balanced_fixed | 57.9% | 69.0% | 63.4% | 67.2% | 26.2% | 36.1% | 0.708 | 0.378 | 11/8/69/31 |
| glare_excluded_regional_balanced_selected | 42.1% | 79.0% | 60.6% | 73.1% | 27.6% | 33.3% | 0.708 | 0.378 | 8/11/79/21 |
| original_selected | 47.4% | 81.0% | 64.2% | 75.6% | 32.1% | 38.3% | 0.694 | 0.452 | 9/10/81/19 |
| original_selected_fixed | 52.6% | 83.0% | 67.8% | 78.2% | 37.0% | 43.5% | 0.694 | 0.452 | 10/9/83/17 |
| glare_excluded_selected | 47.4% | 78.0% | 62.7% | 73.1% | 29.0% | 36.0% | 0.667 | 0.356 | 9/10/78/22 |
| glare_excluded_selected_fixed | 36.8% | 84.0% | 60.4% | 76.5% | 30.4% | 33.3% | 0.667 | 0.356 | 7/12/84/16 |
| full_selected | 47.4% | 78.0% | 62.7% | 73.1% | 29.0% | 36.0% | 0.667 | 0.356 | 9/10/78/22 |
| full_selected_fixed | 36.8% | 84.0% | 60.4% | 76.5% | 30.4% | 33.3% | 0.667 | 0.356 | 7/12/84/16 |

Glare-excluded robust unweighted detects the same10 anemia cases as original robust, with one additional false positive:52.6%/83%, AUC0.710/AP0.392. Weighted glare-excluded robust yields10/19 at84%, AUC0.721/AP0.392; original weighted robust also10/19 at84%, AUC0.713/AP0.380. Small ranking changes do not establish a sensitivity gain. Glare-excluded regional unweighted9/19 at79%, weighted8/19 at79%; regional analysis again adds no recall benefit. Full-eight and excluded-only inner selections coincide in this run:9/19 at78%, AUC0.667/AP0.356. Original-group selection9/19 at81% remains separate.

Masked robust unweighted minus unchanged robust sensitivity delta0, conditional interval[0,0] because their19 positive decisions are identical; this does not include model-refit uncertainty or establish universal equivalence. Specificity delta-1 point, conditional interval-3 to0; AUC delta+0.000526, interval-0.013158 to+0.012645; AP delta+0.006449, interval-0.009627 to+0.030129. Full-selected sensitivity delta-5.3 points(-15.8 to0), specificity-6 points(-11 to-2). No tested training-selected configuration improves sensitivity over the preserved10/19 comparator. Fixed0.5 weighting trades specificity for recall and is not a matched operating-point improvement.

The owner-review gate is clear and the strict/full analyses are complete. No further mask review is required for this exact accepted package. Independent clinical validation and quantitative detector accuracy remain unestablished. All 111 tests run: 108 pass/3 unavailable-worktree-fixture skips; source compilation and whitespace checks pass. Source images, labels, anatomy, earlier outputs and review hashes remain unchanged.

Final reproduction: all160 reviewed-comparison checkpoints,222 checksum-covered files, exact inner/outer subject partitions, training-only scaler statistics, selections/cutoffs, independent pooled/per-fold sklearn metrics, conditional intervals and coefficient stability reproduce without fitting. All119 original/excluded features were freshly re-extracted before this training. The preserved anatomy-onlyv2 independently reproduced80 models,438 covered files and119 fresh masks/features. Every prior experiment/source/review bundle remains unchanged. Reproduction evidence and the readable comparison PNG/SVG are in the separate verification companion; the111-test suite is108 passed/3 skipped.

For reviewed checkpoints:

```bash
python -m src.eyelid_glare_reviewed --reproduce --source-root /Users/madz/Documents/aima-screening --output /Users/madz/Documents/aima-screening/data/processed/local_eyelid_exp04b_reviewed_comparison_2026-10-12_v1
```
