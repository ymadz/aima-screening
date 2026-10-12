# Experiment 4 verified results — 2026-10-12

The predefined joint nested-selection policy did not improve anemia sensitivity: 7/19 detected, 83% specificity, ROC-AUC 0.657/AP 0.352. A predefined matched feature comparison using original rendering, robust18 and unchanged unweighted C=1 logistic regression detected 10/19 at 84% specificity; it recovered 3 original false negatives and retained all 7 original true positives. This is promising exploratory evidence. Its paired sensitivity interval includes zero improvement, its color-managed counterpart detects7/19, and no independent confirmation or clinical validity follows.

## Readiness and preservation

Canonical frozen CSV SHA-256 `1e704fb8d575757f16fc4f690cdc3315bc4063a5cf258d6c2e1b5f784d19596d`, mtime 2026-10-07T14:55:39.254357+08:00; protocol SHA-256 `1b575406391f017d246d505704d3e077745c526f59ffef4b266b02ad026f61d0`, mtime 2026-09-22T18:39:41.367195+08:00. Both were verified before related source reading and rechecked before training. Metadata schema/trailing fields, duplicate IDs, Hb units/consistency, age/sex and 275 image references passed. Fresh audit validated 140 approved manual eyelids and 560 source/TIFF/ROI/mask hashes; eligible 119 (19 anemia/100 non-anemia). Twenty missing-Hb and one age exclusion remain unchanged. The existing46 similarity candidates retain owner-confirmed distinct-source adjudication.

Labels remain unadjusted adult15–65 male<13/female<12 g/dL; non-pregnancy is owner-attested outside the Sheet. The exact five outer folds, source labels and masks are unchanged. All119 are previously owner-authorized classifier development subjects; no untouched classifier test set exists. The separate segmentation reservation is preserved. Saved Experiments 1–3 were reproduced without fitting before this experiment, and all historical checksums and preservation references remain unchanged after verification.

## Diagnostic findings

The 12 original LAB nested-threshold misses include 8 mild and 4 moderate cases under the descriptive [WHO2024 adult severity table](https://www.ncbi.nlm.nih.gov/books/NBK602183/table/fm-ch1.tab2/). Median Hb 11.55 g/dL, versus 10.8 in 7 detected cases (3 mild,2 moderate,2 severe). Median distance below the applicable cutoff is0.6 versus 1.2 g/dL. This supports investigating overlap near the label boundary, without changing labels.

Native ROI Laplacian variance medians 7.57 versus 7.58 and gradient medians 11.43 versus 11.59 are almost equal in missed/detected groups. These resolution/texture-dependent proxies do not establish focus quality or a blur cause. Mean-L medians 66.34 versus 71.67 and mean-a45.22 versus 36.63 describe brighter/less-red detected crops. These are uncalibrated associations conditioned on existing model predictions and do not distinguish capture effects from tissue physiology.

Mask pixel medians 107,633 versus 132,779; full-frame occupied fractions 0.009 versus 0.011. Approved anatomical masks are unchanged; these quantities cannot independently validate tissue coverage. Visual review of all 19 private masked crops found specular-looking spots in both groups and variable brightness/redness. Any-channel clipping medians 46.9%/52.6%, versus 32.7% in non-anemia, are much larger than the bright-neutral proxy; clipping cannot be called glare. No glare or quality exclusion was applied. The private report contains pseudonymous missed-case IDs, Hb distributions, profiles and per-subject proxies; direct linkage remains in a separate private key.

## Rendering audit and implemented corrections

Eligible inputs contain 115 DNG and 4 HEIC originals; their recorded source profiles are Display P3. TIFF histories:81 Photos exports,17 Preview/ColorSync exports,11 embedded DNG-preview copies and 10 unrecorded pilot exports. All annotation TIFFs carry sRGB tags. For 11 DNGs, oriented preview pixels match TIFF pixels exactly, proving the preview ICC should supersede the misleading tag; B performs a real relative-colorimetric conversion to sRGB. None of these 11 is an original false negative (one original TP, ten non-anemia). The correction can affect training distributions but does not directly change the twelve missed input images.

A reproduces stored TIFF RGB and the historical quantized OpenCV LAB path. B uses actual available ICC conversion while retaining exactly the same native raster and mask. Other 108 exports use their embedded export ICC; undocumented tone mapping/retagging cannot be reconstructed from tags alone. This harmonizes declared profiles, not camera calibration. Original sensor development was not substituted for aligned annotation rasters. No normalization, white-balance correction, enhancement, resizing, inpainting or automatic segmentation. RGB uint8, L=encodedL×100/255, a/b=encoded−128 and population standard deviations match the historical units.

## Frozen feature, model and evaluation design

Three bounded configurations: original 6 means/stds; robust18 adds medians, IQRs, selected tails and log((R+1)/(G+1)) location/spread; regional21 adds three right-minus-left median contrasts in image-coordinate mask halves. Definitions and limitations are in [the methods contract](eyelid-exp04-color.md). No Hb/sex/age/quality/export/identity predictor enters classification.

36 candidates = two renderings × three feature sets × six model settings. Logistic L2 C=1/.1 with none/balanced weights and training-only StandardScaler; LightGBM none/balanced with 100 shallow trees, learning_rate.03, depth2/leaves3, min_child10, L2=5, min_split_gain.05, max_bin31 and deterministic single-thread. No oversampling, augmentation, early-stopping search, calibration or learned feature selection.

Three inner subject folds, seed 20261011; outer folds reuse seed 20261010. Inner pooled AP selects configuration, deterministic ties follow frozen order. Thresholds maximize inner sensitivity at specificity≥.80, ties higher specificity then cutoff; .70/.90 are predefined secondary rules. All outer labels are excluded from fitting/selection APIs. A/B/joint policy outputs are predefined, not chosen from outer results. Inner specificity does not guarantee achieved outer specificity. Prior experiments informed this design on these same subjects, so all comparisons remain exploratory development.

## Unified comparison

| Model / rule | Accuracy | Sensitivity | Specificity | Balanced acc. | Precision | F1 | ROC-AUC | AP | TP/FN/TN/FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Exp1 LAB / .5 | 0.848739 | 0.105263 | 0.990000 | 0.547632 | 0.666667 | 0.181818 | 0.719474 | 0.387866 | 2/17/99/1 |
| Historical LAB / inner .80 | 0.764706 | 0.368421 | 0.840000 | 0.604211 | 0.304348 | 0.333333 | 0.719474 | 0.387866 | 7/12/84/16 |
| Exp2 hybrid / inner .80 | 0.739496 | 0.368421 | 0.810000 | 0.589211 | 0.269231 | 0.311111 | 0.575789 | 0.247161 | 7/12/81/19 |
| Exp3 transfer / inner .80 | 0.714286 | 0.052632 | 0.840000 | 0.446316 | 0.058824 | 0.055556 | 0.463684 | 0.143881 | 1/18/84/16 |
| Exp4 A policy / inner .80 | 0.764706 | 0.368421 | 0.840000 | 0.604211 | 0.304348 | 0.333333 | 0.655263 | 0.351693 | 7/12/84/16 |
| Exp4 B policy / inner .80 | 0.756303 | 0.368421 | 0.830000 | 0.599211 | 0.291667 | 0.325581 | 0.653684 | 0.352379 | 7/12/83/17 |
| Exp4 joint policy / .5 | 0.764706 | 0.368421 | 0.840000 | 0.604211 | 0.304348 | 0.333333 | 0.656842 | 0.352228 | 7/12/84/16 |
| Exp4 joint policy / inner .80 | 0.756303 | 0.368421 | 0.830000 | 0.599211 | 0.291667 | 0.325581 | 0.656842 | 0.352228 | 7/12/83/17 |
| Exp4 A robust18 LR C=1 / inner .80 | 0.789916 | 0.526316 | 0.840000 | 0.683158 | 0.384615 | 0.444444 | 0.709474 | 0.385148 | 10/9/84/16 |
| Exp4 B robust18 LR C=1 / inner .80 | 0.781513 | 0.368421 | 0.860000 | 0.614211 | 0.333333 | 0.350000 | 0.696842 | 0.365129 | 7/12/86/14 |
| Exp4 joint / inner .70 | 0.705882 | 0.631579 | 0.720000 | 0.675789 | 0.300000 | 0.406780 | 0.656842 | 0.352228 | 12/7/72/28 |
| Exp4 joint / inner .90 | 0.798319 | 0.263158 | 0.900000 | 0.581579 | 0.333333 | 0.294118 | 0.656842 | 0.352228 | 5/14/90/10 |

The historical LAB selected rule is the separate nested threshold analysis completed during Experiment 3; Experiment 1 probabilities/artifacts remain unchanged. All rules use the same119 subjects and exact outer folds. Original fixed thresholds are.5; selected rules use inner-only cutoffs.

## Controlled interpretation and uncertainty

At matched C=1/unweighted/original 6, changing A to B gives 7→8 detected and 16→18 false positives, ROC.719→.723/AP .388→.384: no dependable rendering benefit. At A/C=1/unweighted, original 6→robust18 gives 7→10 detected with 16 false positives in both; regional21 returns 7 detected with 15 false positives. Robust feature engineering plus its training-selected cutoff improves this operating point, while ROC.709/AP .385 do not improve ranking. Fixed.5 robust18 still detects only 2/19 with 4 false positives.

At original A/6/C=1, balanced weighting gives 9 detected with 18 false positives versus 7/16 unweighted. At A/robust18/C=1, balanced and unweighted both detect10 with 16 false positives; weighting adds no count gain in that comparison. Other fixed candidates have different results, all saved in candidate_results.csv/candidate_metrics.json. No highest outer candidate is promoted to the primary winner.

The joint policy selects original 6 in every outer fold: C=1 balanced in folds1–2, B/C=1 unweighted in 3, A/C=0.1 balanced in 4, A/C=1 unweighted in 5. Its AP-driven inner selection did not choose the robust-feature sensitivity gain. Configuration/score-scale switching and few positives limit pooled interpretation; no calibration was studied.

Joint sensitivity 95% Wilson interval19.1–59.0%, specificity 74.5–89.1%, ROC conditional interval.522–.786/AP .213–.550. Joint-minus-historical-LAB paired sensitivity 0 percentage points (−15.8 to15.8), specificity−1 point (−4 to2), balanced accuracy−0.5 points (−8.4 to6.9), ROC−.0626 (−.1748 to.0505), AP−.0356 (−.1581 to.0719).

Robust-A sensitivity 52.6% (31.7–72.7%). Paired sensitivity gain15.8 points (0 to31.6); balanced accuracy gain7.9 points (0 to16.8); ROC difference−.010 (−.0626 to.0353), AP−.0027 (−.0691 to.0605). These conditional 2,000 stratified subject bootstraps reuse draws across models, seed 20261011. They fix prevalence, omit refit/overlapping-training uncertainty, and do not correct repeated experiments. An interval including zero is not evidence of confirmed superiority.

The 70% rule is a false-positive tradeoff: joint12 detected/28 false positives; the unchanged6-feature A/C=1 candidate under the same predefined70% inner rule detects13/19 at 73% specificity (27 false positives). This descriptive matched-rule calculation refits nothing and selects no outer winner. Higher recall from lowering the constraint cannot be attributed to new features or models.

## Fold stability

| Fold | Historical LAB sensitivity/spec./AUC | Exp4 joint sensitivity/spec./AUC | A robust18 LR sensitivity/spec./AUC |
| --- | --- | --- | --- |
| 1 | 0.000/0.650/0.425 | 0.250/0.600/0.412 | 0.250/0.600/0.462 |
| 2 | 0.750/0.950/0.950 | 0.500/0.950/0.912 | 1.000/0.950/0.962 |
| 3 | 0.500/0.900/0.850 | 0.500/0.850/0.850 | 0.750/0.850/0.863 |
| 4 | 0.500/0.900/0.787 | 0.500/0.950/0.812 | 0.500/1.000/0.787 |
| 5 | 0.000/0.800/0.500 | 0.000/0.800/0.500 | 0.000/0.800/0.550 |

Joint AUC spans.413–.913 versus historical.425–.950; robust-A spans.462–.962. Fold 5 still misses all 3 positives for these rules. Fold-wise steps of 25–33 points reflect only 3–4 positive subjects per validation fold; this does not demonstrate dependable stability.

## Optional Hb regression

After primary classification, fixed managed robust18 Ridge(alpha=10), training-only StandardScaler: MAE 1.3274 g/dL, RMSE 1.7022, R² 0.1341. Training-mean baseline MAE 1.3750, RMSE 1.8403, R²-0.0121. The MAE reduction is only .0476 g/dL. Sex supplies the existing anemia cutoff only for separate evaluation of predicted Hb, never a regression input. The private regression metrics include conditional error intervals, per-fold errors and anemia ranking/decision information. This is a feasibility output, not a clinical Hb measurement.

## Reproduction and artifacts

Accepted private run: `/Users/madz/Documents/aima-screening/data/processed/local_eyelid_classification_exp04_color_optimization_2026-10-12_v1`. Training implementation `2a0f971f909e388e82e2ffb216767eed0f3b1a91`, native features/training/evaluation219.1 seconds,810 checksum-covered files plus checksum manifest,8,063,998 bytes. Saved540 inner+180 outer classifier models and 10 regression models reproduce without fitting. Every source-based feature vector re-extracted exactly; all candidate/policy/inner/outer metrics, selections, cutoffs, train-only scalers and preserved experiments/refs verified. Final suite90 tests:87 passed/3 unavailable-data skips; compilation and whitespace checks passed.

The initial verifier refused object-dtype sex metadata. The sealed output and recorded implementation are unchanged. The corrected independent verifier validates exact archive hashes before loading trusted legacy M/F strings, and verifies copied code against its recorded Git commit. Future outputs use Unicode metadata. This command supersedes the original guide retained inside the sealed run:

```sh
PYTHONPYCACHEPREFIX=/private/tmp/aima-exp04-pycache OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
/Users/madz/Documents/aima-screening/aima_env/bin/python -m src.eyelid_color_verify \
 --source-root /Users/madz/Documents/aima-screening \
 --output /Users/madz/Documents/aima-screening/data/processed/local_eyelid_classification_exp04_color_optimization_2026-10-12_v1 --fresh-features
```

Independent proof/logs, final tests, matched-rule analysis, corrected guide and aggregate comparison figures are in separate ignored `/Users/madz/Documents/aima-screening/data/processed/exp04_verification_2026-10-12_v1`. The twelve-case clinical diagnostic stays private under the run diagnostics folder. New Experiment 4 commits add no model, clinical row, source photograph, mask or patient prediction to Git; no upload/push/deployment.

## Recommendations

Preserve the negative primary-selection result and treat robust18 as a bounded hypothesis for confirmation. Freeze one compact model and decision rule before evaluating newly collected subjects with more anemia cases, including mild/borderline cases. Standardize capture and retain correct color profiles; the promising A result depends on historical rendering and the B counterpart does not reproduce its count gain. Investigate unknown export histories and clipping with acquisition references rather than adding per-image normalization that could erase tissue color. A future prespecified study may align inner configuration selection with sensitivity at a fixed specificity, but must not retune this completed experiment from outer results. No additional architecture search is justified by this result.

## Git implementation handoff

Worktree `/Users/madz/Documents/aima-screening-exp04-color`, branch `experiment/eyelid-exp04-color-optimization`, verified base `66afa8e4611a755a96b6d177016d3888a839c5f5`. Training uses `2a0f971`; the new verifier correction uses `c9fbacf`. Feature commits:

| Commit | Message | Files / feature |
| --- | --- | --- |
| 31397a2 | feat(preprocessing): add masked color variants and anemia miss diagnostics | src/eyelid_color_data.py, src/eyelid_color_diagnostics.py, tests/test_eyelid_color_data.py, methods and core documentation |
| 68a668e | feat(classifier): add bounded nested color model comparison | src/eyelid_color_models.py, tests/test_eyelid_color_models.py, diagnostic whitespace and core implementation notes |
| 2a0f971 | feat(evaluation): seal and reproduce nested color experiment artifacts | src/eyelid_color_run.py, src/eyelid_color_reproduce.py, tests/test_eyelid_color_run.py, methods/core documentation |
| c9fbacf | fix(experiments): reproduce verified color string metadata | src/eyelid_color_verify.py, Unicode cache fix in src/eyelid_color_run.py, tests/test_eyelid_color_verify.py, method/log correction |

The final documentation commit includes this report, registry, model card and docs index/current architecture/log. Its exact hash and every commit's full file list are recorded in the private final_git_handoff.json and final response. New commits contain source/tests/aggregate documentation only. Existing legacy public result CSVs are untouched. The separate Q&A worktree's four untracked documents are unrelated and left untouched. No push or history rewrite.
