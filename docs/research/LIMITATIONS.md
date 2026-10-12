# AIMA limitations and interpretation boundaries

These limitations apply to the current evidence record and should travel with
any thesis table or manuscript paragraph that uses it. Negative results are
preserved because they constrain the next design and prevent unsupported
claims about added complexity, transfer, or color handling.

## Cohort and label limitations

- The local anemia cohort contains 119 eligible eyelid subjects but only 19
  anemia cases and 100 non-anemia cases. One additional detection changes
  sensitivity by about 5.3 percentage points.
- The local cohort is small, class-imbalanced, and drawn from one local
  collection context. The public cohort is also small and imbalanced, and the
  public-to-local experiment has substantial age, sex, capture, and prevalence
  differences.
- Local Hb labels use the implemented adult 15–65 rule (male <13 g/dL,
  non-pregnant female <12 g/dL) and an owner attestation for non-pregnancy
  outside the Sheet. This is a documented research label policy, not a claim
  that all clinical contexts share the same threshold.
- The local classification analyses reused the same development subjects while
  comparing four experiments. Nested CV protects the inner/outer mechanics of
  each run, but repeated model development on the same people still limits
  confirmation of generalization.
- Public eligibility and label harmonization are a separate protocol from the
  legacy public baseline. Public pregnancy information is incomplete, and the
  stricter transfer filter retains only labels consistent under the documented
  rule.

## Image, rendering, and anatomical limitations

- Local images have mixed rendering histories: Photos exports,
  Preview/ColorSync renders, embedded DNG previews, and pilot exports without
  recorded settings. Stored RGB is not equivalent to calibrated color.
- Some DNG-derived previews carry Display-P3 provenance while annotation TIFFs
  carry sRGB tags. Experiment 4’s profile-aware conversion corrects a declared
  profile path but cannot recover unknown tone mapping, camera response, white
  balance, or a physical color reference.
- Public and local images differ in capture geometry, illumination,
  magnification, device, background, and domain. Public success cannot be
  treated as local transfer success.
- Manual conjunctival ROI/mask review is a dependency. Structural hash,
  geometry, binary-value, and owner review checks do not establish clinician
  agreement or clinical anatomical validity.
- Only a small number of nail masks are approved. Automated ExtraTrees and
  SAM proposals may include adjacent skin or partial tissue; all pending
  proposals remain unverified.
- The119 eligible eyelid glare proposals received owner visual acceptance in Experiment4B (96 nonempty,23 no-glare), retained separately from anatomy. Historical candidates remain unreviewed. Owner acceptance is not an independent clinician/reference-mask accuracy study; 4B exclusion did not improve sensitivity.

## Evaluation and statistical limitations

- The local analyses use internal subject-level CV but no untouched classifier
  holdout, external cohort, prospective cohort, or clinical reference study.
  The 29-subject segmentation reservation is not a classifier test set.
- The public baseline uses ten-fold stratified CV over one row per
  country-subject key. It is not a group-separated external test and should not
  be compared directly with the local subject-level CV as if protocols were
  identical.
- Training-selected thresholds are selected from inner information and can
  change sensitivity/specificity substantially. Their outer estimates remain
  development evidence, not approved operating thresholds.
- Conditional bootstrap intervals fix observed prevalence and saved scores;
  they omit model-refit uncertainty and dependence from overlapping CV
  training partitions. Wilson intervals are likewise sample-size limited.
- Per-fold validation contains only three or four positive local subjects, so
  the wide fold variation is expected and does not establish stability.
- Accuracy is dominated by the majority class. Precision can be undefined for
  all-negative predictions, and trapezoidal PR-AUC can show an endpoint
  artifact for a constant majority predictor; average precision is retained as
  the primary ranking summary where available.
- Model comparison differences are correlated because the same subjects,
  masks, labels, and folds recur. An apparent improvement is not independent
  confirmation.

## Model and feature limitations

- Experiment 2’s CNN embedding path and Experiment 3’s neural transfer path
  introduce preprocessing choices that differ from the historical stored-RGB
  LAB path. Negative transfer cannot be attributed to public pretraining alone.
- The hybrid model adds frozen embeddings and PCA but no embedding-only or
  PCA-only ablation was completed. The local cohort is too small to justify a
  broad architecture search.
- Robust18 and regional21 feature differences are exploratory. Regional halves
  are image-coordinate regions rather than guaranteed anatomical halves.
- Class weighting changes the learned score scale and calibration; the bounded
  weighted/unweighted comparisons do not establish a universally preferable
  weighting policy.
- The optional Ridge Hb analysis uses the same limited local development
  subjects, and its error is far too large and uncertain to be called a
  laboratory substitute.
- No calibration study, device latency benchmark, quantization study,
  prospective monitoring, or subgroup clinical performance study was completed.

## Established findings, hypotheses, and required validation

| Classification | Current statement | Evidence boundary / required next evidence |
| --- | --- | --- |
| Established within this repository | Exp1 fixed LAB detects 2/19 at 0.5; accuracy is misleading under the local prevalence. | Verified Exp1 artifact; confirm on new subjects before generalizing. |
| Established within this repository | Exp2 hybrid increased some fixed/selected detections but reduced specificity and ranking metrics versus Exp1. | Verified paired OOF comparison; a new cohort is needed for generalization. |
| Established within this repository | Exp3 public-only and selected transfer did not improve the local development result. | Verified v3 transfer artifact; matched preprocessing is needed to isolate causes. |
| Established within this repository | Exp4 primary joint policy did not improve sensitivity over nested LAB. | Verified Exp4 metrics; no independent confirmation. |
| Hypothesis | Original-rendering robust18 features may improve sensitivity. | Exp4 predefined comparison detected 10/19, but the paired interval includes zero and B detected 7/19; confirm on new data. |
| Hypothesis | Profile-aware rendering may reduce domain variation. | Only declared profile differences were corrected; camera calibration and prospective capture references are needed. |
| Hypothesis | Manual ROI standardization may improve transfer. | Public/local anatomical geometry remains visibly different; clinician-reviewed multi-site masks are needed. |
| Not established | Any model is clinically accurate, safe, calibrated, or deployable. | Requires independent clinical reference labels, prospective/external validation, governance, and device testing. |

## Why unsuccessful approaches matter

The hybrid and transfer results are useful negative evidence: they show that
adding frozen visual features or public pretraining did not automatically
improve this local development protocol. The poor public U-Net local overlap
also prevents silently treating automated masks as equivalent to manual
anatomy. Preserving these outcomes reduces the risk of selecting a favorable
comparison after observing outer results and gives the thesis a scientifically
honest account of why complexity and transfer are not automatically beneficial.

## Experiment5 input limitations under audit

CP-AnemiC describes Ghanaian children6-59 months, while AIMA retains its verified adult15-65 cohort. Public per-record age/sex/patient linkage, source RGB calibration, antialiased alpha edges and release differences must be audited. Preserve pediatric Hb<11 and adult sex-specific labels separately; no local outer labels may guide public objective or preprocessing selection. The same119 development subjects are repeatedly used; this is not the planned independent confirmation study.
