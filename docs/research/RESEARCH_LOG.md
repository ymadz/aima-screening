# AIMA research log

This is the thesis-oriented chronological record of research milestones. It
extends the implementation-focused [`docs/core/PROJECT_LOG.md`](../core/PROJECT_LOG.md)
without replacing it. The log records observed implementation and artifact
evidence; it does not approve a new research method and it does not turn an
internal cross-validation result into clinical or external validation.

## Evidence vocabulary

- **Observed:** directly supported by a committed source file, a dated report,
  a configuration, a checksum manifest, or a verified private artifact.
- **Decision:** an owner-approved or explicitly recorded methodological choice.
- **Limitation:** a condition that constrains interpretation of an observed
  result.
- **Planned:** a proposal or future experiment that has not produced a verified
  run artifact.

Patient-level rows, images, annotations, predictions, and checkpoints remain
in ignored local paths. This document records their containing artifact paths
and non-sensitive aggregate results only.

## 2026-08-21 — Public inner-eyelid baseline implemented

- **Experiment/task:** `PUB-BASELINE-2026-08`.
- **Research objective:** establish the original public-data feature-fusion
  baseline used for repository development.
- **Dataset:** public Eyes Defy Anemia-derived India/Italy data; the later
  reproduction records 184 country-subject keys, 115 non-anemic and 69
  anemic.
- **Changes/methods:** public palpebral image loading, six CIE L\*a\*b*
  statistics, 576 frozen ImageNet MobileNetV3-Small features, one fixed-margin
  ITA feature, and LightGBM classification (583 numeric inputs).
- **Result:** the original implementation and its later reproduction use
  ten-fold stratified cross-validation. The reproduction reports mean-fold
  accuracy 0.804094, sensitivity 0.769048, specificity 0.827273, ROC-AUC
  0.872276; pooled OOF ROC-AUC is 0.869187.
- **Finding:** this is a public-data internal CV baseline, not local performance
  and not external validation.
- **Limitations:** one row per country-subject key is stratified rather than
  explicitly group-split; the ITA patch is not a validated dermatological
  measurement; public subgroup support is small and imbalanced.
- **Decision/next step:** preserve the baseline and state its scope explicitly
  before local experiments.
- **Git evidence:** `1d7ec4b`, `21f79c1`; later reproduction `47d8fc7`.
- **Artifacts:** `data/processed/public_baseline_reproduction_2026-10-07/`;
  [`docs/models/FINDINGS_AND_EVALUATION.md`](../models/FINDINGS_AND_EVALUATION.md).

## 2026-09-23 — Historical local cohort superseded

- **Experiment/task:** `LOCAL-COHORT-HISTORICAL`.
- **Research objective:** explore an early local metadata-derived cohort.
- **Dataset/result:** historical artifact recorded 102 rows (17 anemia, 85
  non-anemia).
- **Finding/limitation:** it used an older Downloads/working metadata source
  and inclusion rule. It is preserved as historical evidence, not a current
  training input.
- **Decision/next step:** use only the frozen 2026-10-07 metadata source for
  current local builds.
- **Git evidence/artifact:** recorded in `docs/core/PROJECT_LOG.md`; private
  historical files under `data/processed/local/`.

## 2026-10-07 — Canonical local metadata frozen and public baseline reproduced

- **Experiment/task:** `DATA-FREEZE-2026-10-07` and
  `PUB-BASELINE-REPRO-2026-10-07`.
- **Research objective:** make the local cohort reproducible and verify the
  public baseline against saved predictions.
- **Dataset/provenance:** the frozen values export contains 140 subjects and
  275 image references (140 eyelid, 135 nail). The canonical CSV SHA-256 is
  `1e704fb8d575757f16fc4f690cdc3315bc4063a5cf258d6c2e1b5f784d19596d`.
  The adjacent `provenance.json` and original Drive export were preserved.
- **Changes/methods:** source-hash verification, strict metadata validation,
  public feature-table validation, ten-fold stratified public reproduction,
  and exact OOF prediction comparison.
- **Result:** the reproduction matched the previously saved public OOF
  probabilities and labels exactly; metrics are recorded above and in
  `data/processed/public_baseline_reproduction_2026-10-07/metrics.json`.
- **Problems/limitations:** the Sheet has no pregnancy-status field and some
  Hb values are missing; the non-pregnancy status is an owner attestation
  outside the Sheet.
- **Decision/next step:** never overwrite the source export; derive ignored
  manifests with source path and hash.
- **Git evidence:** `58881dd` for local data validation and `47d8fc7` for
  public reproduction.
- **Artifacts:** `data/raw/local/metadata/frozen_2026-10-07/`,
  `data/raw/local/manifest/frozen_2026-10-07/`, and the public reproduction
  bundle.

## 2026-10-07 — Local segmentation proposal pilot

- **Experiment/task:** `SEG-EXTRATREES-PILOT-2026-10-07`.
- **Research objective:** test whether a small pixel-classification model could
  provide correction proposals for manually annotated eyelid and nail regions.
- **Dataset/method:** ten distinct subjects per modality, manual masks as
  ground truth, separate ExtraTrees pixel models, five subject-held-out folds,
  and three additional unscored proposals per modality.
- **Result:** eyelid mean Dice 0.694 / IoU 0.555; nail mean Dice 0.746 / IoU
  0.602. The unannotated proposals have no truth metric.
- **Finding:** proposals can assist review but included adjacent tissue in
  observed failures; they are not accepted anatomy and not anemia results.
- **Decision/next step:** retain proposal-only status and collect/review more
  manual masks before any acceptance study.
- **Git evidence:** `0cf891e`.
- **Artifact:** `data/processed/segmentation_prototype_2026-10-07/report.json`.

## 2026-10-08 — Local cohort and annotation readiness audited

- **Experiment/task:** `DATA-LOCAL-READINESS-2026-10-08` and
  `ANNOTATION-CONSOLIDATION-2026-10-08`.
- **Research objective:** consolidate annotation work, verify image-to-subject
  linkage, and determine whether local supervised training was ready.
- **Dataset:** 140 metadata subjects; 275 resolved references; current WHO
  adult 15–65 eligibility produced 119 labeled eyelid subjects (19 anemia,
  100 non-anemia), with 20 missing-Hb exclusions and one age exclusion.
- **Changes/methods:** one active annotation workspace under
  `data/raw/local/annotations/`, hash-linked manifests and pending queue,
  reviewed manual anatomy separate from glare, and subject-level fold
  preparation.
- **Result:** the initial readiness snapshot had insufficient reviewed positive
  masks for local training; training was explicitly not run at that stage.
  The later registration step brought the current approved inventory to 150
  anatomical ROI/mask pairs (140 eyelid, 10 nail), without changing source
  images or labels.
- **Limitations:** structural linkage and owner anatomical review are not
  clinician validation; glare remains a separate unresolved QC layer; nail
  positive support is inadequate for a paired/nail model.
- **Decision/next step:** preserve one consolidated queue and require reviewed,
  hash-linked manual masks before local model fitting.
- **Git evidence:** `58881dd`, `d80c8c3`, `b14eee9`.
- **Artifacts:** `data/processed/local_cohort_who2024_2026-10-08/` and
  `data/raw/local/annotations/manifests/`.

## 2026-10-08 — Missing nail TIFFs rendered without changing originals

- **Experiment/task:** `RENDER-LOCAL-ANNOTATION-TIFFS-2026-10-08`.
- **Research objective:** make the 135 frozen nail references available for
  annotation while preserving original DNG/HEIC files.
- **Changes/methods:** rendered 121 DNG previews and four HEIC originals to
  full-size RGB TIFFs, checked source hashes and pixel equality, and registered
  the results in the annotation inventory.
- **Result:** all 275 frozen image references resolved to annotation TIFFs;
  243 still required manual anatomical annotation at that stage.
- **Finding/limitation:** DNG previews declared Display P3 while some stored
  TIFFs carried sRGB tags; this is rendering provenance, not evidence of
  calibrated color equivalence.
- **Decision/next step:** retain the render history and settle color policy
  before treating local pixels as comparable color measurements.
- **Git evidence:** `b14eee9`.
- **Artifacts:** ignored annotation TIFFs and manifests under
  `data/raw/local/annotations/`.

## 2026-10-09 — Eyelid mask registration and segmentation transfer checks

- **Experiment/task:** `ANNOTATION-EYELID-REVIEW-2026-10-09` and
  `SEG-PUBLIC-UNET-LOCAL-2026-10-09`.
- **Research objective:** register owner-reviewed eyelid anatomy and measure
  whether the public segmentation model transferred to local images.
- **Changes/methods:** registered 140 reviewed eyelid ROIs; created 118 masks
  and reused 22 pixel-identical masks; retained ten reviewed nail masks;
  evaluated the existing public U-Net on 111 local development references
  without local training labels.
- **Result:** public U-Net local development mean Dice 0.0027, IoU 0.0014,
  precision 0.0021, recall 0.0075; 110/111 predictions triggered QC flags.
- **Finding:** the public-only segmenter did not transfer adequately to local
  full-frame eyelid images. It is not an anemia classifier and its predictions
  did not replace manual masks.
- **Limitations/next step:** the 29-subject segmentation reservation was not
  scored in that development run; future segmentation work remains separate
  from the anemia classifier and must preserve manual ground truth.
- **Git evidence:** `e9b091b`, `d80c8c3`.
- **Artifact:** `data/processed/segmentation/local_unet_dev_eval_fullframe_2026-10-09/summary.json`.

## 2026-10-09 — Nail SAM 2.1 assistance pilot

- **Experiment/task:** `SEG-SAM2-NAIL-PILOT-2026-10-09`.
- **Research objective:** evaluate optional prompt/refinement assistance for
  nail-plate annotation.
- **Dataset/method:** ten approved manual nail masks in leave-one-subject-out
  evaluation; ExtraTrees prompt masks plus official SAM 2.1 Hiera Tiny
  refinement; full pending queue proposal generation.
- **Result:** SAM-assisted mean Dice 0.794, IoU 0.669, precision 0.680,
  recall 0.987; ExtraTrees prompt baseline Dice 0.737 / IoU 0.591. The
  queue generated 116 proposals and 21 simple area flags.
- **Finding/limitation:** observed nail/skin leakage and partial masks remain;
  all 116 queue proposals are `predicted_unverified` and none is anatomical
  ground truth.
- **Decision/next step:** manually review/correct candidates and keep glare
  separate; do not update the canonical manifest from proposals alone.
- **Git evidence:** `81cc9ec`.
- **Artifact/report:** `data/processed/segmentation/sam2_nail_lopo_areaqc_v1/`
  and [`sam2-nail-pilot-2026-10-09.md`](../experiments/sam2-nail-pilot-2026-10-09.md).

## 2026-10-10 — Experiment 1 local manual-eyelid CIELAB baseline prepared

- **Experiment/task:** `AIMA-EXP01`.
- **Research objective:** establish a fixed, reproducible local eyelid-only
  anemia classifier using approved manual masks.
- **Dataset:** 119 owner-authorized labeled subjects (19 anemia, 100
  non-anemia); all five classifier folds are development folds and there is no
  untouched classifier holdout.
- **Changes/methods:** six masked LAB mean/population-standard-deviation
  features; current TIFF RGB interpreted as sRGB for an explicitly
  uncalibrated quantized OpenCV LAB conversion; glare retained; training-fold
  StandardScaler; L2 logistic regression C=1 with no class weighting; five
  subject folds, seed `20261010`; fixed research threshold 0.5.
- **Result:** preparation passed source, mask, label, fold, and duplicate gates;
  the first fitting smoke check was blocked until the owner completed the
  duplicate review.
- **Problems/limitations:** mixed rendering histories and only 19 positive
  subjects; owner anatomical review is not clinician validation.
- **Decision/next step:** complete the recorded duplicate review before fitting
  and preserve the fixed threshold as a research rule only.
- **Git evidence:** `31c2946`.
- **Artifacts:** `data/processed/local_eyelid_readiness_2026-10-10_v2/`.

## 2026-10-10 — Experiment 1 fixed-threshold five-fold evaluation completed

- **Experiment/task:** `AIMA-EXP01-FIXED-0.5`.
- **Research objective:** quantify the prespecified local CIELAB baseline after
  the owner cleared the duplicate gate.
- **Decision:** all 46 perceptual candidates were recorded as
  `distinct_sources` from the owner's explicit confirmation; no subjects were
  removed, merged, or reassigned.
- **Methods:** same six LAB features and five subject folds, training-fold
  scaling, no tuning, no holdout, no full-cohort deployment model.
- **Result:** pooled OOF accuracy 0.848739, sensitivity 0.105263, specificity
  0.990000, balanced accuracy 0.547632, precision 0.666667, F1 0.181818,
  ROC-AUC 0.719474, AP 0.387866; confusion matrix `[[99, 1], [17, 2]]`.
  The model detected 2/19 anemia subjects and missed 17.
- **Finding:** accuracy was close to the 0.840336 majority baseline and is
  dominated by the 100/119 non-anemia subjects; this result does not establish
  screening performance.
- **Limitations/next step:** conditional intervals omit refit and overlapping
  fold uncertainty. Improve provenance/rendering and positive-case support
  before any new local experiment.
- **Git evidence:** `4bf6e25` after completion.
- **Artifact/report:** `data/processed/local_eyelid_classification_2026-10-10_v1/`
  and [`local-eyelid-classifier.md`](../experiments/local-eyelid-classifier.md).

## 2026-10-11 — Experiment 1 preserved before Experiment 2

- **Experiment/task:** `PRESERVE-EXP01-2026-10-11`.
- **Research objective:** make the baseline recoverable before adding hybrid
  features.
- **Changes:** preservation branch `archive/eyelid-exp01-cielab-baseline`,
  annotated tag `eyelid-exp01-cielab-baseline-2026-10-11`, private archive,
  and read-only reproduction of ten saved fold models without refitting.
- **Result:** 97 archive files and the original Experiment 1/readiness hashes
  verified; no private data or checkpoints were committed.
- **Decision/next step:** all later experiments must use new versioned output
  directories and leave Experiment 1 immutable.
- **Git evidence:** commit `4bf6e25`; tag object resolves to commit
  `4bf6e2577d02b5e3338e1937140b7eaff31369b4`.
- **Artifact:** `docs/experiments/eyelid-exp01-preservation.md` and the
  owner-only archive recorded there.

## 2026-10-11 — Experiment 2 hybrid feature comparison completed

- **Experiment/task:** `AIMA-EXP02`.
- **Research objective:** test whether frozen MobileNetV3 features add useful
  signal to the six-feature local LAB model.
- **Dataset:** same 119 subjects and exact Experiment 1 folds.
- **Methods:** six LAB values plus 576 frozen ImageNet1K_V1 MobileNetV3-Small
  embeddings; training-fold embedding scaling/PCA8; 14 fused features;
  conservative LightGBM; nested five outer × three inner subject CV; two
  candidates (unweighted/balanced) selected by inner AP; inner threshold
  maximized sensitivity subject to specificity ≥0.80.
- **Result:** fixed 0.5 accuracy 0.756303, sensitivity 0.263158, specificity
  0.850000, balanced accuracy 0.556579, precision 0.250000, F1 0.256410,
  ROC-AUC 0.575789, AP 0.247161; matrix `[[85, 15], [14, 5]]`.
  Inner-selected thresholds yielded accuracy 0.739496, sensitivity 0.368421,
  specificity 0.810000, balanced accuracy 0.589211, precision 0.269231,
  F1 0.311111, ROC-AUC 0.575789, AP 0.247161; matrix `[[81, 19], [12, 7]]`.
- **Finding:** positive detections increased from 2 to 5/7, but false positives
  increased from 1 to 15/19 and ranking metrics declined versus Experiment 1.
  The added hybrid complexity was not justified by this run.
- **Limitations/next step:** same development cohort, no external/untouched
  classifier test, mixed rendering, few positives, and conditional intervals.
  Transfer was recorded as a separate future decision rather than silently
  combined with this result.
- **Git evidence:** feature commits `68720df`, `c4c77c7`, `01cb856`,
  `9d88cc7`; branch handoff `f4b2cb5`.
- **Artifact/report:** `data/processed/local_eyelid_classification_exp02_hybrid_2026-10-11_v1/`;
  reports in that directory and the Experiment 2 worktree.

## 2026-10-11 — Experiment 3 public-to-local transfer completed

- **Experiment/task:** `AIMA-EXP03`.
- **Research objective:** test Eyes Defy Anemia public pretraining and partial
  MobileNetV3 fine-tuning for local anemia classification.
- **Dataset:** public 218 image/mask pairs; 183 eligible public subjects (64
  anemia, 119 non-anemia) after the documented age/Hb/pregnancy-consistency
  filter, with 146 public training and 37 public validation subjects; local
  evaluation remained the same 119 subjects (19/100).
- **Methods:** ICC-aware RGB-to-sRGB rendering, mask-bounding-box input,
  ImageNet-mean background, 224 letterboxing and ImageNet normalization;
  public head/final-two-block adaptation; local strategy A head-only (577
  parameters) and B head/late-block (351,121 parameters), frozen earlier
  weights and BatchNorm statistics; training-only positive weights, early
  stopping, nested subject selection, and comparison with nested LAB.
- **Result:** selected transfer accuracy 0.714286, sensitivity 0.052632,
  specificity 0.840000, balanced accuracy 0.446316, precision 0.058824,
  F1 0.055556, ROC-AUC 0.463684, AP 0.143881; matrix `[[84, 16], [18, 1]]`.
  Public-only local inference had ROC-AUC 0.451053 and detected 0/19 under
  both reported rules. The separately nested LAB threshold analysis detected
  7/19 at 0.840000 specificity.
- **Finding:** public pretraining did not improve this local development
  result; it performed worse than the nested LAB comparison. Because neural
  preprocessing differed from the historical LAB path, the result does not
  isolate public pretraining as the sole cause.
- **Limitations/next step:** no external validation, same local development
  subjects, few positives, controlled public capture versus local capture, and
  uncalibrated scores. Preserve the negative result; any matched ImageNet-only
  ablation is a new experiment.
- **Git evidence:** `b206b39`, `724da9d`, `f114a2e`, `4908020`, `0515e52`;
  branch handoff `66afa8e`.
- **Artifact/report:** `data/processed/local_eyelid_classification_exp03_transfer_2026-10-11_v3/`
  and its report bundle.

## 2026-10-12 — Experiment 4 color/feature optimization completed

- **Experiment/task:** `AIMA-EXP04`.
- **Research objective:** examine rendering provenance and bounded feature/model
  comparisons while preserving the earlier experiments.
- **Dataset:** same 119 local development subjects and five outer folds; no
  classifier holdout.
- **Methods:** original stored-TIFF rendering (A) versus embedded-profile-aware
  sRGB conversion (B); six original, 18 robust, and 21 regional features;
  bounded logistic/LightGBM settings with class-weighted and unweighted
  variants; three inner subject folds; AP-based configuration selection; inner
  sensitivity at specificity constraints 0.80, 0.70, and 0.90. After
  classification, a fixed robust18 Ridge regression was assessed separately.
- **Primary result:** predefined joint nested-selected policy detected 7/19,
  with accuracy 0.756303, sensitivity 0.368421, specificity 0.830000,
  balanced accuracy 0.599211, precision 0.291667, F1 0.325581, ROC-AUC
  0.656842, AP 0.352228; matrix `[[83, 17], [12, 7]]`.
- **Predefined comparison result:** original-rendering robust18 unweighted
  C=1 logistic comparison detected 10/19, with accuracy 0.789916, sensitivity
  0.526316, specificity 0.840000, balanced accuracy 0.683158, precision
  0.384615, F1 0.444444, ROC-AUC 0.709474, AP 0.385148; matrix
  `[[84, 16], [9, 10]]`.
- **Important distinction:** the robust18 row is a predefined matched
  exploratory comparison, not the automatically selected Experiment 4 primary
  model. Its conditional paired sensitivity interval includes zero and its
  color-managed counterpart detected 7/19.
- **Regression result:** robust18 Ridge(alpha=10), training-only scaling, MAE
  1.3274 g/dL, RMSE 1.7022 g/dL, R² 0.1341; training-mean baseline MAE
  1.3750 g/dL, RMSE 1.8403 g/dL, R² −0.0121. This is feasibility evidence,
  not a clinical Hb measurement.
- **Findings/limitations:** color-profile correction harmonizes declared
  profiles but does not perform camera calibration; the same subjects informed
  four experiments; no result is independently confirmed.
- **Git evidence:** feature commits `31397a2`, `68a668e`, `2a0f971`,
  `c9fbacf`; branch handoff `0b50504`.
- **Artifacts/report:** `data/processed/local_eyelid_classification_exp04_color_optimization_2026-10-12_v1/`,
  `data/processed/exp04_verification_2026-10-12_v1/`, and the Experiment 4
  report in the Experiment 4 worktree.

## 2026-10-12 — Preservation audit and centralized documentation

- **Experiment/task:** `RESEARCH-CONSOLIDATION-2026-10-12`.
- **Research objective:** inventory Git/worktrees/artifacts and consolidate the
  verified research record before Experiment 4B or any new training.
- **Audit scope:** main `main` branch, `archive/eyelid-exp01-cielab-baseline`,
  `experiment/eyelid-exp02-hybrid`, `experiment/eyelid-exp03-public-transfer`,
  `experiment/eyelid-exp04-color-optimization`, and one detached Codex
  worktree. Main and the three experiment worktrees were clean. The detached
  worktree has four untracked documents and is associated with a separate idle
  “AIMA Thesis Q&A” task; it was left untouched.
- **Changes implemented:** central research log, experiment registry,
  methodology, results, limitations, thesis handoff, future-experiment guide,
  and a read-only documentation/artifact validator. No model training, source
  image/annotation/label changes, branch merge, reset, push, or upload occurred.
- **Result:** the records distinguish verified completion from planned work and
  preserve the Experiment 4 primary/robust comparison distinction.
- **Problems/limitations:** private artifacts remain local and ignored; no
  single portable archive contains every private artifact; the registry cannot
  substitute for owner-controlled backup of ignored data.
- **Decision/next step:** use the guide before any new experiment and require a
  new experiment ID, immutable versioned output directory, configuration,
  provenance, verification, registry entry, log entry, and feature-based Git
  commits.
- **Git evidence:** consolidation commits `71f1955` (log/registry), `f981e31`
  (methodology/limitations), `eb489b1` (verified results), `6ce11af` (thesis
  handoff/guide), and `dd2640a` (validator). No merge or push was performed.

## 2026-10-12 - Verified Experiment4B and Experiment5 initialization

The existing research records from committed main717da2c were incorporated into the isolated5 worktree without merging or modifying earlier worktrees. Reconciled their provisional4B/5 placeholders with the owner-approved actual experiment IDs. Authoritative metrics/checksums confirm4B160 models/222 files, weighted masked robust10/19 at84%, AUC0.720526/AP0.392086; full-selected9/19 at78%. Baselines1-4 and4B are preserved, with no independent validation claim. Experiment5 source/public availability audits are in progress; the existing V2 folder has251 unique RGBA images,110/141 labels, Hb4.2-15.0 and no cutoff mismatch; official release authenticity/grouping remains under review before training.
