# AIMA implemented methodology

This document describes what the repository actually implemented through
2026-10-12. It separates implemented methods from proposals. The current
protocol manuscript is a research reference, not proof that a proposed step
was implemented: `/Users/madz/Downloads/Research Protocol_Proposal with Cover Page.pdf`
(SHA-256 recorded during the 2026-10-12 audit as
`1b575406391f017d246d505704d3e077745c526f59ffef4b266b02ad026f61d0`).

## Evidence and scope

The primary implemented anemia analyses are local eyelid-only experiments on
119 development subjects. The public baseline and segmentation studies are
separate data flows. A subject is the split group for local experiments; a
local classifier result is never described as an external or clinical test.

The canonical local metadata source is the frozen values export:

`data/raw/local/metadata/frozen_2026-10-07/AIMA Patients Metadata - Sheet1 values.csv`

Its verified SHA-256 is
`1e704fb8d575757f16fc4f690cdc3315bc4063a5cf258d6c2e1b5f784d19596d`. The
source is not overwritten. The non-pregnancy status used for local female
eligibility is an owner attestation outside the Sheet, not a Sheet field.

## Cohorts and eligibility

### Local AIMA cohort — implemented

- The frozen metadata contains 140 subjects and 275 image references: 140
  eyelid and 135 fingernail.
- The implemented local adult label rule applies to ages 15–65 inclusive,
  with Hb strictly below 13.0 g/dL for males and 12.0 g/dL for non-pregnant
  females. Missing or inconsistent Hb is not imputed.
- The current derived cohort contains 119 labeled eyelid subjects: 19 anemia
  and 100 non-anemia. Twenty subjects lack eligible laboratory Hb and one falls
  outside the configured age range.
- Missing nails do not exclude an otherwise eligible eyelid record. A nail or
  paired classifier has not been completed because approved positive nail
  support is insufficient.
- Local supervised inputs require resolved source links, approved manual
  anatomical masks, valid binary masks, verified labels, and subject-level
  fold membership. Unreviewed automated proposals are not training truth.

### Public classification cohort — implemented

- The original public baseline uses 184 unique country-subject keys: 115
  non-anemic and 69 anemic, with no separate external test set.
- The public-to-local transfer experiment applied its stricter documented
  filter to 218 public image/mask pairs and retained 183 subjects (64 anemia,
  119 non-anemia), split into 146 public training and 37 public validation
  subjects. Pregnancy-dependent labels were excluded when they were not
  consistent under the selected public rule.
- Public subjects are keyed by country plus subject number because a number is
  not globally unique. Public baseline CV is stratified over the one-row-per-
  key table; it is not described as an external validation.

### Public and local segmentation cohorts — implemented separately

- The public segmentation audit found 218 original JPEG/palpebral pairs and
  218 candidate palpebral PNGs. Those PNGs can be alpha- or white-background
  masked images and are not automatically treated as conventional binary
  masks.
- The local ExtraTrees segmentation pilot used ten distinct subjects per
  modality and manual masks as ground truth. The later public U-Net local
  development check evaluated 111 local references and left 29 reserved for a
  later segmentation comparison. The nail SAM run evaluated ten approved
  masks and generated unverified queue proposals.
- Segmentation metrics are anatomical overlap metrics, not anemia-classifier
  metrics.

## Labels and privacy — implemented

- Hb values remain continuous source measurements in private derived tables.
  Binary anemia labels are derived using the recorded policy; source values
  and original exports are not edited.
- Black subject-ID cards are identity-verification aids only and never model
  inputs. Direct identity references are kept separate from pseudonymous
  `subject_id` values.
- Source images, clinical metadata, annotation files, predictions,
  checkpoints, feature caches, and row-level outputs are ignored local files.
  Git contains code, tests, aggregate documentation, and non-sensitive summary
  metrics only.
- No implementation supports clinical diagnosis, treatment decisions,
  regulatory claims, encryption claims, or secure deployment claims.

## Image formats and rendering

### Public data — implemented

The public classification pipeline consumes the repository's public palpebral
color PNGs, with public JPEG/workbook provenance. The public segmentation
loader validates subject-key linkage, geometry, alpha/white-background rules,
spelling variants, non-empty masks, and orientation before producing ignored
binary derivatives. Original public files are not rewritten.

### Local data — implemented, with rendering limitations

Local originals include DNG previews and HEIC files. Missing nail annotation
TIFFs were rendered as full-size RGB TIFFs while retaining the original source
files. Recorded local eyelid histories in Experiment 1 comprise 99 Photos
exports, 18 Preview/ColorSync renders, 11 embedded DNG previews, and 12 pilot
exports with settings not recorded.

Experiment 1 and the historical LAB portions of Experiments 2–3 interpret
stored uint8 RGB as sRGB for an explicitly uncalibrated quantized OpenCV LAB
conversion. Experiment 3 neural input and Experiment 4 rendering B use an
available embedded profile to convert to sRGB; for 11 pixel-identical DNG
preview/TIFF cases, the genuine preview profile supersedes a misleading TIFF
tag. Rendering B harmonizes declared profiles but does not perform camera
calibration, recover unknown tone mapping, or prove colorimetric equivalence.

Experiments1-4 retain glare and do not perform learned image-level normalization, histogram equalization, white balance, inpainting or enhancement. Experiment4B subsequently used separately owner-reviewed glare proposals as a pixel-exclusion layer; it did not change anatomical masks or claim a validated automatic glare detector. These are experiment-specific policies.

## ROI, masks, and glare

- The primary local anatomical target is the exposed palpebral conjunctiva;
  nail plate is a separate target. Manual human-reviewed Fiji ROIs and binary
  masks are the local segmentation ground truth.
- Current canonical inventory records 150 approved anatomical ROI/mask pairs:
  140 eyelid and 10 nail. Masks are same-size, non-empty, 8-bit 0/255 PNGs
  linked to the annotation TIFF, original filename, subject, and hashes.
- Experiment 1–4 local classification features use only pixels inside the
  approved manual inner-eyelid mask. The historical LAB path uses all selected
  anatomical pixels, including glare-like pixels.
- Glare is a separate QC/annotation layer. Rejected or unreviewed glare
  candidates are not anatomical masks and were not used to silently exclude
  local classification pixels.
- Automated ExtraTrees and SAM proposals remain proposal-only. They do not
  replace approved manual masks and do not change the canonical manifest
  without human review and registration.

## Feature extraction

### Public baseline — implemented

The public baseline creates:

1. Six LAB statistics: mean and population standard deviation for L\*, a\*, and
   b\*.
2. A 576-value frozen MobileNetV3-Small ImageNet embedding.
3. One fixed image-margin ITA value.

The LightGBM classifier therefore receives 583 numeric features. The ITA
implementation is an exploratory fixed patch proxy, not a validated skin-tone
protocol.

### Local Experiment 1 — implemented

Six masked LAB statistics (mean/std for L\*, a\*, b\*) are extracted from the
full-resolution manual-mask pixels. Population standard deviation (`ddof=0`)
is used. No resize or background pixels enter the LAB statistics. A training-
fold `StandardScaler` precedes L2 logistic regression (`C=1`, `lbfgs`, no
class weight, `max_iter=2000`).

### Local Experiment 2 — implemented

The six LAB values are concatenated with 576 frozen ImageNet1K_V1
MobileNetV3-Small features. The neural input mean-fills outside the manual
mask, crops to the mask bounding box, preserves aspect ratio, letterboxes to
224×224, and applies ImageNet normalization. Scaling and PCA with eight
components are fit within each training fold. The resulting 14 features enter
a conservative LightGBM model.

### Local Experiment 3 — implemented

The public-to-local transfer path uses profile-aware RGB-to-sRGB conversion,
zero-margin mask bounding boxes, ImageNet-mean background, 224 letterboxing,
ImageNet normalization, and training-only horizontal flip augmentation.
MobileNetV3-Small has a dropout 0.2 binary head. Public training adapts the
head and final two blocks while freezing earlier weights and BatchNorm
statistics. Local strategy A adapts 577 head parameters; strategy B adapts
351,121 head/late-block parameters. Training uses AdamW, weight decay 0.001,
training-only positive weights, inner validation stopping, and no local
demographic or Hb predictors.

### Local Experiment 4 — implemented

The bounded feature families are:

- **Original 6:** historical means and population standard deviations.
- **Robust 18:** original six plus medians, IQRs, selected L/a/b tails, and
  median/IQR of `log((R+1)/(G+1))`.
- **Regional 21:** robust18 plus right-minus-left median L/a/b contrasts in
  image-coordinate mask halves. These halves are not anatomically homologous
  across captures.

No age, sex, Hb, export method, quality proxy, or identity value is a
classification predictor. Feature definitions, units, and scaling are stored
in the Experiment 4 configuration.

## Models and training design

| Component | Implemented configuration | Status |
| --- | --- | --- |
| Public classical classifier | LightGBM on 583 fused features; 10-fold stratified CV; fixed 0.5 rule | Implemented and reproduced |
| Local LAB baseline | L2 logistic regression, C=1, no class weight, six features | Implemented and verified in Exp1 |
| Local hybrid | Frozen MobileNet features + LAB + training PCA8 + conservative LightGBM | Implemented and verified in Exp2 |
| Public-to-local transfer | MobileNetV3-Small public training; head-only and partial late-block local adaptation | Implemented and verified in Exp3 |
| Color/feature comparison | Bounded logistic/LightGBM grid across A/B rendering and 6/18/21 features, with none/balanced weights | Implemented and verified in Exp4 |
| Hb regression | Ridge(alpha=10) on managed robust18, training-only scaling, after classification | Implemented as Exp4 feasibility analysis |
| Nail-only / paired anemia classifier | Manual nail support, paired features, approved positive-class coverage | Planned/blocked |
| Independent external/prospective test | New subject collection and frozen model evaluation | Planned |

## Cross-validation, selection, and thresholds

- **Local Experiment 1:** five subject folds, seed `20261010`, every subject
  scored once by a model that excluded it; no tuning and no classifier holdout.
- **Experiments 2–4:** the same five outer subject folds with three inner
  subject folds, seed `20261011`, training-only transformations and candidate
  selection. Outer labels are excluded from model and threshold selection APIs.
- **Public baseline:** ten-fold stratified CV, seed 42, with no separate
  external test set. It must not be described as patient-group external
  validation.
- **Fixed threshold:** probability `>=0.5` is the fixed research rule in
  Experiment 1 and the like-for-like comparisons. It is not a clinical
  operating threshold.
- **Nested selected threshold:** inner OOF scores maximize sensitivity subject
  to inner specificity ≥0.80, with documented tie rules. Experiments 2 and 4
  also report predefined 0.70/0.90 trade-off constraints. Achieved outer
  specificity is not guaranteed to equal the inner constraint.
- **Class imbalance:** the local development cohort has 19 positive and 100
  negative subjects. Exp1 is unweighted. Exp2–4 compare explicitly bounded
  unweighted and class-weighted candidates; weighting is fit inside training
  folds and does not create new samples.

## Evaluation and uncertainty

The classification reports include accuracy, sensitivity/recall,
specificity, balanced accuracy, precision, F1, ROC-AUC, average precision,
and confusion matrices. Counts use non-anemia/anemia rows and columns in that
order where matrices are printed. Reports state detected and missed anemia
subjects as TP and FN.

Proportion intervals use Wilson 95% intervals where implemented. Ranking and
paired comparison intervals use 2,000 stratified subject bootstrap draws with
the recorded seed, conditional on saved OOF predictions. They fix prevalence
and omit model-refit uncertainty and dependence from overlapping CV training
sets. An interval is not evidence of clinical significance or generalization.

Segmentation reports use Dice, IoU, precision, and recall on held-out manual
masks where available. Unannotated proposals have no truth metric.

## Implemented versus planned matrix

| Method or claim | Actual repository status | Evidence |
| --- | --- | --- |
| Local eyelid-only classifier | Implemented and verified (Exp1–4) | `docs/research/EXPERIMENT_REGISTRY.md`, private run manifests |
| Fixed 0.5 classification comparison | Implemented | Exp1/2/3/4 metrics bundles |
| Nested training-selected threshold analysis | Implemented | Exp2/3/4 inner-selection records |
| MobileNetV3 feature extraction | Implemented in public baseline and Exp2–4 | `src/feature_extractor.py`, experiment configs |
| CIELAB feature combination | Implemented | Public baseline, Exp1–4 reports |
| LightGBM classification | Implemented | Public baseline and Exp2/4 configs |
| Public Eyes Defy Anemia pretraining | Implemented in Exp3 | Exp3 v3 transfer bundle |
| Public-only local evaluation | Implemented in Exp3 | Exp3 metrics/report |
| Head-only and partial backbone adaptation | Implemented in Exp3 | Exp3 A/B configurations |
| Original vs color-managed rendering | Implemented in Exp4 | Exp4 rendering audit and A/B metrics |
| Six-, eighteen-, twenty-one-feature comparison | Implemented in Exp4 | Exp4 candidate metrics/configuration |
| Class-weighted and unweighted comparison | Implemented in Exp2/Exp4 bounded grids | Inner candidate records |
| Hb regression | Implemented as exploratory Exp4 feasibility output | Exp4 `regression/metrics.json` |
| Automated segmentation as ground truth | Not implemented; explicitly prohibited | Annotation workflow and audits |
| Nail-only/paired anemia model | Not completed; blocked by support/review | Current annotation queue and registry |
| Independent clinical validation | Not implemented | No external/prospective artifact exists |

## Experiment4B completed methods and Experiment5 scope

Experiment4B added four ROI-distance core/rim color contrasts and separately reviewed96 glare/23 no-glare masks, then compared eight bounded original/excluded C1 logistic configurations on the same nested folds. The strict reviewed subset equals the full119-person cohort. Every160-model checkpoint/decision/metric and all source features were verified; no detection improvement was established. Experiment5 is owner-authorized CP-AnemiC public pediatric classification/Hb-regression pretraining with separate adult heads, head-only/late-block/frozen-embedding+robust-LAB adaptation, matched ImageNet controls and unchanged outer folds. Release/input auditing is in progress; training methods are not yet completed.
