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

Experiment4B added four ROI-distance core/rim color contrasts and separately reviewed96 glare/23 no-glare masks, then compared eight bounded original/excluded C1 logistic configurations on the same nested folds. The strict reviewed subset equals the full119-person cohort. Every160-model checkpoint/decision/metric and all source features were verified; no detection improvement was established. Experiment5 is owner-authorized CP-AnemiC public pediatric classification/Hb-regression pretraining with separate adult heads, head-only/late-block/frozen-embedding+robust-LAB adaptation, matched ImageNet controls and unchanged outer folds. Release/input auditing, both public objectives and all nested local adaptations are completed and reproduced; see the completed methods below.

## Experiment 5 — completed methods

CP-AnemiC Version 2 was selected after official archive authentication (DOI 10.17632/m53vz6b7fx.2). All 503 local files match the release byte-for-byte. Its 251 canonical RGBA images contain 110 anemia and 141 non-anemia labels (43.8247% anemia); the 251 copies in All Categories are excluded. The true patient count is unavailable. The 251 curation IDs plus five near-image pairs form 246 conservative split groups. The grouped split has 199 training images (90 anemia/109 non-anemia) and 52 validation images (20 anemia/32 non-anemia), seed 20261012. Public validation is checkpoint-selected development evidence on image groups.

Version 1 was authenticated and inspected: 710 image files, 498 unique file hashes, 485 decoded rasters and 87 repeated-image groups. Of those groups, 86 contain conflicting Hb measurements and one crosses binary class labels. All 251 Version 2 images are byte-identical Version 1 subsets with matching source IDs and Hb; none belongs to a conflicting exact-image group. The releases were not combined. The original public pediatric target, Hb <11 g/dL, is retained separately from local male <13/female <12 labels. Observed local ages are 16–61, and non-pregnancy remains an owner attestation outside the Sheet. No individual demographics or patient linkage were borrowed from Version 1.

Public alpha=255 tissue and local approved manual 0/255 masks use the same tight bounding box, ImageNet-mean background before resizing, aspect-preserving 224-square bilinear-antialiased resize, and mean letterboxing. The input has three RGB channels. Partially transparent boundary pixels are excluded. Anatomical masks, glare and clinically meaningful color remain otherwise unchanged. Public untagged RGB is explicitly assumed sRGB for computation without evidence of calibration. Local neural RGB reuses verified ICC/DNG profile handling; classical robust18 exactly matches the historical original-rendering manual-mask features. Common processing reduces artificial transparency/background cues but cannot erase acquisition or geometry differences.

Both public objectives initialize verified ImageNet1K_V1 MobileNetV3-Small weights. Earlier blocks and BatchNorm statistics remain frozen; original and flipped early maps are cached without labels. Classification uses training-only weighted BCE and validation AP stopping. Regression uses training-only Hb mean/std, standardized SmoothL1 and physical-g/dL validation MAE stopping. Both use head LR 0.001, tail LR 0.0001, AdamW decay 0.001, batch size 32, maximum 24 epochs and patience 4. Both real public runs stopped after five epochs, selecting epoch 1. The bounded procedure was fixed before local evaluation.

For each CP objective and the matched ImageNet control, A replaces the public output with a new adult head and freezes the backbone (head LR 0.001). B replaces the head and adapts only the last two blocks (head LR 0.0003/tail LR 0.00001). A/B use fixed BatchNorm statistics, weighted BCE, maximum 12 epochs, patience 4, inner AP stopping and rounded-median inner best epochs for the outer refit. C uses frozen 576-dimensional embeddings plus exact robust18 color features: training-only embedding StandardScaler/PCA8, then a fitted StandardScaler on 26 inputs and L2 logistic regression with C=0.1, balanced weights, lbfgs and maximum 2,000 iterations. New adult heads use identical seeds across matched objectives; pediatric and regression heads are discarded.

The 119 subjects (19 anemia/100 non-anemia) and five outer folds are unchanged. Three inner folds use seed 20261011. Nine candidates and tie order were fixed before fitting. Per-objective and joint policies select pooled inner AP; the training API has no outer labels. Both fixed 0.5 and inner-selected thresholds are reported. The latter maximize sensitivity subject to specificity ≥0.80, with specificity/higher-cutoff ties. All scaling, PCA, class weights, checkpoint epochs and threshold decisions use the relevant training folds. No calibration, outer tuning or classifier holdout is introduced.

Uncertainty uses 95% Wilson intervals for class rates and 2,000 conditional stratified paired subject bootstrap draws for ranking metrics, F1, balanced accuracy and differences, seed 20261011. The same draws compare the robust color benchmark and matched CP versus ImageNet adaptations. Intervals condition on saved scores, fix prevalence and omit refit, selection and overlapping-fold uncertainty. Public-only pediatric scores on AIMA remain exploratory; Hb regression is reported in g/dL with bias and training-mean baselines. Local Hb and outer labels never guide public objective selection.

Source: [frozen design](../experiments/eyelid-exp05-cp-transfer.md), [verified report](../experiments/eyelid-exp05-results-2026-10-13.md), recorded clean implementation commit and private experiment_config.json. All182 saved models reproduce without training, with all370 inputs/prefix maps freshly checked.

## Experiment 6 — implemented complementarity and ensemble methods

The compared bases are E4 `A_robust_lr_c1_none` (predefined candidate, not E4 primary) and E5 `cp_classification_C` (candidate, not joint primary). Read-only, checksum-first verification confirms exact 119 subjects/labels/folds, anemia positive-class ordering, source lineage and saved-model reproducibility. Both candidates share robust18 inputs; CP additionally has frozen backbone features, train-only PCA8/fused scaling and class-balanced logistic regression. All labels/masks and original folds are preserved.

After the exploratory historical overlap gate, the [frozen protocol](../experiments/eyelid-exp06-ensemble.md) defines equal raw-probability averaging and LAB weights0.50/0.75/0.25. Highest pooled inner AP selects the weighted policy; ties prefer that fixed order. For each unchanged outer fold, saved base inner scores arise from the same three subject folds entirely within outer training, seed 20261011. Forty relevant saved pipelines are reproduced; no model is refitted. The selection API accepts training labels only. Stored full-cohort OOF scores never train or select weights, thresholds, calibration or a stacker.

Cutoffs maximize inner sensitivity at specificity≥0.80(primary),0.85 or0.90, with specificity/higher-cutoff ties. Base endpoints use identical policies for comparable targets. Fixed0.5 remains separate. No calibration or stacking is implemented. Historical primary policies are displayed separately. These unequal uncalibrated score scales are not clinical risks; inner constraints do not guarantee outer specificity. Equal averaging and the chosen bases were proposed after earlier outcomes, so even fixed averaging is exploratory, not independently prespecified validation.

Report pooled/fold confusion counts and all eight classification metrics. Wilson class-rate intervals and 2000 shared stratified conditional paired bootstrap draws (seed 20261011) omit refit, selection, overlapping-training dependence and prevalence uncertainty. Prediction overlap reports class-specific paired counts/correlations/agreement without claiming an optimized union. A label-preserving descriptive image audit checks geometry, private context thumbnails, registered originals/possible repeats, verified Hb severity and prior rendering/glare evidence. WHO2024 severity descriptions do not change the cohort labels; non-pregnancy remains owner-attested outside the Sheet. No anatomy edits or feature search follows.

Implementation: `src/eyelid_ensemble.py`, `src/eyelid_ensemble_run.py`, `src/eyelid_information_audit.py`; exact copied code, model references, source hashes and reproduction instructions are in `data/processed/local_eyelid_classification_exp06_ensemble_v1/`. See [Experiment 6 report](../experiments/eyelid-exp06-results-2026-10-13.md).
