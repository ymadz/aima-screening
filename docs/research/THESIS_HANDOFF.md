# AIMA thesis handoff

This is the manuscript-facing summary of the verified repository record. It is
organized by likely thesis chapters and points back to experiment IDs and local
artifact paths so every quantitative statement can be checked. It is a handoff
for revision, not a claim that AIMA is clinically validated.

## How to use this document

- Treat [`RESULTS.md`](RESULTS.md) as the numerical source and
  [`METHODOLOGY.md`](METHODOLOGY.md) as the implementation source.
- Treat [`LIMITATIONS.md`](LIMITATIONS.md) as mandatory context for every
  performance table.
- Use the dated experiment reports and private `metrics.json` files for exact
  values, thresholds, confusion counts, and uncertainty.
- Do not copy patient images, row-level predictions, direct identity keys,
  checkpoints, or private diagnostic sheets into a manuscript repository or
  presentation.

## Chapter I — Research design and motivation

AIMA is an exploratory study of image-derived conjunctival and fingernail
appearance for anemia-related research. The implemented work contains a public
benchmark, local manual-eyelid classification experiments, separate
segmentation pilots, and a preliminary Hb-regression feasibility analysis.
The design does not establish a diagnostic device, treatment rule, or clinical
screening performance.

The verified research questions were:

1. Can a public color/visual feature baseline be reproduced?
2. Can a small local manual-eyelid LAB model provide an interpretable baseline?
3. Do frozen CNN features or public-to-local transfer improve it?
4. Do rendering-aware features and bounded robust statistics improve the local
   development result?
5. Can the same features provide a preliminary continuous-Hb feasibility
   signal?

The initial continuous-Hb question was implemented as an optional Experiment4 analysis. Experiment5 subsequently completed CP classification/Hb pretraining and local adaptation; independent-cohort confirmation has not begun. Source: registry IDs
`PUB-BASELINE-2026-08`, `AIMA-EXP01-FIXED-0.5`, `AIMA-EXP02`,
`AIMA-EXP03-PUBLIC-TRANSFER`, and `AIMA-EXP04-HB-REGRESSION`.

## Chapter II — Data sources and cohort construction

### Local AIMA cohort

| Quantity | Verified value | Source |
| --- | ---: | --- |
| Frozen metadata subjects | 140 | `data/raw/local/metadata/frozen_2026-10-07/` provenance |
| Frozen image references | 275 | `docs/core/PROJECT_LOG.md`, local manifest audit |
| Eyelid / nail references | 140 / 135 | local annotation manifest |
| Eligible labeled eyelid subjects | 119 | local cohort and Exp1 readiness bundle |
| Anemia / non-anemia | 19 / 100 | Exp1 `metrics.json` |
| Missing-Hb exclusions | 20 | local cohort summary |
| Age exclusion | 1 | local cohort summary |
| Approved eyelid / nail ROI-mask pairs | 140 / 10 | current annotation manifest |

The local label rule is Hb strictly below 13.0 g/dL for males or 12.0 g/dL
for non-pregnant females, applied to ages 15–65. Non-pregnancy is an owner
attestation outside the Sheet. Missing nails do not remove an otherwise
eligible eyelid record. See `METHODOLOGY.md` “Cohorts and eligibility” and
`data/processed/local_cohort_who2024_2026-10-08/summary.json`.

### Public data

The original public classification reproduction contains 184 country-subject
keys (115 non-anemic, 69 anemic). The transfer filter retains 183 public
subjects (64 anemia, 119 non-anemia), using 146 for public training and 37 for
public validation. The public segmentation audit found 218 original
JPEG/palpebral pairs. These public counts come from the public reproduction
bundle and Experiment 3 readiness report, not from the local Sheet.

### Privacy and handling

Original local images and clinical source files remain in ignored local paths.
Manual ROIs/masks and private predictions are not committed. Subject IDs in
the committed narrative are pseudonymous or aggregate-only. Black identity
cards are verification aids and never model inputs. Source: repository
`AGENTS.md`, `docs/policies/git-policy.md`, and the private run manifests.

## Chapter III — Image acquisition and ROI annotation

The public pipeline uses public JPEG/palpebral PNG pairs and validates subject
keys, geometry, orientation, alpha/white-background mask behavior, and known
filename variants. The local pipeline preserves DNG/HEIC originals and uses
full-size RGB TIFF derivatives for annotation. Local TIFF histories are mixed:
99 Photos exports, 18 Preview/ColorSync renders, 11 embedded DNG previews,
and 12 pilot exports with unrecorded settings in the Experiment 1 audit.

Manual Fiji ROIs and binary masks are the primary local anatomical truth. The
current approved inventory contains 140 eyelid and ten nail pairs. Anatomical
approval is an owner review and structural validation, not clinician
validation. Glare is a separate QC layer; no rejected or unreviewed glare
candidate was promoted to anatomical truth or silently used as an exclusion.

The ExtraTrees and SAM 2.1 segmentation studies are annotation-assistance
pilots. Their proposals remain unverified and must not be presented as the
final ROI method. Sources: `SEG-EXTRATREES-PILOT-2026-10-07`,
`SEG-PUBLIC-UNET-LOCAL-2026-10-09`, `SEG-SAM2-NAIL-PILOT-2026-10-09`, and
`docs/workflows/local-annotation-workflow.md`.

## Chapter III — Preprocessing and feature extraction

### Color rendering

Experiment 1 and historical LAB comparisons use stored RGB interpreted as
sRGB for an explicitly uncalibrated quantized OpenCV LAB conversion. Experiment
3 neural input and Experiment 4 rendering B use available embedded ICC
profiles for conversion to sRGB. For 11 pixel-identical DNG preview/TIFF cases,
the genuine preview profile supersedes a misleading TIFF tag. This corrects a
declared profile path but is not camera calibration.

### Features

- Public baseline: six LAB statistics + 576 frozen MobileNetV3-Small values +
  one fixed-margin ITA value (583 inputs).
- Exp1: six masked LAB mean/std values, no resize or background pixels in the
  color statistics.
- Exp2: six LAB values + 576 frozen ImageNet embeddings, training-fold PCA8,
  then conservative LightGBM.
- Exp3: profile-aware MobileNetV3-Small public pretraining with head-only and
  partial late-block local adaptation, compared with nested LAB.
- Exp4: original6, robust18, and regional21 feature families; bounded
  logistic/LightGBM and weighting comparisons.

No implemented classification run uses age, sex, Hb, identity, export method,
or quality proxy as a classifier predictor. Full parameter definitions are in
[`METHODOLOGY.md`](METHODOLOGY.md) and the versioned experiment configurations.

## Chapter III — Model architectures and training procedures

The public baseline uses LightGBM over the 583 fused inputs. The local baseline
uses L2 logistic regression (C=1) with training-fold StandardScaler. The local
hybrid adds frozen MobileNetV3 embeddings, training-fold scaling/PCA, and
conservative LightGBM. Transfer uses MobileNetV3-Small with public head/final
block adaptation and local head-only/late-block strategies. The optional Hb
model uses Ridge(alpha=10) on robust18 features.

Local classification uses five subject folds, seed `20261010`; Exp2–4 use
three inner folds, seed `20261011`. The fixed 0.5 rule and inner-selected
specificity-constrained thresholds are reported separately. All local metrics
are internal subject CV, and no final classifier holdout was available. Public
baseline CV is ten-fold stratified and has no separate external test.

## Chapter IV — Verified results for manuscript tables

Use the complete table in [`RESULTS.md`](RESULTS.md). A thesis-ready compact
selection is:

| Model/rule | Accuracy | Sensitivity | Specificity | Balanced accuracy | ROC-AUC | AP | Detected / missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Exp1 LAB, fixed 0.5 | 84.9% | 10.5% | 99.0% | 54.8% | 0.719 | 0.388 | 2 / 17 |
| Exp2 hybrid, inner selected | 73.9% | 36.8% | 81.0% | 58.9% | 0.576 | 0.247 | 7 / 12 |
| Exp3 transfer A/B, inner selected | 71.4% | 5.3% | 84.0% | 44.6% | 0.464 | 0.144 | 1 / 18 |
| Exp3 nested LAB comparator | 76.5% | 36.8% | 84.0% | 60.4% | 0.719 | 0.388 | 7 / 12 |
| Exp4 joint primary, inner ≥0.80 | 75.6% | 36.8% | 83.0% | 59.9% | 0.657 | 0.352 | 7 / 12 |
| Exp4 robust18 predefined comparison | 79.0% | 52.6% | 84.0% | 68.3% | 0.709 | 0.385 | 10 / 9 |

The confusion matrices are available in `RESULTS.md`. The Exp4 robust18 row
must be labeled “predefined exploratory comparison,” not “selected primary
model.”

## Chapter IV — Interpretation of findings

### Accuracy and class imbalance

The local Exp1 accuracy of 84.9% is close to the 84.0% majority baseline, but
the LAB model detects only 2/19 anemia subjects and misses 17. This is the
clearest evidence that overall accuracy is not an adequate summary under this
cohort's prevalence. Source: Exp1 report, “At threshold 0.5” and `metrics.json`.

### Hybrid features

The hybrid model detects more positives at fixed or selected rules but loses
specificity and ranking performance. This does not support a claim that frozen
CNN features improve local screening. Source: Exp2 report, “Interpretation
and limitations,” and paired metrics.

### Public-to-local transfer

Public-only local inference detects zero local anemia subjects, while selected
transfer detects 1/19. The negative result is useful evidence about domain and
preprocessing mismatch, but it does not isolate public pretraining as the sole
cause. Source: Exp3 report, “Unified local metrics” and “Uncertainty and
interpretation.”

### Rendering and robust features

The Exp4 primary joint nested-selected policy does not improve over the nested
LAB comparator. A predefined original-rendering robust18 comparison detects
10/19, but its conditional paired interval includes zero and its managed-color
counterpart detects 7/19. Treat robust18 as a hypothesis for confirmation.
Source: Exp4 report, “Unified comparison” and “Controlled interpretation and
uncertainty.”

### Hb regression

Ridge robust18 MAE is 1.3274 g/dL, compared with 1.3750 g/dL for the training-
mean baseline; R² is 0.1341. This is not a clinical Hb measurement. Source:
Exp4 private `regression/metrics.json`, report section “Optional Hb regression”.

## Chapter V — Discussion and limitations

The discussion should emphasize small local sample size, only 19 positive
cases, mixed rendering, manual ROI dependence, owner-only glare QC,
public/local domain differences, repeated development on the same subjects,
internal rather than independent validation, and sensitivity/specificity
trade-offs. See [`LIMITATIONS.md`](LIMITATIONS.md) for the full wording and
the established/hypothesis/external-validation distinction.

Negative hybrid, transfer, and public segmentation results should remain in
the thesis: they show that added complexity and public pretraining did not
automatically solve local generalization, and they justify the narrower future
work plan.

## Chapter VI — Conclusions, future work, and remaining validation

### Supported conclusions

- A reproducible public-data baseline and a reproducible local manual-eyelid
  research pipeline were implemented.
- The local fixed LAB baseline has high specificity but very low sensitivity at
  the fixed 0.5 rule; accuracy is misleading for this cohort.
- Frozen hybrid features and public-to-local transfer did not establish a
  reliable improvement over the local LAB comparator.
- Rendering provenance and robust feature statistics are important hypotheses,
  not resolved clinical methods.
- No current result demonstrates clinical performance or deployment readiness.

### Practical limitations

The current artifacts are private, source-linked, and reproducible only in the
configured local environment. Patient-level outputs cannot be shared through
this repository. Manual mask review and a controlled capture protocol remain
operational dependencies.

### Future work

1. Complete and review the pending nail annotations without promoting model
   proposals automatically.
2. Freeze one prespecified rendering and feature policy before a new run.
3. Collect more positive and borderline cases and reserve genuinely new
   subjects for external confirmation.
4. Experiment4B is now completed: reviewed glare/region features did not improve sensitivity. Experiment5 CP-AnemiC transfer is completed and reproduced; retain separate pediatric/adult heads and frozen inner-only decisions. Neither supplies an independent cohort.
5. Evaluate calibration, subgroup robustness, device variability, and
   prospective workflow only after the reference-label and privacy governance
   is defined.

### Remaining validation requirements

- independent subject-level cohort with reliable Hb reference labels;
- standardized acquisition, correct color-profile handling, and ideally a
  capture color reference;
- clinician-reviewed anatomical ROI agreement and a resolved glare protocol;
- fixed model/threshold frozen before the independent test;
- uncertainty analysis that addresses refitting and repeated development;
- safety, governance, privacy, device, and prospective workflow review.

## Figures and tables for manuscript revision

Existing aggregate figures should be reviewed from the private bundles rather
than regenerated:

- Exp1 internal CV summary: `data/processed/local_eyelid_classification_2026-10-10_v1/internal_cv_summary.png`.
- Exp2 evaluation: `data/processed/local_eyelid_classification_exp02_hybrid_2026-10-11_v1/evaluation.png`.
- Exp3 comparison: `data/processed/local_eyelid_classification_exp03_transfer_2026-10-11_v3/comparison.png`.
- Exp4 evaluation/trade-off/comparison figures under
  `data/processed/local_eyelid_classification_exp04_color_optimization_2026-10-12_v1/`
  and `data/processed/exp04_verification_2026-10-12_v1/`.

These files remain ignored because they may expose patient-derived visual
information. The central tables in `RESULTS.md` are safe aggregate substitutes
for the committed record.

## Citation-needed note

`[CITATION NEEDED]` Add verified peer-reviewed sources for general claims about
conjunctival/nail pallor, smartphone anemia screening, colorimetry, and the
clinical interpretation of Hb thresholds when revising the literature review.
No external literature claim is added here without a verified source. The
protocol manuscript is a project reference, not an independent evidence
source for the empirical metrics above.

## Verified4B update and authorized5 planning

For Chapters3-5, add4B owner-reviewed96-glare/23-no-glare preprocessing, unchanged nested subject folds and the negative detection finding: masked robust10/19 at83% unweighted/84% weighted, full-selected9/19 at78%; AUC for weighted masked robust0.720526. The prior robust10/19 at84% remains a predefined comparison, not Experiment4 primary. Experiment5 completed classification/Hb pretraining, new adult heads, matched ImageNet controls and nine bounded local candidates. Use the completed tables and interpretation below; no independent benefit is established.

## Experiment 5 manuscript integration — Chapters 3–5

Chapter3: use the completed methods in METHODOLOGY.md and the frozen CP contract: authenticated V2/unknown patient count, original pediatric Hb target, new adult heads, common opaque tissue processing, classification versus Hb pretraining, A/B/C matched controls, train-only nested decisions and unchanged119 subjects/folds. Distinguish the segmentation29 reservation from the absence of a classifier holdout.

Chapter4: use [verified Experiment5 tables](../experiments/eyelid-exp05-results-2026-10-13.md) and RESULTS.md for both thresholds, public-only/classification/regression, all adaptation candidates/policies, TP/FN/TN/FP, per-fold metrics and conditional intervals. Keep E4 primary separate from its predefined robust comparator.

Chapter5 conclusions/discussion: CP-AnemiC pretraining did not establish an improvement that justifies its added complexity on AIMA. The predefined joint policy detected **11/19 anemia cases (57.9% sensitivity) at 78% specificity**, compared with **10/19 (52.6%) at 84% specificity** for the predefined Experiment 4 robust color comparison. One additional detection came with six additional false positives. Balanced accuracy was slightly lower (67.95% versus 68.32%), ROC-AUC was 0.734211 versus 0.709474, and AP was 0.372546 versus 0.385148. The paired sensitivity difference had a conditional 95% interval of −21.1 to +31.6 percentage points; the specificity difference was −6.0 points [−11.0, −2.0]. This does not establish independent or clinically acceptable improvement.

The predefined CP-classification frozen-feature plus color candidate C detected **12/19 (63.2%) at 78% specificity**, with ROC-AUC 0.734211 and AP 0.372166. Its sensitivity 95% Wilson interval was 41.0–80.9%; specificity was 68.9–85.0%. Against the robust color comparator, its +10.5-point sensitivity difference had an interval of [−10.5, +36.8], while specificity fell by 6.0 points [−11.0, −2.0]. This is a predefined exploratory candidate comparison, not a replacement primary policy chosen from outer results. CP classification's actual inner-selected A/B/C policy detected 10/19 at 77% specificity; it selected C in four folds and B in one.

CP Hb-regression C and its inner-selected policy detected **11/19 at 79% specificity** (ROC-AUC 0.730000; AP 0.370429), with exactly the same binary OOF decisions as matched ImageNet C (11/19 at 79%; ROC-AUC 0.730526; AP 0.368331). CP classification C added one detection and one false positive versus matched ImageNet C; its paired ROC-AUC and AP intervals included zero. The frozen-feature performance is therefore compatible with ImageNet features plus robust color statistics without useful added CP pretraining. Head-only and partial CP adaptations with training-selected thresholds detected only 2/19 and 0/19 for classification, or 4/19 and 0/19 for regression. These outer results did not guide any redesign.

The public-only pediatric classifier detected **0/19 at 96% specificity** (ROC-AUC 0.459474; AP 0.150111). Public classification validation was weak (ROC-AUC 0.478125), and both bounded public objectives selected epoch 1. Public-only Hb prediction on AIMA had MAE 2.225671 g/dL, bias −1.905735 g/dL and R² −1.069180. Although its MAE was below the public-training-mean baseline of 2.801160, substantial adult underprediction and negative R² provide no evidence of a usable Hb estimator. Weak public fitting and observed acquisition/label differences are plausible explanations, not proven causes.

Retain robust color-only classification as the practical candidate for independent confirmation, with a prespecified acquisition, model and operating-rule policy and genuinely new positive and borderline cases. Frozen embeddings can remain a predefined comparator in that study. These repeated-cohort results do not justify additional CP fine-tuning complexity or choosing a new winner by outer sensitivity. The inner specificity constraint of 0.80 was a research threshold policy and did not guarantee 0.80 outer specificity. No clinical acceptability threshold or independent validation has been established.


No clinical/external validity follows. Candidate-level numerical differences are exploratory and cannot justify picking a new winner from outer scores. Refer to LIMITATIONS.md for small positive support, repeated development, source patient-linkage, calibration and domain-shift limits. All quantitative values trace to the sealed E5 artifacts and independent proof; no patient images/rows should be incorporated in public thesis materials.

## Experiment 6 manuscript integration — Chapters 3–5

**Chapter3:** describe the historical prediction-alignment/complementarity gate, candidate/primary distinction, unchanged 119 subjects 19/100 and original five folds. Explain equal averaging and three bounded weights, matched saved inner cross-fitted predictions, inner-AP weight selection, training-selected sensitivity thresholds at specificity constraints0.80/0.85/0.90 and fixed0.5. Forty saved pipelines reproduce without refitting. No full-cohort OOF training, stacking, calibration, new public data or automatic segmentation was introduced. The gate and candidate choice followed prior results, so all estimates remain exploratory.

**Chapter4:** report8 shared anemia detections,2 LAB-only,4 CP-only and 5 shared misses;16 shared false positives and 6 CP-only, with 78 shared non-anemia clearances. Scores correlate0.801841 and binary agreement is107/119. Equal averaging detects 11/19 at 79% specificity, ROC-AUC0.737895/AP0.379070, matrix[[79,21],[8,11]]. Weighted averaging detects 10/19 at 79%, ROC-AUC0.731579/AP0.371157, matrix[[79,21],[9,10]]. Compare robust LAB10/19 at 84% and CP C12/19 at 78%; E5 primary11/19 at 78% stays separate. Include fixed thresholds, secondary targets, fold variability and exact conditional intervals from [Experiment 6 report](../experiments/eyelid-exp06-results-2026-10-13.md) and RESULTS.md.

**Chapter5:** complementary individual errors did not translate into sufficient ensemble benefit. Equal averaging adds one detection and five false alarms versus robust LAB; weighted averaging adds none and five false alarms. Ranking-difference intervals span zero and the sensitivity gain interval includes zero. Do not promote the more favorable secondary75%-LAB0.85 candidate into a primary result after seeing outer performance. Prefer robust color-only as the practical exploratory confirmation hypothesis, without treating it as a validated final clinical model.

The image audit preserves all anatomy/labels and finds variable tissue extent, orientation and hair/eyelash occlusion handling. All119 masks are structurally valid; thumbnail review is not clinician validation. Five shared misses include3 mild and 2 moderate cases, contradicting an exclusively near-threshold explanation. One verified eyelid original per subject and 154 unassigned files do not support safe repeated-photo analysis. Mixed rendering/exposure/clipping persists;4B did not provide a recall gain and was not retrained. Use only aggregate figures in public thesis materials; private context sheets, prediction rows and the identity key remain excluded.

Trace every value to `data/processed/local_eyelid_classification_exp06_ensemble_v1/` and its109-file checksum manifest, with independent evidence in`data/processed/exp06_verification_2026-10-13_v1/`. The six existing research records carry the update on the new6 branch; previous worktrees/main remain preserved. Clinical/generalization claims require a fixed protocol and genuinely new subjects, with stronger positive-case support.
