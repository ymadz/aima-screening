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

## Experiment5 audited input limitations

CP-AnemiC describes Ghanaian children6-59 months, while AIMA retains its verified adult15-65 cohort. Audits confirmed unavailable public per-record age/sex/patient linkage, unknown source RGB calibration, antialiased alpha edges and release-integrity differences. Preserve pediatric Hb<11 and adult sex-specific labels separately; no local outer labels may guide public objective or preprocessing selection. The same119 development subjects are repeatedly used; this is not the planned independent confirmation study.

## Experiment 5 — verified limitations and interpretation

Public V2 has251 unique images, not251 verified patients; conservative246 groups screen curation/near duplicates but cannot establish unseen-patient validation. V1 conflicting repeated-image/Hb linkage makes it unsuitable for this comparison; its exclusion does not identify which historical clinical measurement is correct. Public pediatric Hb<11 targets and local sex-specific adult/adolescent targets differ. Missing public individual demographics, unknown color calibration, source algorithmic alpha versus local manual anatomy, lower public resolution, clipping and geometry remain measured/known domain differences. Shared mean-fill/crop policy does not prove domain invariance.

Both public objectives selected epoch1 under their fixed bounded early-stopping design. This constrains conclusions to the implemented partial-backbone objectives; it does not prove that every CP pretraining architecture would fail. Public validation was used for checkpoint selection. The same119 AIMA subjects informed all earlier experiments; neither nested CV nor saved-model reproduction makes this an independent validation cohort. Only19 positive labels give wide intervals; many correlated candidate rows, overlapping folds and weighted uncalibrated scores limit performance interpretation. Diagnostic clipping/brightness proxies do not establish glare or causality.

CP-AnemiC pretraining did not establish an improvement that justifies its added complexity on AIMA. The predefined joint policy detected **11/19 anemia cases (57.9% sensitivity) at 78% specificity**, compared with **10/19 (52.6%) at 84% specificity** for the predefined Experiment 4 robust color comparison. One additional detection came with six additional false positives. Balanced accuracy was slightly lower (67.95% versus 68.32%), ROC-AUC was 0.734211 versus 0.709474, and AP was 0.372546 versus 0.385148. The paired sensitivity difference had a conditional 95% interval of −21.1 to +31.6 percentage points; the specificity difference was −6.0 points [−11.0, −2.0]. This does not establish independent or clinically acceptable improvement.

The predefined CP-classification frozen-feature plus color candidate C detected **12/19 (63.2%) at 78% specificity**, with ROC-AUC 0.734211 and AP 0.372166. Its sensitivity 95% Wilson interval was 41.0–80.9%; specificity was 68.9–85.0%. Against the robust color comparator, its +10.5-point sensitivity difference had an interval of [−10.5, +36.8], while specificity fell by 6.0 points [−11.0, −2.0]. This is a predefined exploratory candidate comparison, not a replacement primary policy chosen from outer results. CP classification's actual inner-selected A/B/C policy detected 10/19 at 77% specificity; it selected C in four folds and B in one.

CP Hb-regression C and its inner-selected policy detected **11/19 at 79% specificity** (ROC-AUC 0.730000; AP 0.370429), with exactly the same binary OOF decisions as matched ImageNet C (11/19 at 79%; ROC-AUC 0.730526; AP 0.368331). CP classification C added one detection and one false positive versus matched ImageNet C; its paired ROC-AUC and AP intervals included zero. The frozen-feature performance is therefore compatible with ImageNet features plus robust color statistics without useful added CP pretraining. Head-only and partial CP adaptations with training-selected thresholds detected only 2/19 and 0/19 for classification, or 4/19 and 0/19 for regression. These outer results did not guide any redesign.

The public-only pediatric classifier detected **0/19 at 96% specificity** (ROC-AUC 0.459474; AP 0.150111). Public classification validation was weak (ROC-AUC 0.478125), and both bounded public objectives selected epoch 1. Public-only Hb prediction on AIMA had MAE 2.225671 g/dL, bias −1.905735 g/dL and R² −1.069180. Although its MAE was below the public-training-mean baseline of 2.801160, substantial adult underprediction and negative R² provide no evidence of a usable Hb estimator. Weak public fitting and observed acquisition/label differences are plausible explanations, not proven causes.

Retain robust color-only classification as the practical candidate for independent confirmation, with a prespecified acquisition, model and operating-rule policy and genuinely new positive and borderline cases. Frozen embeddings can remain a predefined comparator in that study. These repeated-cohort results do not justify additional CP fine-tuning complexity or choosing a new winner by outer sensitivity. The inner specificity constraint of 0.80 was a research threshold policy and did not guarantee 0.80 outer specificity. No clinical acceptability threshold or independent validation has been established.


Recommendations require new subjects, fixed acquisition/color handling and a prespecified independently tested rule; no threshold or model is clinically validated. See [verified Experiment5 report](../experiments/eyelid-exp05-results-2026-10-13.md).

## Experiment 6 — complementarity did not establish useful ensemble improvement

Historical errors are partially complementary, but Pearson correlation0.802 and shared robust18 features limit mechanistic independence. All16 LAB false positives are shared with CP; CP contributes six more. The gate and strongest candidate choice use prior development outcomes, and the same 119 subjects have supported repeated experiments. The new nested weight/threshold calculations exclude outer labels, yet they cannot make the overall research trajectory independent. Descriptive union counts are not an ensemble result. Optional stacking was not implemented; no learned method was trained/evaluated on full-cohort OOF scores.

Equal averaging adds only one net detection at the cost of five false positives; its paired sensitivity interval includes zero, and ranking/AP difference intervals span zero. Weighted averaging adds no net detection and five false positives. Selected weights vary by fold. Only19 positive cases, small per-fold support, specificity as low as50% in one fold and conditional intervals prevent reliable superiority claims. More attractive secondary75%-LAB/0.85 numbers must not be selected as a new primary result after inspecting outer performance.

Unequal class weighting and score distributions make raw averages hard to calibrate. The CP representation requires CNN inference and an additional fitted projection/pipeline; averaging arithmetic is cheap but model storage, input processing and interpretation are more complex than LAB. No measured deployment efficiency or clinical specificity standard was established.

Thumbnail inspection cannot verify every anatomical boundary clinically. Visible tissue extent, camera orientation, margin handling and hair/eyelash occlusion vary; structural binary/single-component checks do not establish tissue equivalence. Existing owner approvals are preserved. There is one verified original eyelid image per subject;154 unassigned files cannot support repeated-image analysis without new identity/role/annotation evidence. Shared misses include moderate as well as mild anemia, so a near-cutoff explanation is incomplete. Hb subgroup results are descriptive with small support and unknown measurement uncertainty; no relabeling is justified.

Mixed rendering, unrecorded camera tone processing, exposure/clipping and absent color-reference capture remain unresolved. Previous4B reviewed glare exclusions added no robust-model recall; no new evidence justified repeating glare experiments. Retain robust color-only as an exploratory candidate for genuinely new subjects and fixed acquisition/anatomical/operating policies. Source: [Experiment 6 report](../experiments/eyelid-exp06-results-2026-10-13.md) and its private source/verification bundle. No external or clinical validation follows.


## Experiment 7 additions — 2026-10-14

Normal-only fitting is distinct from anemia diagnosis. The tested robust18
novelty distributions can flag capture/anatomical variation and miss mild or
moderate anemia whose colors overlap normals. All local A85 models detect
only4/19; Mahalanobis misses10/11 mild and5/6 moderate cases. The two severe
cases cannot support a severity-specific claim. Training specificity targets
are research rules and do not guarantee outer specificity or clinical
acceptability. Label-assisted B must be identified separately from normal-only A.

Public normal transfer has severe calibration shift: Eyes direct A85 flags
all119 subjects, CP produces77 FP and combined produces75 FP. Local normal
margin adaptation lowers FP but does not establish useful ranking/detection
benefit. Public/manual ROI extent, resolution, intensity-selected strips,
color capture, pediatric/adult distributions and source label policies differ.
Shared18-feature functions and negative duplicate screens do not establish
anatomical equivalence, true CP patient identity or domain independence.
Original Eyes redistribution terms remain unverified; no dataset files were
published. Missing individual public metadata prevents reliable subgroup claims.

Score normalization and cutoffs use inner normal cross-fits; outer refits can
shift score scale. Pooled raw rankings are descriptive. Candidate comparison,
secondary nested selection and repeated119-subject research remain dependent
exploration. Paired intervals condition on fixed OOF models/prevalence and
exclude refitting/overlapping training uncertainty and multiple-comparison
adjustments. Nineteen positives mean5.26pp per detection.

Exposure/pose/hair-occlusion grades and standardized skin tone are unavailable.
Brightness/clipping/area/aspect/glare proxies and rendering groups cannot
prove causal failure modes or fairness. No ITA or skin-tone groups were added.
A future study needs measured standardized acquisition/subgroups, frozen
methods and independent positive/borderline subjects.

A concurrent nail reset changed three workflow-table hashes; Experiment 7
stopped before fitting, sealed v1, verified exact archived context and complete
unchanged140-eyelid rows, and ran identical-feature v2 with strict guards.
Future source changes still require provenance verification. Private artifacts
are owner-local; no independent encrypted backup is verified. See
[full limitations and results](../experiments/eyelid-exp07-results.md).
