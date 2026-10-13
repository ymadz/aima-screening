# Experiment 7: normal-only color anomaly detection

Protocol frozen before new model fitting, 2026-10-13. Owner requested this
exploratory study and its bounded models, threshold policies and comparisons.
Branch `experiment/eyelid-exp07-one-class`; clean baseline `3b74755` (Exp6).
The preservation gate independently passed Exp1–6 and 4B. See
[the checkpoint](../audits/eyelid-exp07-preservation-2026-10-13.md).

## Question and scope

Does fitting normal conjunctival color distributions improve detection of the
19 anemia subjects among the existing 119 local development subjects? No
benefit is assumed. High anomaly scores describe color novelty, potentially
including lighting or acquisition effects; they are not anemia diagnoses.
All results are internal development CV on the repeatedly reused cohort.

## Fixed inputs

Retain exactly the original five subject folds (seed 20261010), labels,
approved masks, glare retention and Exp4 original-rendering A robust18 vectors.
Reuse `eyelid_color_data.color_features`, including feature order, quantized
OpenCV LAB units, linear percentiles and natural-log red/green statistics.
No imputation, ITA, additional feature, image enhancement or mask editing.
Freshly verify canonical metadata/provenance, schema, labels, all image links,
reviewed masks, duplicate adjudication, and exact feature-cache equality.
Pregnancy status remains the owner attestation outside the Sheet.

## Candidate models and score direction

Six fixed configurations, in this tie-breaking order:

1. Mahalanobis squared distance with normal-only standardization and empirical
   covariance shrunk 20% toward its spherical trace target.
2. RBF One-Class SVM: nu 0.05, gamma scale.
3. RBF One-Class SVM: nu 0.05, gamma 0.1.
4. RBF One-Class SVM: nu 0.10, gamma scale.
5. RBF One-Class SVM: nu 0.10, gamma 0.1.
6. Isolation Forest: 256 trees, max_samples 64 (bounded by available normals),
   max_features 1, contamination auto, one CPU worker, seed 20261013.

Fit scalers and detectors exclusively to normal training subjects. Negate SVM
decision_function and Forest score_samples so higher always means more novel.
PCA and neural representation transfer are omitted from this initial study;
additional searches would require a separately recorded rationale/protocol.

## Inner procedure and threshold policies

For each outer fold, shuffle the normal outer-training subjects into three
KFold partitions using seed 20261013. Cross-fit normal scores: every normal's
calibration score must exclude that subject. For label-assisted analysis only,
assign the anemic outer-training subjects across the same three inner models
by a separate seeded shuffle; they are scored and never fitted. This normal
partitioning does not depend on anemia values or anemic feature vectors.

Policy A uses only cross-fitted normal scores. For specificity targets 0.80,
0.85 (primary) and 0.90, choose the lowest observed cutoff achieving the target
empirical normal specificity under score >= cutoff, including a cutoff above
the maximum if needed. These are training targets, not guaranteed outer rates.
Each candidate is predefined; Policy A does not select a candidate with labels.

Policy B separately maximizes cross-fitted training sensitivity at each target
specificity; ties prefer specificity, then higher cutoff. A secondary nested
policy selects the configuration by training inner AP (ties protocol order).
No outer label is accepted by fitting/selection APIs or used to optimize any
candidate, cutoff, score direction, source weight or reported operating point.

After the inner procedure, refit the detector on all eligible outer-training
normals and score held-out subjects once. Record raw scores and fold-specific
calibration (normal inner mean/std). Pooled ranking metrics use the unbounded
training-normal z score to reduce scale differences between outer models;
within-fold raw ranking and pooled raw ranking are also descriptive outputs.
Calibration uncertainty and inner-versus-refit score shifts remain limitations.

## Public tracks and compatibility gate

Public tracks use only the predefined Mahalanobis model to bound comparisons.
Retain historical public training groups; public validation groups are not
fitted or used for adaptation. Average same-group normal image vectors once;
exclude any group with conflicting anemia labels. Namespaces and source/pixel
duplicate screens must separate public datasets and local subjects.

Eyes Defy retains the verified Exp3 adult eligibility policy (15–65, M<13,
F<12; pregnancy-dependent female records excluded) as an explicitly derived
policy, separate from the source Hb and the originating paper's <=10.5 rule.
CP preserves its original Hb<11 labels and pediatric population. Public RGB
is untagged and assumed sRGB for calculation; neither source is calibrated.
Use only palpebral masks for Eyes Defy; CP uses opaque alpha==255 tissue core.
The same 18 functions operate at the registered tissue raster resolution.
Public mask extraction and resolution differ from local manual anatomy.

Before pooling, confirm anatomical target from primary source descriptions,
verify source hashes, valid masks, exact feature schema, non-conflicting groups
and no screened cross-source duplication. Failed gates block the affected
track; a shared schema alone cannot justify pooling. Passing allows a bounded
exploratory mixture, not anatomical/pixel equivalence or domain invariance.

| Track | Fitting / adaptation |
|---|---|
| B1 / C1 | Eyes / CP training-normal distribution, applied directly to local subjects |
| B2 / C2 | Public normal correlation retained; mean and marginal scales replaced by local outer-training normal estimates |
| B3 / C3 | Source-balanced public + local normals; each source gets half the weight, each conservative group equal within source |
| D1 | Both compatible public normal sources, half weight per source |
| D2 | D1 public correlation with local outer-training normal mean/scales |

For public-only Policy A, threshold/calibration use three group-disjoint public
normal cross-fits; no local labels determine that rule. For adapted/pooled
Policy A, use cross-fitted local training normals with unchanged public training
groups. Policy B may calibrate thresholds on local outer-training labels;
public detector fitting remains normal-only. Public-only local threshold
selection is explicitly a label-assisted decision policy, not direct transfer.

## Evaluation and preservation

Report every predefined candidate/track and all three targets under both
policies, plus the secondary local nested Policy B. Report sensitivity,
specificity, ROC-AUC, AP, TP/FP/TN/FN, confusion matrices, score distributions,
fold cutoffs and fold stability. No candidate is declared a newly validated
winner from outer results. Compare matched subjects/labels/folds against Exp4
robust, Exp5 CP-C, Exp5 joint primary and Exp6 equal-average at its original
training-selected 0.80 rule; disclose their different threshold policies.

Use 2,000 paired stratified subject bootstrap draws (seed 20261013) for ranking
and paired deltas; Wilson proportion intervals. Report overlap of detections,
additional TP/FP and conditional interval limitations (no refitting, overlapping
training dependence, fixed prevalence). With 19 positives, one detection is
5.26 percentage points. Descriptive Hb-severity and measured brightness,
clipping, glare, tissue area and rendering associations are exploratory.
Unknown lighting, orientation, occlusion or retake metadata remain unknown;
no patient-specific tuning or causal claims. Skin-tone fairness is unassessed.

Versioned owner-private output:
`data/processed/local_eyelid_classification_exp07_one_class_v1/` in this worktree.
Save source/plan commits and hashes, audit, exact schema/features/folds, configs,
seeds, all inner/outer models and memberships, calibration/cutoffs, scores,
metrics, comparisons, error analysis, runtime/environment, code snapshot,
checksum manifest and checksum-first saved-model reproduction instructions.
Run deterministic synthetic smoke and leakage/integrity tests before fitting.
Never overwrite completed or failed attempts. Publish only reviewed source,
tests and aggregate documentation; retain the worktree and all older branches.
