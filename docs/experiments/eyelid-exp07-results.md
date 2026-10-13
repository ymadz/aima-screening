# Experiment 7: normal-only anemia novelty detection

Completed and independently reproduced, **2026-10-14**. This is exploratory
internal development cross-validation on the existing 119 subjects, including
19 anemia cases. It does not establish clinical screening performance.

## Objective and hypothesis

The owner requested a bounded test of whether learning non-anemic conjunctival
color distributions could improve detection with few anemia examples. The
hypothesis was that anemia-related pallor would depart from the normal feature
distribution. Acquisition, anatomy and unrelated appearance can also produce
novelty; a high score is an anomaly flag, not a clinical anemia diagnosis.

**The tested approach did not improve detection.** At the primary 0.85 training
specificity target, all six predefined local configurations detected 4/19
anemia cases, with 12–15 false positives among 100 non-anemic subjects. The
preserved Experiment 4 robust CIELAB comparator detected 10/19 with 16 false
positives. Eyes normal adaptation detected 5/19 with 12 false positives. Direct
public normal thresholds flagged all 119 subjects for Eyes, 89 for CP and 91
for the combined source. The secondary nested label-assisted policy detected
3/19 with eight false positives. There is no newly validated winning model.

## Preservation and isolated workspace

The mandatory [preservation checkpoint](../audits/eyelid-exp07-preservation-2026-10-13.md)
passed before Experiment 7 preparation or fitting. All **2,717 files across
14 registered historical artifact/readiness/verification/archive directories**
matched their registered hashes. Original validators reproduced Exp1's ten
models, Exp2's 35 pipelines, Exp3's 41 neural/20 LAB models, Exp4's 720
classifiers/ten regressors, both Exp4B bundles (80/160 models), Exp5's 122
neural/60 classical models and Exp6's 40 relevant base pipelines and decisions.
The phase-0 Exp5 check included fresh 370 inputs; the earlier reports retain
the scope of each historical fresh-pixel check. No historical model was refit.

| Preserved experiment | Verified branch | Handoff commit |
|---|---|---|
| 1 | `archive/eyelid-exp01-cielab-baseline` | `4bf6e25` |
| 2 | `experiment/eyelid-exp02-hybrid` | `f4b2cb5` |
| 3 | `experiment/eyelid-exp03-public-transfer` | `66afa8e` |
| 4 | `experiment/eyelid-exp04-color-optimization` | `0b50504` |
| 4B | `experiment/eyelid-exp04b-glare-regional` | `fc64eed` |
| 5 | `experiment/eyelid-exp05-cp-transfer` | `0943363` |
| 6 | `experiment/eyelid-exp06-ensemble` | `3b74755` |

Previously unsynchronized verified branches were safely pushed to the existing
`ymadz/aima-screening` remote after outgoing-history review; 4B was already
synchronized. Main was not pushed or merged. Exp1's annotated tag and separate
owner-local archive remain intact. Private bundles and that archive are on the
same local volume; no independent external encrypted backup was verified.

Experiment 7 uses branch `experiment/eyelid-exp07-one-class`, based on clean
Exp6 `3b74755`, at
`/Users/madz/.codex/worktrees/eyelid-exp07-one-class/aima-screening`.
Unrelated main-worktree edits and detached draft files were left untouched.
The [protocol](eyelid-exp07-plan.md) was committed as `340d3be` before fitting.

## Dataset audit and compatibility

The [complete compatibility audit](../audits/eyelid-exp07-dataset-compatibility.md)
records schema, labels, raw links, duplicates, masks, rendering and licenses.
The fixed local feature vectors and five subject folds exactly match the
historical robust18 inputs. Missing nails do not remove eligible eyelids.

| Source | Verified cohort / grouping | Non-anemic / anemic | Normal fitting support |
|---|---|---|---|
| Local AIMA | 140 metadata subjects; 119 eligible approved eyelids, ages 16–61 | 100 / 19 | 80 normals in each outer-training fold |
| Eyes Defy | 218 source images; 183 eligible records in 180 conservative groups; eligible ages 19–65 | 119 / 64; public train 92 / 54, validation 27 / 10 | 91 normal training groups; two conflicting-label groups excluded |
| CP-AnemiC V2 | 251 unique canonical images, 246 conservative groups; true patient count unavailable | 141 / 110; public train 109 / 90, validation 32 / 20 | 108 normal training groups |

Local metadata SHA-256 is
`1e704fb8d575757f16fc4f690cdc3315bc4063a5cf258d6c2e1b5f784d19596d`,
mtime `2026-10-07T14:55:39.254357+08:00`, recorded before reading. The frozen
CSV/provenance, header/trailing columns, uniqueness, Hb consistency, age/sex,
all 275 references, all 429 original inventory hashes and all 140 approved
eyelid source/TIFF/ROI/mask/review records passed. Twenty missing-Hb exclusions
and one age exclusion remain unchanged. The adult rule is Hb <13 g/dL for
males or <12 for non-pregnant females, ages 15–65. Non-pregnancy is an owner
attestation outside the Sheet.

Eyes retains the existing Exp3 derived adult policy, which differs from the
originating paper's Hb ≤10.5 target. Source Hb was preserved. Only provided
palpebral masks were used. Its untagged RGB JPEGs and registered mask rasters
passed existing alignment checks; no forniceal or combined region was added.
The [original paper](https://doi.org/10.1016/j.artmed.2022.102477) describes
controlled magnification and LED capture. Original IEEE DataPort license
terms could not be independently retrieved; no permissive redistribution
right is asserted and no public dataset bytes are published.

CP's official V2 archive SHA-256 is
`a976b9c2365fe9b0d07c61dc1a94ffcc08a1918e617e510529cbbf5ecbb0a3b0`;
all 503 local files matched it. The 251 duplicate All Categories copies were
excluded. Conservative curation/near-image groups reduce repeated-image risk
without establishing true patient independence. Original pediatric Hb <11
labels were retained, including equality 11 as non-anemia. Individual age/sex
and patient identity are unavailable. The
[official V2 description](https://data.mendeley.com/datasets/m53vz6b7fx/2)
describes ages 6–59 months, tablet ambient-light capture and triangle/entropy
tissue extraction. Untagged RGBA strips use the opaque alpha==255 core.
CC BY4 is listed alongside an academic/research-only author disclaimer; this
academic study redistributes no dataset files.

The gross palpebral target and identical feature functions support a bounded
exploratory B/C/D comparison. Public extracted strips, local manual masks,
resolution, intensity selection and acquisition are materially different.
Untagged public pixels and unchanged local stored RGB are assumed sRGB for
calculation; no retrospective color calibration or rendering equivalence is
claimed. Source-byte and cross-source perceptual screens found no candidates,
which does not prove true patient independence. All public validation groups
were excluded from fitting, adaptation and public normal calibration.

### Stopped v1 and accepted v2

The first preparation passed, but the final source guard stopped v1 **before
any model fitting** when a separate canonical nail reset changed three active
workflow tables. Exact archived pre-reset copies match their historical hashes;
all columns of all 140 eyelid rows match current tables. Every other historical
source/code hash, approved eyelid input and all 2,717 old artifact files remained
unchanged. Current nail status is 135 pending; this was another task's action.

The narrow version bridge requires those exact archived bytes plus complete
eyelid-row equality, followed by the fresh current source/mask/feature audit.
It does not ignore arbitrary source changes. v1 is sealed with its stop record;
new v2 pins both contexts and has exactly the same feature arrays. No clinical
label, mask, fold, candidate, threshold rule or comparison changed, and no outer
outcome existed before this provenance repair. Current workflow snapshots are
retained privately alongside the independent verification record.

## Implemented methodology

`eyelid_one_class.py`, `eyelid_one_class_data.py`,
`eyelid_one_class_evaluation.py` and `eyelid_one_class_run.py` implement the
predefined study. The exact ordered 18-feature schema is:

```text
mean_L, std_L, mean_a, std_a, mean_b, std_b,
median_L, median_a, median_b, iqr_L, iqr_a, iqr_b,
p10_L, p10_a, p10_b, p90_a,
median_log_red_green, iqr_log_red_green
```

Feature extraction reuses `eyelid_color_data.color_features`: quantized OpenCV
LAB units, linear percentiles and natural-log red/green summaries. Glare is
retained as in Exp4 original rendering. There is no imputation, new feature,
ITA, image enhancement, mask edit or neural training.

Six fixed local configurations were evaluated: squared Mahalanobis distance
with 20% spherical covariance shrinkage and a 1e-10 ridge; four RBF One-Class
SVM combinations (nu 0.05/0.10, gamma scale/0.1); and Isolation Forest with
256 trees, max_samples=min(64,n), all features, contamination auto and one
CPU worker. Scalers and detectors fit only verified normal training groups.
SVM decision_function and Forest score_samples are negated so higher means
more anomalous. Seed is 20261013; original five outer folds use seed 20261010.

Each outer-training set uses three shuffled normal inner folds. Every normal
calibration score excludes that subject from detector fitting. For Policy B,
anemic training subjects are separately assigned to inner models and scored;
they never enter fitting or preprocessing. No held-out label is accepted by
fitting or selection APIs. Every outer detector is then refit on its 80 local
training normals where the track uses local fitting/adaptation.

**Policy A:** choose the lowest cross-fitted normal cutoff attaining training
specificity 0.80, **0.85 primary**, or 0.90 under score ≥ cutoff. No anemic
label selects its threshold or candidate. **Policy B:** fit the same normal
detector, then maximize inner training sensitivity subject to specificity;
ties prefer specificity and higher cutoff. This is a label-assisted decision
policy. A secondary nested B policy chooses among six local configurations by
inner AP, with ties in frozen protocol order. Outer results choose no winner.
An inner target does not guarantee the same outer specificity or clinical
acceptability. Inner cross-fit versus outer-refit score shifts remain possible.

All public tracks use predefined Mahalanobis only. Group-mean vectors prevent
multiple images from overweighting a verified group. B1/C1 fit Eyes/CP normals
directly; B2/C2 retain public precision/correlation and replace marginal means
and scales using local training normals. B3/C3 pool public/local normals with
equal total source weight and equal group weight within source. D1 balances
both public normal sources; D2 adapts D1 margins with local normals. Direct
public Policy A uses group-disjoint public normal cross-fits; adapted/pooled A
uses local normal cross-fits. Public Policy B uses local training labels only
for cutoff selection and is not an entirely normal-only transfer evaluation.

Pooled ranking uses fold-specific normal cross-fit mean/std standardized scores,
as frozen in the protocol. Raw scores, calibrations and raw pooled rankings
are also saved; normalization is not fitted on held-out subjects. No score
direction was flipped after seeing results. PCA and neural representation
transfer were omitted in the initial scope. These negative findings provide
no reason for further complexity or searching on the same evaluation cohort.

## Local-only results: Experiment 7A

| Candidate / policy | TP / FP / TN / FN | Sensitivity | Specificity | ROC-AUC | AP |
| --- | --- | --- | --- | --- | --- |
| mahalanobis A | 4 / 13 / 87 / 15 | 21.1% | 87.0% | 0.537368 | 0.258772 |
| mahalanobis B | 3 / 8 / 92 / 16 | 15.8% | 92.0% | 0.537368 | 0.258772 |
| svm_005_scale A | 4 / 12 / 88 / 15 | 21.1% | 88.0% | 0.531053 | 0.343953 |
| svm_005_scale B | 4 / 7 / 93 / 15 | 21.1% | 93.0% | 0.531053 | 0.343953 |
| svm_005_01 A | 4 / 13 / 87 / 15 | 21.1% | 87.0% | 0.535263 | 0.327175 |
| svm_005_01 B | 4 / 7 / 93 / 15 | 21.1% | 93.0% | 0.535263 | 0.327175 |
| svm_010_scale A | 4 / 12 / 88 / 15 | 21.1% | 88.0% | 0.531579 | 0.344039 |
| svm_010_scale B | 4 / 7 / 93 / 15 | 21.1% | 93.0% | 0.531579 | 0.344039 |
| svm_010_01 A | 4 / 13 / 87 / 15 | 21.1% | 87.0% | 0.535263 | 0.327116 |
| svm_010_01 B | 4 / 7 / 93 / 15 | 21.1% | 93.0% | 0.535263 | 0.327116 |
| isolation_forest A | 4 / 15 / 85 / 15 | 21.1% | 85.0% | 0.498421 | 0.246435 |
| isolation_forest B | 3 / 12 / 88 / 16 | 15.8% | 88.0% | 0.498421 | 0.246435 |

Mahalanobis had the highest observed local standardized ROC-AUC (0.537368);
SVM nu=0.10/gamma=scale had the highest observed local AP (0.344039). Both
detected 4/19 at the primary normal-only operating target. Those descriptions
identify predefined candidates, not an unbiased estimate for a selected winner.
The SVM's high-scoring tail does not yield better primary sensitivity. Policy B
usually reduced false positives while retaining or losing detections.

The nested B policy selected, by fold: SVM 0.10/scale, Mahalanobis, SVM 0.05/0.1,
Mahalanobis, SVM 0.05/0.1. At 0.85 it detected 3/19 at 92% specificity,
ROC-AUC 0.514737 and AP 0.252140. Its higher specificity is accompanied by
16 missed anemia cases. This remains the separate predefined nested policy.

## Eyes Defy transfer results: Experiment 7B

| Candidate / policy | TP / FP / TN / FN | Sensitivity | Specificity | ROC-AUC | AP |
| --- | --- | --- | --- | --- | --- |
| B1 A | 19 / 100 / 0 / 0 | 100.0% | 0.0% | 0.550526 | 0.184737 |
| B1 B | 2 / 8 / 92 / 17 | 10.5% | 92.0% | 0.550526 | 0.184737 |
| B2 A | 5 / 12 / 88 / 14 | 26.3% | 88.0% | 0.541053 | 0.295872 |
| B2 B | 5 / 12 / 88 / 14 | 26.3% | 88.0% | 0.541053 | 0.295872 |
| B3 A | 4 / 17 / 83 / 15 | 21.1% | 83.0% | 0.519474 | 0.302284 |
| B3 B | 3 / 2 / 98 / 16 | 15.8% | 98.0% | 0.519474 | 0.302284 |

B1's direct public normal threshold flagged all 119 local subjects at every
target: 19 TP and 100 FP. It cannot be interpreted as improved detection.
Label-assisted local cutoff calibration reduced B1 to 2 TP/eight FP at 0.85.
B2 local normal adaptation produced 5 TP/12 FP with either primary policy.
B3 source pooling produced 4 TP/17 FP under A or 3 TP/two FP under B. Public
normal support did not restore useful separation; adaptation principally
changed calibration and the false-positive burden.

## CP-AnemiC transfer results: Experiment 7C

| Candidate / policy | TP / FP / TN / FN | Sensitivity | Specificity | ROC-AUC | AP |
| --- | --- | --- | --- | --- | --- |
| C1 A | 12 / 77 / 23 / 7 | 63.2% | 23.0% | 0.378421 | 0.129228 |
| C1 B | 0 / 7 / 93 / 19 | 0.0% | 93.0% | 0.378421 | 0.129228 |
| C2 A | 4 / 12 / 88 / 15 | 21.1% | 88.0% | 0.500526 | 0.261827 |
| C2 B | 2 / 5 / 95 / 17 | 10.5% | 95.0% | 0.500526 | 0.261827 |
| C3 A | 3 / 15 / 85 / 16 | 15.8% | 85.0% | 0.496842 | 0.202217 |
| C3 B | 1 / 11 / 89 / 18 | 5.3% | 89.0% | 0.496842 | 0.202217 |

C1's direct pediatric normal cutoff produced 12 TP/77 FP. Its local ROC-AUC
0.378421 is below chance in the predefined anomaly direction; the direction
was preserved. Local label-assisted cutoff selection produced zero TP/seven
FP at 0.85. C2 normal adaptation produced 4 TP/12 FP under A, or 2 TP/five
FP under B. C3 pooling produced 3 TP/15 FP or 1 TP/11 FP. These findings are
consistent with strong domain mismatch but do not isolate age, labels, anatomy
or acquisition as its cause.

## Combined-public results: Experiment 7D

| Candidate / policy | TP / FP / TN / FN | Sensitivity | Specificity | ROC-AUC | AP |
| --- | --- | --- | --- | --- | --- |
| D1 A | 16 / 75 / 25 / 3 | 84.2% | 25.0% | 0.468947 | 0.150368 |
| D1 B | 1 / 12 / 88 / 18 | 5.3% | 88.0% | 0.468947 | 0.150368 |
| D2 A | 4 / 13 / 87 / 15 | 21.1% | 87.0% | 0.482632 | 0.216244 |
| D2 B | 5 / 12 / 88 / 14 | 26.3% | 88.0% | 0.482632 | 0.216244 |

D1 direct transfer produced 16 TP/75 FP, with ROC-AUC 0.468947/AP 0.150368.
D2 adaptation produced 4 TP/13 FP under A or 5 TP/12 FP under B. The larger
normal mixture did not establish better local discrimination. B/C/D completed
their bounded exploratory gates; no track was skipped for a failed gate.
Optional PCA/neural representations were not run.

## Historical model comparison

All four matched reference predictions were verified against the same subjects,
labels and five outer folds. References and their original operating rules
remain unchanged. Normal-only thresholds, training-label-assisted thresholds,
fixed 0.5 and nested-selected policies are different research procedures.

| Reference / original policy | TP / FP / TN / FN | Sensitivity | Specificity | ROC-AUC | AP |
| --- | --- | --- | --- | --- | --- |
| Exp4 robust18 predefined | 10 / 16 / 84 / 9 | 52.6% | 84.0% | 0.709474 | 0.385148 |
| Exp5 CP classification C predefined | 12 / 22 / 78 / 7 | 63.2% | 78.0% | 0.734211 | 0.372166 |
| Exp5 joint nested primary | 11 / 22 / 78 / 8 | 57.9% | 78.0% | 0.734211 | 0.372546 |
| Exp6 equal-average, inner target 0.80 | 11 / 21 / 79 / 8 | 57.9% | 79.0% | 0.737895 | 0.379070 |

Exp1 fixed 0.5 detected 2/19 at 99% specificity (AUC 0.719474/AP 0.387866);
Exp2's selected hybrid detected 7/19 at 81% (0.575789/0.247161); Exp3's selected
transfer detected 1/19 at 84% (0.463684/0.143881). Exp4's joint primary policy
detected 7/19 at 83% (0.656842/0.352228), distinct from robust18's predefined
10/19 comparison. Exp4B's reviewed glare exclusion and regional policies did
not improve sensitivity over robust18. This historical context is not newly
paired evidence for all of those earlier rows; detailed reports remain intact.

| Matched comparison | Net TP | Net FP | Δ sensitivity pp [95%] | Δ specificity pp [95%] | Δ ROC-AUC [95%] | Δ AP [95%] |
| --- | --- | --- | --- | --- | --- | --- |
| mahalanobis_A_085 vs exp04_robust | -6 | -3 | -31.6 [-57.9, -5.3] | +3.0 [-7.0, +13.0] | -0.172105 [-0.361053, +0.021066] | -0.126376 [-0.271858, +0.030073] |
| mahalanobis_A_085 vs exp05_cp_C | -8 | -9 | -42.1 [-73.7, -10.5] | +9.0 [-2.0, +19.0] | -0.196842 [-0.374342, -0.009461] | -0.113395 [-0.249467, +0.036877] |
| mahalanobis_A_085 vs exp05_joint_primary | -7 | -9 | -36.8 [-63.2, -5.3] | +9.0 [-2.0, +19.0] | -0.196842 [-0.376355, -0.008408] | -0.113775 [-0.250336, +0.036142] |
| mahalanobis_A_085 vs exp06_equal_spec080 | -7 | -8 | -36.8 [-63.2, -10.5] | +8.0 [-3.0, +18.0] | -0.200526 [-0.378434, -0.012092] | -0.120298 [-0.260936, +0.029040] |
| svm_010_scale_A_085 vs exp04_robust | -6 | -4 | -31.6 [-57.9, -5.3] | +4.0 [-5.0, +13.0] | -0.177895 [-0.366868, +0.015789] | -0.041108 [-0.231542, +0.130000] |
| B2_A_085 vs exp04_robust | -5 | -4 | -26.3 [-52.6, +5.3] | +4.0 [-6.0, +14.0] | -0.168421 [-0.367921, +0.038987] | -0.089276 [-0.248193, +0.071164] |
| nested_B_085 vs exp04_robust | -7 | -8 | -36.8 [-63.2, -10.5] | +8.0 [-1.0, +18.0] | -0.194737 [-0.378434, -0.001579] | -0.133007 [-0.283897, +0.028800] |

Mahalanobis A85 loses six net detections and saves three false positives against
robust18. Of 19 anemia cases, both detect three, Mahalanobis alone detects one,
robust18 alone detects seven and both miss eight. SVM 0.10/scale A85 has the
same detection overlap and saves four FP. B2 A85 shares three detections,
adds two and loses seven (net five fewer), with seven shared misses and four
fewer FP. Nested B85 shares two, adds one and loses eight, with eight shared
misses (net seven fewer detections/eight fewer FP). All 87 policies' matched
overlap and paired deltas against four references remain in private metrics.

Compared with local Mahalanobis under the same A85 rule, B2 adds one net
detection and removes one FP: sensitivity +5.3 percentage points [0.0,15.8],
specificity +1.0 [−4.0,6.0], AUC +0.003684 [−0.038947,0.046329], AP +0.037100
[−0.019739,0.088899]. The paired ranking intervals include zero and neither
approach approaches the robust benchmark's detection. All 48 public-versus-local
matched-policy comparisons are saved separately; they did not select a model.

## Fold stability, cutoffs and score distributions

Mahalanobis A85's five fold sensitivities were 0%, 50%, 0%, 25%, 33.3%;
specificities 90%, 70%, 95%, 85%, 95%; fold ROC-AUC 0.5375, 0.6625, 0.3875,
0.5000, 0.6667. Four folds contain four positives and one contains three.
The Appendix gives each candidate's primary fold ranges, all selected fold
cutoffs at all three targets, and standardized normal/anemia score summaries.
Exact unrounded cutoffs, every fold metric and full score quantiles are sealed
privately. Cutoffs are in each model's raw score units and are not comparable
as probabilities. Primary Mahalanobis normal/anemia medians are nearly equal
(−0.3067/−0.3072), despite the anemia group's longer high-score tail.

The private `comparison.png` and `comparison.svg` show primary operating
points against robust18 and the substantial Mahalanobis distribution overlap.
The figure was visually inspected. No private image is committed.

## Error analysis and image-quality limits

Error analysis covers every candidate/policy, with private FN/FP identities,
severity tables, rendering-group summaries and Spearman associations with Hb,
mean RGB brightness, LAB L, clipping, glare proxy, tissue-pixel count and
bounding-box aspect. No known missed subject guided a change or refitting.

At A85, Mahalanobis and SVM 0.10/scale detect 1/11 mild, 1/6 moderate and 2/2
severe anemia cases; they miss ten mild and five moderate cases. B2 detects
2/11 mild, 1/6 moderate, 2/2 severe; nested B detects 0/11, 1/6, 2/2. The two
severe cases are insufficient for a severity-specific performance claim.

Local Mahalanobis associations are weak: Hb rho −0.042 (p=0.650), brightness
−0.109 (p=0.238), LAB L −0.122 (p=0.186), clipping −0.100 (p=0.279), glare
proxy +0.144 (p=0.118), tissue pixels −0.006 (p=0.949), aspect +0.020 (p=0.832).
SVM 0.10/scale's Hb association is −0.010 (p=0.912). These unadjusted
exploratory statistics do not show that scores reliably track lower Hb.

Direct Eyes transfer has much stronger brightness/clipping correlations
(rho +0.731/+0.732), tissue area +0.491 and aspect −0.421, while Hb is +0.083.
Direct combined public scores correlate with brightness +0.527 and clipping
+0.553; direct CP's Hb association is +0.257, opposite to an assumed lower-Hb
anomaly relationship. Capture and tissue differences are plausible confounders;
these correlations do not establish causation. Glare is a proxy measured on
unchanged approved anatomy, not new clinical image-quality ground truth.

Exposure settings, validated lighting grades, camera pose, hair/eyelash
occlusion grades and retake labels are unavailable. Tissue area/aspect and
rendering histories provide limited checks for unusual inputs; they cannot
quantify an orientation or occlusion failure mechanism. Existing Exp6 visual
review found variation in tissue extent/orientation/hair handling, without
changing masks. Authoritative subgroup claims for those categories cannot be
made from the available metadata. This remains a documented incomplete
measurement dimension, not an invented annotation. No ITA group or skin-tone
fairness result was computed. Future standardized measurements could support
a prespecified subgroup study on independent data.

## Statistical uncertainty

Wilson sensitivity/specificity intervals and 2,000 paired stratified subject
bootstrap draws (seed 20261013) are reported. Bootstraps hold class prevalence
at 19/119 and reuse fixed OOF predictions; they do not refit the overlapping
training models, include all model-selection uncertainty or correct the many
comparisons. They are conditional development intervals, not clinical bounds.
One positive subject changes sensitivity by **5.26 percentage points**.

Mahalanobis A85 sensitivity has a 95% Wilson interval of **8.5–43.3%**,
specificity **79.0–92.2%**, AUC bootstrap **0.391579–0.689487**, and AP
**0.148596–0.452798**. B2 A85 sensitivity is **11.8–48.8%**, specificity
**80.2–93.0%**. Nested B85 sensitivity is **5.5–37.6%**, specificity
**85.0–95.9%**. The full paired table shows which differences span zero.
Repeated study design on the same cohort can create optimism even with
leakage-safe inner selection. The observed local AUCs near 0.5 and poor
sensitivity are negative evidence for these tested normal distributions, not
proof that every future one-class method must fail.

## Complexity and reproducibility

Mahalanobis is a compact 18-variable model: covariance fitting approximately
O(n p²+p³), O(p²) scoring, plus training-only feature scaling. Kernel SVM
retains support vectors and has a potentially quadratic kernel-memory/cubic
optimization cost; per-example scoring depends on support-vector count.
Isolation Forest stores 256 bounded-sample trees; inference traverses those
trees. Public adaptation retains an 18×18 public precision estimate and
replaces local margins. No neural encoder is needed, but supervised robust
CIELAB logistic regression is already comparably compact. Less use of anemia
labels did not produce a useful accuracy/complexity trade-off.

The accepted run saved **289 models and 87 policy outputs**: 14 tracks × five
outer × (three inner + one outer) = 280, plus nine public calibration models.
The secondary nested policy selects existing local models rather than fitting
additional ones. Cached-feature fitting and evaluation took **19.369 seconds**;
this excludes source auditing/feature preparation and independent verification.
Peak process RSS was **430,637,056 bytes**; saved models total **26,152,701
bytes**. These are this machine's run observations, not deployment benchmarks.
No final full-cohort local or deployment model was fitted.

Execution commit: `6513c9c2aeadc57e436ec55b292f362ff1316fa8`.
v2 preparation commit: `cbf832f7daafe4ca46a5e4973bf86b7e774ed66d`.
The immutable v2 bundle contains copied implementation/tests/requirements and
protocol, source hashes, exact ordered features/labels/folds, all model scopes,
public calibration, inner scores, training decisions, outer anomaly scores,
OOF predictions, metrics/paired uncertainty, error analysis, plots, environment,
runtime, reproduction instructions and a complete checksum manifest.

Private accepted output (relative to this worktree):
`data/processed/local_eyelid_classification_exp07_one_class_v2/`.
Independent proof/logs/snapshots:
`data/processed/exp07_verification_2026-10-14_v1/`.
Stopped sealed attempt: `data/processed/local_eyelid_classification_exp07_one_class_v1/`.
Phase-0 audit/logs remain in canonical main's ignored
`data/processed/exp07_preservation_2026-10-13_v1/`.

From this worktree, using the canonical Python3.10 environment:

```bash
/Users/madz/Documents/aima-screening/aima_env/bin/python -m src.eyelid_one_class_run \
  --verify --fresh-features \
  --output data/processed/local_eyelid_classification_exp07_one_class_v2
/Users/madz/Documents/aima-screening/aima_env/bin/python -m unittest discover -s tests
/Users/madz/Documents/aima-screening/aima_env/bin/python -m compileall src
/Users/madz/Documents/aima-screening/aima_env/bin/python scripts/validate_research_docs.py \
  --source-root /Users/madz/Documents/aima-screening
git diff --check
```

Checksum-first saved-model reproduction passed without fitting: all 289 model
scopes, fitted scaler/covariance/source weights, inner/public calibration,
thresholds, nested choices, held-out scores, matched references, metrics,
intervals and error analysis reproduced. All 119 local feature vectors were
freshly re-extracted and matched exactly; historical artifacts remained intact.
The full suite completed **151 tests: 148 passed, three expected legacy-source
fixture skips**, with no failures. The 15 new tests include deterministic
all-track smoke, normal-only fitting, subject/source exclusion, scaling and
threshold leakage perturbations, schema/orientation, independent tied-metric
checks, saved-statistic reproduction, checksum-before-deserialization and the
narrow archived-workflow guard. Compilation and whitespace checks passed.

## Research conclusion and handoff

The tested normal-only robust18 detectors did not improve local sensitivity.
They often achieved higher observed specificity by missing more anemia cases;
there is no approved clinical acceptability criterion for these research
targets. Public normal examples did not establish improved discrimination.
Local normal adaptation mitigated direct-transfer false flags, with little
ranking benefit and inadequate detection. The direct-transfer failures and
capture-feature associations provide evidence of domain mismatch, with its
specific causes unresolved.

Retain the preserved robust CIELAB logistic candidate for **prespecified
independent confirmation**, including new positive and borderline subjects and
controlled acquisition/annotation measurements. It is a practical research
candidate, not a clinically validated model. Further searches on these same
119 subjects would add dependent exploratory evidence. A future one-class
study would need a new rationale, measurement improvements, frozen protocol
and independent cohort; these results do not justify neural complexity now.

All earlier branches/worktrees and artifacts are preserved. Experiment 7
source/tests and aggregate reports use focused commits; private files remain
ignored. The final Git publication receipt and exact handoff commit are kept
in the separate private verification record. There is no merge into main.
Current canonical nail-workflow changes and missing standardized fairness/QC
measurements remain separate operational/research limitations.

## Appendix: complete operating results and diagnostics

The tables below are transcribed programmatically from the independently
verified aggregate metrics and training records. All rows use the same
119-subject cohort. Counts are **TP / FP / TN / FN**. Confusion matrices are
`[[TN,FP],[FN,TP]]`. A/B denote the separate policies; 080/085/090 denote
training specificity targets, not measured outer specificity. Ranking values
use predefined training-normal standardization. Display rounding does not
change stored scores or decisions.

| Policy / target | TP / FP / TN / FN | Sensitivity | Specificity | ROC-AUC | AP |
| --- | --- | --- | --- | --- | --- |
| mahalanobis_A_080 | 5 / 17 / 83 / 14 | 26.3% | 83.0% | 0.537368 | 0.258772 |
| mahalanobis_A_085 | 4 / 13 / 87 / 15 | 21.1% | 87.0% | 0.537368 | 0.258772 |
| mahalanobis_A_090 | 4 / 7 / 93 / 15 | 21.1% | 93.0% | 0.537368 | 0.258772 |
| mahalanobis_B_080 | 3 / 8 / 92 / 16 | 15.8% | 92.0% | 0.537368 | 0.258772 |
| mahalanobis_B_085 | 3 / 8 / 92 / 16 | 15.8% | 92.0% | 0.537368 | 0.258772 |
| mahalanobis_B_090 | 3 / 6 / 94 / 16 | 15.8% | 94.0% | 0.537368 | 0.258772 |
| svm_005_scale_A_080 | 4 / 14 / 86 / 15 | 21.1% | 86.0% | 0.531053 | 0.343953 |
| svm_005_scale_A_085 | 4 / 12 / 88 / 15 | 21.1% | 88.0% | 0.531053 | 0.343953 |
| svm_005_scale_A_090 | 4 / 11 / 89 / 15 | 21.1% | 89.0% | 0.531053 | 0.343953 |
| svm_005_scale_B_080 | 4 / 10 / 90 / 15 | 21.1% | 90.0% | 0.531053 | 0.343953 |
| svm_005_scale_B_085 | 4 / 7 / 93 / 15 | 21.1% | 93.0% | 0.531053 | 0.343953 |
| svm_005_scale_B_090 | 4 / 6 / 94 / 15 | 21.1% | 94.0% | 0.531053 | 0.343953 |
| svm_005_01_A_080 | 5 / 16 / 84 / 14 | 26.3% | 84.0% | 0.535263 | 0.327175 |
| svm_005_01_A_085 | 4 / 13 / 87 / 15 | 21.1% | 87.0% | 0.535263 | 0.327175 |
| svm_005_01_A_090 | 4 / 11 / 89 / 15 | 21.1% | 89.0% | 0.535263 | 0.327175 |
| svm_005_01_B_080 | 4 / 13 / 87 / 15 | 21.1% | 87.0% | 0.535263 | 0.327175 |
| svm_005_01_B_085 | 4 / 7 / 93 / 15 | 21.1% | 93.0% | 0.535263 | 0.327175 |
| svm_005_01_B_090 | 4 / 6 / 94 / 15 | 21.1% | 94.0% | 0.535263 | 0.327175 |
| svm_010_scale_A_080 | 4 / 14 / 86 / 15 | 21.1% | 86.0% | 0.531579 | 0.344039 |
| svm_010_scale_A_085 | 4 / 12 / 88 / 15 | 21.1% | 88.0% | 0.531579 | 0.344039 |
| svm_010_scale_A_090 | 4 / 11 / 89 / 15 | 21.1% | 89.0% | 0.531579 | 0.344039 |
| svm_010_scale_B_080 | 4 / 10 / 90 / 15 | 21.1% | 90.0% | 0.531579 | 0.344039 |
| svm_010_scale_B_085 | 4 / 7 / 93 / 15 | 21.1% | 93.0% | 0.531579 | 0.344039 |
| svm_010_scale_B_090 | 4 / 6 / 94 / 15 | 21.1% | 94.0% | 0.531579 | 0.344039 |
| svm_010_01_A_080 | 5 / 16 / 84 / 14 | 26.3% | 84.0% | 0.535263 | 0.327116 |
| svm_010_01_A_085 | 4 / 13 / 87 / 15 | 21.1% | 87.0% | 0.535263 | 0.327116 |
| svm_010_01_A_090 | 4 / 10 / 90 / 15 | 21.1% | 90.0% | 0.535263 | 0.327116 |
| svm_010_01_B_080 | 4 / 13 / 87 / 15 | 21.1% | 87.0% | 0.535263 | 0.327116 |
| svm_010_01_B_085 | 4 / 7 / 93 / 15 | 21.1% | 93.0% | 0.535263 | 0.327116 |
| svm_010_01_B_090 | 4 / 6 / 94 / 15 | 21.1% | 94.0% | 0.535263 | 0.327116 |
| isolation_forest_A_080 | 5 / 16 / 84 / 14 | 26.3% | 84.0% | 0.498421 | 0.246435 |
| isolation_forest_A_085 | 4 / 15 / 85 / 15 | 21.1% | 85.0% | 0.498421 | 0.246435 |
| isolation_forest_A_090 | 4 / 10 / 90 / 15 | 21.1% | 90.0% | 0.498421 | 0.246435 |
| isolation_forest_B_080 | 4 / 13 / 87 / 15 | 21.1% | 87.0% | 0.498421 | 0.246435 |
| isolation_forest_B_085 | 3 / 12 / 88 / 16 | 15.8% | 88.0% | 0.498421 | 0.246435 |
| isolation_forest_B_090 | 3 / 9 / 91 / 16 | 15.8% | 91.0% | 0.498421 | 0.246435 |
| B1_A_080 | 19 / 100 / 0 / 0 | 100.0% | 0.0% | 0.550526 | 0.184737 |
| B1_A_085 | 19 / 100 / 0 / 0 | 100.0% | 0.0% | 0.550526 | 0.184737 |
| B1_A_090 | 19 / 99 / 1 / 0 | 100.0% | 1.0% | 0.550526 | 0.184737 |
| B1_B_080 | 3 / 19 / 81 / 16 | 15.8% | 81.0% | 0.550526 | 0.184737 |
| B1_B_085 | 2 / 8 / 92 / 17 | 10.5% | 92.0% | 0.550526 | 0.184737 |
| B1_B_090 | 2 / 8 / 92 / 17 | 10.5% | 92.0% | 0.550526 | 0.184737 |
| B2_A_080 | 6 / 17 / 83 / 13 | 31.6% | 83.0% | 0.541053 | 0.295872 |
| B2_A_085 | 5 / 12 / 88 / 14 | 26.3% | 88.0% | 0.541053 | 0.295872 |
| B2_A_090 | 4 / 8 / 92 / 15 | 21.1% | 92.0% | 0.541053 | 0.295872 |
| B2_B_080 | 5 / 14 / 86 / 14 | 26.3% | 86.0% | 0.541053 | 0.295872 |
| B2_B_085 | 5 / 12 / 88 / 14 | 26.3% | 88.0% | 0.541053 | 0.295872 |
| B2_B_090 | 2 / 3 / 97 / 17 | 10.5% | 97.0% | 0.541053 | 0.295872 |
| B3_A_080 | 4 / 17 / 83 / 15 | 21.1% | 83.0% | 0.519474 | 0.302284 |
| B3_A_085 | 4 / 17 / 83 / 15 | 21.1% | 83.0% | 0.519474 | 0.302284 |
| B3_A_090 | 4 / 8 / 92 / 15 | 21.1% | 92.0% | 0.519474 | 0.302284 |
| B3_B_080 | 3 / 5 / 95 / 16 | 15.8% | 95.0% | 0.519474 | 0.302284 |
| B3_B_085 | 3 / 2 / 98 / 16 | 15.8% | 98.0% | 0.519474 | 0.302284 |
| B3_B_090 | 3 / 2 / 98 / 16 | 15.8% | 98.0% | 0.519474 | 0.302284 |
| C1_A_080 | 13 / 79 / 21 / 6 | 68.4% | 21.0% | 0.378421 | 0.129228 |
| C1_A_085 | 12 / 77 / 23 / 7 | 63.2% | 23.0% | 0.378421 | 0.129228 |
| C1_A_090 | 9 / 60 / 40 / 10 | 47.4% | 40.0% | 0.378421 | 0.129228 |
| C1_B_080 | 0 / 7 / 93 / 19 | 0.0% | 93.0% | 0.378421 | 0.129228 |
| C1_B_085 | 0 / 7 / 93 / 19 | 0.0% | 93.0% | 0.378421 | 0.129228 |
| C1_B_090 | 0 / 7 / 93 / 19 | 0.0% | 93.0% | 0.378421 | 0.129228 |
| C2_A_080 | 5 / 17 / 83 / 14 | 26.3% | 83.0% | 0.500526 | 0.261827 |
| C2_A_085 | 4 / 12 / 88 / 15 | 21.1% | 88.0% | 0.500526 | 0.261827 |
| C2_A_090 | 3 / 8 / 92 / 16 | 15.8% | 92.0% | 0.500526 | 0.261827 |
| C2_B_080 | 4 / 14 / 86 / 15 | 21.1% | 86.0% | 0.500526 | 0.261827 |
| C2_B_085 | 2 / 5 / 95 / 17 | 10.5% | 95.0% | 0.500526 | 0.261827 |
| C2_B_090 | 2 / 5 / 95 / 17 | 10.5% | 95.0% | 0.500526 | 0.261827 |
| C3_A_080 | 4 / 21 / 79 / 15 | 21.1% | 79.0% | 0.496842 | 0.202217 |
| C3_A_085 | 3 / 15 / 85 / 16 | 15.8% | 85.0% | 0.496842 | 0.202217 |
| C3_A_090 | 2 / 10 / 90 / 17 | 10.5% | 90.0% | 0.496842 | 0.202217 |
| C3_B_080 | 3 / 17 / 83 / 16 | 15.8% | 83.0% | 0.496842 | 0.202217 |
| C3_B_085 | 1 / 11 / 89 / 18 | 5.3% | 89.0% | 0.496842 | 0.202217 |
| C3_B_090 | 1 / 4 / 96 / 18 | 5.3% | 96.0% | 0.496842 | 0.202217 |
| D1_A_080 | 17 / 81 / 19 / 2 | 89.5% | 19.0% | 0.468947 | 0.150368 |
| D1_A_085 | 16 / 75 / 25 / 3 | 84.2% | 25.0% | 0.468947 | 0.150368 |
| D1_A_090 | 8 / 55 / 45 / 11 | 42.1% | 45.0% | 0.468947 | 0.150368 |
| D1_B_080 | 1 / 12 / 88 / 18 | 5.3% | 88.0% | 0.468947 | 0.150368 |
| D1_B_085 | 1 / 12 / 88 / 18 | 5.3% | 88.0% | 0.468947 | 0.150368 |
| D1_B_090 | 0 / 10 / 90 / 19 | 0.0% | 90.0% | 0.468947 | 0.150368 |
| D2_A_080 | 6 / 18 / 82 / 13 | 31.6% | 82.0% | 0.482632 | 0.216244 |
| D2_A_085 | 4 / 13 / 87 / 15 | 21.1% | 87.0% | 0.482632 | 0.216244 |
| D2_A_090 | 4 / 7 / 93 / 15 | 21.1% | 93.0% | 0.482632 | 0.216244 |
| D2_B_080 | 5 / 15 / 85 / 14 | 26.3% | 85.0% | 0.482632 | 0.216244 |
| D2_B_085 | 5 / 12 / 88 / 14 | 26.3% | 88.0% | 0.482632 | 0.216244 |
| D2_B_090 | 3 / 5 / 95 / 16 | 15.8% | 95.0% | 0.482632 | 0.216244 |
| nested_B_080 | 3 / 13 / 87 / 16 | 15.8% | 87.0% | 0.514737 | 0.252140 |
| nested_B_085 | 3 / 8 / 92 / 16 | 15.8% | 92.0% | 0.514737 | 0.252140 |
| nested_B_090 | 3 / 7 / 93 / 16 | 15.8% | 93.0% | 0.514737 | 0.252140 |

### Primary fold ranges

| Policy / primary target | Fold sensitivity range | Fold specificity range | Fold ROC-AUC range |
| --- | --- | --- | --- |
| mahalanobis_A_085 | 0.0%–50.0% | 70.0%–95.0% | 0.3875–0.6667 |
| mahalanobis_B_085 | 0.0%–33.3% | 85.0%–100.0% | 0.3875–0.6667 |
| svm_005_scale_A_085 | 0.0%–50.0% | 85.0%–95.0% | 0.3750–0.7500 |
| svm_005_scale_B_085 | 0.0%–50.0% | 85.0%–100.0% | 0.3750–0.7500 |
| svm_005_01_A_085 | 0.0%–50.0% | 75.0%–95.0% | 0.3375–0.7375 |
| svm_005_01_B_085 | 0.0%–50.0% | 85.0%–100.0% | 0.3375–0.7375 |
| svm_010_scale_A_085 | 0.0%–50.0% | 85.0%–95.0% | 0.3750–0.7500 |
| svm_010_scale_B_085 | 0.0%–50.0% | 85.0%–100.0% | 0.3750–0.7500 |
| svm_010_01_A_085 | 0.0%–50.0% | 75.0%–95.0% | 0.3375–0.7375 |
| svm_010_01_B_085 | 0.0%–50.0% | 85.0%–100.0% | 0.3375–0.7375 |
| isolation_forest_A_085 | 0.0%–50.0% | 80.0%–90.0% | 0.3625–0.7000 |
| isolation_forest_B_085 | 0.0%–33.3% | 85.0%–90.0% | 0.3625–0.7000 |
| B1_A_085 | 100.0%–100.0% | 0.0%–0.0% | 0.3667–0.7625 |
| B1_B_085 | 0.0%–25.0% | 80.0%–100.0% | 0.3667–0.7625 |
| B2_A_085 | 0.0%–50.0% | 75.0%–95.0% | 0.3375–0.6875 |
| B2_B_085 | 0.0%–50.0% | 75.0%–95.0% | 0.3375–0.6875 |
| B3_A_085 | 0.0%–50.0% | 70.0%–90.0% | 0.3500–0.7000 |
| B3_B_085 | 0.0%–33.3% | 95.0%–100.0% | 0.3500–0.7000 |
| C1_A_085 | 50.0%–100.0% | 10.0%–35.0% | 0.2125–0.6500 |
| C1_B_085 | 0.0%–0.0% | 85.0%–100.0% | 0.2125–0.6500 |
| C2_A_085 | 0.0%–50.0% | 80.0%–100.0% | 0.3250–0.6750 |
| C2_B_085 | 0.0%–25.0% | 85.0%–100.0% | 0.3250–0.6750 |
| C3_A_085 | 0.0%–50.0% | 70.0%–95.0% | 0.3125–0.6625 |
| C3_B_085 | 0.0%–33.3% | 75.0%–100.0% | 0.3125–0.6625 |
| D1_A_085 | 50.0%–100.0% | 15.0%–35.0% | 0.3000–0.5875 |
| D1_B_085 | 0.0%–33.3% | 70.0%–95.0% | 0.3000–0.5875 |
| D2_A_085 | 0.0%–50.0% | 80.0%–95.0% | 0.3000–0.6750 |
| D2_B_085 | 0.0%–50.0% | 80.0%–100.0% | 0.3000–0.6750 |
| nested_B_085 | 0.0%–33.3% | 85.0%–100.0% | 0.3375–0.7333 |

### Selected raw cutoffs by fold

| Track / policy / target | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
| --- | --- | --- | --- | --- | --- |
| mahalanobis A 0.8 | 13.5361864 | 13.083689 | 14.1433986 | 14.7803949 | 13.4553833 |
| mahalanobis A 0.85 | 16.1808201 | 13.9191024 | 15.3492982 | 16.3812566 | 15.0375176 |
| mahalanobis A 0.9 | 22.3640545 | 15.4891671 | 21.9303591 | 18.1306701 | 22.2932444 |
| mahalanobis B 0.8 | 16.8370333 | 21.7615796 | 21.613629 | 17.9064959 | 21.5935886 |
| mahalanobis B 0.85 | 16.8370333 | 21.7615796 | 21.613629 | 17.9064959 | 21.5935886 |
| mahalanobis B 0.9 | 24.7303098 | 21.7615796 | 21.613629 | 17.9064959 | 21.5935886 |
| svm_005_scale A 0.8 | 0.0774649702 | 0.0448196265 | 0.0556512012 | 0.0892649274 | 0.0461452308 |
| svm_005_scale A 0.85 | 0.097057198 | 0.0831546433 | 0.0867174038 | 0.112346655 | 0.0732100517 |
| svm_005_scale A 0.9 | 0.151132582 | 0.0948868857 | 0.151640954 | 0.142312341 | 0.113838546 |
| svm_005_scale B 0.8 | 0.134035174 | 0.128701039 | 0.0728320237 | 0.152418487 | 0.0558893106 |
| svm_005_scale B 0.85 | 0.134035174 | 0.128701039 | 0.0728320237 | 0.152418487 | 0.229190816 |
| svm_005_scale B 0.9 | 0.134035174 | 0.128701039 | 0.199076692 | 0.152418487 | 0.229190816 |
| svm_005_01 A 0.8 | 0.0783390069 | 0.0590961641 | 0.0695584131 | 0.0956777763 | 0.0677936281 |
| svm_005_01 A 0.85 | 0.101909629 | 0.0782129542 | 0.121842459 | 0.102778728 | 0.0861717005 |
| svm_005_01 A 0.9 | 0.135638642 | 0.0960177272 | 0.140332714 | 0.132660887 | 0.111608089 |
| svm_005_01 B 0.8 | 0.0820526627 | 0.121253717 | 0.0720781677 | 0.134288341 | 0.0749506926 |
| svm_005_01 B 0.85 | 0.124637172 | 0.121253717 | 0.122995807 | 0.134288341 | 0.183970312 |
| svm_005_01 B 0.9 | 0.164144075 | 0.121253717 | 0.173002986 | 0.134288341 | 0.183970312 |
| svm_010_scale A 0.8 | 0.155612335 | 0.0899870147 | 0.111354525 | 0.178350905 | 0.0924232197 |
| svm_010_scale A 0.85 | 0.193884071 | 0.166207802 | 0.173726726 | 0.224301198 | 0.147027268 |
| svm_010_scale A 0.9 | 0.302222116 | 0.190376735 | 0.30328777 | 0.284309464 | 0.227853257 |
| svm_010_scale B 0.8 | 0.268072116 | 0.257546998 | 0.145053749 | 0.30569615 | 0.111740642 |
| svm_010_scale B 0.85 | 0.268072116 | 0.257546998 | 0.145053749 | 0.30569615 | 0.458277071 |
| svm_010_scale B 0.9 | 0.268072116 | 0.257546998 | 0.398529094 | 0.30569615 | 0.458277071 |
| svm_010_01 A 0.8 | 0.156568899 | 0.117865974 | 0.138489384 | 0.191343603 | 0.135666817 |
| svm_010_01 A 0.85 | 0.203826449 | 0.156590867 | 0.243708221 | 0.20577822 | 0.172321096 |
| svm_010_01 A 0.9 | 0.271248803 | 0.191681918 | 0.280612786 | 0.265150073 | 0.223431232 |
| svm_010_01 B 0.8 | 0.163887003 | 0.242088077 | 0.144014783 | 0.268634208 | 0.150404223 |
| svm_010_01 B 0.85 | 0.249287613 | 0.242088077 | 0.24626857 | 0.268634208 | 0.368021384 |
| svm_010_01 B 0.9 | 0.328175028 | 0.242088077 | 0.345970566 | 0.268634208 | 0.368021384 |
| isolation_forest A 0.8 | 0.524408662 | 0.521143024 | 0.520689577 | 0.514586437 | 0.532547178 |
| isolation_forest A 0.85 | 0.545737414 | 0.54101634 | 0.53188319 | 0.536806248 | 0.539010725 |
| isolation_forest A 0.9 | 0.559373211 | 0.544026326 | 0.551183484 | 0.56413124 | 0.548489561 |
| isolation_forest B 0.8 | 0.546402041 | 0.510530285 | 0.533177032 | 0.56385942 | 0.551544162 |
| isolation_forest B 0.85 | 0.546402041 | 0.556838295 | 0.533177032 | 0.56385942 | 0.551544162 |
| isolation_forest B 0.9 | 0.565137064 | 0.556838295 | 0.570325644 | 0.56385942 | 0.551544162 |
| B1 A 0.8 | 15.198877 | 15.198877 | 15.198877 | 15.198877 | 15.198877 |
| B1 A 0.85 | 17.0185337 | 17.0185337 | 17.0185337 | 17.0185337 | 17.0185337 |
| B1 A 0.9 | 20.3645383 | 20.3645383 | 20.3645383 | 20.3645383 | 20.3645383 |
| B1 B 0.8 | 94.5582954 | 112.745623 | 91.9254608 | 91.9254608 | 111.027867 |
| B1 B 0.85 | 111.027867 | 112.745623 | 111.027867 | 111.027867 | 111.027867 |
| B1 B 0.9 | 111.027867 | 112.745623 | 111.027867 | 111.027867 | 111.027867 |
| B2 A 0.8 | 17.2551413 | 14.5453826 | 16.5021278 | 16.2894463 | 17.6563041 |
| B2 A 0.85 | 20.1381594 | 17.3584378 | 19.3217288 | 17.1250292 | 19.6577909 |
| B2 A 0.9 | 24.5203315 | 17.7785849 | 22.138485 | 20.5333391 | 22.8691741 |
| B2 B 0.8 | 21.8818072 | 15.4620341 | 16.5561151 | 16.4608586 | 18.8386888 |
| B2 B 0.85 | 21.8818072 | 17.5944295 | 20.4599458 | 17.147201 | 18.8386888 |
| B2 B 0.9 | 24.1963445 | 26.3502581 | 30.8114783 | 33.9447601 | 51.3311628 |
| B3 A 0.8 | 12.780597 | 11.3717534 | 13.3676671 | 12.3691647 | 13.267382 |
| B3 A 0.85 | 13.5407097 | 12.3733246 | 15.3882671 | 12.9489166 | 14.1173408 |
| B3 A 0.9 | 17.1792704 | 16.0872592 | 18.1128895 | 18.5463514 | 18.1648323 |
| B3 B 0.8 | 24.0281748 | 22.0478908 | 12.6188332 | 22.140598 | 24.0079182 |
| B3 B 0.85 | 24.0281748 | 22.0478908 | 24.0833187 | 22.140598 | 24.0079182 |
| B3 B 0.9 | 24.0281748 | 22.0478908 | 24.0833187 | 22.140598 | 24.0079182 |
| C1 A 0.8 | 14.466457 | 14.466457 | 14.466457 | 14.466457 | 14.466457 |
| C1 A 0.85 | 15.3151703 | 15.3151703 | 15.3151703 | 15.3151703 | 15.3151703 |
| C1 A 0.9 | 18.8477407 | 18.8477407 | 18.8477407 | 18.8477407 | 18.8477407 |
| C1 B 0.8 | 43.1418626 | 43.1418626 | 43.1418626 | 43.1418626 | 71.2997996 |
| C1 B 0.85 | 43.1418626 | 43.1418626 | 43.1418626 | 43.1418626 | 71.2997996 |
| C1 B 0.9 | 43.1418626 | 43.1418626 | 43.1418626 | 43.1418626 | 71.2997996 |
| C2 A 0.8 | 17.1084843 | 13.442578 | 16.7998084 | 15.3291583 | 15.464275 |
| C2 A 0.85 | 19.4885823 | 16.9625461 | 19.3426362 | 18.165203 | 19.6934985 |
| C2 A 0.9 | 20.7759321 | 18.5380431 | 21.2584114 | 20.1636751 | 22.003225 |
| C2 B 0.8 | 18.9303138 | 13.6611411 | 16.4494162 | 20.5189032 | 15.9775803 |
| C2 B 0.85 | 21.3792904 | 24.9095377 | 21.3010709 | 20.5189032 | 37.3057445 |
| C2 B 0.9 | 21.3792904 | 24.9095377 | 21.3010709 | 20.5189032 | 37.3057445 |
| C3 A 0.8 | 7.81157292 | 7.57976315 | 7.69078369 | 6.63794897 | 8.18487153 |
| C3 A 0.85 | 8.44688171 | 8.05051423 | 8.87380056 | 8.21552418 | 9.21619223 |
| C3 A 0.9 | 10.0933437 | 9.02008182 | 9.44658576 | 9.7180892 | 9.84864828 |
| C3 B 0.8 | 8.17647297 | 7.97566951 | 8.00693899 | 8.87880565 | 8.13635274 |
| C3 B 0.85 | 9.37602215 | 18.7336991 | 8.68259066 | 8.87880565 | 12.8610512 |
| C3 B 0.9 | 9.37602215 | 18.7336991 | 12.4377573 | 12.6694483 | 12.8610512 |
| D1 A 0.8 | 12.5796132 | 12.5796132 | 12.5796132 | 12.5796132 | 12.5796132 |
| D1 A 0.85 | 14.1667343 | 14.1667343 | 14.1667343 | 14.1667343 | 14.1667343 |
| D1 A 0.9 | 18.6426047 | 18.6426047 | 18.6426047 | 18.6426047 | 18.6426047 |
| D1 B 0.8 | 33.1583802 | 29.7764266 | 29.7764266 | 29.7764266 | 29.7764266 |
| D1 B 0.85 | 33.1583802 | 29.7764266 | 29.7764266 | 29.7764266 | 29.7764266 |
| D1 B 0.9 | 33.1583802 | 33.1583802 | 33.1583802 | 29.7764266 | 46.5670056 |
| D2 A 0.8 | 18.2533843 | 15.2761343 | 15.4636749 | 15.2824363 | 16.8650335 |
| D2 A 0.85 | 19.9767018 | 17.1102611 | 19.4506827 | 19.6486942 | 20.9849416 |
| D2 A 0.9 | 26.4466973 | 20.279506 | 22.6121294 | 24.2597327 | 25.5151605 |
| D2 B 0.8 | 24.4180144 | 16.6558461 | 15.4832662 | 17.2253901 | 17.1658105 |
| D2 B 0.85 | 24.4180144 | 17.2706922 | 19.4266668 | 17.2253901 | 20.9632363 |
| D2 B 0.9 | 26.7622183 | 28.1157463 | 27.3422883 | 25.2074095 | 24.7813693 |

For nested B, take the selected candidate's B cutoff in each fold; selected
configurations are recorded above. No cutoff is determined from pooled outer
predictions. Public-only A cutoffs are identical across local folds because
their calibration is exclusively public normal cross-fitting.

### Standardized score distribution summaries

Each row uses A85's scores; A/B and targets do not alter a fixed detector's
ranking. The secondary nested policy has its own fold-specific model scores.
Normal/anemia medians and interquartile ranges are descriptive held-out
summaries, never calibration inputs.

| Score source | Normal median | Normal Q25–Q75 | Anemia median | Anemia Q25–Q75 |
| --- | --- | --- | --- | --- |
| mahalanobis_A_085 | -0.3067 | -0.6786–0.1244 | -0.3072 | -0.6332–0.4081 |
| svm_005_scale_A_085 | -0.4980 | -1.1338–0.1768 | -0.4394 | -1.0458–0.2427 |
| svm_005_01_A_085 | -0.4813 | -1.0354–0.3436 | -0.4728 | -1.2211–0.5637 |
| svm_010_scale_A_085 | -0.4979 | -1.1350–0.1770 | -0.4437 | -1.0458–0.2444 |
| svm_010_01_A_085 | -0.4809 | -1.0338–0.3459 | -0.4742 | -1.2203–0.5651 |
| isolation_forest_A_085 | -0.2602 | -0.7617–0.6092 | -0.2536 | -0.9192–0.8060 |
| B1_A_085 | 7.2386 | 4.7927–10.3048 | 7.3495 | 6.1602–9.9378 |
| B2_A_085 | -0.2464 | -0.7229–0.3507 | -0.2590 | -0.6029–0.8990 |
| B3_A_085 | -0.3070 | -0.7021–0.3743 | -0.2388 | -0.7500–0.2298 |
| C1_A_085 | 0.7252 | 0.3594–1.2126 | 0.5535 | 0.2023–0.7627 |
| C2_A_085 | -0.2450 | -0.7682–0.2893 | -0.3111 | -0.8158–0.6598 |
| C3_A_085 | -0.3065 | -0.7265–0.2798 | -0.4214 | -0.6897–0.3194 |
| D1_A_085 | 0.8353 | 0.3991–1.3697 | 0.7245 | 0.5918–1.1143 |
| D2_A_085 | -0.3015 | -0.6715–0.1846 | -0.4502 | -0.8523–0.8028 |
| nested_B_085 | -0.4575 | -0.8378–0.2721 | -0.3156 | -0.8860–0.3006 |

### Raw pooled ranking sensitivity check

| Track | Raw ROC-AUC | Raw AP |
| --- | --- | --- |
| mahalanobis | 0.543158 | 0.259953 |
| svm_005_scale | 0.537895 | 0.345165 |
| svm_005_01 | 0.541579 | 0.341374 |
| svm_010_scale | 0.537895 | 0.345165 |
| svm_010_01 | 0.541579 | 0.341374 |
| isolation_forest | 0.496842 | 0.246384 |
| B1 | 0.550526 | 0.184737 |
| B2 | 0.541579 | 0.297281 |
| B3 | 0.516842 | 0.301646 |
| C1 | 0.378421 | 0.129228 |
| C2 | 0.502632 | 0.264799 |
| C3 | 0.491579 | 0.221579 |
| D1 | 0.468947 | 0.150368 |
| D2 | 0.488947 | 0.220972 |

Raw pooled rankings are descriptive because cross-fold model scales can differ.
Their similar weak results do not support a different conclusion or authorize
changing the predefined primary ranking representation.
