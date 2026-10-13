# Experiment 7 dataset eligibility and compatibility

Audit completed before new fitting, 2026-10-13. Private source-linked evidence
is under the Experiment 7 bundle's `audit/`; original exports and historical
artifacts were read only. Status: `PASS_WITH_MANUAL_REVIEW`, with existing
owner duplicate adjudication verified and released separately from the old
auditor's historical pending-review wording.

| Source | Available / eligible | Classes non-anemia / anemia | Normal training groups used |
|---|---|---|---:|
| Local AIMA | 140 metadata subjects; 119 eligible labeled eyelids | 100 / 19 | 80 in each outer-training fold |
| Eyes Defy | 218 image/metadata subjects; 183 retained under unchanged Exp3 derived policy; 180 conservative eligible groups | 119 / 64; historical public training 92 / 54, validation 27 / 10 | 91 |
| CP-AnemiC V2 | 251 unique canonical images; 246 conservative groups; true patient count unavailable | 141 / 110; historical public training 109 / 90, validation 32 / 20 | 108 |

Local canonical CSV SHA-256 remains
`1e704fb8d575757f16fc4f690cdc3315bc4063a5cf258d6c2e1b5f784d19596d`;
mtime `2026-10-07T14:55:39.254357+08:00`, recorded before reading. Provenance,
original Drive export, header/trailing cells, IDs, Hb fields/units, age/sex,
previous manifests and labels passed. All 275 image references resolve; all
429 original archive files match their inventory hashes. All 140 approved
eyelids passed source/TIFF/ROI/mask and review-preview checks. Local eligible
ages are 16–61. Original five folds and all 119 robust18 vectors match exactly.
Twenty missing-lab-Hb and one age exclusion remain unchanged. Missing nails
do not affect eyelid eligibility. Non-pregnancy remains an owner attestation.

Eyes source ages span 19–88; eligible ages span 19–65. Keep the existing adult
15–65 sex-specific derived rule, excluding 15 pregnancy-dependent records,
19 age-ineligible records and one missing Hb. The originating paper's
<=10.5 g/dL target is a different research definition, not the label policy
of this transfer subset. Source Hb is preserved. Two mixed-label conservative
groups are excluded from normal fitting; public validation is never fitted.
Public normal-group means reduce repeated-image influence.

CP uses its original pediatric Hb<11 g/dL labels, not adult relabeling or a
claim of current age-specific WHO equivalence. Its official V2 ZIP hash is
`a976b9c2365fe9b0d07c61dc1a94ffcc08a1918e617e510529cbbf5ecbb0a3b0`;
all 503 local files match. The 251 All Categories copies are excluded.
Individual patient identity, age/sex and acquisition metadata are unavailable.
Five near-image pairs and curation IDs define conservative groups. Dataset
level age 6–59 months and acquisition details come from the
[official V2 description](https://data.mendeley.com/datasets/m53vz6b7fx/2).

Eyes uses provided palpebral segmentation, never forniceal/combined tissue.
Its untagged JPEGs align with the registered mask rasters; existing alignment
checks pass. The [originating paper](https://doi.org/10.1016/j.artmed.2022.102477)
describes controlled magnification and LED acquisition. Local features use
unchanged native manual TIFF masks and stored RGB, with mixed Photos/Preview/
DNG histories and no retrospective calibration.

CP V2 documents palpebral conjunctiva captured with tablets under ambient
light, followed by triangle/entropy ROI extraction. The opaque alpha==255
core of its untagged RGBA strips is used. Gross tissue target matches; ROI
extent and intensity selection may differ materially from local manual masks.
All sources use the identical ordered 18 functions and quantized LAB units,
with explicit assumed-sRGB computation. Resolution/rendering equivalence is
not established.

No source-byte or cross-source perceptual duplicate candidate was found.
These screens and separate namespaces do not prove true patient independence,
especially for CP. All B/C/D gates pass for bounded exploratory testing.
Pooling is justified only as a source-balanced mixture over the documented
gross target; it cannot conceal pediatric/adult, acquisition or ROI differences.
Adaptation uses local training normals only. No public validation group is
used for fitting, adaptation or public normal calibration.

CP lists CC BY4.0 alongside an academic/research-only author disclaimer; this
study is academic and redistributes no data. Original Eyes IEEE DataPort
license terms could not be independently retrieved; no permissive license or
redistribution right is asserted. Existing local research copies are used
without downloading or publishing dataset bytes. Neither license establishes
permission to publish local patient data.

Unresolved limitations: true CP patient grouping, public ROI equivalence,
uncalibrated mixed color capture, unavailable graded exposure/occlusion/pose
metadata, standardized skin tone and independent local validation. These are
reported limitations, not newly invented clinical or anatomical annotations.

## 2026-10-14 source-workflow version bridge

The first run stopped at its final source guard before fitting any model.
A separate canonical nail reset changed the annotation manifest, image-ready
inventory and anatomical review register. Current nail status is 135 pending;
this change is outside Experiment 7's eyelid scope. All 140 eyelid rows,
including every column, are exactly unchanged in each of the three tables.
The reset's `manifests_before_reset/` copies match all three historical hashes.
All 2,717 registered historical artifact files remain unchanged.

The stopped preparation is sealed as v1. Accepted preparation/training used
a new v2 and independently reproduced. Historical Exp1 context is verified
against those exact archived
table bytes, with a mandatory 140-row full-field eyelid equality check; every
other historical source/code hash must still match. Fresh current clinical,
image, approval, mask and feature checks remain mandatory. New v2 pins the
current workflow tables and archived originals. Neither historical verifiers
nor another task's source files are modified. No model setting, cohort, label,
mask, fold or threshold rule changed, and no outer result existed at this time.
