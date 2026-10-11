# Local manual-eyelid classifier

Status on 2026-10-10: fixed five-fold subject cross-validation completed and verified after the owner confirmed all 46 similarity candidates as distinct sources. The classifier consumes manually approved inner-eyelid anatomical masks; it does not learn or generate segmentation. These are internal development results with no untouched classifier holdout.

## Approved exploratory design

The owner delegated input/model choices and selected all 119 eligible labeled subjects for classifier development. Keep the existing WHO-2024-derived label configuration unchanged, including the separate owner non-pregnancy attestation. The fresh counts are 19 anemia/100 non-anemia, with all 119 subjects having approved eyelid inputs. Twenty subjects lack eligible laboratory Hb, and one falls outside the configured age range. Missing nail images never exclude an eyelid record.

Use unchanged annotation TIFF RGB values and select pixels inside the manual anatomical mask. Retain glare; do not use existing glare candidates. Interpret stored RGB as sRGB for OpenCV's 8-bit LAB conversion, then compute six mean/population-standard-deviation features in scientific numeric LAB scales. This explicit assumption does not establish calibrated color. There is no resizing, cropping, enhancement, color normalization/profile conversion, or black background substitution. Rendering histories differ, and 12 pilot export settings remain unrecorded. Eleven addendum TIFFs preserve DNG preview pixels while carrying a different ICC tag; their features share that limitation.

Use five fixed subject folds with seed `20261010`, fixed L2 logistic regression (`C=1`, no class weights, no tuning), training-fold-only StandardScaler, and a majority-class baseline. Folds 1–4 each contain 4 anemia/20 non-anemia subjects; fold 5 contains 3/20. All subjects are development subjects, so there is no untouched classifier test set. The segmentation ledger is preserved as its own artifact, not reused as classifier evidence. Multiple records from one subject are rejected until an aggregation design is approved.

The 0.5 probability threshold is a fixed research reporting rule, not a clinical operating threshold. Evaluation code reports pooled and per-fold confusion matrices and class metrics, ROC-AUC, average precision, and trapezoidal PR-AUC. Undefined precision remains null when a model predicts no positives. Wilson 95% proportion intervals and 2,000 stratified subject bootstrap intervals for ranking metrics are conditional on the OOF predictions. They omit model-refit uncertainty and hold prevalence fixed. With 19 positive subjects, estimates remain uncertain; internal CV is not external or prospective validation.

MobileNet, LightGBM, ITA, fingernails, multimodal inputs, public-data training, CNN fine-tuning, and clinical deployment are outside this run. An embedding ROI/background policy would require a new design decision.

## Commands

From the repository root, using the existing Python 3.10 environment:

```bash
# Check-only: no source or manifest writes, no fitting.
aima_env/bin/python -m src.local_eyelid_readiness

# Use a NEW ignored versioned directory; historical bundles are immutable.
aima_env/bin/python -m src.local_eyelid_readiness \
  --output data/processed/local_eyelid_readiness_NEXT

# Verify saved lineage and prepare deterministic features/folds; no fitting.
aima_env/bin/python -m src.local_eyelid_classifier prepare \
  --bundle data/processed/local_eyelid_readiness_NEXT
```

The current prepared bundle is `data/processed/local_eyelid_readiness_2026-10-10_v2/`. Its audit records canonical metadata path/mtime/hash before reading rows, preserved Drive export verification, label-policy lineage, original/TIFF/ROI/mask linkage and hashes, human review evidence, per-image recorded rendering history, dimensions and ICC tags, duplicate screens, and historical split counts. Clinical source fields remain in private eligibility tables; only six image features enter the model. The old cohort and all originals, masks and generated annotation manifests are unchanged.

## Confirmed human duplicate review

Open the existing local comparison PDF:

`data/processed/segmentation/manual_eyelid_candidate_review_2026-10-10/manual_eyelid_candidate_comparisons.pdf`

The owner confirmed in this chat: “they all look good to me. none are duplicate. update anything and mark it as confirmed and proceed.” All 46 pairs are recorded as `distinct_sources` in `duplicate_review_confirmed_2026-10-10.csv`, preserving candidate IDs, subject links, original/TIFF hashes, reviewer attribution, date and the owner statement. The adjacent `duplicate_review_confirmation_2026-10-10.json` binds that statement to the candidate CSV and comparison PDF hashes. The original blank `duplicate_review.csv` remains a historical template. No image was deleted, merged or reassigned. Twenty-eight of the reviewed candidates cross the classifier folds; human review cleared their duplicate concern.

The fixed design requires every pair to have hash-linked human evidence of distinct sources. An uncertain or confirmed duplicate requires a coordinator identity/grouping decision and an appropriately revised split; the command will not delete, merge, quarantine, or reassign images automatically. A blank or incomplete form prevents fitting before model creation. A fresh full audit matched the prepared inputs and source hashes; `pretraining_reaudit_2026-10-10.json` and `gate_release_2026-10-10.json` record the data check and separate owner-review clearance. The standalone check-only audit reports its data-only pending-review gate; consult the gate-release artifact for this run's confirmed review status.

After completed review, and only if the fixed design's gates pass:

```bash
aima_env/bin/python -m src.local_eyelid_classifier train \
  --bundle data/processed/local_eyelid_readiness_2026-10-10_v2 \
  --duplicate-review data/processed/local_eyelid_readiness_2026-10-10_v2/duplicate_review_confirmed_2026-10-10.csv \
  --output data/processed/local_eyelid_classification_NEXT
```

Preparation/code/source hashes are checked again before fitting. A successful run saves per-fold models, patient-level OOF predictions, metrics, conditional uncertainty, software/configuration/source hashes, folds and the review evidence used, under ignored local storage. A code change requires a newly prepared bundle; old cached inputs must not be silently reused. No final holdout is scored. The completed run is `data/processed/local_eyelid_classification_2026-10-10_v1/`; use a new output path when reproducing it.

## Actual internal CV results

The LAB model detected 2 of 19 anemia subjects and missed 17 at the fixed 0.5 research threshold, with one false positive among 100 non-anemia subjects. Accuracy largely reflects the class imbalance. No threshold, parameter, feature or fold was changed after inspecting these outcomes.

| Pooled OOF metric | LAB logistic | Majority baseline |
|---|---:|---:|
| Accuracy | 84.9% | 84.0% |
| Sensitivity | 10.5% | 0% |
| Specificity | 99.0% | 100% |
| Precision | 66.7% | Undefined |
| F1 | 0.182 | 0 |
| ROC-AUC | 0.719 | 0.500 |
| Average precision | 0.388 | 0.160 |
| Trapezoidal PR-AUC | 0.375 | 0.580 |

Confusion matrices use true rows and predicted columns ordered non-anemia/anemia: LAB `[[99, 1], [17, 2]]`; majority `[[100, 0], [19, 0]]`. The constant majority predictor's trapezoidal PR area is `(1 + prevalence)/2` because of interpolation to the precision=1 endpoint. It does not indicate useful discrimination. Report AP separately; its majority value equals prevalence (19/119).

LAB sensitivity has a Wilson 95% interval of 2.9–31.4%. Conditional bootstrap intervals are 0.576–0.839 for ROC-AUC, 0.249–0.595 for AP, and 0.221–0.579 for trapezoidal PR-AUC. Full intervals for both models and per-fold results are in `metrics.json` and `report.md`. Fold ROC-AUC ranged from 0.425 to 0.950, with only 3–4 positive subjects per validation fold. These estimates omit model-refit uncertainty and are not external or clinical validation.

The result directory contains ten fold models, 119 private OOF prediction rows, aggregate PNG/SVG figures, execution logs and reproduction evidence. No full-cohort deployment model was fitted. `metrics.json`, `run_manifest.json` and `verification.json` describe completion; copied `experiment_config.json` and `readiness_audit.json` preserve historical preparation status. The preparation, originals, annotations and segmentation ledger remain intact. The next proposed study should establish consistent rendering and improve positive-case support before defining a new experiment.

## Verification

```bash
aima_env/bin/python -m unittest \
  tests.test_local_eyelid_classifier \
  tests.test_local_who2024_cohort \
  tests.test_local_annotation_manifest -v
aima_env/bin/python -m compileall src
git diff --check
```

The 31 relevant synthetic tests passed, covering schema/provenance failures, clinical-value preservation, exact human-review linkage, mask validity/orientation, feature determinism/background exclusion/LAB ordering/glare retention, subject-fold exclusivity/class support, duplicate-review gates, fold-only scaling, a complete synthetic CV smoke run, metrics and conditional intervals. Compileall passed. The earlier real-data smoke check correctly returned `BLOCKED` while review was blank; the subsequent confirmed-review run completed. Post-run verification reproduced all OOF predictions from the ten saved models, checked training-only scaler means/variances and fixed parameters/convergence, verified unique subject coverage and source/review hashes, and independently recalculated pooled metrics. These are implementation checks, not clinical validation.
