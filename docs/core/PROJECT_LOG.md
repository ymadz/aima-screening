# AIMA Project Log

Chronological record of meaningful implementation, data, methodology, and readiness milestones. Dates are based on repository history or dated artifacts. This log records status; it does not itself approve research-method changes. Patient-level data remain in ignored paths and are not reproduced here. See the [documentation index](../README.md) for document navigation.

## 2026-08-21 — Public eyelid baseline added

- **Area:** public preprocessing, feature extraction, classification, and evaluation.
- **What changed / why:** Git history records the initial dataset-builder, training, evaluation, and dependency work. This established an exploratory public-data baseline.
- **Files/artifacts:** `src/dataset_builder.py`, `src/preprocess.py`, `src/feature_extractor.py`, `src/train_classifier.py`, `src/evaluate.py`, `requirements.txt`.
- **Result:** current implementation selects public palpebral PNGs, extracts six CIE L*a*b* statistics, 576 frozen ImageNet MobileNetV3-Small features, and one fixed-margin ITA value, then trains a LightGBM classifier (583 numeric features).
- **Limitations/blockers:** public cross-validation uses row-stratified folds, not a group-aware splitter; it is not independent/external validation. The ITA patch is not validated as a standardized skin measurement.
- **Next action:** keep the baseline reproducible, disclose scope, and do not use its result as local-model performance.

## 2026-09-23 — Earlier local cohort artifact created (superseded)

- **Area:** local metadata/label exploration.
- **What changed / why:** an earlier derived eyelid cohort was generated from the then-used Downloads/working metadata and an earlier inclusion rule.
- **Files/artifacts:** `data/processed/local/local_supervised_eyelid_cohort_v1.csv` and its provenance JSON.
- **Result:** 102 rows (17 anemic, 85 non-anemic) were recorded in that historical artifact.
- **Limitations/blockers:** it predates the frozen live-Sheet export and current WHO-2024 cohort procedure. It is not a current training input and must not be treated as authoritative.
- **Next action:** use only the versioned live-Sheet export and current derived cohort for new local work.

## 2026-09-23 — Repository and Git operating rules added

- **Area:** agent workflow and sensitive-data safeguards.
- **What changed / why:** repository guidance and a Git policy were added.
- **Files/artifacts:** `AGENTS.md`, `docs/policies/git-policy.md`.
- **Result:** canonical-source verification, sensitive-data exclusions, commit checks, and research-change approval rules are documented.
- **Blockers:** none recorded.
- **Next action:** all agents must read these rules before acting; research-level changes require the main AIMA coordinator's approval before commit.

## 2026-10-07 — Live-Sheet metadata frozen

- **Area:** local data provenance.
- **What changed / why:** a versioned values export of the finalized live Sheet was frozen so derived local audits and cohorts can be tied to stable bytes.
- **Files/artifacts:** `data/raw/local/metadata/frozen_2026-10-07/AIMA Patients Metadata - Sheet1 values.csv`, adjacent original Drive export and `provenance.json`; subject manifest under `data/raw/local/manifest/frozen_2026-10-07/`.
- **Result:** the frozen CSV records 140 subjects, 275 image references (140 eyelid, 135 nail), mtime `2026-10-07T14:55:39+08:00`, and SHA-256 `1e704fb8d575757f16fc4f690cdc3315bc4063a5cf258d6c2e1b5f784d19596d`. Its provenance identifies the live Sheet. The earlier Downloads file is historical for new local builds.
- **Limitations/blockers:** the Sheet has no pregnancy-status field and has missing Hb for some subjects; source data are not overwritten to address either issue.
- **Next action:** verify the frozen hash/provenance before every related build; derive labels without altering source measurements.

## 2026-10-07 — Public baseline reproduced

- **Area:** public baseline verification and saved results.
- **What changed / why:** a separate reproduction/validation script checked the public feature table and reran the configured cross-validation, producing a dated results bundle.
- **Files/artifacts:** `src/reproduce_public_baseline.py`; `data/processed/public_baseline_reproduction_2026-10-07/metrics.json` and associated ignored outputs.
- **Result:** mean-of-fold accuracy 80.41%, sensitivity 76.90%, specificity 82.73%, and ROC-AUC 0.8723; pooled OOF ROC-AUC 0.8692. These are internal CV estimates, not independent or external validation.
- **Limitations/blockers:** very small, imbalanced ITA subgroups; current fold assignment is row-stratified. Reproduction does not establish clinical validity.
- **Next action:** preserve the run configuration and describe metrics only as public-dataset cross-validation results.

## 2026-10-07 — Local segmentation prototype evaluated

- **Area:** optional anatomical segmentation assistance.
- **What changed / why:** a small per-modality ExtraTrees pixel-classification prototype was evaluated against pilot manual masks using held-out subjects.
- **Files/artifacts:** `src/segmentation_prototype.py`; ignored outputs under `data/processed/segmentation_prototype_2026-10-07/report.json`.
- **Result:** eyelid mean Dice 0.694 / IoU 0.555 (minimum Dice 0.320); nail mean Dice 0.746 / IoU 0.602 (minimum Dice 0.539). Three unseen proposals per modality have no truth metric.
- **Limitations/blockers:** only ten annotated examples per modality, leakage/overinclusive proposals observed, and no accepted automated segmenter. This is not an anemia classifier.
- **Next action:** retain as proposal-only; gather and review more manual annotations before any acceptance study.

## 2026-10-08 — Current local cohort and training readiness audited

- **Area:** local cohort, masks, and model readiness.
- **What changed / why:** the current derived cohort and registered annotation inventory were audited against the frozen metadata and file hashes.
- **Files/artifacts:** `data/processed/local_cohort_who2024_2026-10-08/summary.json`, `training_config.json`, `subject_folds.csv`; `data/raw/local/annotations/manifests/annotation_audit.json` and canonical manifest/queue.
- **Result:** 119 labeled subjects (19 anemia, 100 non-anemia); 21 excluded (20 missing laboratory Hb, one outside the configured 15–65 age range). A ten-fold subject assignment is prepared with seed 20261007. Of 32 structurally valid manual ROI/mask pairs, 30 are in the labeled cohort: 20 eyelid (one positive, 19 negative), 10 nail (zero positive, 10 negative). Training is explicitly `not_run_insufficient_reviewed_positive_masks`.
- **Limitations/blockers:** no local anemia classifier has been trained; structural file checks are not clinician validation; positive-class coverage for nail/paired experiments is absent; glare review remains separate and unresolved for some candidates.
- **Next action:** continue prioritized manual annotation and quality review. Recheck each modality's class/fold counts before deciding whether an exploratory model can run.

## 2026-10-08 — Consolidated local annotation workspace

### User-approved direction

- Use one consolidated annotation workspace rather than separate Batch 1 and Batch 2 working folders, to avoid parallel queues and confusion.
- Manually annotate all usable local eyelid and fingernail images as primary anatomical ground truth. Automated segmentation may be evaluated later as optional assistance; it must not be treated as ground truth.
- Keep glare as a separate annotation/QC layer from the anatomical eyelid or nail ROI.
- Do not train a model as part of this consolidation or audit.

### Implemented

- Consolidated Batch 1, Batch 2, and paired-pilot artifacts under `data/raw/local/annotations/`; updated active documentation and source references to use that location.
- Preserved original imports under `data/raw/local/incoming/`. Retained source manifests, guide files, intermediate renders, and historical experiments for provenance.
- Removed only ten TIFF copies that were byte-for-byte verified duplicates; one canonical copy per image remains, with duplicate evidence recorded in the merge audit.
- Added a canonical per-image annotation manifest and prioritized pending queue, both linked to frozen metadata, `subject_id`, original filename, and file hashes.
- Updated the repository instructions and photo-import guide to direct future work to the consolidated workspace.
- Added this project log and a reference index. The core planning and architecture documents were standardized in a later entry below.

### Observed audit results

- Frozen metadata source: `data/raw/local/metadata/frozen_2026-10-07/AIMA Patients Metadata - Sheet1 values.csv`; SHA-256 `1e704fb8d575757f16fc4f690cdc3315bc4063a5cf258d6c2e1b5f784d19596d`.
- Metadata contains 275 image references: 140 eyelid and 135 fingernail. All original-file references resolve uniquely in `incoming`.
- The consolidated annotation workspace contains 150 unique standardized annotation TIFFs: 140 eyelid and 10 fingernail. No metadata-unreferenced annotation image was found.
- Thirty-two existing anatomical Fiji ROI/binary-mask pairs passed filename/subject linkage, hash, geometry, nonempty, and binary-value checks. No anatomical pair was flagged as potentially mismatched by those structural checks. This is not clinician validation or a substitute for visual boundary review.
- The canonical manifest records 243 pending image references: 118 eyelid TIFFs ready for manual annotation and 125 fingernail originals that still need standardized TIFF copies. Of those pending references, 37 are anemia-positive, 166 non-anemia, and 40 have no WHO-eligible label.
- Five subjects have no fingernail reference in frozen metadata and are not missing-file errors or queue items.
- Separate glare status remains unresolved for parts of the pilot: paired-pilot glare ROIs are absent, eyelid candidate/rule masks are not all validated, and ten overinclusive nail glare-threshold candidates were rejected. These are not anatomical annotation failures.
- No annotations were created during the audit, and no local model training was run.

### Remaining work

1. Work from `data/raw/local/annotations/manifests/pending_annotation_queue.csv`, prioritizing labeled image pairs and the listed positive-fold coverage tier.
2. Render standardized full-size RGB TIFF annotation copies for raw-only fingernail originals without changing or renaming incoming files.
3. Save each manual ROI in the documented modality folder and tell the project agent when a set is ready; the agent currently has to reconcile new file paths/statuses in the helper inventory before rebuilding the canonical manifest. Automatic discovery is not implemented yet.
4. Visually review boundaries and verify subject, source filename, image/ROI/mask dimensions, hashes, binary values, glare handling, and subject-level split separation before any local training.
5. Resolve the glare protocol as a separately documented decision before using glare exclusions in model inputs.
6. Use the consolidated queue and complete review before local training; follow the current core documents linked from the README.

## 2026-10-08 — Core project references standardized

- **Area:** project requirements, architecture, and agent onboarding.
- **What changed / why:** the root PRD, detailed architecture, architecture essentials, and root agent instructions were created/updated so future agents can distinguish evidence, implementation, experiments, and proposals without repeated rediscovery. README navigation and the reference index were updated.
- **Files/artifacts:** `docs/core/PRD.md`, `docs/core/ARCHITECTURE.md`, `docs/core/ARCHITECTURE_ESSENTIALS.md`, `docs/core/PROJECT_LOG.md`, `AGENTS.md`, `README.md`, `docs/README.md`.

## 2026-10-09 — Local fingernail SAM 2.1 assistance pilot

- **Area:** local nail-plate proposal generation; separate from eyelid annotation and anemia modeling.
- **What changed / why:** added a local SAM 2.1 Hiera Tiny refinement workflow using an ExtraTrees model trained from approved manual nail masks to generate prompt boxes and points. Kept raw and cleaned candidate masks distinct and linked each prediction to subject, original filename/source, TIFF, hashes, checkpoint, and inference settings.
- **Files/artifacts:** `src/sam2_nail_segmentation.py`, `tests/test_sam2_nail_segmentation.py`; ignored outputs under `data/processed/segmentation/sam2_nail_lopo_areaqc_v1/` and `data/processed/segmentation/sam2_nail_candidate_batch_2026-10-09/`; [pilot report](../experiments/sam2-nail-pilot-2026-10-09.md).
- **Result:** ten-subject leave-one-subject-out mean Dice 0.794, IoU 0.669, precision 0.680, recall 0.987. ExtraTrees prompt-mask baseline Dice/IoU 0.737/0.591. The full current pending queue generated 116 proposals; 21 triggered simple area flags.
- **Limitations/blockers:** only ten approved references; visually observed nail/adjacent-skin leakage and partial masks; area checks miss boundary errors. All 116 remain `predicted_unverified`, with no manifest changes or claims about correction time. Nine ROI files from the earlier 67-item review snapshot use a non-contractual suffix and remain unrasterized/unregistered pending naming resolution.
- **Next action:** manually review/correct candidate overlays against original images, begin with flagged cases, preserve eyelid manual annotation, and validate ROI filename convention before registering the nine alternate-suffix nail ROIs.
- **Result:** public and local workflows are separately documented; the frozen live-Sheet export remains the local metadata authority; current blockers and model status are explicit; existing supporting documents were retained.
- **Limitations/blockers:** numeric clinical success thresholds, final glare rules, local model design/acceptance, transfer-learning approach, and deployment requirements remain unapproved or unimplemented.
- **Next action:** use these documents as shared references; update architecture/log when verified implementation or decisions change, and update the PRD only when approved requirements change.

## 2026-10-08 — Missing nail annotation TIFFs staged

- **Area:** local image rendering and annotation readiness.
- **What changed / why:** the frozen 140-subject metadata referenced 125 nail originals without TIFF annotation copies. Added a reproducible renderer and registered those copies in the single active annotation inventory.
- **Files/artifacts:** `src/render_local_annotation_tiffs.py`, the inventory-size check in `src/local_annotation_manifest.py`, and ignored outputs under `data/raw/local/annotations/pending/nail/images/` and `data/raw/local/annotations/manifests/`.
- **Result:** 121 DNG previews and four HEIC originals produced 125 full-size RGB TIFFs using lossless Deflate/predictor-2 compression (1,976,347,076 output bytes). The refreshed manifest resolves all 275 frozen image references to annotation TIFFs: 140 eyelid and 135 nail. The 32 approved manual ROI/mask pairs are unchanged; 243 images still need manual anatomical annotation.
- **Verification:** source hashes and exact subject/original-filename links were checked before rendering. Every new TIFF was decoded and compared pixel-for-pixel with its intended oriented RGB render before registration. The manifest/queue rebuild passed with 275 image records and 243 pending records.
- **Limitation:** DNG previews declare Display P3 while the existing pilot nail TIFFs carry an sRGB ICC tag despite retaining the same preview pixel values. The new DNG TIFFs follow that pilot annotation method; they are not evidence of calibrated or uniform-color model inputs. HEICs were rendered through macOS ColorSync to sRGB. No new anatomical ROIs, accepted glare masks, or local anemia model were created.
- **Next action:** use the refreshed priority queue for manual anatomical ROIs and review. Settle a separate color-rendering policy before using these annotation TIFF pixel values for color features or model comparisons.

## 2026-10-09 — Local eyelid ROI review and mask registration

- **Area:** local eyelid anatomical annotation and manifest integrity.
- **What changed / why:** after the study owner reviewed the 140 anatomical contact-sheet entries, rasterized all manual Fiji eyelid ROIs into the canonical mask folder and recorded approval separately from glare status. Added an explicit, hash-linked anatomical review register and a safe batch-registration command. Removed the manifest builder's fixed 32-mask approval invariant and its dependency on the older WHO cohort's reviewed-mask snapshot.
- **Files/artifacts:** `src/register_local_annotation_reviews.py`, `src/local_annotation_manifest.py`, `tests/test_local_annotation_manifest.py`; ignored registration inputs and 118 newly created eyelid masks under `data/raw/local/annotations/`; updated annotation workflow documentation.
- **Result:** all 140 local eyelid ROIs are registered as approved; 118 masks were created and 22 pixel-identical existing masks reused. Ten pre-existing approved nail masks were preserved. The canonical queue now contains 125 pending nail images and no pending eyelids. All 275 frozen image references remain uniquely linked to their originals and annotation TIFFs.
- **Limitations/blockers:** contact-sheet approval is anatomical boundary review, not clinician validation. No glare ROI was approved or created by this step; glare remains separate. The dated 2026-10-08 WHO cohort snapshot still describes its earlier 32-mask inventory and must be regenerated to use current mask counts.
- **Verification:** all 140 ROIs passed rasterizer dry-run; final canonical masks matched the reviewed mask pixels. The canonical builder validated 150 approved ROI/mask pairs and regenerated the manifest, queue, and audit. The read-only local data audit returned `PASS_WITH_MANUAL_REVIEW`, with 275/275 image references resolved and no blocking metadata, duplicate, or Hb-unit issues. No model training was run.
- **Next action:** validate the public Eyes Defy Anemia palpebral image-mask pairs and run a public U-Net baseline only if the pairs pass the documented alignment checks; keep local eyelid masks for later local evaluation/fine-tuning readiness.

## 2026-10-09 — Public eyelid U-Net local development check

- **Area:** local anatomical segmentation transfer evaluation.
- **What changed / why:** ran image-only inference with the existing Eyes Defy Anemia U-Net checkpoint against the 111 approved local development references (80 train subjects and 31 validation subjects) after the 140 manual eyelid masks were registered. Full-frame preprocessing was used; no local labels were used for training or prompting.
- **Files/artifacts:** `src/local_segmentation_pilot.py`; private outputs under `data/processed/segmentation/local_unet_dev_eval_fullframe_2026-10-09/` with predictions, overlays, hashes, split ledger, and run summary.
- **Result:** mean Dice 0.0027, IoU 0.0014, precision 0.0021, recall 0.0075; 110/111 predictions triggered one or more QC flags. Mean inference time was 0.073 seconds per image on CPU. This is a local segmentation comparison, not anemia-screening performance.
- **Limitations/blockers:** the public-only model does not transfer to local full-frame eyelid images well enough for unattended use. The 29 local test subjects were not scored in this development run and remain reserved for final comparison after model selection. LAB color difference is an uncalibrated rendering-dependent diagnostic. Automated outputs remain unverified and did not alter the approved manual masks.
- **Next action:** implement the local SAM 2.1 prompt-and-correction assistant for eyelid proposals, then train a conservative local fine-tune using only the 80 train and 31 validation subjects; compare public-only and fine-tuned checkpoints once on the reserved 29-subject test set.

## 2026-10-10 — Local manual-eyelid classifier prepared; duplicate review gate remains

- **Area:** local eyelid-only anemia classification readiness and implementation. No nail, multimodal, automatic segmentation, public-training, transfer-learning, or deployment work was performed.
- **Owner decisions:** the owner delegated rendering and model choices, and explicitly chose all 119 eligible labeled subjects for classifier development after the existing 29-subject reservation was explained. Under that delegation, selected an exploratory current-TIFF/manual-mask LAB policy that retains glare, and fixed L2 logistic regression plus a majority-class baseline. The classifier uses five subject folds, seed `20261010`, with no tuning and no final classifier holdout. The segmentation split ledger was preserved. The 0.5 classification rule has research scope only; no clinical operating threshold was approved.
- **Fresh audit:** canonical metadata mtime `2026-10-07T14:55:39.254357+08:00`, SHA-256 `1e704fb8d575757f16fc4f690cdc3315bc4063a5cf258d6c2e1b5f784d19596d`; provenance and preserved original Drive export checked. Recomputed 140 subjects, 120 with numeric Hb, 119 eligible labeled eyelids (19 anemia/100 non-anemia), and 21 exclusions (20 without eligible laboratory Hb, one outside configured age range). All 275 image references resolve uniquely; no schema, duplicate subject-ID, Hb-unit, prior frozen-manifest, or label discrepancy was found. Non-pregnancy remains the existing owner attestation outside the Sheet.
- **Input integrity:** all 140 approved eyelids passed 560 original/TIFF/ROI/mask hash checks, human-register linkage, same-size nonempty binary mask checks, and equality with reviewed-preview mask pixels. Structural checks and owner anatomical approval are not clinician validation. Source and decoded-TIFF exact-duplicate groups were zero. Recomputed all 46 full-frame pHash candidates; 27 cross the old ten-fold table, 34 cross segmentation partitions, and 28 cross the newly prepared classifier folds. None was declared a confirmed duplicate or reassigned.
- **Rendering evidence:** histories comprise 99 Photos exports, 18 Preview/ColorSync renders, 11 embedded DNG-preview copies, and 12 pilot exports without recorded tool/settings. The 11 addendum TIFFs matched the oriented DNG preview pixels but had different ICC-profile tags. Existing TIFFs are retained unchanged; LAB features interpret stored RGB as sRGB for an explicitly uncalibrated exploratory conversion. No normalization, enhancement, resize, background filling, glare exclusion, inpainting, or anatomical-mask modification was performed.
- **Implemented:** `src/local_eyelid_validation.py`, `src/local_eyelid_readiness.py`, `src/local_eyelid_classifier.py`, and synthetic `tests/test_local_eyelid_classifier.py`; documented in `docs/experiments/local-eyelid-classifier.md` and `docs/core/ARCHITECTURE.md`. The new readiness path replaces stale annotation-count assumptions for this experiment without editing the old cohort builder, label configuration, source data, or generated annotation manifests. Preparation generated six masked LAB features per eligible subject. Folds 1–4 each contain 4 anemia/20 non-anemia subjects; fold 5 contains 3/20. Training uses fold-only StandardScaler and fixed logistic regression, with no validation-based tuning; metrics/conditional uncertainty code is implemented and synthetic-tested.
- **Artifacts:** ignored `data/processed/local_eyelid_readiness_2026-10-10_v2/` contains eligibility, source-linked inputs, features, folds, original label configuration, software/code/source hashes, and a blank hash-linked human duplicate-review form. The earlier `v1` preparation/check snapshot was retained. Patient rows and image-derived features remain local and ignored.
- **Verification/result:** 31 relevant synthetic tests passed, including schema/hash failures, Hb preservation, image-mask linkage, deterministic masked features, subject-level splitting, class support, fold-only scaling, complete synthetic CV, metrics/conditional intervals, and duplicate-review fail-closed behavior. `python -m compileall src` and whitespace checks passed. The real-data audit/preparation passed; the attempted fitting smoke check returned `BLOCKED` before any local model/output directory was created. No local anemia model was trained or evaluated, and no local performance metrics, patient predictions, commit, push, or deployment were produced.
- **Remaining blocker/next action:** owner review of the existing 46-pair comparison PDF and completion of `duplicate_review.csv`. Distinct-source adjudication for every candidate can clear the gate. Any uncertain or confirmed duplicate requires an explicit identity/grouping decision and revised approved partitions. After that gate passes, run the prespecified local internal CV once; no external or clinical-performance claim follows from this cohort.

## 2026-10-10 — Owner duplicate confirmation; local manual-eyelid internal CV completed

- **Owner confirmation:** the owner stated in this chat, “they all look good to me. none are duplicate. update anything and mark it as confirmed and proceed.” Recorded all 46 existing pHash pairs as `distinct_sources`, with owner attribution/date/statement and exact candidate/source/TIFF hashes in `duplicate_review_confirmed_2026-10-10.csv`. The adjacent confirmation JSON records the candidate CSV and comparison PDF hashes. This is a recorded human decision, not a new automated identity determination or clinician validation. The original blank form and historical audits/configurations were preserved; no subject/image was removed, merged or reassigned.
- **Readiness:** READY for the previously fixed exploratory experiment. Verified canonical metadata exists and recorded mtime `2026-10-07T14:55:39.254357+08:00` and SHA-256 `1e704fb8d575757f16fc4f690cdc3315bc4063a5cf258d6c2e1b5f784d19596d` before reading rows; frozen provenance and original Drive export passed. Fresh audit recomputed 140 subjects, 119 labeled eyelids (19 anemia/100 non-anemia), 21 exclusions, 275 resolved image references, all 140 approved eyelids, 560 original/TIFF/ROI/mask hash checks, reviewed-preview mask equality and zero exact source/pixel duplicates. Fresh inputs matched the prepared CSV after normalizing CSV scalar serialization; code, label policy, clinical values, source files and five-fold assignments were unchanged. Saved `pretraining_reaudit_2026-10-10.json` and the separate `gate_release_2026-10-10.json` in the existing private v2 readiness bundle.
- **Run:** fitted the prespecified six-feature LAB logistic regression and majority baseline across five subject folds, seed `20261010`, training-fold StandardScaler, L2/C=1/no class weights, no tuning and research threshold 0.5. Folds 1–4 validate on 4 anemia/20 non-anemia and train on 15/80; fold 5 validates on 3/20 and trains on 16/80. Every subject received one OOF prediction from a model that excluded that subject from training. The segmentation ledger was preserved; all 119 remain classifier-development subjects, with no untouched classifier holdout.
- **Actual pooled LAB results:** accuracy 0.848739, sensitivity 0.105263, specificity 0.990000, precision 0.666667, F1 0.181818, ROC-AUC 0.719474, AP 0.387866, trapezoidal PR-AUC 0.375017. Confusion matrix ordered non-anemia/anemia: `[[99, 1], [17, 2]]`. Detected 2/19 positive subjects and missed 17 at the fixed threshold. Sensitivity Wilson 95% interval 0.029359–0.313941; conditional bootstrap ROC-AUC interval 0.575789–0.838961 and AP interval 0.248723–0.595255. Fold ROC-AUC ranged 0.425–0.950.
- **Actual majority results:** accuracy 0.840336, sensitivity 0, specificity 1, precision undefined (no positive predictions), F1 0, ROC-AUC 0.5, AP 0.159664, trapezoidal PR-AUC 0.579832; matrix `[[100, 0], [19, 0]]`. Its trapezoidal PR area reflects interpolation to the precision=1 endpoint, not useful discrimination. Accuracy improves by only 1/119 net correct subject in the LAB run; there is no claim of statistical superiority or clinical screening readiness.
- **Artifacts:** ignored `data/processed/local_eyelid_classification_2026-10-10_v1/` contains ten fold models, 119 private OOF prediction rows, copied folds/review/configuration/source evidence, pooled/per-fold metrics and intervals, execution logs, PNG/SVG aggregate figures, `report.md`, `verification.json` and `run_manifest.json`. Current completion is recorded separately from the historical pending status retained in copied preparation files. No full-cohort deployment model was fitted.
- **Verification:** 31 synthetic tests and `python -m compileall src` passed. Post-run checks verified source/review hashes, unique label/fold coverage, fixed parameters and convergence, training-only scaler statistics, exact reproduction of every OOF probability from the ten saved models, and independent pooled metric recalculation. Documentation/code whitespace checks are recorded in the final verification artifact. No commit, push or deployment was created.
- **Limits/next step:** no further owner review is needed for this completed fixed run. Mixed uncalibrated TIFF rendering, only 19 positive cases, fold variability and conditional OOF intervals limit interpretation; bootstrap intervals fix prevalence and omit model-refit/overlapping-training uncertainty. No external/prospective validation or clinical threshold exists. No threshold, fold or model was changed from the results. Next proposed study: establish a consistent rendering protocol and increase positive-case support before approving a new experiment.

## 2026-10-11 — Project documentation reorganized

- **Area:** documentation structure and navigation only.
- **What changed / why:** retained `README.md` and `AGENTS.md` at the repository root and grouped the remaining shared documents under `docs/core/`, `docs/models/`, `docs/workflows/`, `docs/experiments/`, `docs/audits/`, and `docs/policies/`. Converted the former reference index into `docs/README.md` and updated internal references.
- **Preserved provenance:** historical annotation guides and local audit records remain beside their source data under `data/raw/local/`; no research data, code, model artifacts, methodology, or results changed.
- **Status correction:** refreshed the PRD header to acknowledge the completed 2026-10-10 local manual-eyelid internal cross-validation while retaining its non-clinical, non-external-validation scope.

## 2026-10-11 — Feature commits and Experiment 1 preservation tooling

- Completed eight coherent commits for previously implemented data, annotation, rendering, segmentation, public evaluation, local CIELAB classification and documentation work. The completed Q&A documentation changes were committed after explicit owner approval; its separate worktree documents remain untouched.
- Added a private archive creator and read-only verifier that reproduce ten saved fold models without fitting. Historical output/readiness hashes remain unchanged. Archive creation and verification are required before Experiment 2 begins; the archive procedure and cleanup commit list are documented in `docs/experiments/eyelid-exp01-preservation.md`.
- Validation: 51 synthetic tests, source compilation, fresh 560-file data audit and original saved-model reproduction passed. No private data were committed or uploaded.

## 2026-10-11 — Experiment 1 archived; isolated hybrid feature path implemented

- Verified the separate owner-only Experiment 1 archive: 97 files, all checksums valid, all ten models reproduced without refitting, original output/readiness bytes unchanged. Preservation branch/tag resolve to `4bf6e2577d02b5e3338e1937140b7eaff31369b4`. Created native linked worktree `../aima-screening-exp02-hybrid` on `experiment/eyelid-exp02-hybrid` from that milestone. No large data were duplicated.
- Implemented frozen ImageNet1K_V1 manual-mask MobileNet embeddings and exact baseline LAB reuse, explicit external source-root and exclusive ignored output guards. Documented the prespecified two-candidate nested design before any hybrid training/outer scoring. No local neural fitting, nails, automated masks or public anemia data are involved.

## 2026-10-11 — Nested hybrid training and paired evaluation implemented

- Added fold-local embedding scaling/eight-component PCA and conservative LightGBM, with exactly two class-weight settings chosen by three-fold inner OOF AP. Selected research thresholds use inner scores only (maximum sensitivity at specificity ≥0.80); the selection API does not accept outer validation labels. No outer early stopping or retuning is used.
- Added fixed/fold-selected metrics and conditional paired stratified bootstrap comparisons against the unchanged baseline. Synthetic tests check training-only statistics, partition exclusivity, model-loading equality, selection invariance to outer features, threshold ties and hand-calculated metrics. Real hybrid training remains pending the artifact runner.

## 2026-10-11 — Private hybrid run and reproduction tooling verified

- Added exclusive ignored owner-only run output, canonical-source snapshot, code/environment/weight provenance, all 35 pipelines, inner decisions, OOF rows, metrics/intervals, aggregate figures/report and complete checksums. Sources and Experiment 1 are read-only and rechecked after execution.
- Added checksum-first offline reproduction of every inner/outer pipeline, training-only scaler/PCA statistics, selected candidate/threshold and pooled/per-fold metrics. A complete synthetic 119-subject/19-positive fixture verifies all 35 pipelines and rejects tampered artifacts; output guards reject baseline/unignored/existing directories. The first guard assertion exposed macOS temporary-path canonicalization; the test now compares resolved paths. No research method changed.
- Real training has not run at this implementation milestone; the new runner requires a clean committed code tree before execution.

## 2026-10-11 — Standalone hybrid reproduction runtime corrected

- The full hybrid run and in-process verification completed. Fresh-process verification crashed with default native threading; matching training's one-thread policy resolved it. Regression tests and standalone source-linked reproduction of all 35 pipelines now pass.
- The original copied implementation also reproduced with the documented thread environment settings. No model/artifact changes, refitting or outer-result retuning occurred.

## 2026-10-11 — Hybrid Experiment 2 completed and compared

- **Run:** one prespecified nested five-outer/three-inner subject CV on the original 119 eligible manual-eyelid records (19 anemia/100 non-anemia), exact baseline folds, unchanged LAB cache, frozen ImageNet1K_V1 MobileNetV3-Small, training-only embedding scaling/PCA8 and two conservative LightGBM weighting candidates. No automatic masks, nails, local neural fitting, public anemia data or outer-result tuning. Training-code commit `01cb856842a637ff8f4ba3ece246e18e904975d7`.
- **Actual fixed-0.5 result:** accuracy 0.756303, sensitivity 0.263158, specificity 0.85, balanced accuracy 0.556579, precision 0.25, F1 0.256410, ROC-AUC 0.575789, AP 0.247161; matrix [[85,15],[14,5]]. Inner-selected thresholds yielded accuracy 0.739496, sensitivity 0.368421, specificity 0.81, balanced accuracy 0.589211, precision 0.269231, F1 0.311111; matrix [[81,19],[12,7]], with unchanged ranking metrics.
- **Comparison/uncertainty:** detection rose from 2/19 to 5/19 or 7/19, with more false positives and lower discrimination. Fixed-rule paired sensitivity delta interval includes zero (−0.052632 to +0.368421); selected-rule interval is +0.052632 to +0.526316 but changes the operating rule. Both balanced-accuracy delta intervals include zero. Fold variability, mixed rendering, uncalibrated weighted scores and 19 positives limit interpretation. No superiority/clinical/generalization claim is justified.
- **Artifacts/verification:** separate ignored owner-only `data/processed/local_eyelid_classification_exp02_hybrid_2026-10-11_v1/` holds 35 pipelines, all selections/assignments, 119 OOF rows, configuration, embeddings/public weights/provenance, report/figures/logs and 86 checksum-covered files plus manifest. Every saved pipeline/inner decision/OOF score and training-only scaler/PCA statistics reproduce. Source-linked standalone verification, independent sklearn metrics and exact embedding recomputation for one real subject per fold passed; 493 source hashes were rechecked. All 41 initial Experiment 1/readiness hashes remain unchanged. No patient images/ROIs/masks were copied or sensitive files committed/uploaded.
- **Documentation:** corrected two remaining stale PRD status statements against verified results; product requirements and clinical acceptance targets are unchanged. See `docs/experiments/eyelid-exp02-results-2026-10-11.md` and the complete Git/file handoff under `docs/audits/`.
- **Next:** harmonize acquisition/rendering, increase positive-case support, obtain new untouched validation data and prespecify ablations/calibration. A separate approved public-anemia transfer design is still required; no transfer training was run.

## 2026-10-11 — Experiment 3 isolated input implementation

Created native `experiment/eyelid-exp03-public-transfer` worktree from verified `f4b2cb5`. Reproduced Experiment 1 archive and Experiment 2 checkpoints without refitting. Public audit found 218 palpebral pairs, 19 age exclusions, one missing Hb and 15 public pregnancy-dependent labels; conservative harmonization leaves 183. Owner reconfirmed non-pregnancy for local AIMA; its 119 labels and folds are unchanged. Added source hashing, manual-mask registration, real ICC conversion and conservative grouped public splits. A pre-training registration check caught an EXIF mismatch in the first preparation attempt; that attempt is preserved as invalid and corrected preparation uses a new v2 output. Three input tests pass. No full training or performance claim yet.

Implemented Experiment 3 neural training with train-only class weights, conservative late-block adaptation, public/inner validation stopping, deterministic prefix caches and guarded checkpoint resume. Six data/neural tests pass, including actual two-epoch smoke training and exact saved prediction reproduction. Full public and local training remain pending.

Corrected Experiment 3 v2 input audit passed: public183 (64 anemia/119 non-anemia), train146/validation37, local119 (19/100), no exact or cross-domain duplicate candidates, 11 pixel-identical Display P3 preview conversions. Implemented nested local strategy/threshold selection and a separate LAB threshold comparator; outer labels are excluded from training APIs. Ten new tests pass. Full suite/compilation precede the first prespecified training run.

Pre-training split QA found four legacy public pHash candidates absent from the aligned-raster screen, including one crossing the initial public split. Stopped and checksum-preserved v2 before comparison. Corrected v3 groups both screens (public train146:54/92; validation37:10/27), reusing only deterministic input tensors and frozen ImageNet prefix maps. Added a regression gate/test; no outer evaluation metrics guided this correction.

Post-training reproduction QA flagged a 1.58e-6 float32 difference from differing convolution batch geometry. Added a separate exact-prefix reproduction entrypoint while preserving the recorded training source and sealed artifacts. Fresh public/local prefix maps (both orientations), all 41 neural/20 LAB checkpoint predictions, selection records, frozen parameters and five local source tensors match exactly; independent sklearn checks match all 72 pooled/per-fold metric sets. Thirteen Experiment 3 tests pass. Final interpretation follows in the results entry.

## 2026-10-11 — Experiment 3 completed and verified

Accepted v3 run completed public training (183 subjects, best epoch 8/12), both local strategies across exact original 119-subject folds, inner-selected A,A,B,B,A, public-only evaluation and a separate nested LAB threshold analysis. Public-only fixed sensitivity 0%/specificity 93%, ROC-AUC0.451/AP0.141. A fixed 21.1%/59%, selected 5.3%/82%; B fixed 21.1%/58%, selected 5.3%/84%. Inner-selected fixed 15.8%/58%, selected 5.3%/84%, ROC-AUC0.464/AP0.144. Nested LAB 36.8%/84%, ROC-AUC0.719/AP0.388, with original outer probabilities reproduced. Selected transfer minus nested LAB sensitivity−31.6 percentage points (conditional paired 95% interval−52.6 to−10.5). This tested transfer design did not improve detection; prioritize simpler LAB investigation with more anemia cases and independent data. Preprocessing/domain differences prevent isolating the effect of public pretraining. All 41 neural/20 LAB models, 216 files, fresh exact prefix maps/five local source tensors and 72 independent metric sets verified without refitting. Training/evaluation invocation 206.8s CPU; originals/checkpoints/archive references unchanged. See the complete results report; no clinical or external-test claim.


## 2026-10-12 — Experiment 4 diagnostic and color input implementation

Started on 2026-10-11 with owner authorization; implementation continues on 2026-10-12. Reproduced Experiments 1–3 from saved models without fitting, freshly checked 560 source/TIFF/ROI/mask hashes and canonical metadata, and verified unchanged 119-subject membership (19 anemia/100 non-anemia) and owner duplicate clearance. The private preflight diagnostic found 12 LAB nested-threshold false negatives: eight mild and four moderate under descriptive unadjusted WHO adult severity. Their median Hb is 11.55 g/dL versus 10.8 in seven true positives; native focus proxies have nearly equal medians, so blur is not established as the cause. Eleven eligible TIFFs have verified pixel-identical Display P3 DNG previews under misleading sRGB tags; none of the twelve missed subjects receives this profile correction. Source/export histories and incomplete calibration remain limitations. Created `experiment/eyelid-exp04-color-optimization` and `../aima-screening-exp04-color` from verified `66afa8e`. Added native masked original/managed color features, private diagnostics and the bounded experiment contract; primary full fitting remains pending. No original labels, masks, data or saved experiments changed.

The bounded nested classifier implementation is complete: 36 rendering/feature/model candidates, inner-only AP and decision thresholds, and 720 planned inner/outer fold pipelines. Ten synthetic color/model checks pass, including actual non-sRGB transformation, masked feature identity, training-only scaler fitting, threshold fallback and outer-feature independence. Corrected a generated blank-line whitespace issue before freezing the training implementation. Full classification remains pending.

Experiment4 artifact execution and independent saved-model verification are implemented. Full suite:87 tests run,84 passed,3 skipped because isolated worktree raw fixtures are absent; source compilation and whitespace checks pass. Synthetic nested fitting exercises all36 candidates and saved pipelines; checksum tampering prevents deserialization. Exact private output/version boundaries and independent metrics are tested. Config/source/code hashes will be frozen before the real classification run; optional fixed Ridge regression runs only after classification.

The accepted Experiment4 v1 full run completed under `2a0f971` in219.1 seconds for source features/training/evaluation, saving720 classifier and10 regression models with810 checksum-covered files. Initial fresh verification caught object-dtype sex metadata rejected by the numeric-only loader. Added an independent checksum-first legacy-string verifier and Unicode serialization for future runs; recorded implementation, source artifacts and every fitted model remain immutable. Verification is pending at this correction commit.


## 2026-10-12 — Experiment4 completed and independently verified

Joint inner-selected original-feature policies A,A,B,A,A detected7/19 at83% specificity (7TP/12FN/83TN/17FP), AUC.656842/AP.352228; no sensitivity improvement. Predefined matched original/robust18/unweighted C1 logistic comparison detected10/19 at84% (10TP/9FN/84TN/16FP), AUC.709474/AP.385148; recovered3 original false negatives and retained all7 original true positives. Conditional paired sensitivity gain15.8 points, interval0–31.6, and the managed robust counterpart7/19 leave this promising but unconfirmed. Joint70% rule12/19 at72%; unchanged original6/C1 under the same predefined70% inner rule13/19 at73%, showing threshold tradeoff rather than a new model advantage. All selected primary folds chose original6 features by inner AP. Fold5 still misses all3 positives in the primary and robust matched rules.

Completed separate fixed Ridge Hb feasibility after classification:MAE1.327386 g/dL, RMSE1.702216, R².134090; training-mean MAE1.375021. Corrected independent verification reproduced720 classifier/10 regression pipelines, all inner selections/cutoffs and pooled/per-fold metrics, checked810 sealed files and freshly re-extracted every source feature exactly without fitting. Original Experiments1–3, source labels/masks and preservation references unchanged. Final90-test suite87 pass/3 unavailable-data skips; compilation and whitespace checks pass. Initial object-array metadata failure is resolved by the independent verified-string reader, preserving the sealed output and recorded training commit2a0f971. Private proof/corrected guide and matched-rule analysis stay in the separate verification bundle. Added current registry, aggregate results and model card; no patient artifact/model/clinical row committed or uploaded. Further confirmation needs new independent subjects and consistent acquisition, not retuning this completed experiment on its outer predictions.

## 2026-10-12 - Experiment4B proposal/review implementation

Owner requested isolated glare, geometry-relative regional color and stability follow-up. Fresh canonical hashes match; cohort audit rechecked560 source/image/ROI/mask hashes,140 approved anatomy pairs and119 eligible19/100 labels. All46 similarity candidates retain owner distinct-source clearance; the reused auditor's historical duplicate-pending wording does not invalidate that separate clearance. Experiment4 reproduced720 classification/10 regression models, all810 covered files and every freshly extracted feature exactly without fitting. Created requested Experiment4B branch/worktree from0b50504; main's newer documentation consolidation is preserved separately. Existing glare audit:22 unvalidated/unreviewed candidates across140, zero accepted glare/no-glare records. Implemented conservative proposals and private review materials; no human approval or exclusion model is claimed.

Implemented controlled regional features and private audit/proposal preparation. Original robust18 remains exact; two distance regions use the original manual anatomy, four median contrasts are flip/90-degree-rotation invariant, and glare-exclusion extraction handles empty proposals without approval. Readiness fails closed on missing hash-linked glare reviews. Synthetic tests cover conversion/feature identity, exclusion, geometry, empty regions, overlap and private output guards. Full-cohort preparation and anatomical-regional fitting follow this verified implementation milestone.

First proposal-preparation invocation stopped before creating its output because fresh JSON labels were integers and preserved CSV labels strings. Corrected representation comparison without weakening source hash or membership checks; added a regression test rejecting changed labels/hashes. Failed log preserved privately; no source/mask/clinical value changed. Preparation will retry without overwriting an attempted artifact directory.

The bounded anatomy-only regional classifier and independent reproduction path are implemented. Seventeen focused synthetic tests pass, including nested subject separation, training-only scaler statistics, outer-feature-independent selection, exact saved predictions, metric checks and checksum tampering before model loading. Four candidates reuse unchanged Experiment4 C1 pipelines; no proposed glare pixels enter training. Full suite and preparation quality audit precede fitting.

Pre-fitting full suite:107 tests,104 passed/3 skipped because isolated worktree raw fixtures are unavailable. Source compilation and whitespace checks pass. The skipped fixtures do not replace the separate real-source readiness audit.

The first80-model fitting attempt stopped before evaluation/sealing at a comparator check: pandas CSV parsing rounded one original cutoff by5.55e-17. Every robust probability and allfive full decision-rule dictionaries already match Experiment4 binary/JSON evidence exactly. Corrected the comparator to authoritative NPZ predictions and JSON thresholds, with a regression test rejecting even a one-ULP threshold change. Preserve the stoppedv1 attempt; copy only verified deterministic preparation assets to a separatev2 and refit unchanged candidates. No threshold/model policy was changed to improve metrics.

Anatomy-only4Bv2 completed and independently reproduced80 checkpoints,438 covered files, all decisions/metrics/intervals/stability and119 fresh native masks/features. Regional unweighted9/19 at82%, weighted8/19 at83%, nested9/19 at81%; unchanged robust10/19 at84% remains better at the selected rules. Owner then reviewed the proposal package and replied “they all look good to me.” to the all119 review question. Recorded96 glare/23 no-glare acceptances in a separate hash-bound register, retaining14 prior uncertainty flags. Strict reviewed subset equals full cohort; added the four glare-excluded counterparts with unchanged parameters/folds/inner rules. Fresh feature validation and160-model comparison follow; no new exclusion performance is claimed yet.

## 2026-10-12 - Experiment4B owner-reviewed comparison completed

Accepted hash-linked96 glare/23 no-glare records; all119 reviewed, strict subset equals full cohort19/100. Fresh mask/source/feature checks passed. Eight predefined candidates on unchanged nested folds saved160 models under1388ec7; masked robust unweighted10/19 at83%, weighted10/19 at84%, regional unweighted9/19 at79%, weighted8/19 at79%; full-eight inner-selected9/19 at78%, AUC.667/AP.356. Original robust10/19 at84% retains exactly the same positive decisions; masking adds one false positive for its unweighted counterpart. No sensitivity improvement demonstrated. Fold5 still misses all3 positives under every training-selected policy. Mean/SD perturbations are small and many robust median/percentile changes zero; geometry-relative features did not improve the operating point. All160 checkpoints,222 covered files, partitions/inner choices/cutoffs, independent sklearn metrics/conditional intervals/stability reproduce without fitting. The separate anatomy-onlyv2 retains80 reproduced models/438 files/119 fresh masks/features; stoppedv1 and all failures remain preserved.111 tests108 passed/3 absent-worktree-fixture skips, compilation and whitespace pass. User review gate is cleared; no remaining modeling/review blocker for this frozen experiment. Independent validation, detector-accuracy reference and capture standardization remain future research. All originals, prior runs and review hashes unchanged; no private artifact/model/image/clinical row committed.

## 2026-10-12 - Experiment5 CP transfer initialized

Owner requested public pediatric CP classification/Hb pretraining, three local adaptations, unchanged subject folds and centralized thesis records, with no external push. Verified all reported1-4B benchmark metrics, preserved eight source/artifact bundles and freshly rechecked local11919/100 cohort/560 hashes and owner distinct-source clearance. Created exact requested5 branch/worktree fromfc64eed. Reused committed main research records, correcting provisional4B/5 placeholders from verified evidence. CP local V2 initially has251 unique RGBA images110/141, Hb4.2-15.0, alpha edges and no ICC/EXIF; official archives/licensing/grouping remain under audit. Public file transfer timed out; all unaffected implementation proceeds, no training or new performance claim yet.

Authenticated official CP archives after resumable public-file transfer; local V2 matches503 official files exactly. V1 contains710 images,498 file hashes/485 rasters,87 repeated-image groups with86 conflicting-Hb groups and one cross-label group. Chose V2 under owner-approved fallback:251 canonical unique RGBA images110/141, no folder/Hb mismatch; every V2 source ID preserves original Hb and no V2 image belongs to an exact conflicting V1 group. Individual public patient/age/sex linkage absent;251 curation IDs plus5 near pairs give246 conservative public split groups, never asserted as251 verified patients. Implemented official release/schema/alias/alpha and source/cohort/domain preparation with configurable paths and private exclusive outputs. Four synthetic data tests, compilation and whitespace pass. Public/adult labels remain separate; no model fitting/performance claim yet.

Public regression/adult-head/frozen-fusion implementation completed. Seven focused tests pass, including actual two-epoch regression, exact checkpoint predictions, frozen prefix parameters, training-only Hb normalization, guarded changed-target resume, explicit new adult heads and train-only PCA/scalers. Matched nine-candidate design is frozen before any real fitting; refined inner seed to prior20261011 for comparable thresholds, retaining public/training seed20261012. No outer CP results motivated any design choice. Full preparation/source diagnostics follow; no new performance claim yet.

- 2026-10-12 — Experiment5 input preparation passed: authenticated V2 has251 unique images/110 anemia/141 non-anemia,246 conservative public split groups; public training199 and validation52 (20 anemia/32 non-anemia). The exact119-subject local cohort, all source/manual-mask checks and original robust18 color features passed, with no screened cross-domain duplicate candidates. The private common-input sheet was inspected. Implemented bounded nested runner, paired comparator and independent reproducer; full model fitting remains pending. Unknown public patient linkage remains explicit. All previous artifacts are unchanged.

## 2026-10-13 — Experiment5 completed/reproduced

CP-AnemiC pretraining did not establish an improvement that justifies its added complexity on AIMA. The predefined joint policy detected **11/19 anemia cases (57.9% sensitivity) at 78% specificity**, compared with **10/19 (52.6%) at 84% specificity** for the predefined Experiment 4 robust color comparison. One additional detection came with six additional false positives. Balanced accuracy was slightly lower (67.95% versus 68.32%), ROC-AUC was 0.734211 versus 0.709474, and AP was 0.372546 versus 0.385148. The paired sensitivity difference had a conditional 95% interval of −21.1 to +31.6 percentage points; the specificity difference was −6.0 points [−11.0, −2.0]. This does not establish independent or clinically acceptable improvement.

The predefined CP-classification frozen-feature plus color candidate C detected **12/19 (63.2%) at 78% specificity**, with ROC-AUC 0.734211 and AP 0.372166. Its sensitivity 95% Wilson interval was 41.0–80.9%; specificity was 68.9–85.0%. Against the robust color comparator, its +10.5-point sensitivity difference had an interval of [−10.5, +36.8], while specificity fell by 6.0 points [−11.0, −2.0]. This is a predefined exploratory candidate comparison, not a replacement primary policy chosen from outer results. CP classification's actual inner-selected A/B/C policy detected 10/19 at 77% specificity; it selected C in four folds and B in one.

CP Hb-regression C and its inner-selected policy detected **11/19 at 79% specificity** (ROC-AUC 0.730000; AP 0.370429), with exactly the same binary OOF decisions as matched ImageNet C (11/19 at 79%; ROC-AUC 0.730526; AP 0.368331). CP classification C added one detection and one false positive versus matched ImageNet C; its paired ROC-AUC and AP intervals included zero. The frozen-feature performance is therefore compatible with ImageNet features plus robust color statistics without useful added CP pretraining. Head-only and partial CP adaptations with training-selected thresholds detected only 2/19 and 0/19 for classification, or 4/19 and 0/19 for regression. These outer results did not guide any redesign.

The public-only pediatric classifier detected **0/19 at 96% specificity** (ROC-AUC 0.459474; AP 0.150111). Public classification validation was weak (ROC-AUC 0.478125), and both bounded public objectives selected epoch 1. Public-only Hb prediction on AIMA had MAE 2.225671 g/dL, bias −1.905735 g/dL and R² −1.069180. Although its MAE was below the public-training-mean baseline of 2.801160, substantial adult underprediction and negative R² provide no evidence of a usable Hb estimator. Weak public fitting and observed acquisition/label differences are plausible explanations, not proven causes.

Retain robust color-only classification as the practical candidate for independent confirmation, with a prespecified acquisition, model and operating-rule policy and genuinely new positive and borderline cases. Frozen embeddings can remain a predefined comparator in that study. These repeated-cohort results do not justify additional CP fine-tuning complexity or choosing a new winner by outer sensitivity. The inner specificity constraint of 0.80 was a research threshold policy and did not guarantee 0.80 outer specificity. No clinical acceptability threshold or independent validation has been established.


Training `cf735ed` completed2 public,120 local neural and60 classical models. Independent checksum-first verification reproduced all182 models, full370 fresh inputs/two-view early maps, train-only statistics/heads/decisions and all pooled/per-fold metrics/intervals.8 prior experiment/review bundles, source masks/images/labels and preservation refs remain unchanged. Full suite122 tests:119 passed/3 legacy-source skips; compilation/whitespace/documentation checks passed. Source/model/metric/reproduction records remain private under the versioned E5 and verification directories; aggregate thesis records updated. No push. See [verified results](../experiments/eyelid-exp05-results-2026-10-13.md).

## 2026-10-13 — Experiment 6 complementarity and bounded ensemble results

Owner requested read-only overlap first, isolated branch/worktree and targeted eyelid-information review. Created requested6 worktree from verified 5 handoff0943363; previous research refs, all earlier worktrees and main remain unchanged. Fresh E4/E5 full reproduction and source/cohort audit passed;119 subjects 19/100 and folds/labels align. Robust-LAB/CP-C overlap is8 shared detections,2 LAB-only,4 CP-only,5 shared misses;16 shared and 6 CP-only false positives. Pearson0.801841, agreement107/119 justified a small exploratory test.

Equal averaging at training specificity≥0.80 detects 11/19 at 79%, AUC0.737895/AP0.379070; weighted selection10/19 at 79%, AUC0.731579/AP0.371157. Against robust10/19 at 84%, equal adds1 detection/5 false positives; weighted adds0/5. Ranking intervals span zero; no convincing complexity benefit. Chosen LAB weights0.50,0.50,0.25,0.25,0.25 come entirely from matched inner scores. No base refitting, stacking, calibration or new datasets. The selected base models and decision gate followed prior outcomes, so no independent selection or clinical-validation claim.

Reviewed119 private anatomy contexts: structural checks pass, but tissue extent/orientation and hair/eyelash handling vary. No masks/labels changed. Shared misses3 mild/2 moderate; repeated-photo consistency unavailable because only one verified eyelid original per subject and 154 files unassigned. Mixed acquisition/color/clipping remains unresolved; previous4B failures do not justify new glare fitting.

136 tests completed 133 passed/3 legacy-source skips; compileall/whitespace passed.109-file sealed private bundle and 40-pipeline/decision/prediction/metric/interval independent reproduction preserve every old source/result. Updated the existing six central research documents and validator on this new branch, including inherited completed 5 results. Protocol/helper commit bc116af precedes new outcomes; evaluator commit f501039 and audit commit 23b04d6 and the documentation commit preserve the verified work after direct in-chat approval resolved automatic review. No push. See [complete report](../experiments/eyelid-exp06-results-2026-10-13.md). Prefer robust color-only for a prespecified comparison on new positive/borderline subjects.


## 2026-10-14 — Experiment 7 preservation, negative results and reproduction

Owner authorized preservation-first normal-only anemia detection/public
transfer and eligible commits/push. Verified Exp1–6/4B refs and original
validators;2,717 historical files matched. Safe previously unsynchronized
experiment branches were pushed after outgoing privacy/remote checks, with
no main push/merge or changes to unrelated work. Created the requested7
branch from verified3b74755 in a retained managed worktree; froze340d3be
protocol before any new models.

Canonical clinical/source/approved-mask audit and exact119 robust18/fold
comparison passed. Eyes91/CP108 normal training groups support the bounded
public tracks with explicit anatomical/rendering/pediatric/label limitations.
Separate nail-workflow reset changed three mutable tables; v1 stopped before
fitting. Archived exact old hashes plus unchanged full140-eyelid rows and
fresh current audit allow strict v2 with identical arrays and design.

At primary A85 all six local models detect4/19 at85–88% specificity; robust
comparison remains10/19 at84%. Mahalanobis AUC/AP.537368/.258772;
SVM.10/scale.531579/.344039. Nested B85 detects3/19 at92%; Eyes normal
adaptation5/19 at88%. Direct public A gives19TP/100FP Eyes,12TP/77FP CP,
16TP/75FP combined, demonstrating calibration mismatch. No useful normal-only
or public benefit established, no outer winner selected, no clinical/fairness
claim. Preserve robust color as the practical independent-confirmation candidate.

Execution6513c9c independently reproduces289 models/87 policies, all train-only
statistics/inner decisions/OOF metrics/paired intervals/error analysis and119
fresh vectors without fitting; historical sources/artifacts remain intact.
151 tests:148 pass/three expected legacy-source skips; compilation/whitespace
pass. Cached-feature run19.369s,430,637,056-byte peak RSS. Completed full
report and six central research documents on this branch only; no protocol
manuscript/PRD change. Private accepted v2 and separate proof retained; backup
independence and measured acquisition/fairness remain limitations. See
[complete handoff](../experiments/eyelid-exp07-results.md).
