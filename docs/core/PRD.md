# AIMA Product Requirements Document

- **Status:** Research prototype; local manual-eyelid internal cross-validation is complete, with no external validation or deployment model.
- **Last reviewed:** 2026-10-11.
- **Authority:** Current project goals and approved requirements. See [architecture invariants](ARCHITECTURE_ESSENTIALS.md), [technical architecture](ARCHITECTURE.md), and [project log](PROJECT_LOG.md).

## Purpose and problem

AIMA explores whether smartphone images of the palpebral conjunctiva and fingernail/nail bed, interpreted with hemoglobin and relevant image-context data, can support anemia pre-screening. The research addresses the need for an accessible preliminary screening aid in community and rural-health settings. AIMA is not a blood test and is not a diagnostic replacement.

## Users and setting

- **Current users:** the AIMA research team, data curators, and developers building and evaluating an offline Python research pipeline.
- **Intended research setting:** image and clinical-reference data collected in community/rural primary-health settings in Zamboanga, Philippines, as described in the protocol.
- **Future intended operators:** trained health workers, if a later system is developed and validated. A user-facing mobile application or clinical workflow is not currently implemented or approved.

## Data and modalities

- **Public dataset:** the current loader uses segmented palpebral images and associated full-face JPEGs plus workbook metadata. It currently produces an eyelid-based baseline; it does not use fingernail images. Public and local cohorts remain separately identifiable.
- **Local dataset:** the authoritative local metadata is the frozen live-Sheet export under `data/raw/local/metadata/frozen_2026-10-07/`. Original local images remain in the ignored `data/raw/local/incoming/` archive. Current metadata references eyelid and nail images; ID-card images are for identity verification only and are never model inputs.
- **Palpebral conjunctiva:** the exposed inner lower-eyelid tissue is the eyelid target, not the eyelid skin, sclera, iris, or fornix beyond the visible exposed region.
- **Nail:** the visible nail plate is the nail target. Nail images are a distinct modality, not a substitute for eyelid images.
- **Annotations:** manually drawn, reviewed anatomical ROIs and their validated binary masks are the local segmentation ground truth. Glare is a separate QC/exclusion layer and is not an anatomical label.
- **Clinical reference:** use only verified laboratory Hb values and the explicitly versioned label policy. Preserve the original recorded Hb value and field; never infer a missing result.

## Research-prototype requirements

1. Keep public and local data provenance, cohort membership, and results distinguishable.
2. Preserve original local images and source filenames. Derived files must remain traceable to `subject_id`, modality, and original filename.
3. Support reproducible, documented preprocessing and color-feature extraction. The existing public baseline computes CIE L\*a\*b\* statistics; this does not imply calibrated color capture.
4. Treat eyelid-only, nail-only, and paired/multimodal experiments as separate comparisons. A missing nail image must not exclude an otherwise eligible subject from an eyelid-only experiment.
5. Use approved manual masks as the anatomical reference. Automated masks are proposals until validated against held-out, human-reviewed annotations; generated proposals must never silently replace reviewed annotations.
6. Preserve source Hb measurements and label provenance in every derived cohort. Do not train on missing, unverified, or invalid labels.
7. Keep every record from a subject in the same validation/test partition. Report cross-validation as cross-validation, not external-test performance.
8. Report class metrics and relevant subgroup results with cohort sizes and limitations. No clinical operating threshold or minimum performance threshold has been approved in this repository.
9. Keep the data workflow runnable offline on the research workstation where practical. Record environment, configuration, seeds, and artifacts for each run.

## Future deployment requirements (not implemented)

The protocol describes offline operation, local image processing, a screening-risk presentation, image-acquisition standardization, and potential use by health workers. Those are future design aims, not claims about a deployed product. Any deployment would require, at minimum, a fixed and validated acquisition/color workflow, usability and device testing, external and prospective clinical validation, calibration and subgroup assessment, clinical governance, and approved privacy/security controls. The repository does not currently demonstrate those controls or a mobile/edge deployment.

The protocol also discusses CNN transfer learning, YOLO-based ROI detection, LightGBM, a possible Hb regression output, FP16, and efficiency testing. These remain proposed unless a corresponding implementation and result are recorded in the architecture and project log.

## Model outputs

- **Implemented public baseline:** an out-of-fold anemia probability and a binary class at a 0.5 threshold. It is a research classification output only.
- **Local research output:** completed exploratory manual-eyelid LAB and frozen MobileNetV3/LAB/LightGBM internal subject cross-validation. These produce private OOF research scores/labels; neither supplies a validated clinical threshold, external-test performance or deployment model. See [the completed comparison](../experiments/eyelid-exp02-results-2026-10-11.md).
- **Future product output:** the protocol suggests a low/high-risk screen. This label, threshold, calibration, and user wording are not finalized. Continuous Hb regression is also discussed in the protocol but is not implemented or established as a product requirement.
- No output may be described as a confirmed diagnosis, Hb measurement, or treatment recommendation.

## Evaluation goals and success criteria

Research evaluation should report sensitivity, specificity, accuracy, ROC-AUC where appropriate, uncertainty, class counts, and subgroup coverage. The current public evaluation also reports ITA-stratified results; its very small and imbalanced subgroups do not establish fairness. The protocol proposes additional regression/MAE, fairness, and efficiency analyses; use them only when the corresponding target, data, and method are implemented and valid.

Prototype work is ready to be evaluated when data lineage and labels are verified, image/ROI/mask alignment passes structural and human review, subject-level partitions are leakage-free, experiments are reproducible, and limitations are reported. There is no approved numeric pass/fail threshold for clinical performance. A usable screening system cannot be declared successful from the current public cross-validation or small local pilot alone.

## Non-functional and privacy requirements

- Preserve raw data; use derived copies for conversion and annotation.
- Store local patient images, ID references, row-level metadata, masks, and outputs only in approved ignored/access-controlled research storage. Never commit or share them through Git.
- Retain source hashes, filenames, modality, and subject linkage in derived manifests. Keep direct identity keys separate from pseudonymous study records.
- Make transformations deterministic where possible, document color-space assumptions, and avoid silent resizing, color conversion, or overwrite.
- The intended research workflow should function offline after dependencies and required model weights are available; network-dependent setup is not evidence of offline deployment readiness.
- Any claim of encryption, regulatory compliance, clinical safety, or secure deployment requires separate verification; the repository does not certify these properties.

## Explicitly out of scope

- Replacing CBC/laboratory testing or making autonomous diagnostic, triage, or treatment decisions.
- Claiming clinical deployment, prospective validation, or external-test performance from existing artifacts.
- Treating an unreviewed automated ROI, glare mask, or segmentation proposal as ground truth.
- Mixing public and local records without an explicit, documented experiment and provenance controls.
- Inventing or imputing Hb values, subject attributes, image links, or annotations.
- Building a production mobile application, clinical device, or patient-facing workflow as part of the current prototype.
- Making causal claims from LightGBM feature gain or demographic neutrality claims from underpowered subgroup tests.

## Current status snapshot

- Public inner-eyelid feature + LightGBM baseline: implemented; current 184-row cohort has 10-fold stratified CV results documented in `docs/models/MODEL_CARD.md` and `docs/models/FINDINGS_AND_EVALUATION.md`. These are not external-test results.
- Local live-Sheet metadata: frozen and provenance-linked (140 subjects; 140 eyelid and 135 nail image references). A WHO-2024-derived adult cohort contains 119 labeled subjects (19 anemia, 100 non-anemia); source Hb fields are retained. The label run relies on the project owner's non-pregnancy attestation outside the Sheet.
- Local manual annotations: consolidated under `data/raw/local/annotations/`; the current annotation manifest records 150 approved anatomical ROI/mask pairs (140 eyelid, 10 fingernail) and 125 pending fingernail annotations. Approval is for anatomical boundaries, not clinician validation or glare exclusion. Glare review is separate and unresolved.
- Local pixel segmentation: experimental ExtraTrees proposals and a SAM 2.1 nail-plate assistance pilot are available. The ten-subject SAM pilot is promising for assisted tracing but shows boundary leakage; all 116 generated nail candidates remain unverified. A full-frame public-trained eyelid U-Net development check scored mean Dice 0.0027 across 111 local train/validation subjects; it is not suitable for unattended local use. All 140 reviewed eyelid masks remain the local references; no automated prediction is approved as ground truth.
- Local anemia modeling: three independent manual-eyelid experiments completed on the same 119 eligible subjects/five folds. Public-to-local transfer did not improve detection. A separately nested LAB threshold analysis detects7/19 at 84% specificity, compared with1/19 for selected transfer. No untouched classifier holdout exists; see the verified Experiment 3 results.
- Public-anemia-to-local classification transfer learning: implemented as independent experimental work, with verified negative results; this does not establish a usable screening system. Multimodal classification, mobile application and prospective/external validation remain unimplemented.
