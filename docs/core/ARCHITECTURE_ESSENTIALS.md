# AIMA Architecture Essentials

Read this short reference before work that touches data, preprocessing, annotations, labels, splits, or models. For goals see [PRD.md](PRD.md); for implementation details and status see [ARCHITECTURE.md](ARCHITECTURE.md); for dated decisions/results see [PROJECT_LOG.md](PROJECT_LOG.md).

## Non-negotiable invariants

- **Canonical sources:** use the frozen live-Sheet export at `data/raw/local/metadata/frozen_2026-10-07/` and its recorded SHA-256/provenance. Verify source mtime/hash before reading a canonical source. The older Downloads CSV and earlier metadata exports are historical; derived manifests are usable only for their documented role and source snapshot.
- **Provenance:** carry `subject_id`, modality, exact original filename, source path, and source/hash lineage through every derivative. For public rows, use the composite `(country, subject_id)` key; numeric IDs repeat across countries.
- **Preserve originals:** never alter or rename raw local images. Work with derived copies; keep patient-derived files in ignored/access-controlled paths, never Git.
- **Separate data sources:** public and local cohorts, label policies, runs, and claims stay distinguishable. Do not call public cross-validation local validation or external testing.
- **Anatomical target:** eyelid = visible exposed palpebral conjunctiva; nail = visible nail plate. They are distinct modalities and distinct tasks.
- **Manual ground truth:** approved human anatomical Fiji ROIs and reviewed binary masks are the local segmentation reference. Automated masks are proposals only and cannot silently replace or overwrite approved manual masks.
- **Glare is separate:** glare annotations/exclusions are a separate QC layer. Rejected/overinclusive glare candidates and unvalidated pilot rules are not anatomical ground truth and are not accepted glare masks.
- **Hb and labels:** retain original recorded Hb value/field and source. Store normalized Hb and derived labels as additional, versioned fields. Do not guess, impute, or introduce synthetic clinical values. Missing or unverified Hb means no supervised label.
- **Subject-level evaluation:** all records from one subject stay in one train/validation/test fold. The existing public table has one row per `(country, subject_id)`, but its implementation splits by row rather than a group-aware API; preserve that limitation in claims and use explicit subject groups for new multi-image experiments. Fit learned preprocessing only within training folds and recheck class counts for each modality subset.
- **Scope of claims:** cross-validation is not external-test performance. Report cohort size, label balance, uncertainty, and subgroup limitations. The local manual-eyelid LAB experiment has internal five-fold results only; it has no untouched classifier holdout or external validation.
- **Modality availability:** missing nail data must not exclude an otherwise eligible subject from eyelid-only work. Eyelid-only, nail-only, and paired runs must remain explicitly separate.
- **Do not train on invalid inputs:** each experiment must have verified labels, resolved image/source/subject links, valid inputs, and appropriate partitions. A local image model additionally requires the relevant anatomical masks to pass structural and human review. Segmentation annotation approval alone does not establish anemia-model readiness; duplicate adjudication and experiment-specific checks must also pass.

## Current workflow references

- Active annotation workspace: `data/raw/local/annotations/`; read its `README.md`, `annotation_manifest.csv`, and `pending_annotation_queue.csv` before annotation tasks.
- Human anatomical approval evidence: `data/raw/local/annotations/manifests/anatomical_review_register.csv`.
- Annotation workflow: `docs/workflows/local-annotation-workflow.md`.
- Current local cohort: `data/processed/local_cohort_who2024_2026-10-08/`; its label config, summary, provenance, and subject-fold file describe that specific derived snapshot.
- Completed manual-eyelid classifier: see `docs/experiments/local-eyelid-classifier.md`; private readiness/review evidence and internal CV results have separate versioned paths.
- Source and Git policy: follow root `AGENTS.md` and read `docs/policies/git-policy.md` before repository edits or Git-history operations.
