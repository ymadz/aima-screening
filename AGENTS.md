# AIMA Repository Agent Guide

These instructions apply to every agent working in this repository. Read this file before taking action. Before substantial data, preprocessing, architecture, modeling, or research work, read the core project documents in this order:

1. `docs/core/ARCHITECTURE_ESSENTIALS.md` — quick, non-negotiable invariants.
2. `docs/core/PRD.md` — authoritative project purpose, requirements, intended use, and scope.
3. `docs/core/ARCHITECTURE.md` — current implementation versus experiment and proposal.
4. `docs/core/PROJECT_LOG.md` — dated progress, decisions, results, and blockers.

Then inspect the relevant source code, tests, and task-specific guidance. For local annotation work, also read `data/raw/local/annotations/README.md` and `docs/workflows/local-annotation-workflow.md`. For any repository-file modification or Git-history operation, read and follow `docs/policies/git-policy.md`.

Referenced repository policies are mandatory. If applicable instructions conflict, stop and report the conflict rather than silently overriding them.

## Documentation authority and conflict handling

- Treat `docs/core/ARCHITECTURE_ESSENTIALS.md` as the quick source for invariants that must not be violated.
- Treat `docs/core/PRD.md` as the authoritative project-goal and product-requirement reference. Do not promote a protocol proposal into an approved requirement without the main AIMA coordinator's decision.
- Treat `docs/core/ARCHITECTURE.md` as the factual description of code and data flows. Its status labels distinguish implemented, experimental, and planned components.
- Treat `docs/core/PROJECT_LOG.md` as a chronological evidence record, not as permission to change research methods.
- Read the current protocol manuscript as research reference material when relevant. It does not override verified implementation or authorize claims that planned work is complete.
- If source files, code, generated results, and these documents conflict, do not silently choose a convenient interpretation. Verify provenance and state, then report the discrepancy and ask the main AIMA coordinator when a research decision is required.
- Do not invent requirements, labels, clinical values, annotations, approvals, results, or historical events. Separate observed facts, decisions, hypotheses, and future proposals.

## Canonical research sources

- **Frozen local metadata:** `data/raw/local/metadata/frozen_2026-10-07/AIMA Patients Metadata - Sheet1 values.csv` with adjacent `provenance.json`; authoritative finalized Sheet: `https://docs.google.com/spreadsheets/d/1YRMjNyNe7FiNrjfHp5Q8NA9fdHjRjrvvHXwLXGWBCik/edit?gid=0#gid=0`. The byte-for-byte Drive CSV export is preserved beside the frozen snapshot as `AIMA Patients Metadata - original Drive export.csv`; provenance records the freeze and source hash.
- **Current protocol manuscript:** `/Users/madz/Downloads/Research Protocol_Proposal with Cover Page.pdf`; corresponding Drive file: `https://drive.google.com/file/d/1fdWZX70_7buSyjGV6dJzb6ZYbVSHIP42/view?usp=share_link`.
- At the start of each related task, verify that the canonical file exists and record its modified time and SHA-256 **before reading it**. Do not use older `Sheet1-*` exports, the historical Downloads CSV, cached manifests, or earlier manuscript copies as authoritative. If the canonical file is unavailable or a newer user-supplied version is identified, stop and report the conflict rather than guessing.
- For every metadata audit/build, validate header schema, blank/trailing columns, duplicate subject IDs, Hb-field consistency, age/sex values, image-reference resolution, and differences from prior manifests. Never overwrite a source export. Derived manifests must record source path/hash and remain under ignored data paths.
- The non-pregnancy status is an owner attestation outside the Sheet, not a Sheet field. Preserve that distinction in any derived cohort explanation.

## Data and privacy

- `data/raw/` contains public source data and sensitive local research images/metadata; it is ignored. Never commit raw images, ID cards, workbooks, row-level clinical metadata, ROIs/masks, or patient-level predictions.
- Preserve original local images and filenames. Use derived copies for rendering/annotation; keep direct identity references separate from pseudonymous study IDs.
- Black subject-ID cards are identity-verification aids only, never model inputs.
- Verify `.gitignore` and intended paths before data work. Do not move or delete original data unless the task explicitly authorizes it and provenance/duplicate evidence is verified.
- Make no claims that the repository provides encryption, regulatory compliance, medical-device status, or secure deployment without separate evidence.

## Current local annotation workflow

- One active workspace: `data/raw/local/annotations/`. Do not recreate Batch 1 or Batch 2 as parallel active workspaces.
- Current per-image status/linkage is in `manifests/annotation_manifest.csv`; use `manifests/pending_annotation_queue.csv` for pending work.
- The exposed palpebral conjunctiva and visible nail plate are separate anatomical targets. Manual human-reviewed ROIs and masks are primary local segmentation ground truth. Glare is a separate QC layer.
- Automated segmentation is optional experimentation. Rejected/unreviewed glare candidates are not anatomical ground truth; generated proposals do not replace approved masks.
- The annotation manifest builder uses a registered image inventory and does not discover new annotations automatically. Reconcile/register files and validate them before trusting regenerated status.

## Project structure and setup

- `src/` — Python preprocessing, feature extraction, cohort/manifest building, training, and evaluation.
- `data/raw/` — ignored source data, including public archive and local imports.
- `data/processed/` — generated cohort tables and experiment artifacts; patient-level outputs remain private/ignored.
- `results/` — model predictions and feature summaries; inspect sensitivity before sharing.
- Supporting guides remain useful: `README.md`, `docs/models/MODEL_CARD.md`, `docs/models/FINDINGS_AND_EVALUATION.md`, `docs/workflows/LOCAL_PHOTO_IMPORT.md`, and `docs/workflows/local-annotation-workflow.md`.

Use Python 3.10 and the project environment:

```bash
python3.10 -m venv aima_env
source aima_env/bin/activate
python -m pip install -r requirements.txt
```

Public-data commands from repository root:

```bash
python src/dataset_builder.py
python src/train_classifier.py
python src/evaluate.py
python -m compileall src
```

The public builder expects `data/raw/dataset anemia/<country>/` with country workbooks and images. Local cohort/annotation scripts write ignored patient-derived files; inspect their CLI and current source hashes before running. Do not run a local model unless label, annotation, split, and cohort checks pass.

## Implementation and research rules

- Keep preprocessing deterministic where possible; document units, color-space assumptions, and image conversions. Use `pathlib.Path`, type hints for public functions, four-space indentation, `snake_case`, and `PascalCase` classes.
- Subject identity is the split group. Every image/record from one subject belongs in only one train/validation/test partition. Cross-validation must never be described as external-test performance.
- Missing nails do not exclude an otherwise eligible eyelid record. Keep eyelid-only, nail-only, and paired datasets/metrics distinguishable.
- Do not train on missing/unverified Hb, invalid annotations, unresolved source links, or invalid masks. Never synthesize, impute, or guess clinical values or ROIs.
- Research-level decisions include Hb eligibility/labels, inclusion/exclusion rules, segmentation, splitting, evaluation, and model-comparison methodology. Obtain main AIMA coordinator approval before changing or committing such decisions, as required by `docs/policies/git-policy.md`.
- When a substantive implementation or approved design changes, update `docs/core/ARCHITECTURE.md`; when a meaningful result, decision, or blocker changes, add a concise dated entry to `docs/core/PROJECT_LOG.md`. Update `docs/core/PRD.md` only when an approved requirement changes. Keep `docs/core/ARCHITECTURE_ESSENTIALS.md` short and invariant-focused.
- Preserve unrelated changes already present in the worktree. Do not stage, revert, or commit another task's work. Do not create a commit unless explicitly requested or required by the task and permitted by the Git policy.

## Verification and handoff

- Inspect available tests. For Python changes, at minimum run `python -m compileall src` and relevant tests/smoke checks.
- For data changes, verify source hashes, schema, uniqueness, resolved paths, modality counts, class counts, label provenance, mask integrity, and subject-level split exclusivity. Keep patient-level results in ignored paths.
- Run `git diff --check` for documentation/code edits. Report files changed, checks and results, unresolved blockers, and whether a commit was created. Do not claim validation beyond checks actually performed.
