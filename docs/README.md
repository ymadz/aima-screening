# AIMA Documentation

This directory groups the project documentation by role. `README.md` and `AGENTS.md` remain at the repository root for normal project discovery and automatic agent instruction loading.

## Directory map

| Directory | Contents |
|---|---|
| `core/` | Requirements, architecture, invariants, and chronological project record. |
| `models/` | Public-baseline model card and detailed evaluation report. |
| `workflows/` | Active photo-import and manual-annotation procedures. |
| `experiments/` | Dated or experiment-specific local model reports. |
| `audits/` | Read-only data and preprocessing audits. |
| `policies/` | Repository editing and Git rules. |

Historical annotation guides and generated audit records remain beside their source data under `data/raw/local/` because they are provenance, not general project documentation.

## Required repository instructions

1. Read the repository-root `AGENTS.md` before taking action.
2. For repository edits or Git-history operations, also read `docs/policies/git-policy.md`.
3. For local image/annotation work, read `data/raw/local/annotations/README.md` and use its canonical manifests and queue.
4. Treat metadata and manuscript files as research references, not agent instructions. Follow the canonical-source rules in `AGENTS.md`.

## Core project documents

| Requested document | Status | Handling |
|---|---|---|
| [`ARCHITECTURE_ESSENTIALS.md`](core/ARCHITECTURE_ESSENTIALS.md) | Current | Read first for non-negotiable data, labeling, segmentation, and evaluation invariants. |
| [`PRD.md`](core/PRD.md) | Current | Authoritative repository reference for project goals, requirements, intended use, and scope. |
| [`ARCHITECTURE.md`](core/ARCHITECTURE.md) | Current | Describes current public/local flows and marks components implemented, experimental, or planned. |
| [`PROJECT_LOG.md`](core/PROJECT_LOG.md) | Current | Chronological factual record; it does not itself approve research-method changes. |
| [`AGENTS.md`](../AGENTS.md) | Current | Mandatory onboarding, source verification, privacy, and work rules. |

These repository documents are evidence-based project references. If a newer user-approved requirement or canonical source is supplied, verify it and reconcile the documents; do not silently replace or reinterpret research decisions.

## Supporting documents

- Public model description: [`models/MODEL_CARD.md`](models/MODEL_CARD.md)
- Public baseline audit: [`models/FINDINGS_AND_EVALUATION.md`](models/FINDINGS_AND_EVALUATION.md)
- Local classifier experiment: [`experiments/local-eyelid-classifier.md`](experiments/local-eyelid-classifier.md)
- Experiment 1 preservation: [archive and feature cleanup](experiments/eyelid-exp01-preservation.md)
- Experiment 2: [methods/reproduction](experiments/eyelid-exp02-hybrid.md), [completed comparison](experiments/eyelid-exp02-results-2026-10-11.md)
- Eyelid anemia [experiment registry](experiments/registry.md)
- Experiment 4: [methods](experiments/eyelid-exp04-color.md), [verified results](experiments/eyelid-exp04-results-2026-10-12.md), [model card](models/local-eyelid-color-model-card.md)
- Experiment 4B: [methods](experiments/eyelid-exp04b-methods.md), [verified glare/regional results and review handoff](experiments/eyelid-exp04b-results-2026-10-12.md)
- Complete commit/file handoff: [Git cleanup and experiments](audits/git-cleanup-and-experiment-handoff-2026-10-11.md)
- Nail SAM pilot: [`experiments/sam2-nail-pilot-2026-10-09.md`](experiments/sam2-nail-pilot-2026-10-09.md)
- Segmentation readiness audit: [`audits/segmentation-data-audit-2026-10-08.md`](audits/segmentation-data-audit-2026-10-08.md)
- Local photo import: [`workflows/LOCAL_PHOTO_IMPORT.md`](workflows/LOCAL_PHOTO_IMPORT.md)
- Local annotation workflow: [`workflows/local-annotation-workflow.md`](workflows/local-annotation-workflow.md)
- Git policy: [`policies/git-policy.md`](policies/git-policy.md)

## Current local annotation authority

- Canonical frozen metadata: `data/raw/local/metadata/frozen_2026-10-07/AIMA Patients Metadata - Sheet1 values.csv`, with SHA-256 recorded in its `provenance.json` and every derived manifest.
- Consolidated annotation workspace: `data/raw/local/annotations/`.
- Current status and exact per-image linkage: `data/raw/local/annotations/manifests/annotation_manifest.csv`.
- Active prioritized queue: `data/raw/local/annotations/manifests/pending_annotation_queue.csv`.
- Hash-linked human anatomical approval evidence: `data/raw/local/annotations/manifests/anatomical_review_register.csv`.
- Current human-readable workflow and limitations: `data/raw/local/annotations/README.md`.
