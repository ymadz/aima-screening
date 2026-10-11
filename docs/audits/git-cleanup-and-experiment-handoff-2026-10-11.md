# Git cleanup and experiment handoff — 2026-10-11

The original index was inspected before selective staging. All 26 initially modified/untracked paths were reviewed and grouped by implemented purpose, then committed before preservation refs and the hybrid worktree were created. No reset, clean, stash, history rewrite, push or upload was used.

## Commits and complete file ownership

### `58881dde8625a6f8221ffcbf7b315a270b3fca49` — feat(data): validate frozen local labels and subject folds

```text
A	src/local_metadata.py
A	src/local_who2024_cohort.py
A	tests/test_local_who2024_cohort.py
```

### `d80c8c35ed34f6b9be42b91a6a96030209c20281` — feat(annotations): register reviewed ROI manifests and queues

```text
A	src/local_annotation_manifest.py
A	src/register_local_annotation_reviews.py
A	tests/test_local_annotation_manifest.py
```

### `b14eee90921fc7e4a6280df5ae3f1fdd022556fc` — feat(preprocessing): render provenance-linked annotation TIFFs

```text
M	requirements.txt
A	src/render_local_annotation_tiffs.py
```

### `0cf891eb48bebc08d565a036b10f867ce4025f78` — feat(segmentation): add manual-mask proposal experiment

```text
M	requirements.txt
A	src/segmentation_prototype.py
```

### `e9b091bfc714c4dcda8e78c859fb7f844cbe52e1` — feat(segmentation): evaluate public models on local development subjects

```text
A	src/local_segmentation_pilot.py
A	tests/test_local_segmentation_pilot.py
```

### `47d8fc79782a27cbefef407c4e18a21015d18ff6` — feat(evaluation): reproduce public classification baseline

```text
A	src/reproduce_public_baseline.py
```

### `31c29469964bc5b591e616d408cb6affb858715a` — feat(classifier): add verified manual-eyelid CIELAB baseline

```text
M	README.md
A	docs/experiments/local-eyelid-classifier.md
A	src/local_eyelid_classifier.py
A	src/local_eyelid_readiness.py
A	src/local_eyelid_validation.py
A	tests/test_local_eyelid_classifier.py
```

### `cdcd91f594a43e6939ee02e59c1427e94c71a88b` — docs(project): organize verified research references and workflows

```text
M	AGENTS.md
D	FINDINGS_AND_EVALUATION.md
D	LOCAL_PHOTO_IMPORT.md
D	MODEL_CARD.md
M	README.md
A	docs/README.md
D	docs/agent-rules/git-policy.md
A	docs/audits/segmentation-data-audit-2026-10-08.md
A	docs/core/ARCHITECTURE.md
A	docs/core/ARCHITECTURE_ESSENTIALS.md
A	docs/core/PRD.md
A	docs/core/PROJECT_LOG.md
A	docs/experiments/sam2-nail-pilot-2026-10-09.md
A	docs/models/FINDINGS_AND_EVALUATION.md
A	docs/models/MODEL_CARD.md
A	docs/policies/git-policy.md
D	docs/sam2-nail-pilot-2026-10-09.md
D	docs/segmentation-data-audit-2026-10-08.md
A	docs/workflows/LOCAL_PHOTO_IMPORT.md
A	docs/workflows/local-annotation-workflow.md
```

### `4bf6e2577d02b5e3338e1937140b7eaff31369b4` — feat(experiments): archive verified local CIELAB baseline

```text
M	docs/core/PROJECT_LOG.md
A	docs/experiments/eyelid-exp01-preservation.md
A	src/preserve_local_experiment.py
A	tests/test_preserve_local_experiment.py
```

### `68720dfc41bc491c75bbd43c8807cb4bac26f1c6` — feat(preprocessing): extract frozen manual-eyelid hybrid features

```text
M	docs/core/ARCHITECTURE.md
M	docs/core/PROJECT_LOG.md
A	docs/experiments/eyelid-exp02-hybrid.md
A	src/local_eyelid_hybrid_features.py
A	tests/test_local_eyelid_hybrid_features.py
```

### `c4c77c7808799c35f09a3e89dc9e7ce720b635f1` — feat(classifier): add nested hybrid CV and paired evaluation

```text
M	docs/core/ARCHITECTURE.md
M	docs/core/PROJECT_LOG.md
M	docs/experiments/eyelid-exp02-hybrid.md
A	src/local_eyelid_hybrid_evaluation.py
A	src/local_eyelid_hybrid_model.py
A	tests/test_local_eyelid_hybrid_evaluation.py
A	tests/test_local_eyelid_hybrid_model.py
```

### `01cb856842a637ff8f4ba3ece246e18e904975d7` — feat(experiments): save and reproduce private hybrid CV runs

```text
M	docs/core/ARCHITECTURE.md
M	docs/core/PROJECT_LOG.md
M	docs/experiments/eyelid-exp02-hybrid.md
A	src/local_eyelid_hybrid_run.py
A	tests/test_local_eyelid_hybrid_run.py
```

### `9d88cc750dc1d176e21d947a2f6bd40f054ac60c` — fix(experiments): limit native threads during hybrid reproduction

```text
M	docs/core/ARCHITECTURE.md
M	docs/core/PROJECT_LOG.md
M	docs/experiments/eyelid-exp02-hybrid.md
M	src/local_eyelid_hybrid_run.py
M	tests/test_local_eyelid_hybrid_run.py
```

The final aggregate-result documentation milestone includes `README.md`, `docs/README.md`, `docs/core/ARCHITECTURE.md`, `docs/core/PRD.md`, `docs/core/PROJECT_LOG.md`, `docs/experiments/eyelid-exp01-preservation.md`, `docs/experiments/eyelid-exp02-hybrid.md`, `docs/experiments/eyelid-exp02-results-2026-10-11.md` and this handoff. It records completed preservation/comparison, navigation and factual status updates. Its hash is supplied in the final response and Git log. No requirements or clinical acceptance targets are changed.

## Checks and retained private work

Initial full tests: 49 passed. Preservation suite: 51 passed. Isolated hybrid suite: 58 passed, three skips because the public Eyes Defy Anemia source is absent from the linked worktree; all ten hybrid tests passed. Per-feature cohort (4), annotation (4), segmentation pilot (2), classifier (31), preservation (2), hybrid feature (3), model/evaluation (5), and artifact runner (2) checks passed. Rendering and segmentation prototype synthetic smokes and public duplicate rejection passed before their commits. Compilation and whitespace checks passed. Source/data integrity, archive checksum/model reproduction, nested split/model/threshold verification and independent metrics/embedding checks are recorded privately.

Final main checkout remains at preservation commit `4bf6e2577d02b5e3338e1937140b7eaff31369b4` on `main`; the Experiment 2 worktree remains on `experiment/eyelid-exp02-hybrid`. Both are expected clean after the final documentation commit, and final status is checked separately. Preservation branch/tag identify the baseline.

All original audited work was committed. Patient-derived outputs, annotation sources, the local archive, audit receipts and model artifacts are intentionally excluded from Git. The unrelated Q&A worktree at `~/.codex/worktrees/10b0/aima-screening` retains its four untracked documents (PRD.md, PROJECT_LOG.md, architecture.md, architecture essentials.md); they belong to that chat and were untouched. The Q&A's completed shared documentation reorganization was committed only after ownership verification and explicit owner approval.

No new commit contains raw/private images, clinical rows, patient prediction tables, masks/ROIs, checkpoints, credentials or secrets. Existing public summary/row tables were not changed. Path/extension/index checks and inspected changes verify that the new commits contain implementation, synthetic tests, requirements and safe documentation only.

Experiment 1: 97 checksum-covered archive files, ten reproduced models, 41 original hashes unchanged. Experiment 2: 86 checksum-covered files plus manifest, 35 reproduced pipelines, 119 OOF rows, 493 source hashes rechecked. All private experiment/archive descendants have no group/other permissions. No large patient data were copied.

The standalone verifier runtime crash was resolved with the training thread policy; its fix is isolated in `9d88cc7`. The completed original run remains unchanged and its copied older verifier works with documented one-thread environment variables. An approval-review timeout during that fix commit was resolved by the permitted single retry; no user approval is pending.

See [completed scientific comparison](../experiments/eyelid-exp02-results-2026-10-11.md) for actual performance, limitations and next-step readiness.
