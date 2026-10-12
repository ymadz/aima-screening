# Research documentation guide

Every future AIMA experiment must leave a recoverable evidence trail before
its results are discussed as research findings. This guide is the operating
contract for `docs/research/RESEARCH_LOG.md`,
`docs/research/EXPERIMENT_REGISTRY.md`, `METHODOLOGY.md`, `RESULTS.md`,
`LIMITATIONS.md`, and `THESIS_HANDOFF.md`.

## Before starting an experiment

1. Assign a unique ID, for example `AIMA-EXP04B` or
   `SEG-LOCAL-EYELID-2026-11`.
2. Read `AGENTS.md`, the four `docs/core/` documents, and
   `docs/policies/git-policy.md` in the required order.
3. Check `git worktree list`, branch status, and active task/worktree ownership.
   Do not modify another active worktree without confirming it is safe.
4. Verify the canonical data source exists. Record modified time, size, and
   SHA-256 before reading it. Do not use older exports as authority.
5. Decide whether the work is implementation, an experiment, a documentation
   correction, or a planned proposal. Do not describe a proposal as a result.
6. Obtain coordinator approval for research-level decisions involving labels,
   inclusion/exclusion, segmentation truth, splits, thresholds, or comparison
   methodology.

## Required experiment bundle

Use a new, versioned ignored output directory. Never overwrite a previous run.
At minimum, save:

- configuration with experiment ID, date, code commit, runtime/dependency
  versions, random seeds, model parameters, preprocessing, features, split,
  threshold, and stopping/selection rules;
- dataset/cohort provenance, source paths, source hashes, inclusion/exclusion
  counts, label policy, and subject-level grouping;
- image/ROI/mask linkage and hashes where applicable;
- split ledger showing that a subject appears in only one validation partition;
- model checkpoints or saved fold models in private ignored storage;
- private predictions and row-level outputs in ignored storage only;
- aggregate metrics, confusion matrices, uncertainty estimates, and detected /
  missed positive counts;
- execution logs, checksums, and an independent or read-only reproduction
  record that does not silently refit or overwrite the run;
- a human-readable report with methods, results, findings, limitations, and
  next step;
- figure files only when they are safe to preserve and their provenance is
  recorded. Patient-derived figures remain private.

## Required central-record updates

After verification, update the records in this order:

1. Add or update one registry entry with status
   `COMPLETED_VERIFIED`, `COMPLETED_UNVERIFIED`, `IN_PROGRESS`, `PLANNED`, or
   `BLOCKED`.
2. Add a dated entry to `RESEARCH_LOG.md` containing date, ID, objective,
   dataset, changes, methods, results, findings, limitations, decisions,
   commit/tag, artifact paths, and next step.
3. Update `METHODOLOGY.md` only when the implementation method actually
   changes; label planned methods explicitly.
4. Update `RESULTS.md` with exact values from the verified artifact. Never
   round away a meaningful distinction between threshold policies or evaluation
   protocols.
5. Update `LIMITATIONS.md` when the result adds a limitation, negative result,
   or external-validation requirement.
6. Update `THESIS_HANDOFF.md` with the manuscript-ready explanation and a
   source path/experiment ID for every quantitative claim.
7. Run `python scripts/validate_research_docs.py` and relevant project tests.
8. Run `git diff --check`, inspect the staged diff, and make feature-based
   Conventional Commits. Never stage the entire worktree blindly.

## Registry entry template

```text
ID/title:
Status:
Research question:
Git branch / commit / tag:
Worktree path:
Dataset/cohort and source hash:
Architecture:
Feature configuration:
Preprocessing version:
Training approach:
Cross-validation and split group:
Threshold-selection method:
Primary metrics and confusion counts:
Reproducibility status:
Artifact location:
Known limitations:
Next step:
```

## Result-writing rules

- Use exact verified values and state the aggregation: fold mean, pooled OOF,
  outer CV, inner selection, or independent test.
- Keep fixed 0.5, training-selected, public-selected, and predefined matched
  thresholds in separate rows.
- Keep Exp4’s predefined robust-feature comparison separate from its primary
  nested-selected policy.
- Do not call a CV fold an external test. Do not call an internal threshold a
  clinical operating threshold.
- Report both detected and missed positive cases when a confusion matrix is
  available.
- State when precision is undefined, when a metric is not reported, and when
  an unannotated proposal has no truth metric. Do not replace missing values
  with zero or a narrative estimate.
- Distinguish observed facts, owner decisions, hypotheses, and future work.
- Keep negative or unsuccessful approaches; they are part of the research
  evidence.

## Privacy and Git rules

- Never commit raw images, workbooks, patient/clinical rows, identity keys,
  ROIs/masks, per-patient predictions, model checkpoints, secrets, or private
  diagnostic figures.
- Do not force-add ignored data. Store source hashes and aggregate summaries in
  tracked documentation instead.
- Preserve existing staged or uncommitted work. If ownership is unclear, leave
  the files untouched and report them.
- Do not reset, delete, overwrite, merge, rewrite history, or force-push as
  part of documentation consolidation.
- Use separate logical commits, for example:
  `docs: add research log and registry`,
  `docs: consolidate methodology and limitations`,
  `docs: consolidate verified results`,
  `docs: add thesis handoff`, and
  `test: validate research documentation`.

## Minimum handoff checklist

Before declaring an experiment preserved, confirm:

- [ ] canonical source path/hash/mtime recorded;
- [ ] code commit, branch, and worktree recorded;
- [ ] configuration, seed, split, threshold, and preprocessing recorded;
- [ ] subject-level split exclusivity checked;
- [ ] labels/Hb provenance and inclusion counts checked;
- [ ] source/ROI/mask linkage and rendering provenance checked where relevant;
- [ ] saved model/prediction/report/checksum paths recorded privately;
- [ ] metrics independently recalculated or saved-model reproduction passed;
- [ ] planned work is not described as completed;
- [ ] private/sensitive files are not staged;
- [ ] registry, log, methodology, results, limitations, and thesis handoff are
      updated as applicable;
- [ ] tests, `git diff --check`, and the documentation validator passed;
- [ ] final Git status and intentionally untouched files are reported.
