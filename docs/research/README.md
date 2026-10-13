# AIMA research records

This directory is the centralized thesis-traceability layer. The core project
log remains the implementation chronology; these records connect experiments,
private evidence bundles, verified aggregate results, and manuscript-ready
language.

| Record | Purpose |
| --- | --- |
| [`RESEARCH_LOG.md`](RESEARCH_LOG.md) | Chronological research milestones, findings, decisions, limitations, and next steps. |
| [`EXPERIMENT_REGISTRY.md`](EXPERIMENT_REGISTRY.md) | One status-bearing entry per experiment or major analysis, plus consolidated comparisons. |
| [`METHODOLOGY.md`](METHODOLOGY.md) | Actual implemented methods, parameter values, splits, thresholds, and planned-method boundaries. |
| [`RESULTS.md`](RESULTS.md) | Verified performance tables, confusion counts, uncertainty, segmentation side-results, and figures. |
| [`LIMITATIONS.md`](LIMITATIONS.md) | Scientific limitations, negative evidence, hypotheses, and external-validation requirements. |
| [`THESIS_HANDOFF.md`](THESIS_HANDOFF.md) | Chapter-organized manuscript handoff with source paths for quantitative claims. |
| [`RESEARCH_DOCUMENTATION_GUIDE.md`](RESEARCH_DOCUMENTATION_GUIDE.md) | Required workflow for every future experiment and documentation update. |

Run the lightweight read-only consistency check from the repository root:

```bash
python scripts/validate_research_docs.py
```

The validator checks central documents, registry statuses/IDs, and the
existence of the non-sensitive report/metric manifests named by the registry.
It does not read or export patient-level tables and it does not train models.

Experiment5 is now completed and verified: [aggregate results and methods](../experiments/eyelid-exp05-results-2026-10-13.md). Use `python scripts/validate_research_docs.py --source-root /Users/madz/Documents/aima-screening` from this isolated worktree to resolve the read-only private artifact root.

Experiment6 is completed and verified: [complementarity, ensemble and image-audit report](../experiments/eyelid-exp06-results-2026-10-13.md). Equal averaging gained one detection and five false positives; the training-selected weighted policy gained no net detections. The simpler robust color candidate remains the practical hypothesis for independent confirmation. Run the existing validator with `--source-root /Users/madz/Documents/aima-screening` to resolve private artifacts from this isolated branch.

Experiment 7 is completed and independently reproduced:
[normal-only color novelty and public transfer](../experiments/eyelid-exp07-results.md).
All six local models detect 4/19 anemia cases at the primary normal-only
target, versus preserved robust color's 10/19. Public transfer did not establish
a benefit. From the retained Experiment 7 worktree, use the same `--source-root`
argument for historical artifacts; the validator resolves Experiment 7's
separate private bundle locally and checks all 87 aggregate report rows.
